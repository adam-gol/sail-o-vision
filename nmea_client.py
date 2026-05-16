#!/usr/bin/env python3
"""
nmea_client.py - NMEA 0183 UDP listener for sail-o-vision.

Receives NMEA 0183 sentences (AIS VDM, radar TTM, own vessel RMC/HDG)
and maintains a live contact picture for the PTZ priority queue.

Designed for PredictWind DataHub Pro UDP/TCP stream but works with
any NMEA 0183 UDP broadcast source including nmea_replay.py.
"""

import socket
import threading
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from pyais import decode as ais_decode
from pyais.exceptions import UnknownMessageException
import pynmea2

logger = logging.getLogger(__name__)

# --- Configuration ---
NMEA_UDP_PORT = 25000
CONTACT_EXPIRY_SECONDS = 120   # contacts older than this are removed
CLEANUP_INTERVAL = 30          # how often to run expiry cleanup


# --- Data structures ---

@dataclass
class AISContact:
    mmsi: int
    lat: float
    lon: float
    sog: float          # speed over ground, knots
    cog: float          # course over ground, degrees true
    heading: int        # true heading, degrees (511 = unavailable)
    name: str = ""
    ship_type: int = 0
    last_update: float = field(default_factory=time.time)

    # Computed fields (filled in by bearing calculator)
    bearing_from_own: Optional[float] = None  # degrees true
    range_nm: Optional[float] = None


@dataclass
class TTMContact:
    target_id: int
    bearing: float      # degrees true
    distance: float     # nautical miles
    course: float       # degrees true
    speed: float        # knots
    cpa: float          # closest point of approach, nm
    tcpa: float         # time to CPA, minutes
    status: str = ""
    last_update: float = field(default_factory=time.time)

@dataclass
class OwnVessel:
    lat: float = 0.0
    lon: float = 0.0
    sog: float = 0.0
    cog: float = 0.0
    heading: float = 0.0          # true heading degrees
    magnetic_heading: float = 0.0
    variation: float = 0.0        # magnetic variation, positive=East
    last_update: float = 0.0

# --- NMEA Client ---

class NMEAClient:
    def __init__(self,
                 port: int = NMEA_UDP_PORT,
                 contact_expiry: int = CONTACT_EXPIRY_SECONDS):

        self.port = port
        self.contact_expiry = contact_expiry

        self.own_vessel = OwnVessel()
        self.ais_contacts: Dict[int, AISContact] = {}   # keyed by MMSI
        self.ttm_contacts: Dict[int, TTMContact] = {}   # keyed by target ID

        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cleanup_thread = None

        # Multi-part AIS message buffer: key = (channel, seq_id)
        self._ais_buffer: Dict[tuple, list] = {}

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="nmea-listener")
        self._thread.start()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="nmea-cleanup")
        self._cleanup_thread.start()
        logger.info(f"NMEA client started, listening on UDP port {self.port}")

    def stop(self):
        self._running = False

    # --- Public interface ---

    def get_own_vessel(self) -> OwnVessel:
        with self._lock:
            return self.own_vessel

    def get_ais_contacts(self) -> Dict[int, AISContact]:
        with self._lock:
            return dict(self.ais_contacts)

    def get_ttm_contacts(self) -> Dict[int, TTMContact]:
        with self._lock:
            return dict(self.ttm_contacts)

    def get_contact_at_bearing(self, bearing: float,
                                tolerance: float = 10.0) -> Optional[object]:
        """Return any known contact within tolerance degrees of bearing."""
        with self._lock:
            for contact in self.ais_contacts.values():
                if contact.bearing_from_own is not None:
                    diff = abs(contact.bearing_from_own - bearing) % 360
                    if diff > 180:
                        diff = 360 - diff
                    if diff <= tolerance:
                        return contact
            for contact in self.ttm_contacts.values():
                diff = abs(contact.bearing - bearing) % 360
                if diff > 180:
                    diff = 360 - diff
                if diff <= tolerance:
                    return contact
        return None

    # --- Internal ---

    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        sock.bind(('', self.port))

        logger.info(f"Listening on UDP port {self.port}")

        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                sentences = data.decode('ascii', errors='ignore').strip()
                for sentence in sentences.splitlines():
                    sentence = sentence.strip()
                    if sentence:
                        self._process_sentence(sentence)
            except socket.timeout:
                continue
            except Exception as e:
                logger.warning(f"Receive error: {e}")

        sock.close()

    def _process_sentence(self, sentence: str):
        try:
            if sentence.startswith('!AIVDM') or sentence.startswith('!AIVDO'):
                self._handle_ais(sentence)
            elif sentence.startswith('$') and 'TTM' in sentence:
                self._handle_ttm(sentence)
            elif sentence.startswith('$') and 'RMC' in sentence:
                self._handle_rmc(sentence)
            elif sentence.startswith('$') and ('HDG' in sentence or 'HDT' in sentence):
                self._handle_hdg(sentence)
        except Exception as e:
            logger.debug(f"Parse error on '{sentence}': {e}")

    def _handle_ais(self, sentence: str):
        """Handle AIS VDM sentences, including multi-part messages."""
        parts = sentence.split(',')
        if len(parts) < 7:
            return

        total_parts = int(parts[1])
        part_num = int(parts[2])
        seq_id = parts[3]
        channel = parts[4]

        if total_parts == 1:
            # Single part message - decode immediately
            self._decode_ais([sentence])
        else:
            # Multi-part - buffer until complete
            key = (channel, seq_id)
            with self._lock:
                if key not in self._ais_buffer:
                    self._ais_buffer[key] = []
                self._ais_buffer[key].append(sentence)
                if len(self._ais_buffer[key]) == total_parts:
                    parts_to_decode = self._ais_buffer.pop(key)
                    self._decode_ais(parts_to_decode)

    def _decode_ais(self, sentences: list):
        try:
            msg = ais_decode(*[s.encode() for s in sentences])

            # Message types 1, 2, 3 = position report Class A
            # Message type 18 = position report Class B
            if msg.msg_type in (1, 2, 3, 18):
                lat = float(msg.lat) if msg.lat else 0.0
                lon = float(msg.lon) if msg.lon else 0.0
                sog = float(msg.speed) if hasattr(msg, 'speed') else 0.0
                cog = float(msg.course) if hasattr(msg, 'course') else 0.0
                heading = int(msg.heading) if hasattr(msg, 'heading') else 511

                contact = AISContact(
                    mmsi=msg.mmsi,
                    lat=lat,
                    lon=lon,
                    sog=sog,
                    cog=cog,
                    heading=heading
                )

                # Update bearing from own vessel
                own = self.own_vessel
                if own.lat != 0.0 and lat != 0.0:
                    contact.bearing_from_own = self._bearing(
                        own.lat, own.lon, lat, lon)
                    contact.range_nm = self._range_nm(
                        own.lat, own.lon, lat, lon)

                with self._lock:
                    if msg.mmsi in self.ais_contacts:
                        # Preserve name from static data message
                        contact.name = self.ais_contacts[msg.mmsi].name
                    self.ais_contacts[msg.mmsi] = contact
                    logger.debug(f"AIS {msg.mmsi}: "
                                 f"lat={lat:.4f} lon={lon:.4f} "
                                 f"sog={sog:.1f} cog={cog:.1f} "
                                 f"bearing={contact.bearing_from_own}")

            # Message type 24 = static data (name)
            elif msg.msg_type == 24:
                if hasattr(msg, 'shipname') and msg.shipname:
                    name = str(msg.shipname).strip()
                    with self._lock:
                        if msg.mmsi in self.ais_contacts:
                            self.ais_contacts[msg.mmsi].name = name
                    logger.debug(f"AIS {msg.mmsi} name: {name}")

            # Message type 5 = static and voyage data (name, destination)
            elif msg.msg_type == 5:
                if hasattr(msg, 'shipname') and msg.shipname:
                    name = str(msg.shipname).strip()
                    with self._lock:
                        if msg.mmsi in self.ais_contacts:
                            self.ais_contacts[msg.mmsi].name = name
                    logger.debug(f"AIS {msg.mmsi} name: {name}")

        except UnknownMessageException:
            pass
        except Exception as e:
            logger.debug(f"AIS decode error: {e}")

    def _handle_ttm(self, sentence: str):
        """Handle TTM (Tracked Target Message) from radar."""
        try:
            msg = pynmea2.parse(sentence)
            target_id = int(msg.target_number)
            bearing = float(msg.bearing) if msg.bearing else 0.0
            distance = float(msg.distance) if msg.distance else 0.0
            course = float(msg.course) if msg.course else 0.0
            speed = float(msg.speed) if msg.speed else 0.0
            cpa = float(msg.cpa) if msg.cpa else 0.0
            tcpa = float(msg.tcpa) if msg.tcpa else 0.0
            status = str(msg.status) if hasattr(msg, 'status') else ""

            contact = TTMContact(
                target_id=target_id,
                bearing=bearing,
                distance=distance,
                course=course,
                speed=speed,
                cpa=cpa,
                tcpa=tcpa,
                status=status
            )

            with self._lock:
                self.ttm_contacts[target_id] = contact

            logger.debug(f"TTM target {target_id}: "
                         f"bearing={bearing:.1f} distance={distance:.2f}nm "
                         f"CPA={cpa:.2f}nm TCPA={tcpa:.1f}min")

        except Exception as e:
            logger.debug(f"TTM parse error: {e}")

    def _handle_rmc(self, sentence: str):
        """Handle RMC (Recommended Minimum) for own vessel position."""
        try:
            msg = pynmea2.parse(sentence)
            if msg.status == 'A':  # valid fix
                with self._lock:
                    self.own_vessel.lat = msg.latitude
                    self.own_vessel.lon = msg.longitude
                    self.own_vessel.sog = float(msg.spd_over_grnd or 0)
                    self.own_vessel.cog = float(msg.true_course or 0)
                    self.own_vessel.last_update = time.time()
                logger.debug(f"Own vessel: "
                             f"lat={msg.latitude:.4f} "
                             f"lon={msg.longitude:.4f} "
                             f"sog={self.own_vessel.sog:.1f}")
        except Exception as e:
            logger.debug(f"RMC parse error: {e}")

    def _handle_hdg(self, sentence: str):
        """Handle HDG/HDT for own vessel heading, converting to true."""
        try:
            msg = pynmea2.parse(sentence)
            if 'HDT' in sentence:
                # Already true heading
                with self._lock:
                    self.own_vessel.heading = float(msg.heading or 0)
            elif 'HDG' in sentence:
                mag_heading = float(msg.heading or 0)
                variation = 0.0
                if hasattr(msg, 'mag_var') and msg.mag_var:
                    variation = float(msg.mag_var)
                    if hasattr(msg, 'mag_var_dir') and msg.mag_var_dir == 'W':
                        variation = -variation
                true_heading = (mag_heading + variation) % 360
                with self._lock:
                    self.own_vessel.heading = true_heading
                    self.own_vessel.magnetic_heading = mag_heading
                    self.own_vessel.variation = variation
                logger.debug(f"Heading: {mag_heading:.1f}M + "
                            f"{variation:.1f} var = {true_heading:.1f}T")
        except Exception as e:
            logger.debug(f"HDG parse error: {e}")

    def _cleanup_loop(self):
        """Remove stale contacts periodically."""
        while self._running:
            time.sleep(CLEANUP_INTERVAL)
            now = time.time()
            with self._lock:
                expired_ais = [
                    mmsi for mmsi, c in self.ais_contacts.items()
                    if now - c.last_update > self.contact_expiry
                ]
                for mmsi in expired_ais:
                    logger.info(f"Expiring AIS contact {mmsi}")
                    del self.ais_contacts[mmsi]

                expired_ttm = [
                    tid for tid, c in self.ttm_contacts.items()
                    if now - c.last_update > self.contact_expiry
                ]
                for tid in expired_ttm:
                    logger.info(f"Expiring TTM contact {tid}")
                    del self.ttm_contacts[tid]

    # --- Geometry ---

    @staticmethod
    def _bearing(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
        """Calculate true bearing from point 1 to point 2."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2) -
             math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _range_nm(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
        """Calculate range in nautical miles using haversine formula."""
        R = 3440.065  # Earth radius in nautical miles
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat/2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        return R * 2 * math.asin(math.sqrt(a))


# --- Standalone test ---

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    client = NMEAClient(
        port=NMEA_UDP_PORT,
        contact_expiry=CONTACT_EXPIRY_SECONDS
    )
    client.start()

    print(f"Listening for NMEA on UDP port {NMEA_UDP_PORT}")
    print("Run nmea_replay.py in another terminal to inject test data")
    print("Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(5)
            own = client.get_own_vessel()
            ais = client.get_ais_contacts()
            ttm = client.get_ttm_contacts()

            print(f"\n--- Contact Picture ---")
            print(f"Own vessel: lat={own.lat:.4f} lon={own.lon:.4f} "
                  f"hdg={own.heading:.1f} sog={own.sog:.1f}kn")
            print(f"AIS contacts: {len(ais)}")
            for mmsi, c in sorted(ais.items()):
                print(f"  {mmsi} {c.name or '(unnamed)':20s} "
                      f"lat={c.lat:.4f} lon={c.lon:.4f} "
                      f"sog={c.sog:.1f}kn cog={c.cog:.1f}° "
                      f"bearing={c.bearing_from_own:.1f}° "
                      f"range={c.range_nm:.2f}nm"
                      if c.bearing_from_own is not None
                      else f"  {mmsi} (no own position for bearing calc)")
            print(f"TTM contacts: {len(ttm)}")
            for tid, c in sorted(ttm.items()):
                print(f"  Target {tid}: bearing={c.bearing:.1f}° "
                      f"range={c.distance:.2f}nm "
                      f"CPA={c.cpa:.2f}nm TCPA={c.tcpa:.1f}min")

    except KeyboardInterrupt:
        print("\nStopped.")
        client.stop()
