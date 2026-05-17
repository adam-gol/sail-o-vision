#!/usr/bin/env python3
"""
simulate_operations.py - Simulates sail-o-vision operation in SF Bay.

Runs the NMEA client against live replay data, injects synthetic camera
detections, and demonstrates the priority queue and PTZ targeting logic.

Usage:
    Terminal 1: python3 nmea_replay.py --file ~/nmea/"AIS Sample 2 San Francisco Bay NMEA0183.xml" --rate 50
    Terminal 2: python3 simulate_operations.py
"""

import time
import math
import logging
import threading
import random
from dataclasses import dataclass
from typing import Optional
from nmea_client import NMEAClient, AISContact, TTMContact

# Suppress debug logging from nmea_client for cleaner output
logging.basicConfig(level=logging.WARNING)

# --- Configuration ---
NMEA_PORT = 25000
CONTACT_EXPIRY = 120
AIS_BEARING_TOLERANCE = 5.0   # degrees — how close a bearing must be to a
                                # known contact to be considered "same target"
AIS_MAX_RANGE_NM = 10.0        # nautical miles
DETECTION_INTERVAL = 8.0       # seconds between simulated camera detections
HIGH_CONF_THRESHOLD = 0.60     # confidence >= this = Priority 1a
LOW_CONF_THRESHOLD = 0.25      # confidence >= this = Priority 1b

# --- PTZ State Machine ---
PTZ_STATE_SCANNING_AHEAD   = "SCANNING_AHEAD"
PTZ_STATE_SCANNING_FLANK   = "SCANNING_FLANK"
PTZ_STATE_VERIFYING_TARGET = "VERIFYING_TARGET"

AHEAD_DURATION  = 30.0   # seconds (shortened for simulation)
FLANK_DURATION  = 5.0    # seconds (shortened for simulation)
VERIFY_DURATION = 6.0    # seconds

# --- Simulated detections ---
# Each entry: (bearing_true, confidence, label, description)
SIMULATED_DETECTIONS = [
    (354.0, 0.88, "boat",   "Near AIS vessel 366771550 (SOG 31.7kn heading NW)"),
    (200.0, 0.72, "boat",   "Open water, no AIS contact — possible debris or unlit vessel"),
    (103.0, 0.45, "boat",   "Near AIS vessel 367305920 (stationary)"),
    (045.0, 0.31, "boat",   "Low confidence, no AIS contact — possible whitecap or small object"),
    (310.0, 0.79, "boat",   "Near AIS vessel 366985330 (stationary)"),
    (270.0, 0.65, "boat",   "Open water, no AIS contact — investigate"),
    (116.0, 0.52, "boat",   "Near AIS vessel 367396710 (stationary)"),
    (180.0, 0.83, "boat",   "Open water, no AIS contact — high confidence unknown contact"),
]


# --- Priority classification ---

@dataclass
class DetectionEvent:
    bearing: float
    confidence: float
    label: str
    description: str
    ais_contact: Optional[AISContact] = None
    ttm_contact: Optional[TTMContact] = None
    priority: str = ""
    priority_reason: str = ""


def classify_detection(bearing: float, confidence: float, label: str,
                        nmea: NMEAClient) -> DetectionEvent:
    event = DetectionEvent(bearing=bearing, confidence=confidence, label=label,
                           description="")

    # Check for known AIS contact at this bearing
    ais = None
    ttm = None
    ais_contacts = nmea.get_ais_contacts()
    ttm_contacts = nmea.get_ttm_contacts()

    for contact in ais_contacts.values():
        if contact.bearing_from_own is not None:
            diff = abs(contact.bearing_from_own - bearing) % 360
            if diff > 180:
                diff = 360 - diff
            if diff <= AIS_BEARING_TOLERANCE and contact.range_nm <= AIS_MAX_RANGE_NM:
                ais = contact
                break

    for contact in ttm_contacts.values():
        diff = abs(contact.bearing - bearing) % 360
        if diff > 180:
            diff = 360 - diff
        if diff <= AIS_BEARING_TOLERANCE:
            ttm = contact
            break

    event.ais_contact = ais
    event.ttm_contact = ttm

    # Classify priority
    if ais is None and ttm is None:
        if confidence >= HIGH_CONF_THRESHOLD:
            event.priority = "1a"
            event.priority_reason = (
                "High confidence detection, no AIS, no radar — "
                "unknown object, immediate zoom-and-verify"
            )
        else:
            event.priority = "1b"
            event.priority_reason = (
                "Low confidence detection, no AIS, no radar — "
                "queued for next PTZ slot"
            )
    elif ttm is not None and ais is None:
        event.priority = "2"
        event.priority_reason = (
            f"Radar contact (target {ttm.target_id}) but no AIS — "
            f"dark vessel, bearing {ttm.bearing:.1f}° range {ttm.distance:.2f}nm"
        )
    else:
        event.priority = "3"
        name = ais.name if ais and ais.name else f"MMSI {ais.mmsi}"
        sog = ais.sog if ais else 0
        cog = ais.cog if ais else 0
        rng = ais.range_nm if ais else 0
        event.priority_reason = (
            f"Known AIS contact: {name}, "
            f"SOG {sog:.1f}kn COG {cog:.1f}° "
            f"range {rng:.2f}nm — opportunistic visual ID"
        )

    return event


# --- PTZ simulator ---

class PTZSimulator:
    def __init__(self):
        self.state = PTZ_STATE_SCANNING_AHEAD
        self.current_bearing = 0.0    # ahead
        self.flank_side = None
        self.state_start = time.time()
        self.active_target: Optional[DetectionEvent] = None
        self.pending_queue: list = []  # sorted by priority
        self._lock = threading.RLock()

    def queue_detection(self, event: DetectionEvent):
        with self._lock:
            self.pending_queue.append(event)
            # Sort by priority: 1a first, then 1b, 2, 3
            priority_order = {"1a": 0, "1b": 1, "2": 2, "3": 3}
            self.pending_queue.sort(
                key=lambda e: (priority_order.get(e.priority, 9),
                               -e.confidence))
            print(f"\n  [QUEUE] Added Priority {event.priority} detection "
                  f"at {event.bearing:.1f}° — queue depth: "
                  f"{len(self.pending_queue)}")

    def update(self):
        now = time.time()
        elapsed = now - self.state_start

        with self._lock:
            if self.state == PTZ_STATE_SCANNING_AHEAD:
                if self.pending_queue:
                    target = self.pending_queue.pop(0)
                    self._start_verify(target, now)
                elif elapsed >= AHEAD_DURATION:
                    self._start_flank(now)

            elif self.state == PTZ_STATE_SCANNING_FLANK:
                if elapsed >= FLANK_DURATION:
                    print(f"\n  [PTZ] Flank scan complete, "
                          f"returning to SCANNING_AHEAD")
                    self.state = PTZ_STATE_SCANNING_AHEAD
                    self.current_bearing = 0.0
                    self.state_start = now

            elif self.state == PTZ_STATE_VERIFYING_TARGET:
                if elapsed >= VERIFY_DURATION:
                    self._complete_verify(now)

    def _start_flank(self, now):
        self.flank_side = "PORT" if not self.flank_side or \
                          self.flank_side == "STARBOARD" else "STARBOARD"
        bearing = 270.0 if self.flank_side == "PORT" else 90.0
        self.state = PTZ_STATE_SCANNING_FLANK
        self.current_bearing = bearing
        self.state_start = now
        print(f"\n  [PTZ] → SCANNING_{self.flank_side} "
              f"(bearing {bearing:.0f}°, {FLANK_DURATION:.0f}s)")

    def _start_verify(self, target: DetectionEvent, now):
        self.state = PTZ_STATE_VERIFYING_TARGET
        self.active_target = target
        self.current_bearing = target.bearing
        self.state_start = now
        zoom = 30 if target.priority in ("1a", "1b") else 15
        print(f"\n  [PTZ] → VERIFYING_TARGET Priority {target.priority} "
              f"at {target.bearing:.1f}° zoom {zoom}x")
        print(f"         {target.priority_reason}")

    def _complete_verify(self, now):
        target = self.active_target
        # Simulate verification outcome — Priority 1 usually confirms,
        # Priority 1b sometimes doesn't
        if target.priority == "1b" and random.random() < 0.4:
            outcome = "NOT CONFIRMED"
            alert = False
        else:
            outcome = "CONFIRMED"
            alert = True

        print(f"\n  [PTZ] Verification complete: {outcome}")
        if alert:
            print(f"  *** ALERT: {target.label.upper()} detected "
                  f"at {target.bearing:.1f}° — Priority {target.priority} ***")

        self.state = PTZ_STATE_SCANNING_AHEAD
        self.current_bearing = 0.0
        self.active_target = None
        self.state_start = now
        print(f"  [PTZ] → SCANNING_AHEAD")

    def status(self) -> str:
        with self._lock:
            elapsed = time.time() - self.state_start
            queue_str = (f", queue: {len(self.pending_queue)}"
                        if self.pending_queue else "")
            return (f"State: {self.state} | "
                    f"Bearing: {self.current_bearing:.0f}° | "
                    f"Elapsed: {elapsed:.1f}s{queue_str}")


# --- Main simulation ---

def main():
    print("=" * 65)
    print("  SAIL-O-VISION OPERATIONS SIMULATION — San Francisco Bay")
    print("=" * 65)
    print(f"\nOwn vessel: 37.8401°N, 122.4067°W (center of SF Bay)")
    print(f"Heading: 000° (North)")
    print(f"Scan cycle: {AHEAD_DURATION:.0f}s ahead, "
          f"{FLANK_DURATION:.0f}s each flank")
    print(f"\nStarting NMEA client...")
    print(f"Make sure nmea_replay.py is running in another terminal.\n")

    nmea = NMEAClient(port=NMEA_PORT, contact_expiry=CONTACT_EXPIRY)
    nmea.start()

    # Change the wait section to:
    print("Waiting 5s for contact picture to build up...")
    for i in range(5):
        time.sleep(1)
        ais = nmea.get_ais_contacts()
        own = nmea.get_own_vessel()
        print(f"  {i+1}s: own={own.lat:.4f},{own.lon:.4f} ais_contacts={len(ais)}")

    print("DEBUG: past wait loop")  # add this line

    # Show initial contact picture
    own = nmea.get_own_vessel()
    ais = nmea.get_ais_contacts()
    print(f"\n--- Initial Contact Picture ---")
    print(f"Own vessel: {own.lat:.4f}°N, {own.lon:.4f}°W "
          f"hdg={own.heading:.1f}° sog={own.sog:.1f}kn")
    print(f"AIS contacts: {len(ais)}")
    for mmsi, c in sorted(ais.items(),
                           key=lambda x: x[1].bearing_from_own or 999):
        if c.bearing_from_own is not None:
            print(f"  MMSI {mmsi:10d} {c.name or '(unnamed)':20s} "
                  f"bearing={c.bearing_from_own:5.1f}° "
                  f"range={c.range_nm:5.2f}nm "
                  f"sog={c.sog:.1f}kn")

    print(f"\n--- Starting PTZ Simulation ---")
    print(f"Injecting {len(SIMULATED_DETECTIONS)} synthetic camera detections\n")

    ptz = PTZSimulator()
    detection_index = 0
    last_detection_time = time.time()
    last_status_time = time.time()

    try:
        while True:
            now = time.time()

            # Update PTZ state machine
            ptz.update()

            # Inject next synthetic detection
            if (now - last_detection_time >= DETECTION_INTERVAL and
                    detection_index < len(SIMULATED_DETECTIONS)):
                bearing, conf, label, desc = \
                    SIMULATED_DETECTIONS[detection_index]
                detection_index += 1
                last_detection_time = now

                print(f"\n{'─' * 65}")
                print(f"[CAMERA] Detection: {label} at {bearing:.1f}° "
                      f"confidence={conf:.2f}")
                print(f"         {desc}")

                event = classify_detection(bearing, conf, label, nmea)
                event.description = desc
                ptz.queue_detection(event)

            # Print PTZ status periodically
            if now - last_status_time >= 5.0:
                last_status_time = now
                print(f"\n  [STATUS] {ptz.status()}")

            # Stop after all detections processed and queue empty
            if (detection_index >= len(SIMULATED_DETECTIONS) and
                    not ptz.pending_queue and
                    ptz.state == PTZ_STATE_SCANNING_AHEAD):
                time.sleep(10)  # let last verify complete
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    print(f"\n{'=' * 65}")
    print("  Simulation complete")
    print(f"{'=' * 65}")
    nmea.stop()


if __name__ == '__main__':
    main()
