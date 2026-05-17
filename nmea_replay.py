#!/usr/bin/env python3
"""
nmea_replay.py - Replays a NMEA 0183 log file as UDP broadcast.
Simulates the PredictWind DataHub Pro TCP/UDP stream for development.

Usage: python3 nmea_replay.py [--file PATH] [--port PORT] [--rate RATE]
"""

import socket
import time
import argparse
import os
import re
import html

from typing import Optional

DEFAULT_FILE = os.path.expanduser("~/nmea/AIS Sample 1 NMEA0183.txt")
DEFAULT_PORT = 25000
DEFAULT_BROADCAST = "127.0.0.1"
DEFAULT_RATE = 10  # lines per second

import re
import html

def extract_sentence(line: str) -> Optional[str]:
    """Extract NMEA sentence from either plain text or XML-wrapped format."""
    line = line.strip()
    if not line:
        return None
    
    # XML format: <N0183Msg TimeStamp="...">!AIVDM,...</N0183Msg>
    match = re.search(r'>([!$][^<]+)<', line)
    if match:
        sentence = html.unescape(match.group(1)).strip()
        return sentence if sentence else None
    
    # Plain text format
    if line.startswith('!') or line.startswith('$'):
        return line
    
    return None

def replay(file_path, broadcast_addr, port, rate):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    delay = 1.0 / rate
    sent = 0
    skipped = 0

    print(f"Replaying {file_path}")
    print(f"Sending to {broadcast_addr}:{port} at {rate} lines/sec")
    print("Ctrl+C to stop\n")

    own_rmc = "$GPRMC,172733.00,A,3750.40600,N,12224.40200,W,0.0,0.0,150526,013.5,E,A*24\r\n"

    # Send own position immediately
    for _ in range(5):
        sock.sendto(own_rmc.encode('ascii'), (broadcast_addr, port))
        time.sleep(0.1)
    print("Sent own vessel position")

    try:
        while True:
            with open(file_path, 'r', errors='ignore') as f:
                for line in f:
                    sentence = extract_sentence(line)
                    if not sentence:
                        skipped += 1
                        continue
                    sock.sendto((sentence + '\r\n').encode('ascii'),
                                (broadcast_addr, port))
                    sent += 1

                    # Send own position every 200 sentences
                    if sent % 20 == 0:
                        sock.sendto(own_rmc.encode('ascii'),
                                    (broadcast_addr, port))

                    if sent % 100 == 0:
                        print(f"Sent {sent} sentences ({skipped} skipped)...")
                    time.sleep(delay)
            print(f"File complete, looping... ({sent} total sent)")
    except KeyboardInterrupt:
        print(f"\nStopped. {sent} sentences sent, {skipped} skipped.")
    finally:
        sock.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay NMEA log as UDP broadcast')
    parser.add_argument('--file', default=DEFAULT_FILE, help='NMEA log file path')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='UDP port')
    parser.add_argument('--broadcast', default=DEFAULT_BROADCAST, help='Broadcast address')
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE, help='Lines per second')
    args = parser.parse_args()

    replay(args.file, args.broadcast, args.port, args.rate)
