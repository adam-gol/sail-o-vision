# sail-o-vision — Design Notes

Architectural decisions and rationale, recorded here so they survive across development contexts.

---

## Camera Hardware

**FLIR M300C** — visible-light HD, 30× optical zoom, gyro-stabilized PTZ, mast-mounted at approximately 73 feet above waterline.

- Visible-light only (no thermal). COCO pretrained weights are a practical starting point for daylight detection. Fine-tuning is relevant primarily for low-light/night operation.
- 30× optical zoom is a meaningful capability advantage. The system architecture actively exploits this via a scan-wide → zoom-to-verify pipeline; do not treat the M300C as a fixed-FOV camera.
- Stabilized by AR200 AHRS. At 30× zoom, vessel pitch/roll correction matters for horizon sweep accuracy.
- If connected to NMEA 0183, the camera has the capability to track 
    - Radar Cursor Tracking, which is implemented using the NMEA Radar System Data (RSD) sentence.
    - Slew to Waypoint, which uses the NMEA Bearing and Distance to Waypoint, Great Circle (BWC) sentence.
    - Radar Tracking, which uses the NMEA Tracked Target Message (TTM) sentence.
    
    But this may not be necessary with Sail-o-Vision

---

## Compute Platform

**NVIDIA Jetson Orin Nano Super Dev Kit** — 67 TOPS, ~$150 plus NVMe storage.

Key benchmarks (measured):
- eWaSR at 192×256 input: ~53ms per frame
- YOLOv8s: ~45ms per frame
- cv2 image resize: ~0.7ms (always use cv2, not PIL — PIL resize measured at 73ms, a major bottleneck)
- Two-stage pipeline mean: ~121ms (8.3 FPS) on video file; estimated ~10 FPS on live RTSP feed

Power budget: ~83–98W continuous (M300C draws 41–56W; account for full system, not just compute).

---

## Detection Pipeline

Two-stage scan-and-verify architecture:

**Stage 1 — eWaSR** (obstacle/water/sky segmentation + blob detection with coastline filtering)
- Input: 192×256
- Purpose: wide-FOV scan to detect that *something* non-water is present
- Fine-tuned on KOLOMVERSE validation set using MobileSAM-generated pseudo-ground-truth masks
- Weights: `~/samples/eWaSR/pretrained/ewasr_kolomverse.pth`

**Stage 2 — YOLOv8s_kolomverse** (classification and bounding box on zoomed crops)
- Purpose: identify and classify what Stage 1 detected
- KOLOMVERSE dataset: 186,419 4K images, 5 classes: ship, buoy, fishnet buoy, lighthouse, wind farm

**Next optimization steps:**
- TensorRT optimization of eWaSR and YOLO (`trtexec` available on Jetson)
- RTSP live feed integration
- Own-vessel masking (when boat is built)
- Horizon scan loop for sub-pixel distant targets

---

## State Machine

States: `SCANNING_AHEAD` → `VERIFYING_TARGET` → `ALERT` → (operator acknowledgement) → action state

**Key parameters (all configurable):**
- Confirmation window before ALERT: N=3 hits above threshold in M=5 seconds (suggested starting point)
- Scan interval in ALERT state: ~30 seconds (see single-camera architecture below)
- ONVIF tilt range min/max (read from GetNode on first connect, override in config)
- Camera height above waterline (see range calculation below)

**Tunnel vision / threat queue:**
PTZ is locked on the primary (highest-priority) contact in ALERT state. The threat queue is sorted by TCPA and re-ranked continuously. Operator acknowledgement gates escalation to an action state — it does not gate re-ranking. If a new contact climbs above the current primary in TCPA priority, the system re-evaluates PTZ lock automatically without waiting for operator input.

---

## Single-Camera Architecture (Phase 1)

The M300C is the only camera. A second wide-FOV camera is deferred to Phase 2.

**In ALERT state**, the camera uses time-sliced scanning: periodic scan breaks (configurable, ~30 seconds) to update the threat queue, then returns to the locked target.

**Residual risk:** Close-in, no-AIS, no-radar contacts (debris, whales, kayaks) are not covered while the M300C is locked on a distant target. This is acceptable because those targets are slow-moving or stationary — they are not fast-developing threats on a 30-second scan cycle timescale. The close-in fast-developing threat that would be missed by this gap (a powered vessel at close range) will almost always have either an AIS signal or a radar return, removing it from the residual risk category.

**Phase 2 option:** A cheap fixed wide-FOV IP camera (~$80–100, RTSP) feeding eWaSR continuously as a tripwire layer. Does not need M300C image quality — its only job is to answer "is something non-water present?" and hand off to the M300C for verification.

---

## Contact Priority Levels

| Priority | Camera | Radar | AIS | Action |
|----------|--------|-------|-----|--------|
| 1a | High-confidence detection | No | No | Unknown object (debris, whale, unlit vessel). Immediate PTZ zoom-and-verify. |
| 1b | Low-confidence detection | No | No | Uncertain detection, no corroboration. Queue for next PTZ slot. |
| 2 | Detection | Yes | No | Radar return, no AIS — fishing vessel, small craft, AIS-dark. PTZ zoom for visual ID. |
| 3 | Detection | Yes | Yes | Fully identified contact. PTZ zoom opportunistic — confirm vessel type, visual on close traffic. |

---

## Range Calculation for Camera-Only Contacts

For contacts with no radar and no AIS, range is estimated from masthead geometry:

```
range = h_eff / tan(θ)
```

Where:
- `h_eff` = effective camera height above waterline = `h * cos(roll)`
- `h` = configured camera height above waterline (meters) — measure at normal load waterline
- `roll` = vessel roll angle from PGN 127257 (AR200 AHRS), 10 Hz, ≤1° accuracy
- `θ` = ONVIF PTZ tilt angle below horizontal (degrees), from GetStatus

**ONVIF tilt mapping:** GetStatus returns normalized floats in [-1, 1], not degrees. Map to degrees using GetNode min/max limits read from device capabilities on first connect. Cache on startup; do not re-query per frame.

**Guard:** If `θ < 0.5°` (near-horizontal), return `None` and inject at a nominal range rather than returning a spurious large value.

**Configurable parameters:**
```yaml
camera:
  height_above_waterline_m: 22.25   # ~73 feet; measure at normal load waterline
  onvif_tilt_min_deg: -90.0         # override after reading from GetNode
  onvif_tilt_max_deg: 90.0
  onvif_pan_min_deg: -180.0
  onvif_pan_max_deg: 180.0
  onvif_tilt_normalized_min: -1.0
  onvif_tilt_normalized_max: 1.0
  range_min_tilt_deg: 0.5           # below this, range estimate unreliable
```

**Known error sources:**
- Freeboard: formula assumes target sits at waterline. For vessels with significant freeboard, computed range is slightly longer than true waterline range (conservative — safe direction).
- Atmospheric refraction: standard coefficient ~7% of geometric dip. At 2–3 NM produces ~3–5% range error (conservative direction).
- Roll correction on a catamaran: negligible in practice (cos(2°) ≈ 0.9994), but implemented correctly for monohull generality.

---

## Synthetic AIS Contact Injection

Confirmed contacts are injected onto the N2K bus as synthetic AIS entries so they appear on the Raymarine Axiom MFD chart.

**PGN path:**
- Position: PGN 129039 (Class B Position Report)
- Label: PGN 129809 (Class B CS Static Report Part A) — vessel name field
- PGN 129810 (Class B CS Static Report Part B) — optional, can be sent with dummy values

**Naming convention:** Use descriptive names in the vessel name field so the Axiom displays a labeled contact rather than an unnamed dot. Examples: `"SOV: DEBRIS"`, `"SOV: WHALE"`, `"SOV: VESSEL"`.

**MMSI range:** TBD — pending bench test. 999xxxxxx (ITU AtoN block) may be filtered by Axiom firmware. Test with an arbitrary number (e.g. 123456789) first to confirm Axiom plots it, then select a semantically appropriate range. Avoid 970/972/974/99x blocks.

**AIS700 transponder:** Does not receive AIS vessel PGNs from the N2K bus (confirmed from Raymarine doc 87326 — 129038/129039/129040/129041 are Transmit only on the AIS700). Synthetic contacts injected onto the bus are invisible to the transponder — no ITU re-broadcast risk.

**Axiom receives:** PGNs 129038, 129039, 129040, 129041, 129809, 129810 all confirmed Receive on LightHouse 4 (Raymarine doc 81406).

---

## Alert Path — PGN 126983

The Axiom receives PGN 126983 (Alert) and fires its internal buzzer for `Emergency Alarm` type alerts. Acknowledgement sends PGN 126984 back on the bus.

Practical injection path: Signal K notification API (`PUT /signalk/v1/api/vessels/self/notifications/...`) feeding through to canboat-n2k output plugin. Whether this plugin automatically maps to PGN 126983 is unverified — test with Axiom connected before depending on it. Fallback: encode CAN frame directly via python-can.

---

## NMEA Integration

- AIS targets: NMEA 0183 VDM sentences from Raymarine DataHub Pro TCP stream, port 11102
- Radar auto-acquired targets: NMEA 0183 TTM sentences, same stream
- Own vessel heading/position: HDT + RMC sentences, same stream
- Attitude (pitch/roll/yaw): PGN 127257 from AR200, 10 Hz, ≤1° accuracy
- Radar tracked targets: PGN 128520 from Axiom (also available as alternative to TTM — investigate)

All contacts converted to absolute bearings using own-vessel heading for PTZ slew targeting.

---

## Simulator

Happytimesoft ONVIF simulator: validates state machine transitions and ONVIF command flow but uses static video — cannot test the zoom-verify loop.

**Planned extension (Dragon's Lair approach):** Extend the mock ONVIF server in `simulate_operations.py` to respond to AbsoluteMove commands by switching to a different video clip, with GetStatus returning the commanded position. This closes the zoom-verify-confirm round trip in simulation and validates state machine correctness, ONVIF sequencing, and CPA→PTZ slew timing before the M300C is in hand.

---

## References

- Raymarine AIS700 NMEA 2000 PGN support: doc 87326
- Raymarine Axiom LightHouse 4 NMEA 2000 PGN support: doc 81406
- AR200 AHRS specs: 3-axis accelerometer/compass/gyro, heading/pitch/roll/ROT at 10 Hz, ≤1° pitch/roll/yaw accuracy
- MODS benchmark (Bovcon et al., T-ITS 2021): correct evaluation framework for the two-stage pipeline; use class-agnostic recall within danger zone as primary metric, not mAP50
- MaCVi workshop annual maritime detection benchmarks: macvi.org
- KOLOMVERSE dataset: 186,419 4K images, 5 classes