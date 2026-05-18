# Sail-O-Vision

Real-time marine obstacle and wildlife detection system running on an NVIDIA Jetson Orin Nano Super. 
Designed for liveaboard sailing use — detects vessels, debris, and anything that isn't water in the 
forward arc, with a web-based live view and alert system.

## Hardware

- NVIDIA Jetson Orin Nano Super Dev Kit ($150 + NVMe SSD) — ~9W active power draw
- FLIR M300C visible-light PTZ camera (onboard deployment — driveway camera used for development)
- Raymarine Axiom MFD + AR200 (handles AIS/radar — this system is supplementary)
- PredictWind DataHub Pro (NMEA 2000 gateway — provides AIS and radar target data over TCP)

## What it does

- Runs YOLOv8l object detection compiled to TensorRT FP16 at ~21 FPS on a live RTSP stream
- Web interface at `http://jetson.local:5000` showing live annotated video
- Browser alert (visual + audio) when something is detected above confidence threshold
- Persistent detection log with paginated gallery, click-to-view annotated images
- Starts automatically on boot via systemd
- Auto-reconnects on stream dropout

## What it's for

The Raymarine stack (AIS + radar + AR200) handles vessel collision avoidance. This system covers 
what radar doesn't resolve well at close range:

- Debris, logs, shipping containers low in the water
- Whales and large marine wildlife at the surface
- Small non-AIS vessels (kayaks, dinghies) at close range
- Anything that breaks the water surface texture

## Performance

Validated against the [Singapore Maritime Dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) 
and [MVTD](https://github.com/AhsanBaidar/MVTD) (182 sequences, ~150,000 frames across 
boat, ship, sailboat, and USV classes, both test and train splits):

- **98.5% detection rate** across ~150,000 frames
- **0 false positives** across the entire dataset
- Test set: 92.5% detection (20,386 frames) — harder sequences, more occlusion/haze
- Train set: 99.4% detection (129,666 frames)
- Misses concentrated in genuinely ambiguous frames (heavy occlusion, extreme haze, 
  motion blur, onshore camera looking across beach — conditions unlikely at sea)
- Open water is a remarkably clean detection environment — the false positive profile 
  on land (shadows, foliage, architecture) does not exist at sea

## Target Priority Architecture

Sail-o-vision fuses camera detections with AIS and radar data from the Raymarine/DataHub Pro 
stack to prioritize PTZ attention:

**Priority 1a** — Camera detection (high confidence), no AIS, no radar contact  
Unknown object: debris, whale, shipping container, unlit vessel. Highest threat uncertainty. 
Immediate PTZ zoom-and-verify.

**Priority 1b** — Camera detection (low confidence), no AIS, no radar contact  
Uncertain detection with no corroborating contact. Queued for next available PTZ slot.

**Priority 2** — Radar contact, no AIS  
Something with a radar return but not broadcasting — fishing vessel, small craft, or 
deliberately AIS-dark vessel. PTZ zoom provides visual identification.

**Priority 3** — Radar + AIS contact  
Fully identified contact already tracked by Raymarine. PTZ zoom is opportunistic — confirm 
vessel type, get a visual on close-passing traffic.

AIS targets arrive as NMEA 0183 VDM sentences, radar auto-acquired targets (via Axiom 
automatic target acquisition) as TTM sentences, both from the DataHub Pro TCP stream at 
port 11102. Own vessel heading and position from the same stream allow conversion of all 
contacts to absolute bearings for camera slew targeting.

Note: NMEA 2000 PGN 128520 (Tracked Target Data) is the N2K equivalent of TTM and contains 
the same data plus CPA and TCPA directly. Whether the Axiom transmits PGN 128520 when 
auto-acquisition is active needs verification — if confirmed, direct N2K access via a 
USB-CAN adapter (e.g. Actisense NGT-1) would be an alternative to the DataHub Pro TCP path.

## PTZ Scan Pattern

The M300C has a 63.7° horizontal field of view at wide angle, narrowing to 2.3° at 30x zoom.

Offshore scan cycle (tunable):
- **Ahead**: 5 minutes (default, primary safety function)
- **Port flank**: 30 seconds
- **Starboard flank**: 30 seconds

Flank scans create a blind spot ahead and are kept short. Any detection during a flank scan 
immediately triggers zoom-and-verify for that contact, then returns to ahead scanning. 
After any state, the camera always returns to SCANNING_AHEAD — never directly to a flank scan.

State machine:
```
SCANNING_AHEAD → (timer) → SCANNING_FLANK
SCANNING_AHEAD → (detection) → VERIFYING_TARGET
SCANNING_FLANK → (timer or hard timeout) → SCANNING_AHEAD
SCANNING_FLANK → (detection) → VERIFYING_TARGET
VERIFYING_TARGET → (confirmed) → ALERT + VIDEO_TRACKING
VERIFYING_TARGET → (not confirmed or timeout) → SCANNING_AHEAD
VIDEO_TRACKING → (target lost or resolved) → SCANNING_AHEAD
```

## PTZ Authority and Raymarine Integration

The M300C supports slew-to-cue (radar/AIS integration) which allows the Raymarine Axiom to 
command the camera to point at tracked targets. Sail-o-vision and the Raymarine slew-to-cue 
system cannot both control the PTZ simultaneously — PTZ authority must be managed:

- **Default**: sail-o-vision runs the scan pattern
- **Raymarine slew-to-cue active**: sail-o-vision yields PTZ control
- **Slew-to-cue complete**: sail-o-vision resumes scan pattern

Note: Raymarine has not implemented video tracking on the M300C despite the camera supporting 
it via ONVIF. Sail-o-vision will implement this directly — once a target is confirmed via 
zoom-and-verify, a lightweight OpenCV tracker (CSRT or KCF, running on CPU alongside YOLO 
on GPU) keeps the camera locked on the target -- possibly while inference continues on the 
zoomed view.

## Installation

### Prerequisites

- Jetson Orin Nano Super with JetPack 6.2 (r36.5)
- Ubuntu 22.04
- CUDA 12.6, TensorRT 10.3 (included with JetPack)

### Power mode

The Orin Nano Super ships with the wrong nvpmodel config by default. Fix it first:

```bash
sudo cp /etc/nvpmodel/nvpmodel_p3767_0004_super.conf /etc/nvpmodel.conf
# Reboot when prompted
sudo nvpmodel -m 2   # MAXN_SUPER — full 67 TOPS
sudo jetson_clocks
```

### Python environment

```bash
python3 -m venv ~/venvs/vision --system-site-packages
source ~/venvs/vision/bin/activate
pip install --upgrade pip wheel

# Install NVIDIA PyTorch wheel for JetPack 6.2
# PyPI wheels do not work on Jetson — must use NVIDIA's specific builds
wget "https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl" \
     -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl

wget "https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl" \
     -O torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.18.0-cp310-cp310-linux_aarch64.whl

pip install "numpy<2"
pip install ultralytics flask opencv-python-headless
```

### TensorRT engine

Export the model once — this compiles for your specific hardware and takes ~10 minutes:

```bash
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8l.pt')
model.export(format='engine', half=True, device=0)
"
```

The resulting `yolov8l.engine` file is hardware-specific and will not transfer to another device.

### Configuration

Edit the configuration section at the top of `vision_server.py`:

```python
RTSP_URL = "rtsp://your-camera-ip/stream"   # your camera's RTSP URL
CONF_THRESHOLD = 0.50                        # confidence for alerts
SAVE_THRESHOLD = 0.25                        # confidence for saving images
EXCLUDED_CLASSES = {"frisbee", "bench", "train"}  # false positive suppression
ALERT_COOLDOWN = 3.0                         # seconds between alerts
```

### Run as a service

```bash
sudo bash -c 'cat << EOF > /etc/systemd/system/vision.service
[Unit]
Description=Jetson Vision Server
After=network.target

[Service]
User=adam
WorkingDirectory=/home/adam
ExecStart=/home/adam/venvs/vision/bin/python3 -u /home/adam/sail-o-vision/vision_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable vision
sudo systemctl start vision
```

## Usage

Open `http://jetson.local:5000` in any browser on your local network.

- **Live stream**: annotated video with bounding boxes
- **Alert bar**: fires with label and confidence when detection exceeds threshold
- **Audio alert**: plays in browser on detection
- **Gallery**: `http://jetson.local:5000/gallery` — paginated detection log with timestamps, 
  labels, and click-to-view images. Persistent across restarts.

## Planned (requires boat + M300C)

### NMEA 0183 Integration via DataHub Pro
- Connect to DataHub Pro TCP stream (port 11102) for mixed NMEA 0183 data
- Parse AIS targets (VDM sentences) — position, bearing, MMSI, vessel name
- Parse radar auto-acquired targets (TTM sentences) — bearing, range, CPA, TCPA
- Parse own vessel heading (HDG/HDT) and position (RMC/GLL)
- Convert all contacts to absolute bearings for camera slew targeting
- Filter camera detections against known contacts to implement target priority queue

### PTZ Control
- ONVIF PTZ control of FLIR M300C
- Automated 180° horizon scan pattern (5 min ahead, 30s each flank, tunable)
- Zoom-and-verify on flagged bearings per target priority queue
- PTZ authority management between sail-o-vision and Raymarine slew-to-cue
- Video tracking via OpenCV CSRT/KCF tracker to maintain camera lock on confirmed targets
  while YOLO continues inference on the zoomed view

### Collision Alarms (Investigation Required)
For Priority 1 targets (camera-only, no AIS, no radar) on a collision course, options 
for alerting via the Axiom MFD:

- **Standards-based**: NMEA 2000 Alert PGNs 126983-126988 — correct approach, 
  Axiom support needs verification
- **Proprietary**: Raymarine PGN 65228 — documented by Yacht Devices as triggering 
  Raymarine-specific alarms on Axiom (experimental)
- **Fallback**: Audio/visual alert via sail-o-vision web interface only

All N2K transmission options require a USB-CAN adapter (e.g. Actisense NGT-1, ~$200) 
to put messages on the SeaTalkNG bus from the Jetson.

### Interesting Third-Party Hardware - Digital Yacht NavAlarm
[NavAlarm](https://digitalyacht.co.uk/product/navalarm/) is a standalone NMEA 2000 
device that triggers a physical audible alarm when it receives NMEA 2000 Alert PGNs 
(126983-126988). This is relevant to sail-o-vision's collision alarm investigation:

- Provides a standards-based audible alert path that doesn't depend on MFD display behavior
- The NMEA organisation has mandated all MFDs to output alert PGN information onto the 
  NMEA 2000 network, suggesting increasing standardisation of alert PGN support
- If the Axiom supports receiving external alert PGNs (needs verification), sail-o-vision 
  could trigger both a NavAlarm audible alert and an Axiom display alert through the same 
  PGN 126983 transmission
- Requires a USB-CAN adapter (e.g. Actisense NGT-1) for the Jetson to put messages on 
  the SeaTalkNG bus

Open question: does the Axiom display alerts triggered by an *external* device sending 
PGN 126983, or only its own internally-generated alerts?

### Interesting Third-Party Hardware - Digital Yacht NavAlert
[NavAlert](https://digitalyacht.co.uk/product/navalert/) is an NMEA 2000 monitor and 
alert solution with custom anchor and collision alarms, configurable thresholds for any 
N2K parameter, and pop-up alarms on Garmin MFDs (Raymarine compatibility unverified).

Sail-o-vision could implement equivalent collision alarm logic in software for 
camera-detected contacts, without requiring the NavAlert hardware:

- For **Priority 1 contacts** (camera-only, no AIS/radar), own vessel SOG/COG and 
  target bearing/range are known from the NMEA stream and camera detection respectively
- For **stationary targets** (debris, containers, whales at surface), SOG=0 so CPA 
  equals current range minus distance own vessel travels on current heading — 
  straightforward to calculate
- For **moving targets** without AIS, target SOG/COG is unknown — CPA calculation 
  requires tracking bearing rate of change over multiple detections to estimate target 
  motion
- Once CPA/TCPA is estimated, a PGN 126983 alert can be sent to the N2K bus, 
  triggering NavAlarm (audible) and potentially the Axiom display

This would give sail-o-vision genuine collision avoidance capability for the class of 
contacts that Raymarine cannot see — the primary threat scenario the system is designed 
for.

### Marine Tuning
- Confidence threshold tuning against real ocean conditions
- Allowlist/blocklist tuning for marine-specific false positive suppression
- KOLOMVERSE 4K dataset evaluation (access requested, pending approval)
- Fine-tuning on MVTD training set (130,368 labeled frames available locally) 
  if detection rate improvements are needed

## Datasets used for validation

- [Singapore Maritime Dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) — onboard visible light video
- [MVTD](https://github.com/AhsanBaidar/MVTD) — 182 sequences, boat/ship/sailboat/USV classes
- [KOLOMVERSE](https://github.com/MaritimeDataset/KOLOMVERSE) — 4K imagery, pending evaluation

## License

MIT