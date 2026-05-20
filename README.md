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

- Two-stage AI detection pipeline: eWaSR segmentation (Stage 1) + YOLO classification on zoomed crops (Stage 2)
- Runs at ~14-17 FPS end-to-end with TensorRT optimization (Jetson Orin Nano Super)
- Web interface at `http://jetson.local:5000` showing live annotated video
- Browser alert (visual + audio) when something is detected above confidence threshold
- Persistent detection log with paginated gallery, click-to-view annotated images
- Starts automatically on boot via systemd
- Auto-reconnects on stream dropout

## Two-Stage Pipeline Architecture

### Stage 1 — eWaSR Wide FOV Scan

[eWaSR](https://github.com/tersekmatija/eWaSR) (embedded Water Segmentation and Recognition, Teršek et al., Sensors 2023) runs continuously on the camera feed, segmenting each frame into obstacle / water / sky. Connected-component analysis finds obstacle blobs in the water region. Coastline blobs are filtered by shape (aspect ratio > 4, area > 5% of image, touching image edges). Surviving blobs trigger Stage 2.

- Input: 192×256 (downsampled from 4K)
- Inference: ~53ms on Orin GPU
- Fine-tuned on KOLOMVERSE for fishnet buoy detection
- Trained on: MaSTr1325 (baseline) + KOLOMVERSE validation set (fine-tune)

### Stage 2 — YOLO Zoom and Verify

When Stage 1 finds a candidate, a 4× zoom crop is extracted from the full-resolution frame centered on the blob centroid (simulating PTZ zoom). YOLOv8s classifies the crop. Confirmed detections above confidence threshold generate alerts.

- Model: yolov8s fine-tuned on KOLOMVERSE (mAP50: 0.830)
- Classes: ship (0), buoy (1), fishnet buoy (2), lighthouse (3), wind farm (4)
- Alert classes: ship, buoy, fishnet buoy
- Inference: ~45ms on Orin GPU

### eWaSR Fine-tuning Pipeline

eWaSR was fine-tuned to improve detection of fishnet buoys and small vessels:

1. KOLOMVERSE bounding box annotations → MobileSAM pixel masks (pseudo-ground-truth)
2. eWaSR baseline predictions supply sky/water labels for unannotated pixels
3. Combined masks used for supervised fine-tuning
4. Training on Google Colab T4 (Jetson cannot train 60M parameter model in-memory)
5. Result: visible improvement on fishnet buoy detection (image 0000068636)

Scripts: `~/sail-o-vision/samples/generate_masks.py`, masks at `~/sail-o-vision/samples/masks_kolomverse/`
Fine-tuned weights: `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_kolomverse.pth`

## What it's for

The Raymarine stack (AIS + radar + AR200) handles vessel collision avoidance. This system covers 
what radar doesn't resolve well at close range:

- Debris, logs, shipping containers low in the water
- Whales and large marine wildlife at the surface
- Small non-AIS vessels (kayaks, dinghies) at close range
- Anything that breaks the water surface texture

## Performance

### Detection pipeline

| Version | Mean latency | FPS | Notes |
|---------|-------------|-----|-------|
| PyTorch, video file | 121ms | 8.3 | Includes disk I/O |
| TensorRT, video file | 70ms | 14.2 | Includes disk I/O |
| TensorRT, live feed (estimated) | ~60ms | ~17 | No disk I/O |

Latency varies with scene complexity — frames with no obstacle blobs run in 15-23ms (eWaSR only); frames with 2-3 blobs run 120-180ms (eWaSR + multiple YOLO crops).

### TensorRT speedups

| Model | PyTorch | TensorRT | Speedup |
|-------|---------|----------|---------|
| eWaSR ResNet-18 (192×256) | 53ms | 17ms | 3× |
| YOLOv8s (640px crop) | 45ms | 39ms | 1.2× |

Engine files (hardware-specific to Orin Nano Super, not transferable):
- eWaSR: `~/sail-o-vision/samples/ewasr_resnet18.engine`
- YOLO: `~/sail-o-vision/samples/KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.engine`

### Detection quality

Evaluated on KOLOMVERSE validation set (1,746 annotated images, 5 classes):

| Model | mAP50 | ship | buoy | fishnet buoy | lighthouse | wind farm |
|-------|-------|------|------|--------------|------------|-----------|
| yolov8s_kolomverse | 0.830 | 0.928 | 0.763 | 0.724 | 0.855 | 0.878 |

Cross-dataset generalization is poor (KOLOMVERSE → SMD: 0.371 mAP) — domain gap between Korean coastal waters and other maritime environments. Mixed-dataset training is the planned fix.

### Evaluation framework

Standard mAP is not the right metric for navigation safety. The correct framework is the **MODS evaluation protocol** (Bovcon et al., IEEE T-ITS 2021):
- Class-agnostic recall within a **danger zone** (not mAP50)
- 15% overlap threshold (vs 50% IoU for standard mAP)
- Asymmetric cost: missed detection >> false positive
- Code: [github.com/bborja/mods_evaluation](https://github.com/bborja/mods_evaluation)

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

# Export eWaSR to ONNX then TensorRT
python3 ~/sail-o-vision/samples/export_ewasr_onnx.py
trtexec \
    --onnx=~/sail-o-vision/samples/ewasr_resnet18.onnx \
    --saveEngine=~/sail-o-vision/samples/ewasr_resnet18.engine \
    --fp16 \
    --memPoolSize=workspace:4096

# Export YOLO to TensorRT (takes ~8 minutes)
python3 -c "
from ultralytics import YOLO
model = YOLO('path/to/yolov8s_kolomverse/best.pt')
model.export(format='engine', half=True, device=0, imgsz=640)
"

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

### Chartplotter and Alert Integration (Investigation Required)

Two complementary approaches planned, both to be tested against the actual Axiom:

**Synthetic AIS targets** — camera-detected contacts injected as fake AIS vessels 
(fixed MMSIs in reserved range per class: FW-SHIP, FW-BOAT, FW-DEBRIS, FW-LOG etc.) 
into the NMEA 0183 stream. Axiom displays them as chart targets with CPA vectors using 
its existing AIS collision avoidance machinery. Approach validated by the 
[signalk-forward-watch](https://github.com/SkipperDon/signalk-forward-watch) project.
**Risk**: fake targets must not reach the AIS transponder for retransmission — 
requires careful NMEA network topology.

**NMEA 2000 Alert PGNs (126983-126988)** — standards-based alert from Jetson to 
Axiom via USB-CAN adapter (e.g. Actisense NGT-1). Triggers audible alarm via 
Digital Yacht NavAlarm. Whether Axiom displays external alerts from PGN 126983 
needs verification.

Both will be implemented and tested — they are complementary: synthetic AIS provides 
visual chart targets, PGN 126983 provides audible alerts.

### NMEA 0183 Integration via DataHub Pro
- Connect to DataHub Pro TCP stream (port 11102) for mixed NMEA 0183 data
- Parse AIS targets (VDM sentences) — position, bearing, MMSI, vessel name
- Parse radar auto-acquired targets (TTM sentences) — bearing, range, CPA, TCPA
- Parse own vessel heading (HDG/HDT) and position (RMC/GLL)
- Convert all contacts to absolute bearings for camera slew targeting
- Filter camera detections against known contacts to implement target priority queue

### Or NMEA 2000 Integration via TBD

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

### Alert Severity Hierarchy
Camera detections with estimated distance will use a three-level severity:
- **Emergency** (≤30m) — imminent collision risk
- **Warn** (≤75m) — close approach, take action  
- **Normal** (>75m) — awareness only

Severity feeds into both the web UI alert bar color and the NMEA 2000 alert PGN 
urgency field.

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
- KOLOMVERSE training set fine-tuning (1.16TB across 87 zips — requires cloud storage/compute; validation set fine-tuning completed)
- Mixed-dataset training (KOLOMVERSE + SMD + MVTD) to improve cross-domain generalization
- Fine-tuning on MVTD training set (130,368 labeled frames available locally) 
  if detection rate improvements are needed

### TensorRT — Complete ✅
eWaSR and YOLOv8s both exported to TensorRT FP16 engines. Pipeline runs at ~14 FPS on video file, ~17 FPS estimated on live feed. See `~/sail-o-vision/samples/pipeline_trt.py`.


## Datasets used for validation

- [Singapore Maritime Dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) — onboard visible light video
- [MVTD](https://github.com/AhsanBaidar/MVTD) — 182 sequences, boat/ship/sailboat/USV classes
- [KOLOMVERSE](https://doi.org/10.1109/TITS.2024.3449122) (Nanda et al., IEEE T-ITS 2024) — 186,419 4K images, 5 classes from 21 Korean territorial waters. Validation set (20,000 images) extracted locally. yolov8s fine-tuned and evaluated (mAP50: 0.830). eWaSR fine-tuned on validation subset via Colab T4.

## License

MIT