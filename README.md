# Sail-O-Vision

Real-time marine obstacle and wildlife detection system running on an NVIDIA Jetson Orin Nano Super. 
Designed for liveaboard sailing use — detects vessels, debris, and anything that isn't water in the 
forward arc, with a web-based live view and alert system.

## Hardware

- NVIDIA Jetson Orin Nano Super Dev Kit ($150 + NVMe SSD)
- FLIR M300C visible-light PTZ camera (onboard deployment — driveway camera used for development)
- Raymarine Axiom MFD + AR200 (handles AIS/radar — this system is supplementary)

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
and [MVTD](https://github.com/AhsanBaidar/MVTD) (182 sequences, 20,386 frames):

- **92.5% detection rate** across all sequences
- **0 false positives** across the entire dataset
- Misses concentrated in genuinely ambiguous frames (heavy occlusion, extreme haze, motion blur)

## Installation

### Prerequisites

- Jetson Orin Nano Super with JetPack 6.2 (r36.5)
- Ubuntu 22.04
- CUDA 12.6, TensorRT 10.3 (included with JetPack)

### Python environment

```bash
python3 -m venv ~/venvs/vision --system-site-packages
source ~/venvs/vision/bin/activate
pip install --upgrade pip wheel

# Install NVIDIA PyTorch wheel for JetPack 6.2 (must be done before ultralytics)
pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-deps
pip install "numpy<2"
pip install ultralytics flask opencv-python-headless onvif-zeep
```

### Power mode

```bash
# Switch to MAXN_SUPER mode for full 67 TOPS
sudo cp /etc/nvpmodel/nvpmodel_p3767_0004_super.conf /etc/nvpmodel.conf
sudo nvpmodel -m 2
sudo jetson_clocks
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
ExecStart=/home/adam/venvs/vision/bin/python3 -u /home/adam/boat-vision/vision_server.py
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

- ONVIF PTZ control of FLIR M300C
- Automated 180° horizon scan pattern
- Zoom-and-verify on flagged bearings
- SignalK integration for chart overlay alerts
- Marine threshold tuning against real ocean conditions

## Datasets used for validation

- [Singapore Maritime Dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) — onboard visible light video
- [MVTD](https://github.com/AhsanBaidar/MVTD) — 182 sequences, boat/ship/sailboat/USV classes
- [KOLOMVERSE](https://github.com/MaritimeDataset/KOLOMVERSE) — 4K imagery, pending evaluation

## License

MIT
