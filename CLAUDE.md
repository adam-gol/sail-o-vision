# sail-o-vision

AI vision system for a sailboat, providing:
1. **Navigation safety** — detecting vessels, unlit boats, fishnet buoys, debris, and other hazards that radar/AIS miss
2. **Wildlife observation** — marine mammals, birds

This file orients Claude Code on a project that has been developed across multiple chat conversations. The boat is under construction; the M300C is not yet installed. Development uses the KOLOMVERSE dataset as a proxy for live camera feed.

For ground truth on any specific claim, read the actual code in this repo and on the Jetson. Items flagged with `<!-- VERIFY -->` should be confirmed against files before being relied on.

---

## Hardware

- **Camera**: FLIR M300C — visible-light HD, 30× optical zoom, gyro-stabilized PTZ. Mast-mounted (planned).
  - **Not yet installed.** Boat is under construction.
  - Visible-light only (no thermal). An earlier guide assumed thermal (M324-class) and had to be rewritten.
  - Power draw: 41–56 W.
  - ONVIF Profile S support for this specific firmware is **unverified** — some early M300C firmware revisions are RTSP-only without the PTZ control channel. Check FLIR docs for the unit's firmware version.
- **Compute**: Jetson Orin Nano Super Dev Kit, 67 TOPS, 8 GB unified memory, ~$249 (released December 2024).
  - Replaces the original Jetson Orin Nano Dev Kit recommendation ($499, 40 TOPS).
- **MFD**: Raymarine (existing install).
- **Total system power**: ~83–98 W continuous (camera + Jetson + ancillaries).

Possible future camera upgrade: FLIR M332 or M364 (thermal capability is the current gap; another buyer of the same boat model is going thermal).

## Architecture: two-stage scan-and-verify pipeline

The PTZ + 30× zoom is treated as an active capability, not a fixed camera. The pipeline scans wide and zooms to verify.

### Stage 1 — eWaSR wide-FOV scan
[eWaSR](https://github.com/tersekmatija/eWaSR) (Teršek et al., *Sensors*, 2023) is an embedded-compute-ready maritime obstacle segmentation network. <!-- VERIFY citation -->

- Segments each frame into three classes: **obstacle / water / sky**.
- Connected-component analysis finds obstacle blobs in the water region.
- Coastline filtering suppresses shore clutter: blobs with aspect ratio > 4, area > 5% of image, or touching image edges are filtered out.
- Surviving blobs trigger Stage 2.
- **Input**: 192×256 (downsampled from 4K).
- **Latency**: ~53 ms on Orin Nano Super GPU.
- **Models**:
  - Baseline: `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_resnet18.pth` — pretrained on MaSTr1325.
  - Fine-tuned: `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_kolomverse.pth` — fine-tuned on KOLOMVERSE validation set, shows improved fishnet buoy detection.

### Stage 2 — YOLO zoom and verify
When Stage 1 finds a candidate, a **4× zoom crop** is extracted from the full-resolution frame centred on the blob centroid (simulating PTZ zoom in the dev environment). YOLO classifies the crop.

- **Model**: `yolov8s_kolomverse` at `~/sail-o-vision/samples/KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.pt`. mAP50 = 0.830 on KOLOMVERSE val.
- **Classes**: ship (0), buoy (1), fishnet buoy (2), lighthouse (3), wind farm (4).
- **Alert classes**: {ship, buoy, fishnet buoy}.
- **Confidence threshold**: 0.3.
- **Latency**: ~45 ms on Orin Nano Super GPU.

### Horizon scan loop (planned, parallel)
A systematic PTZ sweep of the horizon band at 30× zoom, running independently of Stage 1. This catches sub-pixel distant targets that eWaSR cannot detect at wide FOV — the fundamental resolution limit of Stage 1. Not yet implemented; requires the M300C.

### Pipeline performance

| Stage | Time | Notes |
|-------|------|-------|
| eWaSR (192×256) | 53 ms | Stage 1 segmentation |
| Blob detection + filtering | 2 ms | Connected components, coastline filter |
| YOLO on zoomed crop | 45 ms | Stage 2 classification |
| **Total (live feed)** | **~100 ms (~10 FPS)** | Image load from disk adds ~75 ms when running on stills |
| Total (on video file) | 121 ms (~8.3 FPS) | |

**Critical perf gotcha**: `cv2.resize` takes 0.7 ms on a 4K image; `PIL` resize takes 73 ms. Always use cv2 for resizing in the hot path. This was a real bottleneck that got caught and fixed.

### eWaSR fine-tuning pipeline
eWaSR was fine-tuned on KOLOMVERSE to improve detection of fishnet buoys and small vessels (Korean coastal context not well-represented in MaSTr1325):

1. MobileSAM generates pixel-level segmentation masks from KOLOMVERSE bounding box annotations.
2. eWaSR baseline predictions supply sky/water labels for unannotated pixels.
3. Combined masks used as **pseudo-ground-truth** for supervised fine-tuning.
4. Training on Google Colab T4 — the Jetson Orin cannot train the ~60M-param model in-memory.
5. Result: visible improvement on fishnet buoy detection (reference image: 0000068636).

- Mask generation: `~/sail-o-vision/samples/generate_masks.py`
- Generated masks: `~/sail-o-vision/samples/masks_kolomverse/`
- Fine-tuned weights: `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_kolomverse.pth`

## File layout

**Jetson (`adam@jetson`)**:
- `~/sail-o-vision/` — main repo (current working directory)
- `~/sail-o-vision/samples/pipeline_live.py` — main two-stage pipeline (live feed entry point)
- `~/sail-o-vision/samples/pipeline_test.py` — earlier static-image version of the pipeline
- `~/sail-o-vision/samples/eWaSR/` — eWaSR clone (model code, weights, `export.py` for ONNX)
- `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_resnet18.pth` — baseline weights
- `~/sail-o-vision/samples/eWaSR/pretrained/ewasr_kolomverse.pth` — fine-tuned weights
- `~/sail-o-vision/samples/KOLOMVERSE/images/validation/0/` — 20,000 validation images extracted
- `~/sail-o-vision/samples/KOLOMVERSE/labels/validation_label.csv` — CSV labels
- `~/sail-o-vision/samples/KOLOMVERSE/scripts/` — dataset scripts including `csv2yolo.py`
- `~/sail-o-vision/samples/KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.pt` — trained YOLO
- `~/sail-o-vision/samples/generate_masks.py` — MobileSAM mask generation
- `~/sail-o-vision/samples/masks_kolomverse/` — generated pseudo-GT masks
- `~/sail-o-vision/samples/kolomverse_test.mp4` — synthetic 30s test video built from 300 validation stills @ 10 FPS (note: frames are *not* sequential — fine for pipeline mechanics, useless for temporal models)
- `~/yolov8l.pt` — baseline COCO model, not used in pipeline
- `~/venvs/vision/` — Python virtualenv

## Development environment

- **JetPack**: 6.2.2 (Jetson Linux 36.5, R36 rev 5.0, January 2026 build) <!-- VERIFY with `cat /etc/nv_tegra_release` -->
- **CUDA**: 12.6 (packages labelled 12.5 — normal NVIDIA naming quirk)
- **TensorRT**: 10.3.0.30 (and Python bindings)
- **cuDNN**: 9.3
- **Python**: 3.10.12
- **virtualenv**: `~/venvs/vision`, created with `--system-site-packages` to inherit JetPack libs
- **Key Python packages**: torch 2.3.0, torchvision 0.18.0a0+6043bc2, ultralytics 8.4.48, pytorch-lightning 2.6.1, timm 0.6.5, albumentations, mobile_sam, segment_anything

### NumPy gotcha
At one point ultralytics 8.4.48 was importable but `ultralytics.checks()` crashed because NumPy 2.2.6 was installed and torch 2.3.0 was compiled against NumPy 1.x. **If `ultralytics.checks()` fails with a NumPy 1.x/2.x compile mismatch, pin `numpy<2` in the venv.** <!-- VERIFY this was resolved and how -->

### PyTorch wheel
torch 2.3.0 came from NVIDIA's JetPack-specific wheel. torchvision 0.18.0 wheel was fetched from `https://nvidia.box.com/shared/static/u0ziu01c0kyji4zz3gxam79181nebylf.whl`. If the venv ever needs rebuilding, use NVIDIA's JetPack-pinned wheels, not PyPI generics — generic ARM wheels won't have CUDA support on Jetson.

### JetPack upgrades
APT upgrades within a JetPack point release should leave the venv intact because it's in `$HOME`. The risk is CUDA/TensorRT `.so` path shifts → `ImportError: libXXX.so not found`. Recovery: `apt search` for the missing library at the new version.

## Datasets

### KOLOMVERSE (in active use)
Korea Open Large-Scale Image Dataset for Object Detection in the Maritime Universe (Nanda et al., IEEE T-ITS 2024). 186,419 4K images, 5 classes, 21 Korean territorial waters.
- DOI: 10.1109/TITS.2024.3449122 <!-- VERIFY this DOI -->
- Validation set (20,000 images) fully extracted locally.
- Full training set is 1.16 TB across 87 zips — fine-tuning on the full set requires cloud storage/compute, not yet done.
- Cross-dataset generalization is **poor**: KOLOMVERSE → SMD = 0.371 mAP. Domain gap between Korean coastal waters and other environments. Mixed-dataset training is the planned fix.

### Detection quality (yolov8s_kolomverse on KOLOMVERSE val, 1,746 images)
| Model | mAP50 | ship | buoy | fishnet buoy | lighthouse | wind farm |
|-------|-------|------|------|--------------|------------|-----------|
| yolov8s_kolomverse | 0.830 | 0.928 | 0.763 | 0.724 | 0.855 | 0.878 |

### Other datasets evaluated
- Singapore Maritime Dataset (SMD)
- MVTD (HuggingFace)
- MaSTr1325 (used by eWaSR baseline)
- MODS benchmark dataset (Bovcon et al., T-ITS 2021) — used for evaluation framework

## Evaluation framework

Standard mAP is **not the right metric for navigation safety**. The chosen framework is the **MODS evaluation protocol** (Bovcon et al., IEEE T-ITS 2021): <!-- VERIFY citation and that the repo below is the right one -->

- Class-agnostic recall within a **danger zone** (not mAP50)
- 15% overlap threshold (vs 50% IoU for standard mAP)
- Separate reporting for danger zone vs full image
- Asymmetric cost: missed detection ≫ false positive
- Evaluation code: `github.com/bborja/mods_evaluation`
- Reference result from literature: WaSR-T reduces false positives 53% vs WaSR
- MaCVi workshop publishes annual maritime detection benchmarks at macvi.org

## Models — planned but not yet in pipeline

- **YOLO26** for navigation (replacing YOLOv8s)
- **YOLOE-26** (open-vocabulary) for wildlife — value prop is no fine-tuning needed; visual prompts as fallback if text prompts fail
- Both confirmed as the target choices in earlier chat <!-- VERIFY current status before changing pipeline -->

## Marine integration

### SignalK
Open marine data bus (JSON over WebSocket/HTTP). Runs on a Raspberry Pi or directly on the Jetson. Detections published as custom SignalK paths, e.g.:

```
vessels.self.sensors.ai_vision.detections.0
  .bearing.value         045.3
  .class                 "boat"
  .confidence            0.82
  .rangeEstimate.value   450    # meters
  .timestamp             "2026-04-23T14:23:11Z"
```

Downstream: OpenCPN chart overlay, NMEA 2000 bridge to chartplotter, Starlink → shore dashboard. **Output plugin not yet built.**

### PTZ control (ONVIF)
Python ONVIF for M300C PTZ control. Camera not yet arrived; firmware ONVIF support unverified.

### Power path (recommended)
48 V house bank → 48 V/12 V step-down → 12 V breaker panel → inline 5 A fuse → wide-input (9–36 V) DC-DC to 5 V @ 5 A → Jetson barrel jack.
- **Do NOT power the Jetson directly from anything that can surge above 5.25 V.** Orin Nano dev kit input protection is minimal.

### Graceful shutdown on low voltage
Wire a Victron Cerbo relay output or VE.Direct-tapped voltage threshold to a Jetson GPIO. Daemon polls the GPIO and `systemctl poweroff` below ~11.8 V. Thousands of uncontrolled power cuts will eventually corrupt the NVMe.

### Thermal (aluminium boat)
Two enclosure options for the Jetson:
1. Vented IP67 enclosure with Gore membrane on intake and exhaust, active airflow.
2. Passive aluminium enclosure thermally coupled to a bulkhead — 2 mm thermal pad between heatsink fins and an inner aluminium plate, then bolt the enclosure to an aluminium bulkhead. The hull becomes a heatsink. Preferred on an aluminium cat.

### EMI mitigation
The 5 kW inverter is the worst offender. The 48 V switching power path is noisy. Mitigations:
- Shielded Cat6 for camera-to-Jetson, shield bonded to case at **both ends**
- Ferrite cores on DC input to Jetson (Würth 742 710 31 or similar for 10 MHz–1 GHz)
- Keep Jetson DC-DC ≥1 m from inverters
- Ground Jetson enclosure to boat ground via a short braided strap, not via DC return alone

## Pipeline runtime (current)

The live pipeline reads from any cv2 video capture source — works identically for RTSP, video files, or webcam:

```bash
source ~/venvs/vision/bin/activate

# Video file (current dev mode)
python3 ~/sail-o-vision/samples/pipeline_live.py /path/to/video.mp4

# RTSP stream (when camera available)
python3 ~/sail-o-vision/samples/pipeline_live.py rtsp://camera-ip/stream
```

Config constants live at the top of `pipeline_live.py`:
- `EWASR_WEIGHTS`, `YOLO_WEIGHTS` — model paths
- `EWASR_SIZE = (192, 256)` (H × W)
- `ZOOM_FACTOR = 4`
- `MIN_BLOB_PX = 25`
- `ALERT_CONF = 0.3`
- `ALERT_CLASSES = {0, 1, 2}` (ship, buoy, fishnet buoy)
- `DISPLAY = False`

## Web interface (referenced in past chat)

There was discussion of a web interface at `http://jetson.local:5000` with live annotated video, browser visual+audio alerts, persistent detection log with paginated gallery, systemd autostart, and stream auto-reconnect. <!-- VERIFY whether this exists in the current repo, what its entry point is, and whether it's the canonical UI -->

## Next steps (priority order, from recent chat)

1. **TensorRT optimization** of eWaSR and YOLO — `trtexec` available on Jetson; eWaSR has `export.py` for ONNX; Ultralytics has built-in TensorRT export. Target: 15–20+ FPS.
2. **RTSP live feed integration**
3. **Own-vessel masking** (when boat built — masks out own rigging/deck from frame)
4. **SignalK output plugin**
5. **PTZ control via ONVIF** when M300C arrives
6. **Horizon scan loop** for sub-pixel distant targets
7. **Mixed-dataset training** (KOLOMVERSE + SMD + MVTD) for cross-domain generalization
8. **WaSR-T evaluation** — temporal variant, expected to reduce false positives ~53% per the literature

## Design principles (carry forward)

- **Exploit the zoom.** The 30× optical zoom is a real advantage; the architecture must scan wide and zoom to verify, not treat the camera as fixed-FOV.
- **Power budget includes the camera**, not just the compute.
- **For visible-light, COCO pretrained weights are a viable starting point.** Fine-tuning matters most for low-light/night.
- **Class-agnostic recall in the danger zone** is the metric that matches the safety use case. mAP50 is misleading here.
- **cv2 over PIL** for any per-frame image op.
- **Diversity over volume** for any future training data collection — different sea states, sun angles, day/night, varying ranges. Target 500–2000 labelled frames per model.

## Commercial systems context (for reference, not in use)

- **Raymarine ClearCruise AR**: only labels AIS targets and chart objects — *not* a true hazard detector.
- **Sea.AI** (sea.ai): Austrian, est. 2018. Strong racing pedigree (Vendée Globe, Transat Jacques Vabre). Documented racing collision avoidances of unlit fishing boats and AIS-less cargo ships.
- **LOOKOUT** (getalookout.com): Newer US system, shipping since 2024.
- Cost range surveyed: $17K (M300C + LOOKOUT Brain Pro) to $47K (Sea.AI Sentry premium).
- **Recreational cruising track record for both is thin.** Almost all "saved by the system" reports are from racing. Only one documented recreational account found in research (LOOKOUT in Boston Harbor fog). Worth weighing against marketing claims.

## Open questions / unverified

- FLIR M300C firmware ONVIF Profile S compliance (some early firmware is RTSP-only)
- Actual measured FPS of YOLO26s on Orin Nano Super at 1080p (the planned model swap)
- Whether to stay custom, adopt Sea.AI/LOOKOUT, or run hybrid
- Whether the Flask web UI described above is the canonical interface in the current repo
- DOI verification for KOLOMVERSE and MODS citations
- Confirmation that `csv2yolo.py` lives where memory says it does

## Key references

- Bovcon, B. et al. "MODS — A USV-oriented object detection and obstacle segmentation benchmark." *IEEE T-ITS*, 2021. arXiv:2105.02359. <!-- VERIFY -->
- Nanda, A. et al. "KOLOMVERSE: Korea Open Large-Scale Image Dataset for Object Detection in the Maritime Universe." *IEEE T-ITS*, 2024. DOI: 10.1109/TITS.2024.3449122. <!-- VERIFY -->
- Teršek, M. et al. "eWaSR: an embedded-compute-ready maritime obstacle detection network." *Sensors*, 2023. github.com/tersekmatija/eWaSR. <!-- VERIFY -->
