import sys, os, time
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, './eWaSR')
import wasr.models as models
from wasr.utils import load_weights
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
EWASR_WEIGHTS = './eWaSR/pretrained/ewasr_resnet18.pth'
YOLO_WEIGHTS  = './KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.pt'
SOURCE        = sys.argv[1] if len(sys.argv) > 1 else './kolomverse_test.mp4'
EWASR_SIZE    = (192, 256)   # H x W
ZOOM_FACTOR   = 4
MIN_BLOB_PX   = 25
ALERT_CONF    = 0.3
CLASSES       = {0:'ship', 1:'buoy', 2:'fishnet buoy', 3:'lighthouse', 4:'wind farm'}
ALERT_CLASSES = {0, 1, 2}
DISPLAY       = False  # set True if you have a display

# ── Load models ─────────────────────────────────────────────────────────────
print("Loading eWaSR...")
ewasr = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
ewasr.load_state_dict(load_weights(EWASR_WEIGHTS))
ewasr = ewasr.cuda().eval()

print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

# ── Open source ──────────────────────────────────────────────────────────────
print(f"Opening: {SOURCE}")
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("ERROR: Could not open source")
    sys.exit(1)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Source resolution: {orig_w}x{orig_h}")

# ── Helper functions ─────────────────────────────────────────────────────────
def preprocess(frame):
    small = cv2.resize(frame, (EWASR_SIZE[1], EWASR_SIZE[0]),
                       interpolation=cv2.INTER_LINEAR)
    img_np = (small.astype(np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    # Note: cv2 is BGR, convert to RGB for model
    img_np = img_np[:, :, ::-1].copy()
    tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0).cuda()
    imu = torch.zeros(1, EWASR_SIZE[0], EWASR_SIZE[1]).cuda()
    return tensor, imu

def run_ewasr(tensor, imu):
    with torch.no_grad():
        out = ewasr({'image': tensor, 'imu_mask': imu})
    pred = out['out'].squeeze(0).argmax(0).cpu().numpy()
    return cv2.resize(pred.astype(np.uint8), (EWASR_SIZE[1], EWASR_SIZE[0]),
                      interpolation=cv2.INTER_NEAREST)

def is_coastline_blob(stats, img_w, img_h):
    x, y, w, h = (stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP],
                  stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT])
    area = stats[cv2.CC_STAT_AREA]
    if (w / max(h, 1)) > 4.0: return True
    if area > (img_w * img_h * 0.05): return True
    if x <= 2 or (x + w) >= img_w - 2: return True
    return False

def find_blobs(mask):
    obstacle = (mask == 0).astype(np.uint8)
    obstacle[:int(EWASR_SIZE[0] * 0.4), :] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle)
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_PX: continue
        if is_coastline_blob(stats[i], EWASR_SIZE[1], EWASR_SIZE[0]): continue
        blobs.append({'cx': centroids[i][0], 'cy': centroids[i][1], 'area': area})
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs

def zoom_crop(frame, blob):
    scale_x = orig_w / EWASR_SIZE[1]
    scale_y = orig_h / EWASR_SIZE[0]
    cx = int(blob['cx'] * scale_x)
    cy = int(blob['cy'] * scale_y)
    cw = orig_w // ZOOM_FACTOR
    ch = orig_h // ZOOM_FACTOR
    x1 = max(0, cx - cw//2)
    y1 = max(0, cy - ch//2)
    x2 = min(orig_w, x1 + cw)
    y2 = min(orig_h, y1 + ch)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

def bearing(cx):
    return ((cx / EWASR_SIZE[1]) - 0.5) * 63.0

# ── Main loop ────────────────────────────────────────────────────────────────
frame_count = 0
alert_count = 0
times = []

print("\nRunning pipeline... (Ctrl+C to stop)\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        # Stage 1: eWaSR
        tensor, imu = preprocess(frame)
        mask = run_ewasr(tensor, imu)
        blobs = find_blobs(mask)

        # Stage 2: YOLO on top blobs
        alerts = []
        for blob in blobs[:3]:
            crop, bbox = zoom_crop(frame, blob)
            if crop.size == 0: continue
            results = yolo(crop, imgsz=640, conf=ALERT_CONF, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls)
                    if cls in ALERT_CLASSES:
                        alerts.append({
                            'class': CLASSES[cls],
                            'conf': round(float(box.conf), 3),
                            'bearing': round(bearing(blob['cx']), 1)
                        })
                        alert_count += 1

        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        frame_count += 1

        if alerts:
            print(f"Frame {frame_count:4d} [{elapsed:5.0f}ms] ALERT: {alerts}")
        elif frame_count % 10 == 0:
            print(f"Frame {frame_count:4d} [{elapsed:5.0f}ms] blobs={len(blobs)} no alert")

        # Stop after 100 frames for this test
        if frame_count >= 100:
            break

except KeyboardInterrupt:
    pass

cap.release()
print(f"\n{'='*50}")
print(f"Frames: {frame_count}  Alerts: {alert_count}")
print(f"Mean: {np.mean(times):.0f}ms  Min: {np.min(times):.0f}ms  Max: {np.max(times):.0f}ms")
print(f"Effective FPS: {1000/np.mean(times):.1f}")
