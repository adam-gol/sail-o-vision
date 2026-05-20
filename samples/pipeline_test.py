import sys, os, time, json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import cv2
from collections import defaultdict

sys.path.insert(0, './eWaSR')

import wasr.models as models
from wasr.utils import load_weights
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
EWASR_WEIGHTS  = './eWaSR/pretrained/ewasr_resnet18.pth'
YOLO_WEIGHTS   = './KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.pt'
IMAGE_DIR      = './KOLOMVERSE/images/validation/0'
OUTPUT_DIR     = './pipeline_output'
EWASR_SIZE     = (192, 256)
ZOOM_FACTOR    = 4
MIN_BLOB_PX    = 25
ALERT_CONF     = 0.3
CLASSES        = {0:'ship', 1:'buoy', 2:'fishnet buoy', 3:'lighthouse', 4:'wind farm'}
ALERT_CLASSES  = {0, 1, 2}

# ── Load models ─────────────────────────────────────────────────────────────
print("Loading eWaSR...")
ewasr = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
ewasr.load_state_dict(load_weights(EWASR_WEIGHTS))
ewasr = ewasr.cuda().eval()

print("Loading YOLO...")
yolo = YOLO(YOLO_WEIGHTS)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helper functions ─────────────────────────────────────────────────────────
def run_ewasr(img_path):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img_small = cv2.resize(img, (EWASR_SIZE[1], EWASR_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    img_np = (np.array(img_small).astype(np.float32)/255.0 - 
              [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0).cuda()
    imu = torch.zeros(1, EWASR_SIZE[0], EWASR_SIZE[1]).cuda()
    with torch.no_grad():
        out = ewasr({'image': tensor, 'imu_mask': imu})
    pred = out['out'].squeeze(0).argmax(0).cpu().numpy()
    pred_up = cv2.resize(pred.astype(np.uint8), (EWASR_SIZE[1], EWASR_SIZE[0]), interpolation=cv2.INTER_NEAREST)
    return pred_up, (orig_w, orig_h)

def is_coastline_blob(stats, img_w, img_h):
    x = stats[cv2.CC_STAT_LEFT]
    y = stats[cv2.CC_STAT_TOP]
    w = stats[cv2.CC_STAT_WIDTH]
    h = stats[cv2.CC_STAT_HEIGHT]
    area = stats[cv2.CC_STAT_AREA]
    # Wide flat blob = coastline
    aspect = w / max(h, 1)
    if aspect > 4.0:
        return True
    # Very large blob = land mass
    if area > (img_w * img_h * 0.05):
        return True
    # Touching left or right edge = likely shoreline
    if x <= 2 or (x + w) >= img_w - 2:
        return True
    return False

def find_obstacle_blobs(mask):
    obstacle = (mask == 0).astype(np.uint8)
    # Only look below horizon (top 40% ignored)
    horizon_y = int(EWASR_SIZE[0] * 0.4)
    obstacle[:horizon_y, :] = 0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle)

    blobs = []
    img_w, img_h = EWASR_SIZE[1], EWASR_SIZE[0]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_PX:
            continue
        if is_coastline_blob(stats[i], img_w, img_h):
            continue
        cx, cy = centroids[i]
        blobs.append({
            'cx': cx, 'cy': cy, 'area': area,
            'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        })

    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs

def extract_zoom_crop(img_full, blob, orig_size):
    orig_w, orig_h = orig_size
    scale_x = orig_w / EWASR_SIZE[1]
    scale_y = orig_h / EWASR_SIZE[0]
    cx = blob['cx'] * scale_x
    cy = blob['cy'] * scale_y
    crop_w = orig_w // ZOOM_FACTOR
    crop_h = orig_h // ZOOM_FACTOR
    x1 = max(0, int(cx - crop_w/2))
    y1 = max(0, int(cy - crop_h/2))
    x2 = min(orig_w, x1 + crop_w)
    y2 = min(orig_h, y1 + crop_h)
    crop = img_full[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)

def run_yolo_on_crop(crop):
    results = yolo(crop, imgsz=640, conf=ALERT_CONF, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            detections.append({
                'class': CLASSES.get(cls, str(cls)),
                'class_id': cls,
                'confidence': conf,
                'alert': cls in ALERT_CLASSES
            })
    return detections

def bearing_from_image_x(cx, img_w, hfov_deg=63.0):
    offset = (cx / img_w) - 0.5
    return offset * hfov_deg

# ── Main pipeline loop ───────────────────────────────────────────────────────
image_paths = sorted(Path(IMAGE_DIR).glob('*.jpg'))[:50]
print(f"\nRunning pipeline on {len(image_paths)} images...")

results_log = []
alert_count = 0

for img_path in image_paths:
    t0 = time.perf_counter()

    mask, orig_size = run_ewasr(str(img_path))
    blobs = find_obstacle_blobs(mask)

    if not blobs:
        results_log.append({'image': img_path.name, 'stage1_blobs': 0, 'alerts': []})
        continue

    # Load full-res image once per frame
    img_full = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    alerts = []
    for blob in blobs[:3]:
        crop, crop_bbox = extract_zoom_crop(img_full, blob, orig_size)
        if crop.size == 0:
            continue
        detections = run_yolo_on_crop(crop)
        bearing = bearing_from_image_x(blob['cx'], EWASR_SIZE[1])

        for det in detections:
            if det['alert']:
                alerts.append({
                    'class': det['class'],
                    'confidence': round(det['confidence'], 3),
                    'bearing_offset_deg': round(bearing, 1),
                    'blob_area_px': blob['area'],
                })
                alert_count += 1

    t1 = time.perf_counter()

    result = {
        'image': img_path.name,
        'stage1_blobs': len(blobs),
        'pipeline_ms': round((t1-t0)*1000),
        'alerts': alerts
    }
    results_log.append(result)

    if alerts:
        print(f"  ALERT {img_path.name}: {[(a['class'], a['confidence'], a['bearing_offset_deg']) for a in alerts]}")
    elif len(blobs) > 0:
        print(f"  {img_path.name}: {len(blobs)} blobs, no confirmed detections")
    else:
        print(f"  {img_path.name}: no blobs after filtering")

# ── Summary ──────────────────────────────────────────────────────────────────
total_with_blobs = sum(1 for r in results_log if r['stage1_blobs'] > 0)
print(f"\n{'='*60}")
print(f"Processed: {len(image_paths)} images")
print(f"Stage 1 triggers (after filtering): {total_with_blobs} ({total_with_blobs/len(image_paths)*100:.0f}%)")
print(f"Total alerts: {alert_count}")
times = [r['pipeline_ms'] for r in results_log if 'pipeline_ms' in r]
if times:
    print(f"Mean pipeline time: {np.mean(times):.0f}ms ({1000/np.mean(times):.1f} FPS)")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)
with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(results_log, f, indent=2, cls=NumpyEncoder)
print(f"Full results saved to {OUTPUT_DIR}/results.json")
