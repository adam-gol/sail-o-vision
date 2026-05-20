"""
visualize_mask.py  —  render eWaSR segmentation + blob analysis on a single frame.

Usage:
    python3 visualize_mask.py <video_or_image> [frame_number]

Outputs:
    mask_vis.jpg  —  4-panel: original | eWaSR mask | blobs (pass/fail) | overlay
"""
import sys, os
import numpy as np
import cv2
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'eWaSR'))
import wasr.models as models
from wasr.utils import load_weights

# ── Config ───────────────────────────────────────────────────────────────────
EWASR_WEIGHTS = str(Path(__file__).parent / 'eWaSR/pretrained/ewasr_resnet18.pth')
EWASR_SIZE    = (192, 256)   # H × W
MIN_BLOB_PX   = 25
OUTPUT        = 'mask_vis.jpg'

# eWaSR class colours:  obstacle=red, water=blue, sky=grey
CLASS_COLOURS = np.array([
    [220,  50,  50],   # 0 obstacle  — red
    [ 50,  50, 220],   # 1 water     — blue
    [180, 180, 180],   # 2 sky       — grey
], dtype=np.uint8)

def is_coastline_blob(stats, img_w, img_h):
    x, y, w, h, area = (stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP],
                        stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT],
                        stats[cv2.CC_STAT_AREA])
    if w == 0 or h == 0:
        return True
    aspect = w / h
    frac   = area / (img_w * img_h)
    # Pipeline only checks left/right edges, not top/bottom
    touches_lr_edge = (x <= 2 or x + w >= img_w - 2)
    return aspect > 4 or frac > 0.05 or touches_lr_edge

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading eWaSR...")
ewasr = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
ewasr.load_state_dict(load_weights(EWASR_WEIGHTS))
ewasr = ewasr.cuda().eval()

# ── Load frame ───────────────────────────────────────────────────────────────
src = sys.argv[1] if len(sys.argv) > 1 else 'video_clips/clip2.mp4'
target_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 15

if src.lower().endswith(('.jpg', '.jpeg', '.png')):
    frame = cv2.imread(src)
    if frame is None:
        print(f"ERROR: cannot read {src}"); sys.exit(1)
else:
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: cannot read frame {target_frame} from {src}"); sys.exit(1)

orig_h, orig_w = frame.shape[:2]
print(f"Frame {target_frame}: {orig_w}x{orig_h}")

# ── Run eWaSR ────────────────────────────────────────────────────────────────
small = cv2.resize(frame, (EWASR_SIZE[1], EWASR_SIZE[0]),
                   interpolation=cv2.INTER_LINEAR)
img_np = (small.astype(np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
img_np = img_np[:, :, ::-1].copy()   # BGR→RGB
tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0).cuda()
imu    = torch.zeros(1, EWASR_SIZE[0], EWASR_SIZE[1]).cuda()

with torch.no_grad():
    out = ewasr({'image': tensor, 'imu_mask': imu})
# Model outputs at 1/4 resolution (48×64) — resize to match pipeline behaviour
pred_raw = out['out'].squeeze(0).argmax(0).cpu().numpy()
pred = cv2.resize(pred_raw.astype(np.uint8), (EWASR_SIZE[1], EWASR_SIZE[0]),
                  interpolation=cv2.INTER_NEAREST)  # now 192×256

# ── Colour mask ──────────────────────────────────────────────────────────────
mask_rgb = CLASS_COLOURS[pred]           # 192×256×3
mask_bgr = mask_rgb[:, :, ::-1].copy()

# ── Blob analysis ────────────────────────────────────────────────────────────
obstacle_map = (pred == 0).astype(np.uint8)
# Match pipeline: zero out top 40% (sky/horizon masking)
obstacle_map[:int(EWASR_SIZE[0] * 0.4), :] = 0
n, labels, stats, centroids = cv2.connectedComponentsWithStats(
    obstacle_map, connectivity=8)

blob_vis = mask_bgr.copy()
blob_info = []
for i in range(1, n):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < MIN_BLOB_PX:
        continue
    killed = is_coastline_blob(stats[i], EWASR_SIZE[1], EWASR_SIZE[0])
    cx, cy = int(centroids[i][0]), int(centroids[i][1])
    x = stats[i, cv2.CC_STAT_LEFT];  y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
    colour = (0, 0, 255) if killed else (0, 255, 0)   # red=killed, green=passes
    cv2.rectangle(blob_vis, (x, y), (x+w, y+h), colour, 1)
    cv2.circle(blob_vis, (cx, cy), 3, colour, -1)
    reason = []
    if stats[i, cv2.CC_STAT_WIDTH] / max(stats[i, cv2.CC_STAT_HEIGHT], 1) > 4:
        reason.append('aspect')
    if stats[i, cv2.CC_STAT_AREA] / (EWASR_SIZE[1]*EWASR_SIZE[0]) > 0.05:
        reason.append('area')
    xi = stats[i, cv2.CC_STAT_LEFT]
    wi = stats[i, cv2.CC_STAT_WIDTH]
    if xi <= 2 or xi + wi >= EWASR_SIZE[1] - 2:
        reason.append('edge')
    blob_info.append({
        'id': i, 'area': area, 'cx': cx, 'cy': cy,
        'killed': killed, 'reason': ', '.join(reason) if reason else 'pass'
    })
    label = f"{area}" + (f" [{','.join(reason)}]" if reason else " [OK]")
    cv2.putText(blob_vis, label, (x, max(y-2, 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, colour, 1)

# Print blob table
print(f"\n{'ID':>3} {'Area':>6} {'cx':>4} {'cy':>4}  Status")
print("-" * 40)
for b in sorted(blob_info, key=lambda x: -x['area']):
    status = f"KILLED ({b['reason']})" if b['killed'] else "PASS"
    print(f"{b['id']:>3} {b['area']:>6} {b['cx']:>4} {b['cy']:>4}  {status}")

# ── Overlay on original ───────────────────────────────────────────────────────
mask_up   = cv2.resize(mask_bgr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
overlay   = cv2.addWeighted(frame, 0.55, mask_up, 0.45, 0)

# ── Assemble 4-panel output ───────────────────────────────────────────────────
# Scale all panels to same height for display
H = 384
def to_h(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*s), h), interpolation=cv2.INTER_LINEAR)

p1 = to_h(frame,    H)
p2 = to_h(mask_bgr, H)
p3 = to_h(blob_vis, H)
p4 = to_h(overlay,  H)

# Labels
for img, label in [(p1,'Original'),(p2,'eWaSR mask'),(p3,'Blobs (green=pass, red=killed)'),(p4,'Overlay')]:
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),       1)

panel = np.hstack([p1, p2, p3, p4])
cv2.imwrite(OUTPUT, panel)
print(f"\nSaved → {OUTPUT}  ({panel.shape[1]}×{panel.shape[0]})")
