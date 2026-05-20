import csv
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import time
import sys

import wasr.models as models
from wasr.utils import load_weights

SEGMENTATION_COLORS = np.array([
    [247, 195, 37],   # obstacle - yellow
    [41, 167, 224],   # water - blue
    [90, 75, 164]     # sky - purple
], np.uint8)

IMG_SIZE = (384, 512)

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE[1], IMG_SIZE[0]))
    orig = np.array(img)
    img_np = orig.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0)
    imu_mask = torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])
    return tensor, imu_mask, orig

def run_ewasr(model, img_path):
    tensor, imu_mask, orig = preprocess(img_path)
    batch = {'image': tensor.cuda(), 'imu_mask': imu_mask.cuda()}
    with torch.no_grad():
        out = model(batch)
    pred = out['out'].squeeze(0).argmax(0).cpu().numpy()
    pred_up = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
        (IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST))
    return pred_up, orig

def main():
    weights = sys.argv[1]
    csv_path = sys.argv[2]
    img_base = sys.argv[3]
    out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
    state_dict = load_weights(weights)
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    # Read CSV, find ship annotations in validation/0, compute bbox area
    ships = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['label'] != 'ship':
                continue
            if not row['image'].startswith('validation/0/'):
                continue
            w = int(row['xmax']) - int(row['xmin'])
            h = int(row['ymax']) - int(row['ymin'])
            area = w * h
            ships.append({
                'image': row['image'],
                'area': area,
                'w': w, 'h': h,
                'xmin': int(row['xmin']), 'ymin': int(row['ymin']),
                'xmax': int(row['xmax']), 'ymax': int(row['ymax']),
            })

    # Sort by area, bin into 6 size categories
    ships.sort(key=lambda x: x['area'])
    
    # Define size bins (area in pixels at 4K resolution)
    bins = [
        ('tiny',       0,      500),
        ('very_small', 500,    2500),
        ('small',      2500,   10000),
        ('medium',     10000,  40000),
        ('large',      40000,  160000),
        ('very_large', 160000, 999999999),
    ]

    print(f"{'Bin':12s} {'Area':>8s} {'BBox WxH':>12s} {'Obs_px':>8s} {'Detected':>10s}  Image")
    print("-" * 90)

    for bin_name, lo, hi in bins:
        # Get up to 3 samples from this bin
        samples = [s for s in ships if lo <= s['area'] < hi]
        if not samples:
            print(f"{bin_name:12s}  (no samples)")
            continue
        # Pick samples spread across the bin
        indices = np.linspace(0, len(samples)-1, min(3, len(samples)), dtype=int)
        for idx in indices:
            s = samples[idx]
            img_path = Path(img_base) / s['image']
            if not img_path.exists():
                continue

            pred_up, orig = run_ewasr(model, img_path)

            # Check obstacle pixels in the ship bounding box region
            # Scale bbox from 4K to eWaSR output resolution
            scale_x = IMG_SIZE[1] / 3840
            scale_y = IMG_SIZE[0] / 2160
            x1 = int(s['xmin'] * scale_x)
            y1 = int(s['ymin'] * scale_y)
            x2 = int(s['xmax'] * scale_x)
            y2 = int(s['ymax'] * scale_y)
            x1, x2 = max(0,x1), min(IMG_SIZE[1]-1, x2)
            y1, y2 = max(0,y1), min(IMG_SIZE[0]-1, y2)

            roi = pred_up[y1:y2, x1:x2]
            obs_in_bbox = (roi == 0).sum()
            bbox_area = max(1, (x2-x1)*(y2-y1))
            detected = obs_in_bbox / bbox_area > 0.15  # >15% of bbox is obstacle

            # Save output image with bbox drawn
            colored = SEGMENTATION_COLORS[pred_up]
            blended = (orig * 0.6 + colored * 0.4).astype(np.uint8)
            # Draw bbox in red
            blended[y1:y2, x1, :] = [255, 0, 0]
            blended[y1:y2, x2, :] = [255, 0, 0]
            blended[y1, x1:x2, :] = [255, 0, 0]
            blended[y2, x1:x2, :] = [255, 0, 0]

            out_file = out_dir / f"{bin_name}_{s['area']}_{Path(s['image']).name}"
            Image.fromarray(blended).save(out_file)

            print(f"{bin_name:12s} {s['area']:>8d} {s['w']:>5d}x{s['h']:<5d} "
                  f"{obs_in_bbox:>8d} {'YES' if detected else 'NO':>10s}  {s['image']}")

if __name__ == '__main__':
    main()
