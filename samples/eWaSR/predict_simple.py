import torch
import numpy as np
from PIL import Image
from pathlib import Path
import time
import sys
import glob

import wasr.models as models
from wasr.utils import load_weights

SEGMENTATION_COLORS = np.array([
    [247, 195, 37],   # obstacle - yellow
    [41, 167, 224],   # water - blue
    [90, 75, 164]     # sky - purple
], np.uint8)

IMG_SIZE = (384, 512)  # H x W

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE[1], IMG_SIZE[0]))
    orig = np.array(img)
    img_np = orig.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0)
    # Dummy IMU mask - zeros (no IMU data)
    imu_mask = torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1])
    return tensor, imu_mask, orig

def run(image_paths, weights, output_dir):
    model = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
    state_dict = load_weights(weights)
    model.load_state_dict(state_dict)
    model = model.cuda().eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = []
    for img_path in image_paths:
        img_path = Path(img_path)
        tensor, imu_mask, orig = preprocess(img_path)
        batch = {
            'image': tensor.cuda(),
            'imu_mask': imu_mask.cuda()
        }

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        pred = out['out'].squeeze(0).argmax(0).cpu().numpy()
        pred_up = np.array(Image.fromarray(pred.astype(np.uint8)).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST))
        colored = SEGMENTATION_COLORS[pred_up]
        blended = (orig * 0.6 + colored * 0.4).astype(np.uint8)

        obstacle_px = (pred_up == 0).sum()
        print(f"{img_path.name}: {(t1-t0)*1000:.1f}ms  obstacle_pixels={obstacle_px}")

        Image.fromarray(blended).save(output_dir / img_path.name)

    print(f"\nMean inference: {np.mean(times)*1000:.1f}ms  ({1/np.mean(times):.1f} FPS)")
    print(f"Output saved to {output_dir}")

if __name__ == '__main__':
    weights = sys.argv[1]
    img_dir = sys.argv[2]
    out_dir = sys.argv[3]
    paths = sorted(glob.glob(f"{img_dir}/*.jpg"))[:20]
    print(f"Running on {len(paths)} images...")
    run(paths, weights, out_dir)
