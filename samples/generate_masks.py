import csv
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import sys
import os
from collections import defaultdict

from segment_anything import sam_model_registry, SamPredictor
import wasr.models as models
from wasr.utils import load_weights

sys.path.insert(0, '/home/adam/samples/eWaSR')

IMG_SIZE = (384, 512)  # eWaSR H x W

def get_ewasr_prediction(model, img_path):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE[1], IMG_SIZE[0]))
    orig = np.array(img)
    img_np = orig.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    tensor = torch.from_numpy(img_np.transpose(2,0,1)).float().unsqueeze(0).cuda()
    imu_mask = torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1]).cuda()
    with torch.no_grad():
        out = model({'image': tensor, 'imu_mask': imu_mask})
    pred = out['out'].squeeze(0).argmax(0).cpu().numpy()
    pred_up = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
        (IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST))
    return pred_up

def get_sam_masks(predictor, img_path, bboxes_4k):
    img = np.array(Image.open(img_path).convert('RGB'))
    predictor.set_image(img)

    # Scale bboxes from 4K to eWaSR resolution
    scale_x = IMG_SIZE[1] / 3840
    scale_y = IMG_SIZE[0] / 2160

    combined_mask = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=bool)

    for bbox in bboxes_4k:
        x1, y1, x2, y2 = bbox
        # Scale to eWaSR resolution for output mask
        sx1 = int(x1 * scale_x)
        sy1 = int(y1 * scale_y)
        sx2 = int(x2 * scale_x)
        sy2 = int(y2 * scale_y)

        # SAM works at original image resolution
        sam_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=sam_box,
            multimask_output=False
        )
        # Resize SAM mask to eWaSR resolution
        sam_mask = masks[0]  # H x W bool at original resolution
        sam_mask_small = np.array(
            Image.fromarray(sam_mask).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)
        )
        combined_mask |= sam_mask_small.astype(bool)

    return combined_mask

def main():
    sam_checkpoint = sys.argv[1]
    ewasr_weights = sys.argv[2]
    csv_path = sys.argv[3]
    img_base = Path(sys.argv[4])
    out_dir = Path(sys.argv[5])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load eWaSR
    print("Loading eWaSR...")
    ewasr = models.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
    state_dict = load_weights(ewasr_weights)
    ewasr.load_state_dict(state_dict)
    ewasr = ewasr.cuda().eval()

    # Load SAM
    print("Loading SAM...")
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam = sam.cuda()
    predictor = SamPredictor(sam)

    # Read CSV, group annotations by image
    print("Reading annotations...")
    annotations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['image'].startswith('validation/0/'):
                continue
            annotations[row['image']].append((
                int(row['xmin']), int(row['ymin']),
                int(row['xmax']), int(row['ymax'])
            ))

    print(f"Processing {len(annotations)} annotated images...")
    
    for i, (img_rel_path, bboxes) in enumerate(annotations.items()):
        img_path = img_base / img_rel_path
        if not img_path.exists():
            continue

        # Get eWaSR baseline (sky=2, water=1, obstacle=0)
        ewasr_pred = get_ewasr_prediction(ewasr, img_path)

        # Get SAM obstacle masks from bounding boxes
        sam_mask = get_sam_masks(predictor, img_path, bboxes)

        # Combine: use eWaSR for sky/water, override with obstacle where SAM fires
        final_mask = ewasr_pred.copy()
        final_mask[sam_mask] = 0  # 0 = obstacle

        # Save mask as PNG
        out_name = Path(img_rel_path).stem + '.png'
        Image.fromarray(final_mask.astype(np.uint8)).save(out_dir / out_name)

        if i % 50 == 0:
            print(f"  {i}/{len(annotations)}: {img_rel_path}")

    print(f"Done. Masks saved to {out_dir}")

if __name__ == '__main__':
    main()
