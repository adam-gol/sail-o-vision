import cv2
import os
from ultralytics import YOLO

DATA_DIR = os.path.expanduser("~/sail-o-vision/samples/MVTD_data/test")
CONF_THRESHOLD = 0.25
EXCLUDED_CLASSES = {"frisbee", "bench", "train"}

model = YOLO(os.path.expanduser("~/yolov8l.engine"), task="detect")

sequences = sorted(os.listdir(DATA_DIR))
total_detected = 0
total_missed = 0
total_fp = 0
total_frames = 0

for seq in sequences:
    seq_dir = os.path.join(DATA_DIR, seq)
    gt_file = os.path.join(seq_dir, "groundtruth.txt")
    if not os.path.exists(gt_file):
        continue

    gt_boxes = []
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                gt_boxes.append(tuple(map(float, parts)))

    frames = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])

    detected = missed = fp = 0
    for i, fname in enumerate(frames):
        img = cv2.imread(os.path.join(seq_dir, fname))
        results = model(img, verbose=False, conf=CONF_THRESHOLD)
        relevant = [
            b for b in results[0].boxes
            if model.names[int(b.cls)] not in EXCLUDED_CLASSES
        ]
        gt = gt_boxes[i] if i < len(gt_boxes) else None
        if relevant:
            if gt:
                detected += 1
            else:
                fp += 1
        else:
            if gt:
                missed += 1

    total_detected += detected
    total_missed += missed
    total_fp += fp
    total_frames += len(frames)

    pct = 100 * detected / len(gt_boxes) if gt_boxes else 0
    print(f"{seq:20s}: {detected:4d}/{len(gt_boxes):4d} ({pct:5.1f}%) detected, {fp:3d} FP")

print(f"\n{'='*60}")
print(f"TOTAL: {total_detected}/{total_detected+total_missed} "
      f"({100*total_detected/(total_detected+total_missed):.1f}%) detected")
print(f"Total false positives: {total_fp}")
print(f"Total frames: {total_frames}")
