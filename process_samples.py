from ultralytics import YOLO
import cv2
import os
import time

VIDEO_DIR = os.path.expanduser("~/samples/VIS_Onboard/Videos")
OUTPUT_DIR = os.path.expanduser("~/samples/detections")
CONF_THRESHOLD = 0.50
SAVE_THRESHOLD = 0.25
EXCLUDED_CLASSES = {"frisbee", "bench", "train"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading TensorRT engine...")
model = YOLO(os.path.expanduser("~/yolov8l.engine"), task="detect")

videos = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')])

for video_file in videos:
    video_path = os.path.join(VIDEO_DIR, video_file)
    video_name = os.path.splitext(video_file)[0]
    video_output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n{'='*60}")
    print(f"Processing: {video_file} ({total_frames} frames @ {fps:.0f}fps)")

    frame_count = 0
    detection_count = 0
    class_summary = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, verbose=False, conf=SAVE_THRESHOLD)
        boxes = results[0].boxes

        # Filter excluded classes
        visible = [
            b for b in boxes
            if model.names[int(b.cls)] not in EXCLUDED_CLASSES
        ]
        alertable = [
            b for b in visible
            if float(b.conf) >= CONF_THRESHOLD
        ]

        if visible:
            labels = [
                f"{model.names[int(b.cls)]}:{float(b.conf):.2f}"
                for b in visible
            ]
            label_str = ", ".join(labels)
            timestamp = frame_count / fps
            print(f"  Frame {frame_count:4d} ({timestamp:5.1f}s): {label_str}")

            # Track class summary
            for b in alertable:
                cls = model.names[int(b.cls)]
                class_summary[cls] = class_summary.get(cls, 0) + 1

            # Save annotated frame
            annotated = results[0].plot()
            fname = f"frame_{frame_count:05d}.jpg"
            cv2.imwrite(os.path.join(video_output_dir, fname), annotated)
            detection_count += 1

    cap.release()
    print(f"  Summary: {detection_count} detection frames out of {frame_count} total")
    if class_summary:
        print(f"  Alert-level classes: {class_summary}")
    else:
        print(f"  No alert-level detections")

print(f"\n{'='*60}")
print(f"Complete. Annotated frames saved to {OUTPUT_DIR}")
