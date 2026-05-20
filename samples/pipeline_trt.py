import sys, os, time
import numpy as np
import cv2
import torch
import tensorrt as trt

sys.path.insert(0, './eWaSR')
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
EWASR_ENGINE  = './ewasr_resnet18.engine'
YOLO_ENGINE   = './KOLOMVERSE/scripts/models/yolov8s_kolomverse/weights/best.engine'
SOURCE        = sys.argv[1] if len(sys.argv) > 1 else './kolomverse_test.mp4'
EWASR_SIZE    = (192, 256)
ZOOM_FACTOR   = 4
MIN_BLOB_PX   = 25
ALERT_CONF    = 0.3
CLASSES       = {0:'ship', 1:'buoy', 2:'fishnet buoy', 3:'lighthouse', 4:'wind farm'}
ALERT_CLASSES = {0, 1, 2}

# ── Load eWaSR TRT ───────────────────────────────────────────────────────────
print("Loading eWaSR TRT engine...")
trt_logger = trt.Logger(trt.Logger.WARNING)
with open(EWASR_ENGINE, 'rb') as f:
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

trt_buffers = {}
trt_input_names = []
trt_output_names = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(engine.get_tensor_shape(name))
    dtype = torch.float32
    buf = torch.zeros(shape, dtype=dtype, device='cuda').contiguous()
    trt_buffers[name] = buf
    context.set_tensor_address(name, buf.data_ptr())
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        trt_input_names.append(name)
    else:
        trt_output_names.append(name)

def run_ewasr_trt(frame_rgb):
    _, C, H, W = trt_buffers[trt_input_names[0]].shape
    img = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / \
          np.array([0.229,0.224,0.225], dtype=np.float32)
    tensor = torch.from_numpy(
        img.transpose(2,0,1).reshape(1,C,H,W)).to(dtype=torch.float32, device='cuda')
    trt_buffers[trt_input_names[0]].copy_(tensor)
    context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    pred = trt_buffers[trt_output_names[0]]  # (1, 3, 48, 64)
    mask = pred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (EWASR_SIZE[1], EWASR_SIZE[0]), interpolation=cv2.INTER_NEAREST)

# ── Load YOLO TRT ────────────────────────────────────────────────────────────
print("Loading YOLO TRT engine...")
yolo = YOLO(YOLO_ENGINE, task='detect')

# ── Helper functions ─────────────────────────────────────────────────────────
def is_coastline_blob(stats, img_w, img_h):
    x, y, w, h = (stats[cv2.CC_STAT_LEFT], stats[cv2.CC_STAT_TOP],
                  stats[cv2.CC_STAT_WIDTH], stats[cv2.CC_STAT_HEIGHT])
    area = stats[cv2.CC_STAT_AREA]
    if (w / max(h,1)) > 4.0: return True
    if area > (img_w * img_h * 0.05): return True
    if x <= 2 or (x+w) >= img_w-2: return True
    return False

def find_blobs(mask):
    obstacle = (mask == 0).astype(np.uint8)
    obstacle[:int(EWASR_SIZE[0]*0.4), :] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle)
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_BLOB_PX: continue
        if is_coastline_blob(stats[i], EWASR_SIZE[1], EWASR_SIZE[0]): continue
        blobs.append({'cx': centroids[i][0], 'cy': centroids[i][1], 'area': area})
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs

def zoom_crop(frame, blob, orig_w, orig_h):
    scale_x = orig_w / EWASR_SIZE[1]
    scale_y = orig_h / EWASR_SIZE[0]
    cx = int(blob['cx'] * scale_x)
    cy = int(blob['cy'] * scale_y)
    cw = orig_w // ZOOM_FACTOR
    ch = orig_h // ZOOM_FACTOR
    x1 = max(0, cx-cw//2); y1 = max(0, cy-ch//2)
    x2 = min(orig_w, x1+cw); y2 = min(orig_h, y1+ch)
    return frame[y1:y2, x1:x2]

def bearing(cx):
    return ((cx / EWASR_SIZE[1]) - 0.5) * 63.0

# ── Open source ──────────────────────────────────────────────────────────────
print(f"Opening: {SOURCE}")
cap = cv2.VideoCapture(SOURCE)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {orig_w}x{orig_h}")

# Warmup
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
for _ in range(3):
    run_ewasr_trt(frame_rgb)
    yolo(frame_rgb[:orig_h//4, :orig_w//4], imgsz=640, conf=ALERT_CONF, verbose=False)

print("\nRunning TRT pipeline... (100 frames)\n")
frame_count = 0
alert_count = 0
times = []

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while frame_count < 100:
    ret, frame = cap.read()
    if not ret: break

    t0 = time.perf_counter()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask = run_ewasr_trt(frame_rgb)
    blobs = find_blobs(mask)

    alerts = []
    for blob in blobs[:3]:
        crop = zoom_crop(frame_rgb, blob, orig_w, orig_h)
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

    elapsed = (time.perf_counter()-t0)*1000
    times.append(elapsed)
    frame_count += 1

    if alerts:
        print(f"Frame {frame_count:4d} [{elapsed:5.0f}ms] ALERT: {alerts}")
    elif frame_count % 10 == 0:
        print(f"Frame {frame_count:4d} [{elapsed:5.0f}ms] blobs={len(blobs)}")

cap.release()
print(f"\n{'='*50}")
print(f"Frames: {frame_count}  Alerts: {alert_count}")
print(f"Mean: {np.mean(times):.0f}ms  Min: {np.min(times):.0f}ms  Max: {np.max(times):.0f}ms")
print(f"Effective FPS: {1000/np.mean(times):.1f}")
print(f"\nSpeedup vs PyTorch pipeline: {455/np.mean(times):.1f}x")
