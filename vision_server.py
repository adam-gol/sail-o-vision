from flask import Flask, Response, jsonify, send_file, request
from ultralytics import YOLO
import cv2
import time
import threading
import os
import json
from datetime import datetime
import base64

# --- Configuration ---
RTSP_URL = "rtsp://192.168.55.29/axis-media/media.amp"
DETECTION_DIR = os.path.expanduser("~/detections")
CONF_THRESHOLD = 0.50        # Confidence for alerts
SAVE_THRESHOLD = 0.25       # Confidence for saving images
MAX_GALLERY_IMAGES = 50     # How many to show in gallery
POLL_INTERVAL_MS = 500      # Browser polls for new detections every 500ms

os.makedirs(DETECTION_DIR, exist_ok=True)

import logging
app = Flask(__name__)
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# --- Shared state ---
latest_frame = None
latest_frame_lock = threading.Lock()
latest_detection = None
latest_detection_lock = threading.Lock()
detection_log = []          # List of {timestamp, filename, labels}
detection_log_lock = threading.Lock()
last_alert_time = 0
ALERT_COOLDOWN = 3.0        # Seconds between alerts
EXCLUDED_CLASSES = {"frisbee", "bench", "train"}

# --- Frame grabber ---
class FrameGrabber:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()

    def _grab(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                print("Stream disconnected, reconnecting in 5s...")
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                print("Reconnected.")

    def get(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.cap.release()

# --- Inference thread ---
def inference_loop():
    global latest_frame, latest_detection, last_alert_time

    print("Loading TensorRT engine...")
    model = YOLO("yolov8l.engine", task="detect")

    print(f"Connecting to {RTSP_URL}...")
    grabber = FrameGrabber(RTSP_URL)

    # Wait for first frame
    for _ in range(50):
        if grabber.get() is not None:
            break
        time.sleep(0.1)

    print("Stream connected. Running inference...")

    while True:
        frame = grabber.get()
        if frame is None:
            time.sleep(0.01)
            continue

        results = model(frame, verbose=False, conf=SAVE_THRESHOLD)
        annotated = results[0].plot()

        # Encode annotated frame for streaming
        _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with latest_frame_lock:
            latest_frame = jpeg.tobytes()

        # Check for alertable detections
        boxes = results[0].boxes
        alert_detections = [
            b for b in boxes if float(b.conf) >= CONF_THRESHOLD
            and model.names[int(b.cls)] not in EXCLUDED_CLASSES
        ]

        if alert_detections:
            now = time.time()
            labels = [
                f"{model.names[int(b.cls)]}:{float(b.conf):.2f}"
                for b in alert_detections
            ]
            label_str = ", ".join(labels)

            # Save annotated frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"det_{ts}.jpg"
            fpath = os.path.join(DETECTION_DIR, fname)
            cv2.imwrite(fpath, annotated)

            # Log detection
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": fname,
                "labels": label_str
            }
            with detection_log_lock:
                detection_log.insert(0, entry)
                if len(detection_log) > MAX_GALLERY_IMAGES:
                    detection_log.pop()

            # Write to CSV
            csv_path = os.path.join(DETECTION_DIR, 'detections.csv')
            with open(csv_path, 'a') as f:
                f.write(f"{entry['timestamp']},{fname},{label_str}\n")

            # Trigger alert if cooldown elapsed
            if now - last_alert_time > ALERT_COOLDOWN:
                last_alert_time = now
                with latest_detection_lock:
                    latest_detection = {
                        "timestamp": entry["timestamp"],
                        "labels": label_str,
                        "t": now
                    }

            print(f"{entry['timestamp']}: {label_str}")
# --- Flask routes ---

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Jetson Vision</title>
    <style>
        body { background: #111; color: #eee; font-family: monospace; margin: 0; padding: 10px; }
        h1 { color: #0f0; margin: 0 0 10px 0; }
        #stream { width: 100%; max-width: 1200px; border: 2px solid #333; display: block; }
        #alert-bar { 
            background: #300; color: #f88; padding: 8px 12px; margin: 8px 0;
            border: 1px solid #f00; border-radius: 4px; display: none;
            font-size: 1.1em;
        }
        #alert-bar.active { display: block; background: #500; color: #ff0; border-color: #ff0; }
        #status { color: #888; font-size: 0.85em; margin: 4px 0 8px 0; }
        a { color: #0af; }
    </style>
</head>
<body>
    <h1>🎥 Jetson Vision</h1>
    <div id="status">Connecting...</div>
    <div id="alert-bar">⚠️ DETECTION: <span id="alert-text"></span></div>
    <img id="stream" src="/stream" alt="Live Stream">
    <p><a href="/gallery">📷 Detection Gallery</a></p>

    <audio id="alert-sound" preload="auto">
        <source src="/alert.wav" type="audio/wav">
    </audio>

    <script>
        var lastAlertTime = 0;
        var alertBar = document.getElementById('alert-bar');
        var alertText = document.getElementById('alert-text');
        var statusDiv = document.getElementById('status');
        var alertSound = document.getElementById('alert-sound');
        var alertTimeout = null;

        function pollDetections() {
            fetch('/latest_detection')
                .then(r => r.json())
                .then(data => {
                    statusDiv.textContent = 'Live — ' + new Date().toLocaleTimeString();
                    if (data.t && data.t !== lastAlertTime) {
                        lastAlertTime = data.t;
                        alertText.textContent = data.labels + ' @ ' + data.timestamp;
                        alertBar.classList.add('active');
                        alertSound.currentTime = 0;
                        alertSound.play().catch(() => {});
                        // Clear alert bar after 5 seconds
                        if (alertTimeout) clearTimeout(alertTimeout);
                        alertTimeout = setTimeout(() => {
                            alertBar.classList.remove('active');
                        }, 5000);
                    }
                })
                .catch(() => {
                    statusDiv.textContent = 'Connection lost — retrying...';
                });
        }

        setInterval(pollDetections, ''' + str(POLL_INTERVAL_MS) + ''');
    </script>
</body>
</html>'''

@app.route('/stream')
def stream():
    def generate():
        while True:
            with latest_frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.04)  # ~25fps cap on stream
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_detection')
def latest_detection_route():
    with latest_detection_lock:
        d = latest_detection
    if d:
        return jsonify(d)
    return jsonify({"t": 0})

@app.route('/alert.wav')
def alert_wav():
    # Generate alert sound on first request if it doesn't exist
    wav_path = '/tmp/alert.wav'
    if not os.path.exists(wav_path):
        os.system(f'sox -n -r 44100 -c 1 {wav_path} synth 0.3 sine 880 synth 0.3 sine 1100')
    return send_file(wav_path, mimetype='audio/wav')

@app.route('/gallery')
def gallery():
    # Load from disk on every request so it survives restarts
    entries = []
    csv_path = os.path.join(DETECTION_DIR, 'detections.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            for line in reversed(f.readlines()):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 2)
                if len(parts) == 3:
                    entries.append({
                        'timestamp': parts[0],
                        'filename': parts[1],
                        'labels': parts[2]
                    })

    # Pagination
    page_sizes = [50, 100, 250, 500, 'all']
    per_page = request.args.get('per_page', '50')
    per_page_int = len(entries) if per_page == 'all' else int(per_page)
    try:
        page = max(1, int(request.args.get('page', '1')))
    except:
        page = 1

    total = len(entries)
    total_pages = max(1, -(-total // per_page_int))
    page = min(page, total_pages)
    start = (page - 1) * per_page_int
    page_entries = entries[start:start + per_page_int]

    size_links = ' | '.join(
        f'<a href="/gallery?per_page={s}&page=1" '
        f'style="color:{"#ff0" if str(s) == str(per_page) else "#0af"}">{s}</a>'
        for s in page_sizes
    )

    def page_link(p, label):
        return f'<a href="/gallery?per_page={per_page}&page={p}" style="color:#0af;margin:0 4px">{label}</a>'

    prev_link = page_link(page - 1, '← Prev') if page > 1 else ''
    next_link = page_link(page + 1, 'Next →') if page < total_pages else ''
    page_info = f'Page {page} of {total_pages}'

    rows = ''
    for e in page_entries:
        label_cell = e['labels'] if e['labels'] else '<span style="color:#555">unknown</span>'
        rows += f"""
        <tr>
            <td style="padding:6px 12px;color:#888;white-space:nowrap">{e['timestamp']}</td>
            <td style="padding:6px 12px;color:#ff0">{label_cell}</td>
            <td style="padding:6px 12px">
                <a href="/detection_image/{e['filename']}" target="_blank"
                   style="color:#0af">view</a>
            </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Detection Log</title>
    <style>
        body {{ background: #111; color: #eee; font-family: monospace; padding: 10px; }}
        h1 {{ color: #0f0; margin: 0 0 8px 0; }}
        a {{ color: #0af; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
        tr:hover {{ background: #1a1a1a; }}
        th {{ color: #888; text-align: left; padding: 6px 12px;
              border-bottom: 1px solid #333; font-weight: normal; }}
    </style>
    <meta http-equiv="refresh" content="15">
</head>
<body>
    <h1>📋 Detection Log</h1>
    <p>
        <a href="/">← Live Stream</a> &nbsp;|&nbsp;
        {total} total detections &nbsp;|&nbsp;
        Show: {size_links}
    </p>
    <p>{prev_link} {page_info} {next_link}</p>
    <table>
        <tr>
            <th>Date/Time</th>
            <th>Detection</th>
            <th>Image</th>
        </tr>
        {rows if rows else '<tr><td colspan="3" style="color:#555;padding:12px">No detections yet.</td></tr>'}
    </table>
    <p style="margin-top:12px">{prev_link} {page_info} {next_link}</p>
</body>
</html>"""

@app.route('/detection_image/<filename>')
def detection_image(filename):
    fpath = os.path.join(DETECTION_DIR, filename)
    if os.path.exists(fpath):
        return send_file(fpath, mimetype='image/jpeg')
    return "Not found", 404

# --- Main ---
if __name__ == '__main__':
    # Start inference in background thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    # Wait for first frame before serving
    print("Waiting for inference to start...")
    time.sleep(5)

    print("Starting web server at http://jetson.local:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
