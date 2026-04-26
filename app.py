import csv
import json
import os
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
SCREENSHOT_DIR = BASE_DIR / "static" / "screenshots"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
HISTORY_FILE = LOG_DIR / "detection_history.csv"
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR))

from ultralytics import YOLO

ALLOWED_IMAGES = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEOS = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_CONFIDENCE = 0.25

model_cache = {}
webcam_lock = threading.Lock()
webcam_capture = None
webcam_active = False
webcam_config = {"camera": 0, "confidence": DEFAULT_CONFIDENCE, "model": DEFAULT_MODEL, "quality": "balanced"}
latest_frame = None
latest_summary = {
    "total": 0,
    "counts": {},
    "fps": 0,
    "detections": [],
    "source": "webcam",
    "updated_at": None,
}


def ensure_directories():
    for folder in [UPLOAD_DIR, OUTPUT_DIR, SCREENSHOT_DIR, MODEL_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        with HISTORY_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "source", "file_name", "total_objects", "class_counts", "output_file"])


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def unique_name(filename):
    stem = Path(secure_filename(filename)).stem or "upload"
    suffix = Path(filename).suffix.lower()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{stamp}_{uuid.uuid4().hex[:8]}{suffix}"


def get_model(model_name=DEFAULT_MODEL):
    model_path = str(model_name or DEFAULT_MODEL)
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]


def class_color(class_id):
    palette = np.array(
        [
            [0, 212, 255],
            [36, 255, 154],
            [255, 177, 0],
            [255, 79, 216],
            [124, 92, 255],
            [255, 92, 92],
            [0, 245, 212],
            [255, 137, 6],
            [160, 231, 229],
            [190, 242, 100],
            [251, 113, 133],
            [56, 189, 248],
        ],
        dtype=np.uint8,
    )
    return tuple(int(value) for value in palette[class_id % len(palette)])


def draw_detection(frame, box, label, confidence, color):
    x1, y1, x2, y2 = [int(value) for value in box]
    line = max(2, round(0.003 * (frame.shape[0] + frame.shape[1]) / 2))
    corner = max(16, min(42, int((x2 - x1) * 0.18)))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)

    # Bright corner rails make the result easier to read than a plain rectangle.
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, line + 1)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, line + 1)
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, line + 1)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, line + 1)
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, line + 1)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, line + 1)
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, line + 1)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, line + 1)

    caption = f"{label} {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(0.8, frame.shape[1] / 1500))
    thickness = max(2, line)
    (text_w, text_h), baseline = cv2.getTextSize(caption, font, font_scale, thickness)
    label_y = max(y1, text_h + 12)
    cv2.rectangle(frame, (x1, label_y - text_h - 12), (x1 + text_w + 18, label_y + baseline - 3), (6, 10, 20), -1)
    cv2.rectangle(frame, (x1, label_y - text_h - 12), (x1 + 5, label_y + baseline - 3), color, -1)
    cv2.putText(frame, caption, (x1 + 10, label_y - 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def quality_settings(quality):
    settings = {
        "fast": {"imgsz": 640, "iou": 0.45, "max_det": 100},
        "balanced": {"imgsz": 832, "iou": 0.5, "max_det": 200},
        "accurate": {"imgsz": 960, "iou": 0.55, "max_det": 300},
    }
    return settings.get(quality or "balanced", settings["balanced"])


def annotate_frame(frame, model_name=DEFAULT_MODEL, confidence=DEFAULT_CONFIDENCE, quality="balanced"):
    model = get_model(model_name)
    settings = quality_settings(quality)
    results = model.predict(frame, conf=float(confidence), verbose=False, **settings)
    detections = []
    counts = Counter()

    annotated = frame.copy()
    for result in results:
        names = result.names
        for box in result.boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            label = names[class_id]
            xyxy = box.xyxy[0].cpu().numpy()
            color = class_color(class_id)
            draw_detection(annotated, xyxy, label, score, color)
            counts[label] += 1
            detections.append(
                {
                    "class": label,
                    "confidence": round(score * 100, 2),
                    "box": [int(value) for value in xyxy],
                }
            )

    return annotated, {
        "total": int(sum(counts.values())),
        "counts": dict(counts),
        "detections": detections,
    }


def append_history(source, file_name, summary, output_file):
    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source,
                file_name,
                summary.get("total", 0),
                json.dumps(summary.get("counts", {}), ensure_ascii=False),
                output_file,
            ]
        )


def read_history(limit=30):
    if not HISTORY_FILE.exists():
        return []
    with HISTORY_FILE.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows[-limit:][::-1]


def relative_static_url(path):
    return "/" + path.relative_to(BASE_DIR).as_posix()


ensure_directories()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect_image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image file."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_IMAGES):
        return jsonify({"error": "Supported image formats: jpg, jpeg, png, bmp, webp."}), 400

    confidence = float(request.form.get("confidence", DEFAULT_CONFIDENCE))
    model_name = request.form.get("model", DEFAULT_MODEL)
    quality = request.form.get("quality", "balanced")
    filename = unique_name(file.filename)
    input_path = UPLOAD_DIR / filename
    output_path = OUTPUT_DIR / f"detected_{Path(filename).stem}.jpg"
    file.save(input_path)

    image = cv2.imread(str(input_path))
    if image is None:
        return jsonify({"error": "Unable to read the uploaded image."}), 400

    start = time.time()
    annotated, summary = annotate_frame(image, model_name, confidence, quality)
    summary["fps"] = round(1 / max(time.time() - start, 1e-6), 2)
    cv2.imwrite(str(output_path), annotated)
    append_history("image", file.filename, summary, output_path.name)

    return jsonify(
        {
            "message": "Image processed successfully.",
            "summary": summary,
            "output_url": relative_static_url(output_path),
            "download_url": f"/download/{output_path.name}",
            "history": read_history(),
        }
    )


@app.route("/detect_video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "Please upload a video file."}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename, ALLOWED_VIDEOS):
        return jsonify({"error": "Supported video formats: mp4, avi, mov, mkv, webm."}), 400

    confidence = float(request.form.get("confidence", DEFAULT_CONFIDENCE))
    model_name = request.form.get("model", DEFAULT_MODEL)
    quality = request.form.get("quality", "balanced")
    filename = unique_name(file.filename)
    input_path = UPLOAD_DIR / filename
    output_path = OUTPUT_DIR / f"detected_{Path(filename).stem}.mp4"
    file.save(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return jsonify({"error": "Unable to open the uploaded video."}), 400

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    aggregate_counts = Counter()
    total_frames = 0
    start = time.time()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated, frame_summary = annotate_frame(frame, model_name, confidence, quality)
        writer.write(annotated)
        aggregate_counts.update(frame_summary["counts"])
        total_frames += 1

    capture.release()
    writer.release()

    elapsed = max(time.time() - start, 1e-6)
    summary = {
        "total": int(sum(aggregate_counts.values())),
        "counts": dict(aggregate_counts),
        "fps": round(total_frames / elapsed, 2),
        "frames": total_frames,
        "detections": [],
    }
    append_history("video", file.filename, summary, output_path.name)

    return jsonify(
        {
            "message": "Video processed successfully.",
            "summary": summary,
            "output_url": relative_static_url(output_path),
            "download_url": f"/download/{output_path.name}",
            "history": read_history(),
        }
    )


@app.route("/upload_model", methods=["POST"])
def upload_model():
    if "model" not in request.files:
        return jsonify({"error": "Please upload a YOLO .pt model file."}), 400
    file = request.files["model"]
    if file.filename == "" or not file.filename.lower().endswith(".pt"):
        return jsonify({"error": "Only .pt model files are supported."}), 400
    filename = unique_name(file.filename)
    model_path = MODEL_DIR / filename
    file.save(model_path)
    return jsonify({"message": "Custom model uploaded.", "model": str(model_path)})


@app.route("/webcam/start", methods=["POST"])
def start_webcam():
    global webcam_capture, webcam_active, webcam_config
    data = request.get_json(silent=True) or {}
    camera_index = int(data.get("camera", 0))
    confidence = float(data.get("confidence", DEFAULT_CONFIDENCE))
    model_name = data.get("model", DEFAULT_MODEL)
    quality = data.get("quality", "balanced")

    with webcam_lock:
        if webcam_capture is not None:
            webcam_capture.release()
        webcam_capture = cv2.VideoCapture(camera_index)
        if not webcam_capture.isOpened():
            webcam_capture = None
            webcam_active = False
            return jsonify({"error": "Camera not found. Check camera index or permissions."}), 400
        webcam_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        webcam_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        webcam_config = {"camera": camera_index, "confidence": confidence, "model": model_name, "quality": quality}
        webcam_active = True
    return jsonify({"message": "Webcam detection started."})


@app.route("/webcam/stop", methods=["POST"])
def stop_webcam():
    global webcam_capture, webcam_active
    with webcam_lock:
        webcam_active = False
        if webcam_capture is not None:
            webcam_capture.release()
            webcam_capture = None
    append_history("webcam", "live_camera", latest_summary, "webcam_session")
    return jsonify({"message": "Webcam detection stopped.", "history": read_history()})


def generate_webcam_frames():
    global latest_frame, latest_summary, webcam_active, webcam_capture
    last_time = time.time()
    while True:
        with webcam_lock:
            active = webcam_active
            capture = webcam_capture
            config = webcam_config.copy()
        if not active or capture is None:
            time.sleep(0.1)
            continue

        ok, frame = capture.read()
        if not ok:
            with webcam_lock:
                webcam_active = False
            break

        annotated, summary = annotate_frame(frame, config["model"], config["confidence"], config.get("quality", "balanced"))
        now = time.time()
        fps = 1 / max(now - last_time, 1e-6)
        last_time = now
        summary.update({"fps": round(fps, 2), "source": "webcam", "updated_at": datetime.now().strftime("%H:%M:%S")})
        latest_summary = summary
        latest_frame = annotated.copy()

        cv2.putText(annotated, f"FPS: {fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 170), 2, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/video_feed")
def video_feed():
    return Response(generate_webcam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/webcam/status")
def webcam_status():
    return jsonify({"active": webcam_active, "summary": latest_summary})


@app.route("/screenshot", methods=["POST"])
def screenshot():
    if latest_frame is None:
        return jsonify({"error": "Start webcam detection before capturing a screenshot."}), 400
    output_path = SCREENSHOT_DIR / f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output_path), latest_frame)
    append_history("screenshot", "webcam", latest_summary, output_path.name)
    return jsonify({"message": "Screenshot saved.", "output_url": relative_static_url(output_path), "download_url": f"/download/{output_path.name}"})


@app.route("/history")
def history():
    return jsonify({"history": read_history()})


@app.route("/download/<filename>")
def download(filename):
    safe_name = secure_filename(filename)
    for folder in [OUTPUT_DIR, SCREENSHOT_DIR, LOG_DIR]:
        file_path = folder / safe_name
        if file_path.exists():
            return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found."}), 404


@app.route("/download_history")
def download_history():
    return send_file(HISTORY_FILE, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
