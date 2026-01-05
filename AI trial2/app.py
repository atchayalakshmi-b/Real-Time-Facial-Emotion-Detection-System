# app.py
import os
from flask import Flask, render_template, Response, jsonify, request, send_file, make_response
from collections import Counter, deque
from datetime import datetime
import cv2
from deepface import DeepFace
import threading
import csv
import io
import numpy as np

# Prevent onednn TF optimizations on some systems causing issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# ---------------- Configuration ----------------
ANALYZE_EVERY_N = 3           # analyze every N frames to save CPU
SMOOTH_WINDOW = 8             # smoothing window for dominant calculation
HISTORY_MAX = 5000            # max stored emotions in memory
TIMELINE_MAX = 40             # items returned to frontend for timeline

# Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Thread-safe shared state
state_lock = threading.Lock()
emotion_history = deque(maxlen=HISTORY_MAX)   # list of (iso_ts, emotion, confidence)
recent_window = deque(maxlen=SMOOTH_WINDOW)   # smoothing window of latest emotions
per_face_ids = {}                              # optional id tracking in future (not persistent)
global_stats = {
    "session_start": None,
    "session_end": None
}

# Quotes per emotion
EMOTION_QUOTES = {
    "happy": "Happiness is contagious — keep smiling!",
    "sad": "It's okay to feel sad. This too shall pass.",
    "angry": "Pause. Breathe. Respond, don’t react.",
    "neutral": "Calmness is the cradle of power.",
    "surprise": "Life is full of surprises!",
    "surprised": "Life is full of surprises!",
    "fear": "Courage grows by facing fears.",
    "fearful": "Courage grows by facing fears.",
    "disgust": "Focus on what brings you joy."
}

# Colors per emotion (BGR)
EMOTION_COLORS = {
    "happy": (0, 200, 0),
    "sad": (200, 0, 0),
    "angry": (0, 0, 200),
    "neutral": (120, 120, 120),
    "surprise": (0, 200, 200),
    "surprised": (0, 200, 200),
    "fear": (128, 0, 128),
    "fearful": (128, 0, 128),
    "disgust": (0, 128, 0)
}

VALID_FILTERS = {"none", "grayscale", "sepia", "cartoon"}

# ---------------- Helper functions ----------------
def apply_filter(img_bgr, filter_name):
    if filter_name == "grayscale":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if filter_name == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img_bgr, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    if filter_name == "cartoon":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray_blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=200, sigmaSpace=200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    return img_bgr

def mode_smooth(values):
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    candidates = [v for v, c in counts.items() if c == max_count]
    for v in reversed(values):
        if v in candidates:
            return v
    return values[-1]

# ---------------- Video & Analysis ----------------
def generate_frames(filter_name="none"):
    # mark session start
    with state_lock:
        if global_stats["session_start"] is None:
            global_stats["session_start"] = datetime.now().isoformat(timespec='seconds')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        analyze_now = (frame_idx % ANALYZE_EVERY_N == 0)

        face_labels = []
        for (x, y, w, h) in faces:
            face_roi_rgb = rgb_frame[y:y+h, x:x+w]
            emotion = None
            confidence = 0.0

            if analyze_now:
                try:
                    res = DeepFace.analyze(img_path=face_roi_rgb, actions=['emotion'], enforce_detection=False)
                    res = res[0] if isinstance(res, list) else res
                    emotion_raw = res.get('dominant_emotion') or next(iter(res.get('emotion', {}).keys()), 'neutral')
                    emotion = str(emotion_raw).lower()
                    probs = res.get('emotion', {})
                    confidence = float(probs.get(emotion, 0.0))
                except Exception as e:
                    # fail gracefully
                    emotion = "neutral"
                    confidence = 0.0

                ts = datetime.now().isoformat(timespec='seconds')
                with state_lock:
                    emotion_history.append((ts, emotion, round(confidence, 2)))
                    recent_window.append(emotion)

            # smoothing dominant for display on each face
            display_emotion = emotion if emotion else "neutral"
            color = EMOTION_COLORS.get(display_emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{display_emotion} {confidence:.0f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            face_labels.append(display_emotion)

        # update global dominant (smoothed)
        with state_lock:
            smooth = mode_smooth(list(recent_window))
            global_dominant = smooth or "neutral"

        # Apply filter and make sure frame is 3-channel BGR for encoding
        filtered = apply_filter(frame, filter_name)
        if isinstance(filtered, np.ndarray) and filtered.ndim == 2:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

        # overlay info pill
        cv2.putText(filtered, f"Dominant: {global_dominant}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        ret, buf = cv2.imencode('.jpg', filtered)
        if not ret:
            continue
        frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_idx += 1

    cap.release()
    with state_lock:
        global_stats["session_end"] = datetime.now().isoformat(timespec='seconds')

# ---------------- Flask routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    filter_name = request.args.get('filter', 'none').lower()
    if filter_name not in VALID_FILTERS:
        filter_name = 'none'
    return Response(generate_frames(filter_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_data')
def emotion_data():
    """
    Returns JSON:
    {
      counts: {emotion: count, ...},
      dominant: "emotion",
      timeline: [{t: "...", e: "emotion", c: confidence}, ...],
      quote: "..."
    }
    """
    with state_lock:
        counts = Counter([e for (_, e, _) in emotion_history])
        timeline = list(emotion_history)[-TIMELINE_MAX:]
        # determine dominant by smoothing recent window
        dom = mode_smooth(list(recent_window)) or (list(counts.keys())[0] if counts else "neutral")
        quote = EMOTION_QUOTES.get(dom, "")
    return jsonify({
        "counts": counts,
        "dominant": dom,
        "timeline": [{"t": t, "e": e, "c": c} for (t, e, c) in timeline],
        "quote": quote
    })

@app.route('/download_report')
def download_report():
    with state_lock:
        data = list(emotion_history)

    proxy = io.StringIO()
    writer = csv.writer(proxy)
    writer.writerow(["timestamp", "emotion", "confidence"])
    for ts, emo, conf in data:
        writer.writerow([ts, emo, conf])

    mem = io.BytesIO()
    mem.write(proxy.getvalue().encode('utf-8'))
    mem.seek(0)
    proxy.close()

    return send_file(mem,
                     as_attachment=True,
                     download_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                     mimetype='text/csv')

@app.route('/session_summary')
def session_summary():
    with state_lock:
        data = list(emotion_history)
    counts = Counter([e for (_, e, _) in data])
    total = sum(counts.values()) or 1
    dominant = counts.most_common(1)[0][0] if counts else "neutral"
    # top confidences average per emotion
    per_emo_conf = {}
    for (_, e, c) in data:
        per_emo_conf.setdefault(e, []).append(c)
    avg_conf = {e: round(sum(lst)/len(lst),2) for e, lst in per_emo_conf.items()}

    return jsonify({
        "session_start": global_stats.get("session_start"),
        "session_end": global_stats.get("session_end"),
        "counts": counts,
        "dominant": dominant,
        "avg_confidence": avg_conf
    })


if __name__ == '__main__':
    # threaded so UI endpoints still respond while video stream runs
    app.run(debug=True, threaded=True)
