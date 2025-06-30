import cv2
import os
import time
import json
import pygame
from flask import Flask, Response, jsonify
from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO

from tts_utils import speak_label_threaded
from firestore_utils import threaded_send_detection_to_firestore

# ====================== #
# 🔧 Inisialisasi
# ====================== #
app = Flask(__name__)
model = YOLO('rambuid.pt')
with open('label_kategori.json', 'r', encoding='utf-8') as f:
    label_to_category = json.load(f)

icons_dir = 'icons'
send_interval = 5
recent_labels = {}
pygame.mixer.init()

# ====================== #
# 🔁 Stream Deteksi
# ====================== #
def draw_wrapped_text_with_background(img, text, origin, font, scale, text_color, thickness, max_width, bg_color=(0, 0, 0), alpha=0.5):
    words = text.split()
    lines, current_line = [], ''
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    x, y = origin
    line_height = h + 10
    total_height = line_height * len(lines)
    max_line_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + max_line_width + 5, y + total_height - h + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, text_color, thickness)
        y += line_height

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, conf=0.5, save=False, imgsz=320, device='cpu')
        waktu_jakarta = datetime.now(ZoneInfo("Asia/Jakarta"))
        timestamp = waktu_jakarta.isoformat()
        formatted_time = waktu_jakarta.strftime("%d-%m-%Y %H:%M:%S") + " WIB"

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]
                kategori = label_to_category.get(label.lower().replace(" ", "-"), "Tidak Diketahui")
                current_time = time.time()

                label_norm = label.lower().replace(' ', '-').strip()
                icon_path = os.path.join(icons_dir, f"{label_norm}.png")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Kategori: {kategori}", (x1, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                if os.path.exists(icon_path):
                    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                    if icon is not None:
                        icon = cv2.resize(icon, (100, 100))
                        x_offset, y_offset = 10, 10
                        y1_icon, y2_icon = y_offset, y_offset + icon.shape[0]
                        x1_icon, x2_icon = x_offset, x_offset + icon.shape[1]
                        if icon.shape[2] == 4:
                            alpha_s = icon[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(3):
                                frame[y1_icon:y2_icon, x1_icon:x2_icon, c] = (
                                    alpha_s * icon[:, :, c] +
                                    alpha_l * frame[y1_icon:y2_icon, x1_icon:x2_icon, c]
                                )
                        else:
                            frame[y1_icon:y2_icon, x1_icon:x2_icon] = icon

                        draw_wrapped_text_with_background(
                            frame,
                            f"Ini adalah rambu {label}",
                            origin=(10, y2_icon + 30),
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            scale=0.7,
                            text_color=(0, 255, 255),
                            thickness=2,
                            max_width=frame.shape[1] - 20
                        )

                if label not in recent_labels or (current_time - recent_labels[label]) >= send_interval:
                    recent_labels[label] = current_time
                    frame_to_send = cv2.resize(frame.copy(), (320, 240))
                    speak_label_threaded(label)
                    threaded_send_detection_to_firestore(
                        label.strip().lower(), kategori,
                        x1, y1, x2, y2,
                        frame_to_send,
                        timestamp, formatted_time
                    )

        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ====================== #
# 🔌 Flask Routes
# ====================== #

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/healthz')
@app.route('/status')
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Deteksi rambu jalan aktif",
        "timestamp": datetime.now().isoformat()
    })

# ====================== #
# 🚀 Main App
# ====================== #

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
