import socket
import cv2
import base64
import json
import asyncio
import sys
import os
import time
import subprocess
import netifaces
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from threading import Thread, Lock
from pathlib import Path
from pydantic import BaseModel
from auto_cleanup import bersihkan_log_lama
from camera import Camera
from firebase_auth import verify_firebase_token
from log_utils import get_log_folder, log_event, tulis_log_csv
from tts_utils import speak_label_threaded
from firestore_utils import threaded_send_detection_to_firestore
from sensor import gps_thread, mpu_thread, sensor_lock, sensor_data
from wifi import (
    scan_wifi,
    connect_to_wifi,
    check_internet,
    get_current_ssid,
    try_connect_and_verify,
    stop_hotspot,
    start_hotspot,
    write_wifi_config
)

app = FastAPI()
camera = Camera()
model = YOLO('models/rambuid.pt')

class WifiConnectRequest(BaseModel):
    ssid: str
    password: str

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

log_folder = get_log_folder()
log_txt_path = os.path.join(log_folder, "log.txt")
log_txt_file = open(log_txt_path, "a", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_txt_file)
sys.stderr = Tee(sys.stderr, log_txt_file)

with open('category/label_kategori.json', 'r', encoding='utf-8') as f:
    label_to_category = json.load(f)

clients = set()
send_interval = 5
recent_labels = {}
latest_payload = None
frame_lock = Lock()
annotated_frame = None  # Untuk streaming frame yang sudah dianotasi
ssid_lock = Lock()
current_ssid = None  # global var untuk menyimpan SSID saat connect
UID = None
def load_uid():
    global UID
    UID = get_uid_from_file()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== #
# 📤 WebSocket Route
# ====================== #
@app.get("/wifi/scan")
def scan():
    ssids = scan_wifi()
    return {"wifi": ssids}


@app.post("/wifi/connect")
def connect_wifi_endpoint(data: WifiConnectRequest):
    global current_ssid
    with ssid_lock:
        current_ssid = data.ssid  # ✅ simpan SSID yang berhasil dicoba
    if not data.ssid or len(data.password) < 8:
        raise HTTPException(status_code=400, detail="SSID/password tidak valid")

    write_wifi_config(data.ssid, data.password)
    stop_hotspot()
    subprocess.run(["sudo", "wpa_cli", "-i", "wlan0", "reconfigure"])
    time.sleep(10)

    for _ in range(6):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {"status": "connected", "fallback": False}
        except OSError:
            time.sleep(5)

    start_hotspot()
    return {"status": "failed", "fallback": True}


def get_ip_address():
    iface = 'wlan0'
    try:
        if iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                return addrs[netifaces.AF_INET][0]['addr']
    except Exception:
        pass
    return None


@app.get("/wifi/status")
def wifi_status():
    connected = False
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        connected = True
    except OSError:
        pass

    fallback = False
    hotspot_ssid = None

    hostapd_status = subprocess.run(
        ["systemctl", "is-active", "hostapd"],
        capture_output=True,
        text=True
    )
    if "active" in hostapd_status.stdout:
        fallback = True
        try:
            with open("/etc/hostapd/hostapd.conf") as f:
                for line in f:
                    if line.startswith("ssid="):
                        hotspot_ssid = line.strip().split("=")[1]
                        break
        except FileNotFoundError:
            pass

    ip_address = get_ip_address() if connected else None

    return {
        "connected": connected,
        "fallback": fallback,
        "ssid": hotspot_ssid if fallback else None,
        "ip": ip_address
    }


@app.get("/wifi/info")
def wifi_info():
    ssid = get_current_ssid()
    return {
        "ssid": ssid,
        "message": "Belum terhubung ke jaringan WiFi" if not ssid else None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            if latest_payload:
                await websocket.send_text(latest_payload)
            await asyncio.sleep(0.05)
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        clients.remove(websocket)

@app.get("/")
def read_root():
    return {"message": "FastAPI server aktif"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/deteksi")
async def post_uid(request: Request):
    try:
        data = await request.json()
        id_token = data.get("id_token")

        uid = verify_firebase_token(id_token)

        with open("device_info.json", "w") as f:
            json.dump({"user_id": uid}, f)

        if not uid:
            return {"status": "unauthorized", "detail": "Token tidak valid"}
        
        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/offline-detections")
async def get_offline_detections():
    try:
        file_path = Path("offline_detections.json")
        if not file_path.exists():
            return JSONResponse(content=[], status_code=200)

        with open(file_path, "r") as f:
            data = json.load(f)

        return JSONResponse(content=data, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ====================== #
# 🔁 Auto Infer Loop
# ====================== #

def get_uid_from_file():
    try:
        with open("device_info.json", "r") as f:
            data = json.load(f)
            return data.get("user_id")
    except Exception as e:
        print(f"[UID] Gagal ambil UID: {e}")
        return None

def get_current_gps_str():
    with sensor_lock:
        gps = sensor_data["gps"]
        return f"{gps.get('lat')},{gps.get('lon')}" if gps else "N/A"

def mjpeg_generator():
    while True:
        if annotated_frame is None:
            print("⚠️ Menunggu annotated_frame...")
            time.sleep(0.05)
            continue
        with frame_lock:
            frame = annotated_frame.copy() if annotated_frame is not None else None
        success, jpeg = cv2.imencode('.jpg', frame)
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("⚠️ Frame tidak valid.")
            continue
        if not success:
            print("❌ Gagal meng-encode JPEG.")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)

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

def infer_loop():
    global latest_payload, recent_labels, annotated_frame
    icons_dir = "icons"
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        results = model.predict(frame, conf=0.5, save=False, imgsz=320, device='cpu')[0]
        waktu_jakarta = datetime.now(ZoneInfo("Asia/Jakarta"))
        timestamp = waktu_jakarta.isoformat()
        formatted_time = waktu_jakarta.strftime("%d-%m-%Y %H:%M:%S") + " WIB"

        if results.boxes is None or len(results.boxes) == 0:
            continue

        detections = []
        for result in results:
            
            with sensor_lock:
                gps_info = sensor_data["gps"]
                mpu_info = sensor_data["mpu"]

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]
                kategori = label_to_category.get(label.lower().replace(" ", "-"), "Tidak Diketahui")
                current_time = time.time()

                # 📝 Tambahkan log ke file
                waktu = formatted_time
                gps_info_str = get_current_gps_str()

                tulis_log_csv(label, kategori, conf, gps_info_str, waktu)

                with ssid_lock:
                    ssid = current_ssid if current_ssid else "unknown"
                log_event(f"Detected label={label}, SSID={ssid}, waktu={waktu}")

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

            detections.append({
                "label": label,
                "confidence": float(conf),
                "kategori": kategori,
                "box": [x1, y1, x2, y2]
            })

            # 🔊 TTS + Firestore (1x per label dalam interval)
            if label not in recent_labels or (current_time - recent_labels[label]) >= send_interval:
                recent_labels[label] = current_time
                frame_to_send = cv2.resize(frame.copy(), (320, 240))
                speak_label_threaded(label)
                if UID:
                    threaded_send_detection_to_firestore(
                        label.strip().lower(), kategori,
                        x1, y1, x2, y2,
                        frame_to_send,
                        timestamp, formatted_time,
                        UID
                    )
                else:
                    print("⚠️ UID tidak tersedia.")

        success, jpeg = cv2.imencode('.jpg', frame)
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("⚠️ Frame tidak valid.")
            continue
        if not success:
            print("❌ Gagal meng-encode JPEG.")
            continue
        with frame_lock:
            annotated_frame = frame.copy()
        print("🔁 Annotated frame updated.")
        b64_image = base64.b64encode(jpeg).decode("utf-8")
        with sensor_lock:
            gps_info = sensor_data["gps"]
            mpu_info = sensor_data["mpu"]

        latest_payload = json.dumps({
            "image": b64_image,
            "detections": detections,
            "gps": gps_info,
            "mpu": mpu_info
        })

        cv2.imwrite("test_frame.jpg", annotated_frame)

        time.sleep(0.05)

# ====================== #
# 🚀 Start Infer Thread
# ====================== #
print("🚀 Infer loop started")
Thread(target=infer_loop, daemon=True).start()
Thread(target=gps_thread, daemon=True).start()
Thread(target=mpu_thread, daemon=True).start()
Thread(target=bersihkan_log_lama, daemon=True).start()