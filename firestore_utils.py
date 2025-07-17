import json
import os
import threading
import cv2
import uuid
import base64
import firebase_admin
import requests
import numpy as np
import logging
from firebase_admin import credentials, firestore
from pathlib import Path
from sensor import sensor_data, sensor_lock

# Inisialisasi logging
logging.basicConfig(
    filename='sensor_error.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Inisialisasi Firebase hanya sekali
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Konstanta file dan koleksi
FIRESTORE_USER_COLLECTION = "users"
DEVICE_INFO_FILE = "device_info.json"
OFFLINE_DETECTIONS_FILE = "offline_detections.json"

# UID hanya diambil sekali setelah boot
CACHED_UID = None

# ===========================
# UID Management
# ===========================

def save_uid_to_file(uid):
    if not uid:
        return
    if not os.path.exists(DEVICE_INFO_FILE):
        data = {"user_id": uid, "device_id": str(uuid.uuid4())}
        with open(DEVICE_INFO_FILE, "w") as f:
            json.dump(data, f)
            logging.info(f"UID {uid} disimpan ke file.")
    else:
        with open(DEVICE_INFO_FILE, "r") as f:
            data = json.load(f)
        if not data.get("user_id"):
            data["user_id"] = uid
            with open(DEVICE_INFO_FILE, "w") as f:
                json.dump(data, f)
            logging.info(f"UID {uid} diperbarui di file.")

def get_user_id():
    global CACHED_UID
    if CACHED_UID:
        return CACHED_UID
    
    if os.path.exists(DEVICE_INFO_FILE):
        try:
            with open(DEVICE_INFO_FILE, "r") as f:
                data = json.load(f)
                uid = data.get("user_id")
                if uid:
                    logging.info(f"[INFO] UID ditemukan: {uid}")
                    CACHED_UID = uid
                    return uid
        except Exception as e:
            logging.error(f"[UID Error] {e}")
    return None

def get_device_id():
    if os.path.exists(DEVICE_INFO_FILE):
        with open(DEVICE_INFO_FILE, "r") as f:
            data = json.load(f)
            return data.get("device_id")
    device_id = str(uuid.uuid4())
    with open(DEVICE_INFO_FILE, "w") as f:
        json.dump({"device_id": device_id}, f)
    return device_id

# ===========================
# Koneksi dan Sinkronisasi
# ===========================

def is_connected():
    try:
        requests.get("https://firestore.googleapis.com", timeout=3)
        return True
    except requests.RequestException:
        return False

def save_offline(data):
    offline_file = Path(OFFLINE_DETECTIONS_FILE)
    existing = []
    if offline_file.exists():
        with open(offline_file, "r") as f:
            existing = json.load(f)
    existing.append(data)
    with open(offline_file, "w") as f:
        json.dump(existing, f, indent=2)
    logging.warning("[OFFLINE] Data disimpan lokal.")

def sync_offline_data():
    if not is_connected():
        logging.warning("[SYNC] Offline, tidak bisa sinkron.")
        return

    try:
        offline_file = Path(OFFLINE_DETECTIONS_FILE)
        if not offline_file.exists():
            return

        with open(offline_file, "r") as f:
            data = json.load(f)

        if not data:
            logging.info("[SYNC] Tidak ada data untuk disinkronkan.")
            return

        for d in data:
            doc_id = str(uuid.uuid4())
            db.collection(FIRESTORE_USER_COLLECTION).document(d["userId"]).collection("detections").document(doc_id).set(d)

        offline_file.unlink()
        logging.info("[SYNC] Semua data offline berhasil disinkronkan dan file lokal dihapus.")

    except Exception as e:
        logging.error(f"[SYNC ERROR] {e}")

# ===========================
# Validasi dan Pengiriman
# ===========================

def is_valid_label(label):
    return isinstance(label, str) and 0 < len(label.strip()) <= 100

def is_valid_frame(frame):
    return frame is not None and isinstance(frame, (np.ndarray,))

def send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time):
    user_id = get_user_id()
    if not user_id:
        logging.warning("[FIREBASE] UID tidak tersedia. Tidak menyimpan deteksi.")
        return

    label = label.strip().lower()
    if not is_valid_label(label):
        logging.warning(f"Label tidak valid: '{label}', tidak dikirim.")
        return

    if not is_valid_frame(frame):
        logging.warning("Frame tidak valid.")
        return

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    if len(jpg_as_text) > 800000:
        logging.warning("Gambar terlalu besar untuk Firestore.")
        return

    with sensor_lock:
        gps = sensor_data["gps"]
        mpu = sensor_data["mpu"]

    data = {
        "userId": user_id,
        "deviceId": get_device_id(),
        "label": label,
        "kategori": kategori,
        "timestamp": timestamp,
        "tanggal": formatted_time,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "lokasi": gps,
        "orientasi": mpu,
        "image_base64": jpg_as_text
    }

    try:
        if is_connected():
            doc_id = str(uuid.uuid4())
            db.collection(FIRESTORE_USER_COLLECTION).document(user_id).collection("detections").document(doc_id).set(data)
            logging.info(f"[ONLINE] Data dikirim ke Firestore: {label} @ {timestamp}")
            sync_offline_data()
        else:
            save_offline(data)
    except Exception as e:
        logging.error(f"[ERROR FIREBASE] {e}")
        save_offline(data)

def threaded_send_detection_to_firestore(*args):
    threading.Thread(target=send_detection_to_firestore, args=args, daemon=True).start()

# Untuk testing manual UID
if __name__ == "__main__":
    uid = get_user_id()
    print("UID aktif:", uid)
