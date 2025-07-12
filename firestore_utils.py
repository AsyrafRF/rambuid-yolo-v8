import json
import os
import threading
import cv2
import uuid
import base64
import firebase_admin
import requests  # Untuk cek koneksi
import numpy as np

from firebase_admin import credentials, firestore
from pathlib import Path
from sensor import sensor_data, sensor_lock

# Inisialisasi Firebase hanya sekali
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# ============================ #
# 📤 Kirim Deteksi ke Firestore
# ============================ #

FIRESTORE_USER_COLLECTION = "users"
DEVICE_INFO_FILE = "device_info.json"
OFFLINE_DETECTIONS_FILE = "offline_detections.json"

def get_user_ids():
    from detectserver import post_uid
    try:
        users = post_uid
        user_ids = [user.id for user in users]
        print(f"[DEBUG] UID aktif ditemukan: {user_ids}")
        return user_ids
    except Exception as e:
        print(f"Error ambil user aktif: {e}")
        return []

def is_connected():
    try:
        requests.get("https://firestore.googleapis.com", timeout=3)
        return True
    except requests.RequestException:
        return False

def get_device_id():
    if os.path.exists(DEVICE_INFO_FILE):
        with open(DEVICE_INFO_FILE, "r") as f:
            data = json.load(f)
            return data.get("device_id")
    device_id = str(uuid.uuid4())
    with open(DEVICE_INFO_FILE, "w") as f:
        json.dump({"device_id": device_id}, f)
    return device_id

def save_offline(data):
    offline_file = Path(OFFLINE_DETECTIONS_FILE)
    if offline_file.exists():
        with open(offline_file, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    existing.append(data)
    with open(offline_file, "w") as f:
        json.dump(existing, f, indent=2)
    print("[OFFLINE] Data disimpan lokal.")

def sync_offline_data():
    if not is_connected():
        print("[SYNC] Offline, tidak bisa sinkron.")
        return

    try:
        offline_file = Path(OFFLINE_DETECTIONS_FILE)
        if not offline_file.exists():
            return

        with open(offline_file, "r") as f:
            data = json.load(f)

        if not data:
                print("[SYNC] Tidak ada data untuk disinkronkan.")
                return

        print(f"[SYNC] Sinkronisasi {len(data)} data lokal ke Firestore...")

        for d in data:
            try:
                doc_id = str(uuid.uuid4())
                db.collection(FIRESTORE_USER_COLLECTION).document(d["userId"]).collection("detections").document(doc_id).set(d)
            except Exception as e:
                print(f"[SYNC ERROR] Gagal kirim data: {e}")
                return  # Hentikan dan jangan hapus file kalau ada yang gagal
        
    except Exception as e:
        print(f"[SYNC ERROR] Terjadi error saat membaca/sinkronisasi: {e}")

    # Hapus file hanya jika semua berhasil
    offline_file.unlink()
    print("[SYNC] Semua data offline berhasil disinkronkan dan file lokal dihapus.")

def is_valid_label(label):
    if not isinstance(label, str):
        return False
    label = label.strip()
    if len(label) == 0 or len(label) > 100:
        return False
    return True

def is_valid_frame(frame):
    return frame is not None and isinstance(frame, (np.ndarray,))

def send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time, uid):

    label = label.strip().lower()

    if not is_valid_label(label):
        print(f"Label tidak valid: '{label}', tidak dikirim.")
        return 
    
    if not is_valid_frame(frame):
        print("Frame tidak valid.")
        return
    
    user_ids = get_user_ids()
    if not user_ids:
        print("UID tidak tersedia. Tidak menyimpan deteksi.")
        return

    # Encode frame ke base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Pastikan size gambar tidak berlebihan
    if len(jpg_as_text) > 800000:
        print("Gambar terlalu besar untuk Firestore.")
        return

    for uid in user_ids():
        print(f"[INFO] Kirim data Firestore: label='{label}', kategori='{kategori}'")
        with sensor_lock:
            gps = sensor_data["gps"]
            mpu = sensor_data["mpu"]

        device_id = get_device_id()
            
        data = {
            "userId": uid,
            "deviceId": device_id,
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

        # print(f"[DEBUG] Data akan dikirim untuk UID {uid}: {data}")
        print(f"[DEBUG] Data akan dikirim untuk UID {uid}")
        # print("[PAYLOAD]", json.dumps(data, indent=2))

        if is_connected():
            try:
                doc_id = str(uuid.uuid4())
                db.collection(FIRESTORE_USER_COLLECTION).document(uid).collection("detections").document(doc_id).set(data)
                print(f"[ONLINE] Data dikirim ke Firestore untuk UID: {uid}")
                print(f"[Firebase] Dikirim: {label} oleh UID {user_ids} @ {timestamp}")
                sync_offline_data()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Error Firebase] Gagal kirim untuk {uid}: {e}")
                save_offline(data)
        else:
            print("[OFFLINE] Tidak ada koneksi. Simpan lokal.")
            save_offline(data)

def threaded_send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time, uid):
    threading.Thread(
        target=send_detection_to_firestore,
        args=(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time, uid),
        daemon=True
    ).start()
