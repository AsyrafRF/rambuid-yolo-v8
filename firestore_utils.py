import json
import threading
import cv2
import uuid
import base64
import firebase_admin

import numpy as np

from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter
from sensor import sensor_data, sensor_lock

# Inisialisasi Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ============================ #
# 📤 Kirim Deteksi ke Firestore
# ============================ #

FIRESTORE_USER_COLLECTION = "users"

def get_active_user_ids():
    try:
        users = db.collection("active_user").where(filter=FieldFilter("active", "==", True)).stream()
        user_ids = [user.id for user in users]
        print(f"[DEBUG] UID aktif ditemukan: {user_ids}")
        return user_ids
    except Exception as e:
        print(f"Error ambil user aktif: {e}")
        return []

def is_valid_label(label):
    if not isinstance(label, str):
        return False
    label = label.strip()
    if len(label) == 0 or len(label) > 100:
        return False
    return True

def is_valid_frame(frame):
    return frame is not None and isinstance(frame, (np.ndarray,))

def send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time):

    label = label.strip().lower()

    if not is_valid_label(label):
        print(f"Label tidak valid: '{label}', tidak dikirim.")
        return 
    
    if not is_valid_frame(frame):
        print("Frame tidak valid.")
        return
    
    user_ids = get_active_user_ids()
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

    for uid in user_ids:
        print(f"[INFO] Kirim data Firestore: label='{label}', kategori='{kategori}'")
        with sensor_lock:
            gps = sensor_data["gps"]
            mpu = sensor_data["mpu"]
            
        data = {
            "userId": uid,
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

        print(f"[DEBUG] Data akan dikirim untuk UID {uid}: {data}")

        print("[PAYLOAD]", json.dumps(data, indent=2))

        try:
            doc_id = str(uuid.uuid4())
            db.collection(FIRESTORE_USER_COLLECTION).document(uid).collection("detections").document(doc_id).set(data)
            print(f"[Firebase] Dikirim: {label} oleh UID {user_ids} @ {timestamp}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Error Firebase] Gagal kirim untuk {uid}: {e}")

def threaded_send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time):
    threading.Thread(
        target=send_detection_to_firestore,
        args=(label, kategori, x1, y1, x2, y2, frame, timestamp, formatted_time),
        daemon=True
    ).start()
