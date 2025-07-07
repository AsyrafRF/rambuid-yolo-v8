import cv2
import os
import time
import json
import pygame

from datetime import datetime
from zoneinfo import ZoneInfo
from ultralytics import YOLO

from tts_utils import speak_label_threaded
from firestore_utils import threaded_send_detection_to_firestore

# ============================ #
# 🔧 Inisialisasi Komponen     
# ============================ #

# Load mapping label → kategori dari file JSON eksternal
with open('label_kategori.json', 'r', encoding='utf-8') as f:
    label_to_category = json.load(f)

# Load model YOLOv8
model = YOLO('rambuid.pt')

# Folder ikon
icons_dir = 'icons'

# Buka webcam
cap = cv2.VideoCapture(0)

# Variabel kontrol pengiriman
recent_labels = {}
send_interval = 5  # detik antar deteksi label yang sama

# Teks
def draw_wrapped_text_with_background(img, text, origin, font, scale, text_color, thickness, max_width, bg_color=(0, 0, 0), alpha=0.5):
    """
    Gambar teks terbungkus dengan latar belakang semi-transparan.
    - origin: posisi awal (x, y)
    - bg_color: warna latar belakang (B, G, R)
    - alpha: transparansi latar (0.0 - 1.0)
    """
    words = text.split()
    lines = []
    current_line = ''

    # Bungkus teks
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
    line_height = h + 10  # spasi antar baris
    total_height = line_height * len(lines)
    max_line_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)

    # Buat latar belakang semi-transparan
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + max_line_width + 5, y + total_height - h + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Gambar teks
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, text_color, thickness)
        y += line_height

# ============================ #
# 🎯 Loop Deteksi Utama
# ============================ #

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka webcam.")
        break

    # Ambil waktu sekarang di zona Asia/Jakarta
    waktu_jakarta = datetime.now(ZoneInfo("Asia/Jakarta"))

     # Format waktu zona   
    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()
    formatted_time = waktu_jakarta.strftime("%d-%m-%Y %H:%M:%S") + " WIB"

    # Prediksi menggunakan model
    results = model.predict(source=frame, conf=0.5, save=False, imgsz=320, device='cpu')
    # results = model.predict(source=frame, conf=0.5, save=False, imgsz=416)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]
            print(f"[DEBUG] Deteksi model: label={label}, class_id={cls_id}")

            # Ambil Kategori
            kategori = label_to_category.get(label.lower().replace(" ", "-"), "Tidak Diketahui")
            if kategori == "Tidak Diketahui":
                print(f"[Warning] Kategori tidak ditemukan untuk label: {label}")

            # Ambil waktu sekarang
            current_time = time.time()

            # --- Normalisasi label ---
            label_norm = label.lower().replace(' ', '-').strip()

            # Path ikon
            icon_path = os.path.join(icons_dir, f"{label_norm}.png")

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Tambahkan kategori di bawah label
            cv2.putText(frame, f"Kategori: {kategori}", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Tampilkan ikon jika ada
            if os.path.exists(icon_path):
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, (100, 100))

                    # Posisi ikon
                    x_offset, y_offset = 10, 10
                    y1_icon, y2_icon = y_offset, y_offset + icon.shape[0]
                    x1_icon, x2_icon = x_offset, x_offset + icon.shape[1]

                    # Cek frame cukup besar
                    if y2_icon <= frame.shape[0] and x2_icon <= frame.shape[1]:
                        if icon.shape[0] > frame.shape[0] or icon.shape[1] > frame.shape[1]:
                            icon = cv2.resize(icon, (min(100, frame.shape[1]), min(100, frame.shape[0])))
                            print("Ukuran ikon lebih besar dari frame, dilewati.")
                            continue
                        if icon.shape[2] == 4:
                            alpha_s = icon[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(0, 3):
                                frame[y1_icon:y2_icon, x1_icon:x2_icon, c] = (
                                    alpha_s * icon[:, :, c] +
                                    alpha_l * frame[y1_icon:y2_icon, x1_icon:x2_icon, c]
                                )
                        else:
                            frame[y1_icon:y2_icon, x1_icon:x2_icon] = icon

                        # Tulis teks di bawah ikon
                        # Tentukan posisi teks agar tidak keluar dari frame
                        text = f"Ini adalah rambu {label}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                        # Coba tampilkan teks di bawah ikon, jika muat
                        if y2_icon + 30 + text_height < frame.shape[0]:
                            text_y = y2_icon + 30
                        else:
                            # Kalau tidak muat, tampilkan di atas ikon
                            text_y = y1_icon - 10
                            if text_y - text_height < 0:
                                text_y = y1_icon  # fallback ke posisi ikon jika terlalu atas

                        # Periksa jika teks terlalu lebar untuk frame
                        max_width = frame.shape[1] - 20  # beri margin 10 px kiri-kanan
                        draw_wrapped_text_with_background(
                            frame,
                            text,
                            origin=(10, text_y),
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            scale=0.7,
                            text_color=(0, 255, 255),
                            thickness=2,
                            max_width=frame.shape[1] - 20,
                            bg_color=(0, 0, 0),        # Hitam
                            alpha=0.6                 # Transparansi
                        )                        
                        if text_width > max_width:
                            while text_width > max_width and len(text) > 3:
                                text = text[:-4] + "..."  # potong dan beri elipsis
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)                       
                    else:
                        print("Frame terlalu kecil untuk menempelkan ikon.")
                else:
                    print(f"Gagal load ikon: {icon_path}")
            else:
                print(f"Tidak ditemukan ikon: {icon_path}")

            print(f"[DEBUG] Label hasil deteksi: '{label}'")

            # Kirim ke Firestore hanya jika label berubah atau waktu sudah lewat dan ucapkan jika perlu
            if label not in recent_labels or (current_time - recent_labels[label]) >= send_interval:
                recent_labels[label] = current_time

                # Buat salinan frame lengkap dengan anotasi
                frame_with_annotation = frame.copy()
                cv2.rectangle(frame_with_annotation, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_with_annotation, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_with_annotation, f"Kategori: {kategori}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) 

                # Resize agar lebih kecil saat dikirim
                frame_to_send = cv2.resize(frame_with_annotation, (320, 240))  # Opsi 160x120 - 320x240

                # Normalisasi label saat kirim
                label = label.strip().lower()
                print(f"🔧 Mengirim ke Firestore: label={label}, kategori={kategori}, coords=({x1},{y1},{x2},{y2})")

                if not isinstance(label, str) or label.strip() == "":
                    print(f"[SKIP] Label kosong atau tidak valid: {label}")
                    continue

                # Pengiriman dan proses TTS                          
                speak_label_threaded(label)                
                threaded_send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame_to_send, timestamp, formatted_time)
                print(f"Dikirim ke Firebase: {label} @ {datetime.now().isoformat()}")

    # Tampilkan hasil
    cv2.imshow('Webcam Deteksi Rambu', frame)

    # Tekan 'q' untuk keluar
    key = cv2.waitKey(1)  # Ganti dari 10 ke 1 atau 5
    if key & 0xFF == ord('q'):
        break

# ============================ #
# 🚪 Bersih-bersih
# ============================ #

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()