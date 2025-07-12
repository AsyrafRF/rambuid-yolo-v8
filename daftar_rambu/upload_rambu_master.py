import csv
import base64
import os
from firebase_admin import credentials, firestore, initialize_app

# Inisialisasi Firebase Admin
cred = credentials.Certificate("firebase-key.json")
initialize_app(cred)
db = firestore.client()

# Path CSV
csv_path = 'rambu_indonesia.csv'

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Gambar utama
        image_base64 = ""
        if os.path.exists(row['file_gambar']):
            with open(row['file_gambar'], 'rb') as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        else:
            print(f"[!] Gambar utama tidak ditemukan: {row['file_gambar']}")

        # Galeri tambahan
        galeri = []
        galeri_paths_raw = row.get('galeri_paths', '').strip().lower()

        # Hanya proses jika isinya bukan 'none', 'tidakada', kosong, atau 'galeri' tanpa path
        if galeri_paths_raw and galeri_paths_raw not in ['none', 'tidakada', 'galeri']:
            paths = [p.strip() for p in row['galeri_paths'].split(',')]
            for idx, path in enumerate(paths):
                if os.path.exists(path):
                    with open(path, 'rb') as img_file:
                        encoded = base64.b64encode(img_file.read()).decode('utf-8')
                        galeri.append({
                            'caption': f"Gambar tambahan {idx + 1}",
                            'ikon_base64': encoded
                        })
                else:
                    print(f"[!] Gambar galeri tidak ditemukan: {path}")

        # Dokumen Firestore
        doc = {
            'id': row['id'],
            'nama': row['nama'],
            'jenis': row['jenis'],
            'deskripsi': row['deskripsi'],
            'ikon_base64': image_base64,
            'galeri': galeri  # ← ditambahkan di sini
        }

        # Pastikan semua item di galeri valid
        galeri_valid = []
        for item in galeri:
            if isinstance(item, dict) and 'caption' in item and 'ikon_base64' in item:
                if isinstance(item['caption'], str) and isinstance(item['ikon_base64'], str):
                    galeri_valid.append(item)

        doc['galeri'] = galeri_valid

        db.collection('master_rambu').document(row['id']).set(doc)
        print(f"[✓] Upload: {row['id']} - {row['nama']} (galeri: {len(galeri)})")
