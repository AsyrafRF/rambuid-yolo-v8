import os
from datetime import datetime

def tulis_log(teks):
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(teks + "\n")

def get_log_folder():
    tanggal = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("logs", tanggal)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def tulis_log_csv(label, kategori, confidence, gps, waktu):
    log_folder = get_log_folder()
    csv_path = os.path.join(log_folder, "deteksi.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Waktu,Label,Kategori,Confidence,GPS\n")
        f.write(f"{waktu},{label},{kategori},{confidence:.2f},{gps}\n")