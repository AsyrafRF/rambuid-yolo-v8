import os
import shutil
from datetime import datetime, timedelta

def bersihkan_log_lama(folder_induk="logs", max_umur_hari=7):
    sekarang = datetime.now()
    batas_waktu = sekarang - timedelta(days=max_umur_hari)

    if not os.path.exists(folder_induk):
        return

    for nama_folder in os.listdir(folder_induk):
        path_folder = os.path.join(folder_induk, nama_folder)
        try:
            # Cek apakah nama folder adalah format tanggal
            tanggal_folder = datetime.strptime(nama_folder, "%Y-%m-%d")
            if tanggal_folder < batas_waktu:
                shutil.rmtree(path_folder)
                print(f"🧹 Menghapus folder log lama: {path_folder}")
        except ValueError:
            # Bukan folder dengan format tanggal
            continue
