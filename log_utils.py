import logging
import os
from datetime import datetime

# Folder log harian
def get_log_folder():
    tanggal = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join("logs", tanggal)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# Logger utama
log_folder = get_log_folder()
main_log_path = os.path.join(log_folder, "main.log")
sensor_log_path = os.path.join(log_folder, "sensor_error.log")

# Setup logging untuk log umum
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(main_log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Logger terpisah untuk sensor
sensor_logger = logging.getLogger("sensor_logger")
sensor_handler = logging.FileHandler(sensor_log_path, encoding='utf-8')
sensor_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
sensor_logger.addHandler(sensor_handler)
sensor_logger.setLevel(logging.WARNING)

def log_event(msg: str):
    logging.info(msg)

def log_error(msg: str):
    logging.error(msg)

def tulis_log_csv(label, kategori, confidence, gps, mpu, waktu):
    csv_path = os.path.join(log_folder, "deteksi.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Waktu,Label,Kategori,Confidence,GPS,MPU\n")
        f.write(f"{waktu},{label},{kategori},{confidence:.2f},{gps},{mpu}\n")
