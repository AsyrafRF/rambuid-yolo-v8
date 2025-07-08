import serial
import pynmea2
import threading
import time
import platform

# ============ MPU6050 Setup (Mock di Windows) ============

class MPU6050_Mock:
    def get_accel_data(self):
        return {'x': 0.0, 'y': 0.0, 'z': 9.8}

    def get_gyro_data(self):
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}

if platform.system() == "Windows":
    print("[INFO] Detected Windows OS - using mock MPU6050")
    mpu = MPU6050_Mock()
else:
    from mpu6050 import mpu6050
    mpu = mpu6050(0x68)

# ============ Sensor Setup ============

gps_port = "/dev/serial0"
try:
    gps = serial.Serial(gps_port, baudrate=9600, timeout=1)
except Exception as e:
    print(f"[GPS] Gagal membuka port GPS: {e}")
    gps = None

sensor_data = {
    "gps": {"lat": None, "lon": None},
    "mpu": {"accel": None, "gyro": None}
}
sensor_lock = threading.Lock()

def gps_thread():
    if gps is None:
        print("[GPS] GPS tidak tersedia, thread tidak dijalankan.")
        return

    while True:
        try:
            line = gps.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                if int(msg.gps_qual or 0) > 0:
                    with sensor_lock:
                        sensor_data["gps"]["lat"] = msg.latitude
                        sensor_data["gps"]["lon"] = msg.longitude
                else:
                    print("[GPS] Belum dapat fix.")
            elif line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                if msg.status == 'A':
                    with sensor_lock:
                        sensor_data["gps"]["lat"] = msg.latitude
                        sensor_data["gps"]["lon"] = msg.longitude
                else:
                    print("[GPS] Status tidak aktif.")
        except Exception as e:
            print(f"[GPS] Error: {e}")
        time.sleep(0.1)

def mpu_thread():
    while True:
        try:
            accel = mpu.get_accel_data()
            gyro = mpu.get_gyro_data()
            with sensor_lock:
                sensor_data["mpu"]["accel"] = accel
                sensor_data["mpu"]["gyro"] = gyro
        except Exception as e:
            print(f"[MPU6050] Error: {e}")
        time.sleep(0.05)
