import serial
import pynmea2
from mpu6050 import mpu6050
import threading
import time

# ============ Sensor Setup ================
gps_port = "/dev/serial0"
gps = serial.Serial(gps_port, baudrate=9600, timeout=1)
mpu = mpu6050(0x68)

sensor_data = {
    "gps": {"lat": None, "lon": None},
    "mpu": {"accel": None, "gyro": None}
}
sensor_lock = threading.Lock()

def gps_thread():
    while True:
        try:
            line = gps.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPGGA'):
                msg = pynmea2.parse(line)
                # GPS fix akan bernilai > 0 jika sudah dapat sinyal
                if int(msg.gps_qual or 0) > 0:
                    with sensor_lock:
                        sensor_data["gps"]["lat"] = msg.latitude
                        sensor_data["gps"]["lon"] = msg.longitude
                else:
                    print("[GPS] Belum dapat fix.")
            elif line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                if msg.status == 'A':  # A = Active, V = Void
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
