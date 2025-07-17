import platform
import cv2
import threading
import time

def find_available_camera(max_index=5):
    import os

    system_platform = platform.system()
    print(f"[INFO] Mendeteksi kamera di platform: {system_platform}")

    if system_platform == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    elif system_platform == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:  # Linux
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # Coba semua kombinasi backend dan index
    for backend in backends:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if cap is not None and cap.isOpened():
                print(f"[INFO] Kamera ditemukan di index {i} dengan backend {backend}")
                return cap
            cap.release()

    # Fallback: coba semua /dev/video* secara eksplisit tanpa backend
    print("[INFO] Fallback ke device eksplisit /dev/video* tanpa backend...")
    for i in range(32):  # misalnya dari /dev/video0 sampai /dev/video31
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            cap = cv2.VideoCapture(device_path)
            if cap is not None and cap.isOpened():
                print(f"[INFO] Kamera ditemukan di {device_path} tanpa backend")
                return cap
            cap.release()

    print("❌ Tidak ada kamera yang tersedia.")
    return None

class Camera:
    def __init__(self):
        self.cap = find_available_camera()

        if self.cap is None or not self.cap.isOpened():
            print("🚫 Kamera tidak tersedia atau gagal dibuka. Program dihentikan.")
            exit(1)

        # Baru di sini aman dipanggil
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
            time.sleep(0.01)  # Hindari over-utilisasi CPU

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
