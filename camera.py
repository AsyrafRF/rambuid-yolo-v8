import platform
import cv2
import threading
import time

def find_available_camera(max_index=5):
    system_platform = platform.system()
    print(f"[INFO] Mendeteksi kamera di platform: {system_platform}")

    # Tentukan backend berdasarkan sistem operasi
    if system_platform == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
    elif system_platform == "Darwin":  # macOS
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:  # Linux
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    # Coba dengan backend eksplisit
    for backend in backends:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, backend)
            if cap is not None and cap.isOpened():
                print(f"[INFO] Kamera ditemukan di index {i} dengan backend {backend}")
                return cap
            cap.release()

    # Fallback terakhir: coba tanpa backend eksplisit
    print("[INFO] Mencoba fallback tanpa backend eksplisit...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            print(f"[INFO] Kamera ditemukan di index {i} tanpa backend")
            return cap
        cap.release()

    print("❌ Tidak ada kamera yang tersedia.")
    return None

class Camera:
    def __init__(self):
        self.cap = find_available_camera()
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if not self.cap.isOpened():
            raise RuntimeError("Kamera tidak bisa dibuka")
        if self.cap is None:
            print("🚫 Tidak ada kamera yang tersedia. Program dihentikan.")
            exit(1)

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
