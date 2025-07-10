#!/usr/bin/env python3
import subprocess
import socket
import time
import os
from datetime import datetime, timedelta

REBOOT_LOG = "../logs/last_reboot.log"
MAX_WAIT = 180  # waktu tunggu sebelum reboot (detik)
REBOOT_COOLDOWN = 3600  # 1 jam (dalam detik)
CHECK_INTERVAL = 5

def check_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def start_hotspot():
    subprocess.run(["sudo", "systemctl", "start", "hostapd"])
    subprocess.run(["sudo", "systemctl", "start", "dnsmasq"])

def reboot_allowed():
    if not os.path.exists(REBOOT_LOG):
        return True
    with open(REBOOT_LOG, "r") as f:
        last_time_str = f.read().strip()
        try:
            last_time = datetime.fromisoformat(last_time_str)
            now = datetime.now()
            if (now - last_time).total_seconds() >= REBOOT_COOLDOWN:
                return True
        except Exception:
            return True  # jika file corrupt, izinkan reboot
    return False

def log_reboot_time():
    with open(REBOOT_LOG, "w") as f:
        f.write(datetime.now().isoformat())

def reboot():
    log_reboot_time()
    subprocess.run(["sudo", "reboot"])

# Mulai
print("⏳ Mengecek koneksi internet...")
time.sleep(5)

elapsed = 0
while elapsed < MAX_WAIT:
    if check_internet():
        print("✅ Internet tersedia. Tidak perlu reboot.")
        exit(0)
    print(f"🔄 Belum ada koneksi... ({elapsed}/{MAX_WAIT}s)")
    time.sleep(CHECK_INTERVAL)
    elapsed += CHECK_INTERVAL

print("❌ Tidak ada koneksi internet.")
start_hotspot()
time.sleep(10)

if reboot_allowed():
    print("🔁 Rebooting Raspberry Pi...")
    reboot()
else:
    print("⛔ Reboot dibatalkan (sudah reboot dalam 1 jam terakhir).")
