import subprocess
import time
import socket
import os

from log_utils import log_event

WPA_CONF = "/etc/wpa_supplicant/wpa_supplicant.conf"

def scan_wifi():
    """Scan jaringan WiFi yang tersedia dan kembalikan daftar SSID."""
    try:
        result = subprocess.run(['sudo', 'iwlist', 'wlan0', 'scan'], capture_output=True, text=True)
        ssids = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('ESSID:'):
                ssid = line.split(':')[1].strip().strip('"')
                if ssid and ssid not in ssids:
                    ssids.append(ssid)
        return ssids
    except Exception as e:
        log_event(f"Error saat scan wifi: {e}")
        return []


def stop_hotspot():
    subprocess.run(['sudo', 'systemctl', 'stop', 'hostapd'])
    subprocess.run(['sudo', 'systemctl', 'stop', 'dnsmasq'])


def start_hotspot():
    subprocess.run(['sudo', 'systemctl', 'start', 'hostapd'])
    subprocess.run(['sudo', 'systemctl', 'start', 'dnsmasq'])


def write_wifi_config(ssid: str, password: str):
    """Tulis konfigurasi jaringan ke wpa_supplicant.conf."""
    config = f'''
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=ID

network={{
    ssid="{ssid}"
    psk="{password}"
}}
'''
    with open(WPA_CONF, 'w') as f:
        f.write(config)


def connect_to_wifi(ssid: str, password: str):
    """Hentikan hotspot dan konek ke WiFi."""
    stop_hotspot()
    write_wifi_config(ssid, password)
    subprocess.run(['sudo', 'wpa_cli', '-i', 'wlan0', 'reconfigure'])
    time.sleep(2)


def check_internet(timeout: int = 10) -> bool:
    """Cek apakah Raspberry Pi memiliki koneksi internet."""
    for _ in range(timeout):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            time.sleep(1)
    return False


def try_connect_and_verify(ssid: str, password: str, timeout: int = 30) -> bool:
    """Coba konek ke jaringan dan verifikasi apakah ada internet."""
    connect_to_wifi(ssid, password)
    if check_internet(timeout):
        return True
    else:
        start_hotspot()
        return False




def get_current_ssid() -> str | None:
    """Ambil SSID dari jaringan WiFi yang sedang terhubung."""
    try:
        result = subprocess.check_output(["iwgetid", "-r"]).decode().strip()
        return result if result else None
    except subprocess.CalledProcessError:
        return None
