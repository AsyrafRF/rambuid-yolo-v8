import pygame
import tempfile
import pyttsx3
import os
import time
import threading
import socket
from gtts import gTTS
import hashlib

# Inisialisasi mixer dan pyttsx3
pygame.mixer.init()
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

tts_lock = threading.Lock()

# Folder cache untuk file suara
CACHE_DIR = "tts_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def is_connected():
    """Cek koneksi internet."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False

def get_cache_filename(text):
    """Hasilkan nama file cache dari teks dengan hash MD5."""
    hash_id = hashlib.md5(text.encode('utf-8')).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_id}.mp3")

def play_mp3(file_path):
    """Mainkan file .mp3 dengan pygame."""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
    except Exception as e:
        print(f"[ERROR] Pemutaran audio gagal: {e}")

def speak_label(label):
    with tts_lock:
        sentence = f"Terdeteksi rambu {label}"
        cache_file = get_cache_filename(sentence)

        if os.path.exists(cache_file):
            # File sudah di-cache, langsung pakai
            play_mp3(cache_file)
            return

        if is_connected():
            # Online: generate dan simpan ke cache
            try:
                tts = gTTS(text=sentence, lang='id')
                tts.save(cache_file)
                play_mp3(cache_file)
                return
            except Exception as e:
                print(f"[ERROR] gTTS gagal: {e} -> fallback ke pyttsx3")

        # Offline atau gagal gTTS: fallback ke pyttsx3
        tts_engine.say(sentence)
        tts_engine.runAndWait()

def speak_label_threaded(label):
    threading.Thread(target=speak_label, args=(label,), daemon=True).start()
