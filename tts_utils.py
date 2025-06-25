import pygame
import tempfile
import pyttsx3
import os
import time
import threading

from gtts import gTTS

# Inisialisasi TTS global
pygame.mixer.init()

# # Text-to-Speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Kecepatan bicara

tts_lock = threading.Lock()

def speak_label(label):
    with tts_lock:
        sentence = f"Terdeteksi rambu {label}"
        tts = gTTS(text=sentence, lang='id')
        
        # Buat file temporer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            temp_mp3 = fp.name

        # Beri waktu file ditulis
        time.sleep(0.2)
        
        try:
            pygame.mixer.music.load(temp_mp3)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)  # tunggu sampai selesai

            pygame.mixer.music.unload()  # pastikan pygame selesai pakai file    

        finally:
            if os.path.exists(temp_mp3):
                try:
                    os.remove(temp_mp3)
                except PermissionError:
                    print(f"[Warning] Tidak bisa hapus {temp_mp3}, masih dipakai.")

def speak_label_threaded(label):
    threading.Thread(target=speak_label, args=(label,), daemon=True).start()