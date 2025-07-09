import firebase_admin
from firebase_admin import credentials, auth

# Inisialisasi Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-key.json")
    firebase_admin.initialize_app(cred)

def verify_firebase_token(id_token: str) -> str:
    """Verifikasi Firebase ID Token dan kembalikan UID"""
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token['uid']
    except Exception as e:
        print(f"[AUTH ERROR] Token tidak valid: {e}")
        return None
