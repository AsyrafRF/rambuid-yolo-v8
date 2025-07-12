import csv
import base64
from firebase_admin import firestore, credentials, initialize_app

cred = credentials.Certificate("firebase-key.json")
initialize_app(cred)
db = firestore.client()

with open('rambu.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        with open(row['file_gambar'], 'rb') as img:
            encoded = base64.b64encode(img.read()).decode('utf-8')

        doc = {
            'id': row['id'],
            'nama': row['nama'],
            'jenis': row['jenis'],
            'deskripsi': row['deskripsi'],
            'ikon_base64': encoded
        }

        db.collection('master_rambu').document(row['id']).set(doc)
