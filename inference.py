import requests
import json

url = "http://localhost:8080/invocations"

# KITA SESUAIKAN DENGAN PERMINTAAN MODEL:
# Hapus 'e' dan 'd_angle'. Cukup 2 kolom saja.
payload = {
    "dataframe_split": {
        "columns": ["I_y", "PF"], 
        "data": [[1.5, 0.8]]
    }
}

headers = {"Content-Type": "application/json"}

try:
    print(f"Mengirim request ke {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("✅ SUKSES! Model Merespon:")
        print(response.json())
    else:
        print(f"❌ GAGAL! Status Code: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error Koneksi: {e}")