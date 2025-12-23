import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- KONFIGURASI PATH ---
# Sesuaikan path agar bisa berjalan baik di lokal maupun GitHub Actions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'SynchronousMachine_raw', 'SynchronousMachine.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessing', 'SynchronousMachine_preprocessing')

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    return pd.read_csv(path)

def preprocess_and_save():
    print("Memulai proses otomatisasi...")
    
    # 1. Load Data
    df = load_data(RAW_DATA_PATH)
    
    # 2. Cleaning (Hapus d_if dan e_PF sesuai temuan EDA)
    # d_if: Data leakage, e_PF: Redundant
    df_clean = df.drop(columns=['d_if', 'e_PF'], errors='ignore')
    
    # 3. Split Features & Target
    X = df_clean.drop(columns=['I_f'])
    y = df_clean['I_f']
    
    # 4. Splitting Data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Susun kembali ke DataFrame
    train_data = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_data['I_f'] = y_train.values
    
    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data['I_f'] = y_test.values
    
    # 7. Simpan Hasil
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_data.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    print(f"✅ Sukses! Data tersimpan di: {OUTPUT_DIR}")

if __name__ == "__main__":
    preprocess_and_save()