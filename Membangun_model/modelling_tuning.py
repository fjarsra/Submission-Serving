import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- KONFIGURASI PENTING ---

# 1. URL DagsHub (Pastikan ini URL repo DagsHub kamu yang benar)
# Tips: Copy dari tombol "Remote" > "MLflow" di DagsHub
DAGSHUB_URI = "https://dagshub.com/fjarsra/Eksperimen_SML_fjarsra.mlflow"

# 2. Lokasi Data
# Kita set path relatif agar bisa dibaca dari Root folder GitHub
# Sesuaikan ini jika struktur folder kamu berubah
DATA_DIR = "Membangun_model/preprocessing/SynchronousMachine_preprocessing"

# 3. Setup MLflow (TANPA dagshub.init)
# Kita pakai cara manual biar GitHub Actions gak bingung masalah Auth
# Auth akan otomatis diambil dari environment variable MLFLOW_TRACKING_USERNAME & PASSWORD
mlflow.set_tracking_uri(DAGSHUB_URI)
print(f"Tracking URI diset ke: {DAGSHUB_URI}")

def load_data():
    print("Loading data...")
    # Cek apakah folder ada (untuk debugging)
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ PERINGATAN: Folder {DATA_DIR} tidak ditemukan!")
        print("Mencoba mencari di folder saat ini...")
        # Fallback: Coba cari di folder local jika script dijalankan dari dalam folder
        local_path = "preprocessing/SynchronousMachine_preprocessing"
        if os.path.exists(local_path):
            current_dir = local_path
        else:
            # Fallback terakhir: Asumsi file ada di folder yang sama
            current_dir = "."
    else:
        current_dir = DATA_DIR

    try:
        train = pd.read_csv(os.path.join(current_dir, "train.csv"))
        test = pd.read_csv(os.path.join(current_dir, "test.csv"))
        
        X_train = train.drop(columns=['I_f'])
        y_train = train['I_f']
        X_test = test.drop(columns=['I_f'])
        y_test = test['I_f']
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        print(f"❌ Error: File CSV tidak ditemukan di {current_dir}")
        print("Pastikan script preprocessing sudah dijalankan dan artifact sudah di-download.")
        raise e

def main():
    # 1. Set Experiment Name
    mlflow.set_experiment("XGBoost_Hyperparameter_Tuning_CI")
    
    X_train, y_train, X_test, y_test = load_data()

    # 2. Definisi Hyperparameter untuk Tuning
    # (Saya kurangi sedikit opsinya biar Training di GitHub Actions gak kelamaan/timeout)
    param_grid = {
        'n_estimators': [100],      # Bisa ditambah jadi [100, 200] kalau mau lama
        'learning_rate': [0.1],     # Bisa ditambah [0.01, 0.1]
        'max_depth': [3, 5],
        'subsample': [0.8]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    print("Mulai Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               cv=3, scoring='neg_root_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    # 3. Mulai Logging ke MLflow
    with mlflow.start_run(run_name="Best_XGBoost_Model_CI"):
        
        # Log Parameter Terbaik
        mlflow.log_params(best_params)
        
        # Prediksi & Evaluasi
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log Metrik
        print(f"Evaluasi -> RMSE: {rmse}, R2: {r2}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log Model
        # Ganti ke 'sklearn' atau 'xgboost' tergantung preferensi, xgboost lebih spesifik
        mlflow.xgboost.log_model(best_model, "model")
        
        # --- ARTEFAK TAMBAHAN ---
        
        # Artefak 1: Feature Importance Plot
        plt.figure(figsize=(10, 6))
        # Cek atribut feature importances
        if hasattr(best_model, 'feature_importances_'):
            sorted_idx = best_model.feature_importances_.argsort()
            plt.barh(X_train.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
            plt.title("XGBoost Feature Importance")
            plt.xlabel("Relative Importance")
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
        
        # Artefak 2: Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.savefig("prediction_plot.png")
        mlflow.log_artifact("prediction_plot.png")
        plt.close()

        print("✅ Selesai! Log tersimpan di DagsHub (atau lokal jika offline).")

if __name__ == "__main__":
    main()