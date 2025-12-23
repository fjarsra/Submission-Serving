import pandas as pd
import xgboost as xgb
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- KONFIGURASI ---
# Ganti dengan URI dari DagsHub kamu!
DAGSHUB_URI = "https://dagshub.com/fjarsra/Eksperimen_SML_fjarsra.mlflow" 
DATA_DIR = "SynchronousMachine_preprocessing"

# Inisialisasi DagsHub agar bisa upload otomatis
dagshub.init(repo_owner='fjarsra', repo_name='Eksperimen_SML_fjarsra', mlflow=True)
mlflow.set_tracking_uri(DAGSHUB_URI)

def load_data():
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    X_train = train.drop(columns=['I_f'])
    y_train = train['I_f']
    X_test = test.drop(columns=['I_f'])
    y_test = test['I_f']
    
    return X_train, y_train, X_test, y_test

def main():
    # 1. Set Experiment Name
    mlflow.set_experiment("XGBoost_Hyperparameter_Tuning")
    
    X_train, y_train, X_test, y_test = load_data()

    # 2. Definisi Hyperparameter untuk Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    print("Mulai Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               cv=3, scoring='neg_root_mean_squared_error', verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Parameter Terbaik: {best_params}")

    # 3. Mulai Logging ke MLflow (Manual Logging)
    with mlflow.start_run(run_name="Best_XGBoost_Model"):
        
        # Log Parameter Terbaik
        mlflow.log_params(best_params)
        
        # Prediksi & Evaluasi
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Akar kuadrat dari MSE
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log Metrik
        print(f"Evaluasi -> RMSE: {rmse}, R2: {r2}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log Model
        mlflow.xgboost.log_model(best_model, "model")
        
        # --- ARTEFAK TAMBAHAN (Syarat Advance: Minimal 2) ---
        
        # Artefak 1: Feature Importance Plot
        plt.figure(figsize=(10, 6))
        sorted_idx = best_model.feature_importances_.argsort()
        plt.barh(X_train.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
        plt.title("XGBoost Feature Importance")
        plt.xlabel("Relative Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png") # Upload ke DagsHub
        plt.close()
        
        # Artefak 2: Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.savefig("prediction_plot.png")
        mlflow.log_artifact("prediction_plot.png") # Upload ke DagsHub
        plt.close()

        print("✅ Selesai! Cek DagsHub kamu.")

if __name__ == "__main__":
    main()