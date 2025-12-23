import pandas as pd
import xgboost as xgb
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# KONFIGURASI
# ==============================================================================
DAGSHUB_URI = "https://dagshub.com/fjarsra/Eksperimen_SML_fjarsra.mlflow" 
DATA_DIR = "Membangun_model/preprocessing/SynchronousMachine_preprocessing"

# Lokasi penyimpanan Output Gambar agar rapi
OUTPUT_DIR = "Membangun_model" 

mlflow.set_tracking_uri(DAGSHUB_URI)

def load_data():
    if not os.path.exists(DATA_DIR):
        local_path = "preprocessing/SynchronousMachine_preprocessing"
        current_dir = local_path if os.path.exists(local_path) else "."
    else:
        current_dir = DATA_DIR

    train = pd.read_csv(os.path.join(current_dir, "train.csv"))
    test = pd.read_csv(os.path.join(current_dir, "test.csv"))
    return train.drop(columns=['I_f']), train['I_f'], test.drop(columns=['I_f']), test['I_f']

def save_plot(fig, filename):
    """Fungsi bantu untuk simpan plot ke lokal & upload ke MLflow"""
    # 1. Simpan ke folder lokal (biar bisa di-push ke GitHub)
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath)
    print(f"📷 Gambar disimpan: {filepath}")
    
    # 2. Upload ke MLflow (Syarat Advance)
    mlflow.log_artifact(filepath)

def main():
    mlflow.set_experiment("XGBoost_Advance_Submission")
    X_train, y_train, X_test, y_test = load_data()

    # Parameter Grid
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'subsample': [0.8]
    }
    
    xgb_model = xgb.XGBRegressor(random_state=42)
    print("Mulai Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    with mlflow.start_run(run_name="Best_Model_Advance"):
        mlflow.log_params(best_params)
        y_pred = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(best_model, "model")

        # --- SIMPAN MODEL MANUAL ---
        model_dir = os.path.join(OUTPUT_DIR, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        best_model.save_model(os.path.join(model_dir, "xgboost_model.json"))

        # 1. Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            sorted_idx = best_model.feature_importances_.argsort()
            fig1 = plt.figure(figsize=(10, 6))
            plt.barh(X_train.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
            plt.title("1. Feature Importance")
            plt.tight_layout()
            save_plot(fig1, "feature_importance.png") # Panggil fungsi bantu
            plt.close(fig1)

        # 2. Actual vs Predicted
        fig2 = plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title("2. Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        save_plot(fig2, "prediction_plot.png")
        plt.close(fig2)

        # 3. Residual Plot
        residuals = y_test - y_pred
        fig3 = plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title("3. Residual Plot")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        save_plot(fig3, "residual_plot.png")
        plt.close(fig3)

        # 4. Distribution of Residuals
        fig4 = plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title("4. Distribution of Residuals")
        plt.xlabel("Residual Value")
        plt.tight_layout()
        save_plot(fig4, "residual_dist.png")
        plt.close(fig4)

        print("✅ Semua artefak gambar & model telah disimpan!")

if __name__ == "__main__":
    main()