import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor # Atau model yang lu pake
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# 1. Load Data
# GANTI path ini sesuai lokasi dataset lu
df = pd.read_csv("SynchronousMachine_raw\SynchronousMachine.csv") 

# 2. Split Data (Sesuaikan nama kolom target lu)
X = df.drop('I_f', axis=1) # Ganti 'I_f' dengan target prediksi lu
y = df['I_f']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set Experiment Name
mlflow.set_experiment("Submission_Basic_Modelling")

# 4. Mulai Run dengan AUTOLOG (Ini SYARAT WAJIB)
with mlflow.start_run():
    # Wajib panggil ini sebelum training!
    mlflow.autolog()

    # 5. Define Model Polos (Tanpa Hyperparameter Tuning)
    # Langsung pake default aja, jangan pake GridSearch
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42)) 
    ])

    # 6. Train Model
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluasi (Optional, karena autolog udah nyatet)
    score = pipeline.score(X_test, y_test)
    print(f"R2 Score: {score}")
    
    print("Selesai! Cek mlruns folder.")