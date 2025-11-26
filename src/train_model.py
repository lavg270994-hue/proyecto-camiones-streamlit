# src/train_model.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== RUTAS ==================
DATA_PATH = Path("data/raw/dataset_camiones_mexico.csv")
MODEL_PATH = Path("model_camiones.pkl")
METRICS_PATH = Path("model_metrics.json")

TARGET_COL = "market_price_mex"

# ================== 1. CARGAR DATOS ==================
df = pd.read_csv(DATA_PATH)

feature_cols = [
    "truck_brand",
    "truck_model",
    "truck_year",
    "engine_model",
    "transmission",
    "axle_type",
    "ubication",
]

X = df[feature_cols].copy()
y = df[TARGET_COL].astype(float).copy()

# ================== 2. TRAIN / TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ================== 3. PIPELINE (PREPROCESAMIENTO + MODELO) ==================
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=16,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf),
    ]
)

# ================== 4. ENTRENAR ==================
print("Entrenando modelo...")
model.fit(X_train, y_train)

# ================== 5. FUNCIÓN DE MÉTRICAS ==================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)   # versión vieja de sklearn, sin squared
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ================== 6. EVALUAR TRAIN Y TEST ==================
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae, train_rmse, train_r2 = compute_metrics(y_train, y_pred_train)
test_mae, test_rmse, test_r2 = compute_metrics(y_test, y_pred_test)

print("\n=== MÉTRICAS ENTRENAMIENTO ===")
print(f"MAE train : {train_mae:,.0f}")
print(f"RMSE train: {train_rmse:,.0f}")
print(f"R² train  : {train_r2:,.3f}")

print("\n=== MÉTRICAS TEST (LAS IMPORTANTES) ===")
print(f"MAE test  : {test_mae:,.0f}")
print(f"RMSE test : {test_rmse:,.0f}")
print(f"R² test   : {test_r2:,.3f}")

# ================== 7. GUARDAR MODELO Y MÉTRICAS ==================
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_PATH}")

metrics = {
    "train": {
        "mae": float(train_mae),
        "rmse": float(train_rmse),
        "r2": float(train_r2),
    },
    "test": {
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "r2": float(test_r2),
    },
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métricas guardadas en: {METRICS_PATH}")
print("\nListo. Usa las métricas de TEST en tu reporte para evitar sobreentrenamiento.")
