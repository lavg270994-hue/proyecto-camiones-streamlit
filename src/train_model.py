# src/train_model.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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


# ================== 3. PREPROCESAMIENTO ==================
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# ================== 4. FUNCIÓN DE MÉTRICAS ==================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


# ================== 5. DEFINIR MODELOS ==================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ),
}


# ================== 6. ENTRENAR Y COMPARAR MODELOS ==================
results = {}
trained_pipelines = {}

print("Entrenando y comparando modelos...")

for model_name, estimator in models.items():
    print(f"\nEntrenando: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    results[model_name] = {
        "train": train_metrics,
        "test": test_metrics,
    }

    trained_pipelines[model_name] = pipeline

    print("Métricas TEST:")
    print(f"MAE : {test_metrics['mae']:,.0f}")
    print(f"RMSE: {test_metrics['rmse']:,.0f}")
    print(f"R²  : {test_metrics['r2']:,.3f}")


# ================== 7. SELECCIONAR MEJOR MODELO ==================
# Criterio: menor RMSE en TEST
best_model_name = min(
    results,
    key=lambda name: results[name]["test"]["rmse"]
)

best_model = trained_pipelines[best_model_name]

print("\n==============================")
print(f"✅ Mejor modelo: {best_model_name}")
print("==============================")


# ================== 8. GUARDAR MEJOR MODELO ==================
joblib.dump(best_model, MODEL_PATH)
print(f"\n✅ Modelo guardado en: {MODEL_PATH}")


# ================== 9. GUARDAR MÉTRICAS ==================
metrics = {
    "best_model": best_model_name,
    "selection_criterion": "Menor RMSE en conjunto de prueba",
    "models": results,
    "train": results[best_model_name]["train"],
    "test": results[best_model_name]["test"],
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métricas guardadas en: {METRICS_PATH}")

print("\nResumen de comparación:")
for model_name, values in results.items():
    test = values["test"]
    print(
        f"- {model_name}: "
        f"MAE={test['mae']:,.0f}, "
        f"RMSE={test['rmse']:,.0f}, "
        f"R²={test['r2']:,.3f}"
    )

print("\nListo. Usa la comparación de modelos en tu reporte y presentación.")
