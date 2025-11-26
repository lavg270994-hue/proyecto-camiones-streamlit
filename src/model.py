import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Cargar dataset
df = pd.read_csv("data/raw/dataset_camiones_mexico.csv")

# 2. Definir variables X (entrada) y y (salida)
X = df[[
    "truck_brand",
    "truck_model",
    "truck_year",
    "engine_model",
    "transmission",
    "axle_type",
    "ubication"
]]

y = df["market_price_mex"]

# 3. Separar numéricas y categóricas
numeric_features = ["truck_year"]
categorical_features = [
    "truck_brand",
    "truck_model",
    "engine_model",
    "transmission",
    "axle_type",
    "ubication"
]

# 4. Transformador para variables categóricas (One-Hot Encoding)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# 5. Definir el modelo (Random Forest)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 6. Pipeline: preprocesamiento + modelo
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# 7. Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Entrenar el modelo
clf.fit(X_train, y_train)

# 9. Hacer predicciones
y_pred = clf.predict(X_test)

# 10. Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (error absoluto medio): {mae:,.0f} MXN")
print(f"R2 (explicación de varianza): {r2:.3f}")
import joblib

# ======== Guardar modelo entrenado ========
joblib.dump(clf, "model_camiones.pkl")
print("Modelo guardado como model_camiones.pkl")
