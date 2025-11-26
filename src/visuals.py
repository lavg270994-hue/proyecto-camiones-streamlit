import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar dataset
df = pd.read_csv("data/raw/dataset_camiones_mexico.csv")

# ========= GRÁFICA 1: Distribución de precios =========
plt.figure()
df["market_price_mex"].hist(bins=30)
plt.title("Distribución de precios de camiones (MXN)")
plt.xlabel("Precio")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# ========= GRÁFICA 2: Precio promedio por marca =========
plt.figure()
precio_por_marca = df.groupby("truck_brand")["market_price_mex"].mean().sort_values()
precio_por_marca.plot(kind="bar")
plt.title("Precio promedio por marca")
plt.ylabel("Precio promedio (MXN)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ========= GRÁFICA 3: Conteo de unidades por año =========
plt.figure()
conteo_anio = df["truck_year"].value_counts().sort_index()
conteo_anio.plot(kind="bar")
plt.title("Número de camiones por año")
plt.xlabel("Año")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.show()

# ========= GRÁFICA 4 (opcional): Precio promedio por año =========
plt.figure()
precio_por_anio = df.groupby("truck_year")["market_price_mex"].mean()
precio_por_anio.plot(kind="line", marker="o")
plt.title("Precio promedio por año de modelo")
plt.xlabel("Año")
plt.ylabel("Precio promedio (MXN)")
plt.tight_layout()
plt.show()
