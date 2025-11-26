import pandas as pd

# Cargar dataset
df = pd.read_csv("data/raw/dataset_camiones_mexico.csv")

print("Columnas del dataset:")
print(df.columns)

print("\nPrimeras filas:")
print(df.head())

print("\nDescripción estadística:")
print(df.describe(include='all'))

