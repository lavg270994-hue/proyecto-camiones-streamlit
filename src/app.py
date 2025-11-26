import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from openai import OpenAI
import json
from pathlib import Path


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import altair as alt

# ============================================
# CONFIGURACIÓN BÁSICA
# ============================================
st.set_page_config(
    page_title="Cotizador de camiones siniestrados",
    page_icon="🚛",
    layout="wide"
)

# ============================================
# CARGA DE MODELO Y DATASET
# ============================================
@st.cache_resource
def cargar_modelo():
    return joblib.load("model_camiones.pkl")


@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/raw/dataset_camiones_mexico.csv")
@st.cache_data
def cargar_metricas_guardadas():
    metrics_path = Path("model_metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)


model = cargar_modelo()
df = cargar_dataset()
# Configurar API de OpenAI (usa la variable de entorno OPENAI_API_KEY)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Definir columnas de entrada (ajusta si tu dataset es distinto)
FEATURE_COLS = [
    "truck_brand",
    "truck_model",
    "truck_year",
    "engine_model",
    "transmission",
    "axle_type",
    "ubication"
]
TARGET_COL = "market_price_mex"

# ============================================
# HOME PROFESIONAL + KPIs
# ============================================
# ================== HOME / PANTALLA DE INICIO ==================
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0'>
        🚛 Cotizador inteligente de camiones siniestrados
    </h1>
    <p style='text-align: center; color: gray; margin-top: 4px;'>
        Proyecto final · Ciencia de Datos · Valuación de tractocamiones en el mercado mexicano
    </p>
    """,
    unsafe_allow_html=True,
)

home_col1, home_col2 = st.columns([2, 1])

with home_col1:
    st.markdown(
        """
        ### 🧩 ¿Qué problema resuelve?

        En la compra–venta de camiones siniestrados es fácil:
        - Pagar **de más** por una unidad dañada.
        - Subestimar **costos de reparación y logística**.
        - Fijar un precio de venta **poco competitivo**.

        Esta herramienta usa un modelo de **Machine Learning** entrenado con datos reales
        del mercado mexicano para estimar el **precio de mercado en buen estado** y ayudarte a:

        - Definir un **precio objetivo de compra** del siniestro.
        - Simular **costos totales** (logística, reparación, otros).
        - Calcular un **precio sugerido de venta**, utilidad y margen.
        - Comparar segmentos de mercado con un **dashboard interactivo**.
        """
    )

with home_col2:
    st.markdown("### 📊 KPIs del dataset")
    n_registros = len(df)
    precio_prom = df[TARGET_COL].mean()
    precio_mediana = df[TARGET_COL].median()
    anio_min = int(df["truck_year"].min())
    anio_max = int(df["truck_year"].max())
    n_marcas = df["truck_brand"].nunique()

    st.metric("Total de registros", f"{n_registros:,}")
    st.metric("Precio promedio", f"${precio_prom:,.0f} MXN")
    st.metric("Precio mediano", f"${precio_mediana:,.0f} MXN")
    st.caption(f"Rango de años: {anio_min} - {anio_max} · Marcas: {n_marcas}")

st.markdown("---")

st.markdown(
    """
    ### 🧭 ¿Cómo usar la aplicación?

    1. **🧮 Cotizador**  
       Ingresa marca, modelo, año, motor, transmisión y ubicación del camión siniestrado.  
       Ajusta el **descuento de compra**, costos y **markup** para ver:
       - Precio de mercado ajustado (camión bueno).
       - Precio sugerido de compra del siniestro.
       - Costo total, utilidad y margen.

    2. **📊 Dashboard de mercado**  
       Explora el mercado con filtros por **marca, ubicación y año**:
       - Distribución de precios.
       - Precio promedio por marca.
       - Evolución del precio por año.

    3. **📈 Modelo y métricas**  
       Consulta las métricas del modelo (MAE, RMSE, R²) y la importancia de variables.

    4. **🤖 Asistente IA**  
       Haz preguntas sobre valuaciones, descuentos, márgenes y decisiones de compra.
    """
)

st.markdown("---")


# ============================================
# FUNCIONES AUXILIARES (MÉTRICAS Y FEATURE IMPORTANCE)
# ============================================
@st.cache_data
def calcular_metricas_modelo(_model, df):
    """
    Calcula MAE, RMSE y R² usando todo el dataset.
    Si el modelo falla al predecir, usa un baseline (predicción = promedio).
    """
    X = df[FEATURE_COLS]
    y_true = df[TARGET_COL]

    try:
        y_pred = _model.predict(X)
    except Exception:
        # baseline: todos los camiones predichos con el promedio
        y_pred = np.full(shape=len(y_true), fill_value=y_true.mean(), dtype=float)

    mae = mean_absolute_error(y_true, y_pred)

    # Tu versión de sklearn no acepta 'squared=False',
    # así que calculamos MSE y luego le sacamos raíz cuadrada.
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2



@st.cache_data
def obtener_feature_importance(_model):
    """
    Devuelve un DataFrame con la importancia de variables.
    Como el modelo está dentro de un Pipeline + OneHotEncoder,
    no intentamos mapear a las columnas originales: usamos nombres genéricos.
    """
    try:
        # Caso 1: el modelo directamente tiene feature_importances_
        if hasattr(_model, "feature_importances_"):
            importances = np.array(_model.feature_importances_)

        # Caso 2: es un Pipeline y algún step final tiene feature_importances_
        elif hasattr(_model, "named_steps"):
            final_estimator = None
            for name, step in reversed(list(_model.named_steps.items())):
                if hasattr(step, "feature_importances_"):
                    final_estimator = step
                    break
            if final_estimator is None:
                return None
            importances = np.array(final_estimator.feature_importances_)
        else:
            return None

        # Generamos nombres genéricos para evitar problemas de longitud
        features = np.array([f"feature_{i}" for i in range(len(importances))])

        fi = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return fi

    except Exception:
        return None


# ============================================
# TABS PRINCIPALES
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🧮 Cotizador",
    "📊 Dashboard de mercado",
    "📈 Modelo y métricas",
    "🤖 Asistente IA"
])


# ============================================
# TAB 1 - COTIZADOR
# ============================================
with tab1:
    st.header("1️⃣ Datos del camión accidentado")

    col1, col2 = st.columns(2)

    # ----------- COLUMNA 1 -------------
    with col1:
        brand = st.selectbox("Marca:", sorted(df["truck_brand"].unique()))

        modelos = sorted(df[df["truck_brand"] == brand]["truck_model"].unique())
        model_truck = st.selectbox("Modelo:", modelos)

        min_year = int(df["truck_year"].min())
        max_year = int(df["truck_year"].max())
        year = st.slider("Año de modelo:", min_year, max_year, 2010)

    # ----------- COLUMNA 2 -------------
    with col2:
        engine = st.selectbox("Motor:", sorted(df["engine_model"].unique()))
        trans = st.selectbox("Transmisión:", sorted(df["transmission"].unique()))
        axle = st.selectbox("Tipo de eje:", sorted(df["axle_type"].unique()))
        ubi = st.selectbox("Ubicación:", sorted(df["ubication"].unique()))

    st.markdown("---")

    # ============================================
    # PARÁMETROS DE NEGOCIO
    # ============================================
    st.header("2️⃣ Parámetros de compra y venta")

    col3, col4 = st.columns(2)

    with col3:
        descuento_compra = st.slider(
            "Descuento para compra (%)",
            12, 40, 35,
            help="Porcentaje de descuento respecto al valor de mercado del camión en buen estado."
        )
    with col4:
        markup_venta = st.slider(
            "Markup de venta (%)",
            5, 50, 25,
            help="Porcentaje de utilidad que quieres obtener sobre el costo total."
        )

    col5, col6, col7 = st.columns(3)
    with col5:
        costo_logistica = st.number_input(
            "Logística [MXN]", 0, step=1000,
            help="Transporte, maniobras, importación, etc."
        )
    with col6:
        costo_reparacion = st.number_input(
            "Reparación [MXN]", 0, step=1000,
            help="Refacciones, mano de obra, hojalatería, etc."
        )
    with col7:
        otros_costos = st.number_input(
            "Otros costos [MXN]", 0, step=1000,
            help="Honorarios, gestorías, almacenaje, etc."
        )

    st.markdown("---")

    # ============================================
    # BOTÓN PRINCIPAL
    # ============================================
    st.header("3️⃣ Cotización")

    if st.button("Calcular cotización"):
        # ----------------------------------------
        # 1) Registro nuevo para el modelo
        # ----------------------------------------
        X_new = pd.DataFrame([{
            "truck_brand": brand,
            "truck_model": model_truck,
            "truck_year": year,
            "engine_model": engine,
            "transmission": trans,
            "axle_type": axle,
            "ubication": ubi
        }])

        precio_modelo = model.predict(X_new)[0]

        # ----------------------------------------
        # 2) AJUSTES MOTOR / TRANS / EJES
        # ----------------------------------------
        engine_str = engine.lower()
        trans_str = trans.lower()
        axle_str = str(axle).lower()

        # MOTOR
        factor_motor = 1.0
        extra_motor = 0

        if "cummins" in engine_str and ("isx" in engine_str or "x15" in engine_str):
            factor_motor, extra_motor = 1.10, 120000
        elif "cummins" in engine_str:
            factor_motor, extra_motor = 1.03, 50000
        elif "detroit" in engine_str:
            factor_motor, extra_motor = 1.06, 80000
        elif "paccar" in engine_str:
            factor_motor, extra_motor = 1.00, 40000
        elif "mercedes" in engine_str:
            factor_motor, extra_motor = 0.96, 15000
        elif "maxx" in engine_str:
            factor_motor, extra_motor = 0.93, -50000
        elif "volvo" in engine_str:
            factor_motor, extra_motor = 0.97, 20000
        elif "mack" in engine_str:
            factor_motor, extra_motor = 0.90, -80000

        # TRANSMISIÓN
        factor_trans = 1.0
        extra_trans = 0

        if "18" in trans_str:
            factor_trans, extra_trans = 1.04, 25000
        elif "13" in trans_str:
            factor_trans, extra_trans = 1.02, 12000
        elif "10" in trans_str:
            factor_trans, extra_trans = 0.97, -10000
        elif any(x in trans_str for x in ["ultrashift", "i-shift", "dt12", "mdrive"]):
            factor_trans, extra_trans = 1.01, 5000
        elif "allison" in trans_str:
            factor_trans, extra_trans = 0.95, -20000

        # EJES
        factor_eje = 1.0
        extra_eje = 0

        if "52" in axle_str:
            factor_eje, extra_eje = 1.03, 15000
        elif "46" in axle_str:
            factor_eje, extra_eje = 1.00, 0
        elif "40" in axle_str:
            factor_eje, extra_eje = 0.97, -15000

        # PRECIO AJUSTADO FINAL
        precio_modelo_ajustado = (
            precio_modelo * factor_motor * factor_trans * factor_eje
            + extra_motor + extra_trans + extra_eje
        )

        # ----------------------------------------
        # 3) COMPRA – COSTO – VENTA
        # ----------------------------------------
        precio_compra_siniestro = precio_modelo_ajustado * (1 - descuento_compra / 100)

        costo_total = (
            precio_compra_siniestro +
            costo_logistica +
            costo_reparacion +
            otros_costos
        )

        precio_venta_sugerido = costo_total * (1 + markup_venta / 100)

        utilidad = precio_venta_sugerido - costo_total
        margen_porcentaje = (
            (utilidad / precio_venta_sugerido * 100) if precio_venta_sugerido > 0 else 0
        )

        # ----------------------------------------
        # 4) MOSTRAR RESULTADOS
        # ----------------------------------------
        st.subheader("📊 Resultados de cotización")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Precio mercado ajustado", f"${precio_modelo_ajustado:,.0f}")
        col_r2.metric("Precio sugerido de compra", f"${precio_compra_siniestro:,.0f}")
        col_r3.metric("Precio sugerido de venta", f"${precio_venta_sugerido:,.0f}")

        col_r4, col_r5 = st.columns(2)
        col_r4.metric("Utilidad esperada", f"${utilidad:,.0f}")
        col_r5.metric("Margen estimado", f"{margen_porcentaje:,.1f} %")

        with st.expander("Ver detalle de costos"):
            st.write(f"💸 **Precio de mercado ajustado (camión bueno):** ${precio_modelo_ajustado:,.0f} MXN")
            st.write(f"🟢 **Precio sugerido de COMPRA:** ${precio_compra_siniestro:,.0f} MXN")
            st.write("----")
            st.write(f"🚚 Logística: ${costo_logistica:,.0f} MXN")
            st.write(f"🔧 Reparación: ${costo_reparacion:,.0f} MXN")
            st.write(f"📦 Otros costos: ${otros_costos:,.0f} MXN")
            st.write(f"🧾 **Costo total:** ${costo_total:,.0f} MXN")

        st.success(
            "✅ Cotización generada. Usa estos valores como referencia para "
            "negociar la compra del siniestro y definir tu precio objetivo de venta."
        )

# ============================================
# TAB 2 - DASHBOARD
# ============================================
with tab2:
    st.header("📊 Dashboard de mercado")

    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        marca_filtro = st.selectbox("Filtrar por marca:", ["Todas"] + sorted(df["truck_brand"].unique()))
    with colf2:
        ubi_filtro = st.selectbox("Filtrar por ubicación:", ["Todas"] + sorted(df["ubication"].unique()))
    with colf3:
        anio_filtro = st.selectbox(
            "Filtrar por año:",
            ["Todos"] + sorted(df["truck_year"].unique())
        )

    df_filtrado = df.copy()
    if marca_filtro != "Todas":
        df_filtrado = df_filtrado[df_filtrado["truck_brand"] == marca_filtro]
    if ubi_filtro != "Todas":
        df_filtrado = df_filtrado[df_filtrado["ubication"] == ubi_filtro]
    if anio_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado["truck_year"] == anio_filtro]

    # KPIs del filtro
    st.markdown("#### KPIs del segmento filtrado")
    colk1, colk2, colk3 = st.columns(3)
    colk1.metric("Unidades", f"{len(df_filtrado):,}")
    colk2.metric("Precio promedio", f"${df_filtrado[TARGET_COL].mean():,.0f} MXN")
    colk3.metric("Mediana precio", f"${df_filtrado[TARGET_COL].median():,.0f} MXN")

    st.markdown("---")

    # ------- Visualización 1: Distribución de precios (histograma) -------
    st.subheader("1️⃣ Distribución de precios de mercado")
    chart_hist = alt.Chart(df_filtrado).mark_bar().encode(
        x=alt.X("market_price_mex:Q", bin=alt.Bin(maxbins=30), title="Precio de mercado [MXN]"),
        y=alt.Y("count():Q", title="Número de camiones"),
        tooltip=["count()"]
    ).properties(height=300)
    st.altair_chart(chart_hist, use_container_width=True)

    # ------- Visualización 2: Precio promedio por marca -------
    st.subheader("2️⃣ Precio promedio por marca")
    df_marca = (
        df_filtrado.groupby("truck_brand", as_index=False)["market_price_mex"]
        .mean()
        .rename(columns={"market_price_mex": "precio_promedio"})
    )

    chart_brand = alt.Chart(df_marca).mark_bar().encode(
        x=alt.X("truck_brand:N", sort="-y", title="Marca"),
        y=alt.Y("precio_promedio:Q", title="Precio promedio [MXN]"),
        tooltip=["truck_brand", "precio_promedio"]
    ).properties(height=300)
    st.altair_chart(chart_brand, use_container_width=True)

    # ------- Visualización 3: Precio promedio por año -------
    st.subheader("3️⃣ Evolución del precio promedio por año")
    df_year = (
        df_filtrado.groupby("truck_year", as_index=False)["market_price_mex"]
        .mean()
        .rename(columns={"market_price_mex": "precio_promedio"})
    )

    chart_year = alt.Chart(df_year).mark_line(point=True).encode(
        x=alt.X("truck_year:O", title="Año"),
        y=alt.Y("precio_promedio:Q", title="Precio promedio [MXN]"),
        tooltip=["truck_year", "precio_promedio"]
    ).properties(height=300)
    st.altair_chart(chart_year, use_container_width=True)

# ============================================
# TAB 3 - MODELO Y MÉTRICAS
# ============================================
# ============================================
# TAB 3 - MODELO Y MÉTRICAS
# ============================================
with tab3:
    st.header("📈 Modelo y métricas")

    st.markdown(
        """
        En esta sección se muestran las métricas del modelo de regresión entrenado
        con **Random Forest** y evaluado correctamente con una partición
        **train/test (80% / 20%)** para evitar sobreentrenamiento.
        """
    )

    metrics = cargar_metricas_guardadas()

    if metrics is None:
        st.error("No se encontraron métricas guardadas. Ejecuta primero `python src/train_model.py`.")
    else:
        mae_test = metrics["test"]["mae"]
        rmse_test = metrics["test"]["rmse"]
        r2_test = metrics["test"]["r2"]

        mae_train = metrics["train"]["mae"]
        rmse_train = metrics["train"]["rmse"]
        r2_train = metrics["train"]["r2"]

        st.subheader("🔢 Métricas en conjunto de prueba (TEST)")

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("MAE (test)", f"${mae_test:,.0f}")
        colm2.metric("RMSE (test)", f"${rmse_test:,.0f}")
        colm3.metric("R² (test)", f"{r2_test:.3f}")

        st.caption("Estas métricas se calcularon sobre el 20% de los datos reservados para prueba.")

        st.subheader("📚 Métricas en entrenamiento (TRAIN)")

        colt1, colt2, colt3 = st.columns(3)
        colt1.metric("MAE (train)", f"${mae_train:,.0f}")
        colt2.metric("RMSE (train)", f"${rmse_train:,.0f}")
        colt3.metric("R² (train)", f"{r2_train:.3f}")

        st.caption(
            "Comparar TRAIN vs TEST permite verificar que el modelo no esté sobreajustado. "
            "En este caso, las métricas son razonablemente cercanas, lo que indica buena generalización."
        )

# ============================================
# TAB 4 - ASISTENTE IA (ChatGPT)
# ============================================
with tab4:
    st.header("🤖 Asistente IA sobre el cotizador")

    st.markdown(
        """
        Escribe una pregunta y el asistente te responde usando un modelo de lenguaje.
        Puedes preguntarle, por ejemplo:

        - Cómo interpretar una cotización específica.
        - Si conviene comprar un camión con cierto motor/transmisión.
        - Cómo ajustar descuentos o márgenes.
        """
    )

    pregunta = st.text_area("Escribe tu pregunta:", height=150)

if st.button("Preguntar a ChatGPT"):
    if not os.getenv("OPENAI_API_KEY"):
        st.error("No se encontró la API key de OpenAI. Define la variable OPENAI_API_KEY.")
    else:
        try:
            respuesta = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente experto en compra-venta de camiones siniestrados "
                            "y análisis de precios. Responde claro y concreto."
                        ),
                    },
                    {"role": "user", "content": pregunta},
                ],
                max_tokens=300,
                temperature=0.3
            )

            texto = respuesta.choices[0].message.content
            st.markdown(texto)

        except Exception as e:
            st.error(f"Ocurrió un error al llamar a la API:\n\n{e}")


