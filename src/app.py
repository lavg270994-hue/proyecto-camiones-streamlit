import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
import altair as alt


# ================== CONFIGURACIÓN ==================
st.set_page_config(
    page_title="Cotizador de camiones siniestrados",
    page_icon="🚛",
    layout="wide"
)


# ================== CARGA DE ARCHIVOS ==================
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
metrics = cargar_metricas_guardadas()

TARGET_COL = "market_price_mex"


# ================== HOME ==================
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0'>
        🚛 Cotizador inteligente de camiones siniestrados
    </h1>
    <p style='text-align: center; color: gray; margin-top: 4px;'>
        Proyecto final · Machine Learning · Toma de decisiones en compra de tractocamiones
    </p>
    """,
    unsafe_allow_html=True,
)

home_col1, home_col2 = st.columns([2, 1])

with home_col1:
    st.markdown(
        """
        ### 🧩 ¿Qué problema resuelve?

        En la compra de camiones siniestrados existe incertidumbre sobre el precio máximo
        que conviene pagar. Un error en la valuación puede generar pérdidas por sobrepago,
        costos mal estimados o márgenes insuficientes.

        Esta aplicación usa Machine Learning para estimar el precio de mercado de un camión
        en buen estado y traducir ese resultado en una recomendación de negocio.
        """
    )

with home_col2:
    st.markdown("### 📊 KPIs del dataset")

    st.metric("Total de registros", f"{len(df):,}")
    st.metric("Precio promedio", f"${df[TARGET_COL].mean():,.0f} MXN")
    st.metric("Precio mediano", f"${df[TARGET_COL].median():,.0f} MXN")
    st.caption(
        f"Rango de años: {int(df['truck_year'].min())} - {int(df['truck_year'].max())} · "
        f"Marcas: {df['truck_brand'].nunique()}"
    )

st.markdown("---")

st.markdown(
    """
    ### 🧭 ¿Cómo usar la aplicación?

    1. Ingresa los datos del camión.
    2. Ajusta descuento, costos y margen deseado.
    3. Calcula la cotización.
    4. Revisa la recomendación automática: **comprar, negociar o no comprar**.
    """
)

st.markdown("---")


# ================== TABS ==================
tab1, tab2, tab3 = st.tabs([
    "🧮 Cotizador",
    "📊 Dashboard de mercado",
    "📈 Modelo y métricas"
])


# ================== TAB 1: COTIZADOR ==================
with tab1:
    st.header("1️⃣ Datos del camión accidentado")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Marca:", sorted(df["truck_brand"].unique()))

        modelos = sorted(df[df["truck_brand"] == brand]["truck_model"].unique())
        model_truck = st.selectbox("Modelo:", modelos)

        min_year = int(df["truck_year"].min())
        max_year = int(df["truck_year"].max())
        year = st.slider("Año de modelo:", min_year, max_year, 2010)

    with col2:
        engine = st.selectbox("Motor:", sorted(df["engine_model"].unique()))
        trans = st.selectbox("Transmisión:", sorted(df["transmission"].unique()))
        axle = st.selectbox("Tipo de eje:", sorted(df["axle_type"].unique()))
        ubi = st.selectbox("Ubicación:", sorted(df["ubication"].unique()))

    st.markdown("---")

    st.header("2️⃣ Parámetros de compra y venta")

    col3, col4 = st.columns(2)

    with col3:
        descuento_compra = st.slider(
            "Descuento para compra (%)",
            12,
            40,
            35,
            help="Descuento aplicado al valor estimado de mercado."
        )

    with col4:
        markup_venta = st.slider(
            "Markup de venta (%)",
            5,
            50,
            25,
            help="Porcentaje de utilidad deseada sobre el costo total."
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        costo_logistica = st.number_input("Logística [MXN]", min_value=0, step=1000)

    with col6:
        costo_reparacion = st.number_input("Reparación [MXN]", min_value=0, step=1000)

    with col7:
        otros_costos = st.number_input("Otros costos [MXN]", min_value=0, step=1000)

    st.markdown("---")

    st.header("3️⃣ Cotización")

    if st.button("Calcular cotización"):

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

        engine_str = engine.lower()
        trans_str = trans.lower()
        axle_str = str(axle).lower()

        # Ajuste por motor
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

        # Ajuste por transmisión
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

        # Ajuste por eje
        factor_eje = 1.0
        extra_eje = 0

        if "52" in axle_str:
            factor_eje, extra_eje = 1.03, 15000
        elif "46" in axle_str:
            factor_eje, extra_eje = 1.00, 0
        elif "40" in axle_str:
            factor_eje, extra_eje = 0.97, -15000

        precio_modelo_ajustado = (
            precio_modelo * factor_motor * factor_trans * factor_eje
            + extra_motor + extra_trans + extra_eje
        )

        precio_compra_siniestro = precio_modelo_ajustado * (1 - descuento_compra / 100)

        costo_total = (
            precio_compra_siniestro
            + costo_logistica
            + costo_reparacion
            + otros_costos
        )

        precio_venta_sugerido = costo_total * (1 + markup_venta / 100)
        utilidad = precio_venta_sugerido - costo_total

        margen_porcentaje = (
            utilidad / precio_venta_sugerido * 100
            if precio_venta_sugerido > 0
            else 0
        )

        st.subheader("📊 Resultados de cotización")

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Precio mercado ajustado", f"${precio_modelo_ajustado:,.0f}")
        col_r2.metric("Precio sugerido de compra", f"${precio_compra_siniestro:,.0f}")
        col_r3.metric("Precio sugerido de venta", f"${precio_venta_sugerido:,.0f}")

        col_r4, col_r5 = st.columns(2)
        col_r4.metric("Utilidad esperada", f"${utilidad:,.0f}")
        col_r5.metric("Margen estimado", f"{margen_porcentaje:,.1f} %")

        st.subheader("🚦 Decisión de compra")

        if margen_porcentaje >= 20:
            st.success("🟢 RECOMENDACIÓN: COMPRAR")
            st.write("La operación presenta un margen atractivo y cumple con el objetivo de rentabilidad.")

        elif margen_porcentaje >= 10:
            st.warning("🟡 RECOMENDACIÓN: NEGOCIAR")
            st.write("El margen es aceptable, pero se recomienda negociar mejor el precio de compra o reducir costos.")

        else:
            st.error("🔴 RECOMENDACIÓN: NO COMPRAR")
            st.write("El margen es bajo y existe riesgo de pérdida en la operación.")

        st.info(
            "💡 Esta recomendación se basa en el margen esperado. "
            "Debe complementarse con inspección física del camión y condiciones reales del mercado."
        )

        with st.expander("Ver detalle de costos"):
            st.write(f"💸 **Precio de mercado ajustado:** ${precio_modelo_ajustado:,.0f} MXN")
            st.write(f"🟢 **Precio sugerido de compra:** ${precio_compra_siniestro:,.0f} MXN")
            st.write("----")
            st.write(f"🚚 Logística: ${costo_logistica:,.0f} MXN")
            st.write(f"🔧 Reparación: ${costo_reparacion:,.0f} MXN")
            st.write(f"📦 Otros costos: ${otros_costos:,.0f} MXN")
            st.write(f"🧾 **Costo total:** ${costo_total:,.0f} MXN")

        st.success(
            "✅ Cotización generada. Usa estos valores como referencia para negociar la compra."
        )


# ================== TAB 2: DASHBOARD ==================
with tab2:
    st.header("📊 Dashboard de mercado")

    colf1, colf2, colf3 = st.columns(3)

    with colf1:
        marca_filtro = st.selectbox(
            "Filtrar por marca:",
            ["Todas"] + sorted(df["truck_brand"].unique())
        )

    with colf2:
        ubi_filtro = st.selectbox(
            "Filtrar por ubicación:",
            ["Todas"] + sorted(df["ubication"].unique())
        )

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

    st.markdown("#### KPIs del segmento filtrado")

    colk1, colk2, colk3 = st.columns(3)
    colk1.metric("Unidades", f"{len(df_filtrado):,}")
    colk2.metric("Precio promedio", f"${df_filtrado[TARGET_COL].mean():,.0f} MXN")
    colk3.metric("Mediana precio", f"${df_filtrado[TARGET_COL].median():,.0f} MXN")

    st.markdown("---")

    st.subheader("1️⃣ Distribución de precios de mercado")

    chart_hist = alt.Chart(df_filtrado).mark_bar().encode(
        x=alt.X(
            "market_price_mex:Q",
            bin=alt.Bin(maxbins=30),
            title="Precio de mercado [MXN]"
        ),
        y=alt.Y("count():Q", title="Número de camiones"),
        tooltip=["count()"]
    ).properties(height=300)

    st.altair_chart(chart_hist, use_container_width=True)

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


# ================== TAB 3: MODELO Y MÉTRICAS ==================
with tab3:
    st.header("📈 Modelo y métricas")

    st.markdown(
        """
        En esta sección se muestra la comparación entre los modelos evaluados y
        las métricas del modelo final seleccionado.
        """
    )

    if metrics is None:
        st.error("No se encontraron métricas guardadas. Ejecuta primero `python src/train_model.py`.")

    else:
        st.subheader("🏆 Mejor modelo seleccionado")

        best_model = metrics.get("best_model", "No especificado")
        criterion = metrics.get("selection_criterion", "Menor error en prueba")

        colb1, colb2 = st.columns(2)
        colb1.metric("Modelo final", best_model)
        colb2.metric("Criterio de selección", criterion)

        st.markdown("---")

        st.subheader("🔍 Comparación de modelos")

        if "models" in metrics:
            comparison_rows = []

            for model_name, values in metrics["models"].items():
                test_metrics = values["test"]
                train_metrics = values["train"]

                comparison_rows.append({
                    "Modelo": model_name,
                    "MAE Test": test_metrics["mae"],
                    "RMSE Test": test_metrics["rmse"],
                    "R² Test": test_metrics["r2"],
                    "MAE Train": train_metrics["mae"],
                    "RMSE Train": train_metrics["rmse"],
                    "R² Train": train_metrics["r2"],
                })

            df_comparison = pd.DataFrame(comparison_rows)

            st.dataframe(
                df_comparison.style.format({
                    "MAE Test": "${:,.0f}",
                    "RMSE Test": "${:,.0f}",
                    "R² Test": "{:.3f}",
                    "MAE Train": "${:,.0f}",
                    "RMSE Train": "${:,.0f}",
                    "R² Train": "{:.3f}",
                }),
                use_container_width=True
            )

            st.markdown(
                """
                **Interpretación:**  
                La comparación permite identificar qué algoritmo predice mejor el precio de mercado.
                Se selecciona el modelo con menor RMSE en el conjunto de prueba, ya que esta métrica
                penaliza con mayor fuerza los errores grandes.
                """
            )

            df_chart = df_comparison.melt(
                id_vars="Modelo",
                value_vars=["MAE Test", "RMSE Test"],
                var_name="Métrica",
                value_name="Valor"
            )

            chart_models = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X("Modelo:N", title="Modelo"),
                y=alt.Y("Valor:Q", title="Error [MXN]"),
                color="Métrica:N",
                tooltip=["Modelo", "Métrica", "Valor"]
            ).properties(height=350)

            st.altair_chart(chart_models, use_container_width=True)

        else:
            st.warning(
                "El archivo de métricas todavía no contiene comparación de modelos. "
                "Ejecuta nuevamente `python src/train_model.py`."
            )

        st.markdown("---")

        st.subheader("🔢 Métricas del modelo final")

        mae_test = metrics["test"]["mae"]
        rmse_test = metrics["test"]["rmse"]
        r2_test = metrics["test"]["r2"]

        mae_train = metrics["train"]["mae"]
        rmse_train = metrics["train"]["rmse"]
        r2_train = metrics["train"]["r2"]

        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("MAE Test", f"${mae_test:,.0f}")
        colm2.metric("RMSE Test", f"${rmse_test:,.0f}")
        colm3.metric("R² Test", f"{r2_test:.3f}")

        st.subheader("📚 Comparación Train vs Test")

        colt1, colt2, colt3 = st.columns(3)
        colt1.metric("MAE Train", f"${mae_train:,.0f}")
        colt2.metric("RMSE Train", f"${rmse_train:,.0f}")
        colt3.metric("R² Train", f"{r2_train:.3f}")

        st.caption(
            "Comparar entrenamiento contra prueba permite revisar si existe sobreajuste o bajo ajuste."
        )

        st.subheader("📌 Interpretación ejecutiva")

        st.markdown(
            """
            El modelo final permite estimar el precio de mercado de un camión en buen estado
            y usar esa estimación como base para calcular un precio máximo recomendable de compra
            de una unidad siniestrada.

            Desde el punto de vista de negocio, el resultado no debe interpretarse como un precio exacto,
            sino como una referencia para negociar y reducir el riesgo de sobrepago.
            """
        )

        st.info(
            "💡 El modelo no sustituye la inspección física ni la experiencia del comprador; "
            "funciona como herramienta de apoyo para tomar decisiones más informadas."
        )


