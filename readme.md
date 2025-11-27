📘 Proyecto: Cotizador Inteligente de Camiones Siniestrados
README – Instrucciones de uso

Este proyecto implementa una aplicación web en Streamlit que predice el precio de mercado de un camión de carga en México y genera una cotización ajustada para compra como siniestrado, considerando descuentos, costos y márgenes de venta.
También incluye un dashboard, análisis del mercado, visualizaciones, importancia de variables y un asistente IA (OpenAI).

🧠 1. ¿Qué hace la aplicación?

La aplicación permite:

Predecir el precio de mercado de un camión según sus características.

Simular escenarios de compra y venta para unidades siniestradas.

Calcular utilidad y margen esperado.

Ver gráficos del mercado: precios por marca, año y distribución.

Consultar la importancia de variables del modelo.

Usar un asistente IA integrado para responder preguntas sobre el análisis y las cotizaciones.

📦 2. Estructura del proyecto
proyecto-camiones-streamlit/
│
├── src/
│   ├── app.py               ← Aplicación Streamlit principal
│   ├── train_model.py       ← Script de entrenamiento del modelo
│   ├── model_camiones.pkl   ← Modelo entrenado
│   ├── model_metrics.json   ← Métricas principales
│
├── data/                    ← Dataset usado (opcional para entrega)
│
├── requirements.txt         ← Librerías necesarias para ejecutar la app
├── README.md                ← Este archivo

▶️ 3. Cómo ejecutar la aplicación localmente
Requisitos previos

Asegúrate de tener instalado:

Python 3.10 o superior

pip

Anaconda (opcional pero recomendado)

Paso 1: Clonar el repositorio
git clone https://github.com/lavg270994-hue/proyecto-camiones-streamlit.git
cd proyecto-camiones-streamlit

Paso 2: Instalar dependencias
pip install -r requirements.txt

Paso 3: Ejecutar Streamlit
streamlit run src/app.py

Paso 4: Abrir la app en tu navegador

Cuando la terminal muestre algo como:

Local URL: http://localhost:8501


Solo debes abrir ese enlace.

🔑 4. Uso del asistente IA (OpenAI)

Si deseas usar el asistente IA dentro de la app, debes configurar tu API key:

Mac / Linux
export OPENAI_API_KEY="tu_clave_aqui"

Windows PowerShell
setx OPENAI_API_KEY "tu_clave_aqui"


Luego reinicia la terminal y vuelve a correr la app.

🌐 5. Cómo desplegar la app en Streamlit Cloud

Ve a: https://share.streamlit.io

Conecta tu cuenta con GitHub.

Selecciona tu repositorio:

lavg270994-hue/proyecto-camiones-streamlit


En el campo “Main file path”, escribe:

src/app.py


Guarda y despliega.

La app se publicará y podrás compartir el enlace.

Si usas el asistente IA, agrega tu API key como Secret:

En Streamlit Cloud → “App settings”

“Secrets” →

OPENAI_API_KEY="tu_clave"

📊 6. Métricas del modelo

Estas métricas se calculan con datos de prueba reales:

MAE: 57,397

RMSE: 81,549

R²: 0.81

El modelo explica el 81% de la variación del precio de mercado.

📈 7. Visualizaciones incluidas

La aplicación muestra al menos tres gráficos obligatorios:

Histograma de precios.

Precio promedio por marca.

Precio promedio por año.

Además incluye:

Importancia de variables del modelo.

KPIs de compra, costo total, rentabilidad y margen.

💬 8. Características interactivas

Selectores y filtros para configurar el camión.

Sliders para definir descuento, costos y markup.

Botón de cálculo completo del escenario.

Asistente IA integrado.

Gráficas dinámicas según las selecciones.

📝 9. Autor

Luis Alberto Vera Castillo
Proyecto final – Curso de Ciencia de Datos
2025

✔️ 10. Licencia

Uso académico y no comercial.
