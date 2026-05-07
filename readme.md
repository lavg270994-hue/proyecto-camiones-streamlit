# Universidad Anáhuac Puebla

# Proyecto Final  
## “Cotizador inteligente de camiones siniestrados”

### Integrantes
- Edgar Eduardo González Amezcua
- Luis Alberto Vera Guevara

### Materia
Machine Learning

### Profesor
Dr. Paulo Daniel Vázquez

---

# Introducción

Decidimos desarrollar un cotizador inteligente de camiones siniestrados debido a que uno de los integrantes del equipo, Luis Alberto Vera Guevara, tiene experiencia y un proyecto relacionado con el negocio de reparación de tractocamiones y venta de refacciones.

Dentro de este sector existe una gran oferta de camiones, y el precio puede variar considerablemente dependiendo de múltiples variables como la marca, el motor, la transmisión, el año, la configuración de ejes, la ubicación y el estado general de la unidad.

Uno de los principales problemas dentro de este negocio es no subestimar los costos de reparación, ya que esto puede reducir significativamente el margen de utilidad esperado. Sin embargo, no todos los tractocamiones tienen el mismo comportamiento en el mercado. Existen marcas como Kenworth que normalmente tienen una rotación de venta más rápida, lo cual puede compensar márgenes menores.

Con este cotizador inteligente, apoyándonos en técnicas de Machine Learning y visualización de datos, es posible estimar el precio de mercado de un tractocamión configurando distintas variables, permitiendo así determinar precios óptimos de compra para unidades siniestradas.

---

# Desarrollo

El proyecto fue desarrollado utilizando técnicas de Machine Learning para estimar el valor de mercado de tractocamiones y apoyar la toma de decisiones en la compra de unidades siniestradas.

Para el proyecto se utilizó un dataset histórico de precios reales proporcionado previamente por el compañero Luis. Adicionalmente, se utilizaron datos sintéticos generados con apoyo de ChatGPT a partir de especificaciones basadas en información histórica real.

El dataset contiene información relacionada con:

- Marca del camión
- Modelo
- Año
- Motor
- Tipo de transmisión
- Tipo de eje o diferenciales
- Ubicación
- Precio de mercado

---

# Modelos utilizados

Inicialmente se implementó un modelo de Random Forest Regressor debido a su capacidad para trabajar con:

- Datos heterogéneos
- Relaciones no lineales
- Interacciones complejas entre múltiples variables

Posteriormente también se realizó una comparación contra un modelo de Linear Regression para evaluar el desempeño entre distintos enfoques de regresión.

---

# Métricas de evaluación

Para evaluar el desempeño del modelo se utilizaron las siguientes métricas:

## MAE (Mean Absolute Error)

El Error Absoluto Medio representa el promedio de error entre las predicciones y el valor real.

El modelo obtuvo aproximadamente:

- MAE = $57,397 MXN

Considerando que muchos tractocamiones superan el millón de pesos en valor comercial, este error es aceptable dentro del contexto del proyecto.

## RMSE (Root Mean Squared Error)

La Raíz del Error Cuadrático Medio penaliza errores grandes elevando las diferencias al cuadrado.

El modelo obtuvo:

- RMSE = $81,549 MXN

## R² (Coeficiente de determinación)

El coeficiente R² obtenido fue:

- R² = 0.81

Esto indica que el modelo logra explicar aproximadamente el 81% de la variabilidad del precio de mercado.

Durante el desarrollo también se detectó un problema de sobreentrenamiento, ya que inicialmente el modelo mostraba una precisión cercana al 99%, lo cual no era realista. Posteriormente se corrigió mediante una mejor validación utilizando métricas sobre el conjunto de prueba.

---

# Dashboard y funcionalidades

El proyecto incluye un dashboard interactivo desarrollado en Streamlit que permite:

- Estimar el precio de mercado de un camión
- Visualizar diferencias de precio entre marcas
- Analizar el incremento de precios por año
- Comparar componentes como motor y transmisión
- Calcular utilidad esperada
- Calcular margen estimado
- Estimar costos totales
- Simular escenarios de negocio

El sistema predice el precio de mercado de un camión en buen estado, posteriormente aplica un descuento por condición de siniestro y agrega costos relacionados con:

- Logística
- Reparación
- Otros gastos

Finalmente calcula:

- Precio sugerido de compra
- Precio sugerido de venta
- Utilidad esperada
- Margen de ganancia

---

# Decisión automática de compra

Se implementó una lógica automática de recomendación para apoyar la toma de decisiones:

- Si el margen es mayor o igual al 20% → **COMPRAR**
- Si el margen está entre 10% y 20% → **NEGOCIAR**
- Si el margen es menor al 10% → **NO COMPRAR**

---

# Conclusión

El cotizador inteligente permite estimar el valor de mercado de un tractocamión utilizando información histórica y datos reales, facilitando la toma de decisiones y reduciendo el riesgo de pagar un sobreprecio.

La aplicación desarrollada en Streamlit brinda información rápida y profesional, permitiendo calcular utilidad, margen real, costos totales y distintos escenarios de negocio.

Además, el dashboard permite visualizar variaciones de precio dependiendo de:

- Marca
- Motor
- Transmisión
- Tren motriz
- Año del camión

El proyecto demuestra cómo el Machine Learning puede aplicarse en sectores con alta incertidumbre y múltiples variables que afectan el valor de un bien, buscando disminuir el riesgo y mejorar la precisión en las decisiones de compra.

---

# Tecnologías utilizadas

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Altair
- Machine Learning
- Random Forest Regressor
- Linear Regression

---

# Enlaces

## GitHub
Repositorio del proyecto:
https://github.com/lavg270994-hue/proyecto-camiones-streamlit

## Streamlit
Aplicación desplegada:
https://proyecto-camiones-app-hxvjy7oub58zqmweyyok4m.streamlit.app/
