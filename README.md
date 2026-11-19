#  Introduccion al machine Learning — Fundamentales(Af) y Proyecto Integrador(Pia)

Este espacio reúne los notebooks, scripts y análisis desarrollados en equipo para la unidad de aprendizaje **Introducción al Aprendizaje Máquina**. Todo el flujo de trabajo está implementado en Python mediante entornos de Google Colab y Jupyter.

---

### Entregables

En lugar de trabajar de forma aislada, las tres actividades fundamentales siguen una línea continua de procesamiento de datos, culminando con un proyecto independiente de visión por computadora:

* **Fase 1 · Limpieza y Exploración (`01_AF1_EDA_y_Preprocesamiento`)**
  Análisis estadístico descriptivo, detección de valores atípicos, manejo de datos faltantes y preparación general del conjunto de datos clínico.
  > 🔗 [Consultar instrucciones en Notion](https://geode-cloud-e9e.notion.site/AF-1-Introducci-n-al-Aprendizaje-M-quina-3de0ddbf1878805b956ed15e1361153f)

* **Fase 2 · Predicción con Redes Neuronales (`02_AF2_Aprendizaje_Supervisado`)**
  Entrenamiento y evaluación de un modelo de clasificación supervisada construido sobre la base de datos previamente depurada en la Fase 1.
  > 🔗 [Consultar instrucciones en Notion](https://geode-cloud-e9e.notion.site/AF-2-Introducci-n-al-Aprendizaje-M-quina-4000ddbf18788024976dcd0570ccc18f)

* **Fase 3 · Segmentación no Supervisada (`03_AF3_Clustering`)**
  Búsqueda de patrones y agrupamiento de registros utilizando el algoritmo **K-Means** a partir de los mismos datos procesados.
  > 🔗 [Consultar instrucciones en Notion](https://geode-cloud-e9e.notion.site/AF-3-Introducci-n-a-Aprendizaje-M-quina-4000ddbf18788057b01bee5c88e90660)

* **Proyecto Integrador · Visión Artificial (`04_Proyecto_Final_CNN`)**
  Diseño y entrenamiento de una Red Neuronal Convolucional (CNN) capaz de reconocer y clasificar señas manuales (*piedra, papel o tijera*).

---

### 📊 Conjuntos de Datos Empleados

1. **[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset):** Utilizado de manera secuencial en las actividades `AF 1`, `AF 2` y `AF 3`. El archivo limpio resultante de la primera práctica sirve como entrada directa para las dos siguientes.
2. **[Rock, Paper, Scissors](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors):** Banco de imágenes utilizado exclusivamente para entrenar y validar la red convolucional del proyecto final.

---

### 🛠️ Stack Tecnológico y Formato de Entrega

* **Lenguaje y Entorno:** Python 3 ejecutado en Google Colab / Jupyter Notebooks.
* **Librerías principales:** `pandas`, `numpy`, `matplotlib`, `scikit-learn` y `TensorFlow / Keras`.
* **Documentación:** Cada carpeta incluye el desarrollo técnico en código para acompañar los reportes formales (portada, introducción, desarrollo con enlace directo al notebook, conclusiones y bibliografía; con entrega final en formato PDF para el proyecto integrador).
