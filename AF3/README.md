# AF 3: Introducción al Aprendizaje Máquina — Aplicación de Clustering (K-Means)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/AF3/AF3.ipynb)

- **Libreta de la actividad (con salidas y gráficas embebidas):** [`AF3.ipynb`](./AF3.ipynb)
- **Reporte en LaTeX (Plantilla FIME–UANL):** [`report/main.tex`](./report/main.tex) y [`report/contenido.tex`](./report/contenido.tex)
- **Paquete ZIP listo para Murfy AI:** [`outputs/AF3_Reporte_LaTeX_MurfyAI.zip`](./outputs/AF3_Reporte_LaTeX_MurfyAI.zip)
- **Dataset:** [Stroke Prediction Dataset (Kaggle — `fedesoriano/stroke-prediction-dataset`)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## Descripción
Utilizando el dataset limpio de la Actividad Fundamental 1 (`N = 5,110` pacientes, imputando los 201 valores faltantes de `bmi` con la mediana `28.10 kg/m²`), pero **omitiendo la columna objetivo (`stroke`) y la columna `id`**, se desarrolla un análisis completo de agrupamiento no supervisado mediante **K-Means**.

---

### Sección 1: Preparación y Escalado de Características para Distancias

- **Objetivo:** Comprender por qué los algoritmos de clustering basados en distancias (como K-Means) exigen un preprocesamiento estricto.
- **Instrucciones:**
  1. Carga el dataset limpio y **elimina** la columna objetivo (`stroke`) y la columna `id`. El algoritmo no debe saber quién sufrió un derrame.
  2. Selecciona las variables numéricas y continuas clave (`age`, `avg_glucose_level`, `bmi`).
  3. Aplica un `StandardScaler` para normalizar los datos ($\mu = 0, \sigma = 1$).
  4. **Investigación:** ¿Por qué si una variable está en años (0 a 100) y otra en miligramos por decilitro (50 a 300) el algoritmo de K-Means fallaría catastróficamente si no se escalan los datos?

---

### Sección 2: Búsqueda del Número Óptimo de Clústeres (Método del Codo y Silueta)

- **Objetivo:** Determinar cuántos perfiles de pacientes existen de forma matemática en lugar de adivinarlo.
- **Instrucciones:**
  1. Utiliza el **Método del Codo (Elbow Method)** iterando K-Means para un $k$ desde 1 hasta 10, calculando la inercia (*inertia* / suma de errores cuadráticos intra-clúster). Grafica los resultados con `matplotlib`.
  2. Calcula también el **Coeficiente de Silueta (Silhouette Score)** para cada valor de $k \in [2, 10]$ para evaluar qué tan bien definidos y separados están los clústeres.
  3. Con base en ambas gráficas, selecciona el valor óptimo de $k$ (**$k = 4$ perfiles clínicos**, donde convergen el codo de la inercia `5,140.29` y el máximo global absoluto del Coeficiente de Silueta `0.3727`).

---

### Sección 3: Entrenamiento del Modelo y Visualización de Centroides

- **Objetivo:** Agrupar a los pacientes e interpretar el significado físico de cada clúster.
- **Instrucciones:**
  1. Entrena el modelo final de `KMeans` utilizando el número óptimo de clústeres seleccionado ($k = 4$).
  2. Obtén las coordenadas de los **centroides** (*cluster centers*) en escala estandarizada y en unidades clínicas originales.
  3. Crea un gráfico de dispersión (`scatter plot`) utilizando `matplotlib` relacionando dos variables clave (**Edad vs. Glucosa** y **Edad vs. IMC**), pintando a cada punto según el clúster asignado y marcando con una **"X"** la posición de los centroides.

---

### Sección 4: Validación Cruzada (El toque maestro del Aprendizaje No Supervisado)

- **Objetivo:** Evaluar si un algoritmo ciego (*sin etiquetas*) logró agrupar de manera natural a las personas con mayor propensión a enfermar.
- **Instrucciones:**
  1. Vuelve a unir la columna `stroke` (guardada al inicio) al DataFrame con las etiquetas de los clústeres asignadas por el modelo.
  2. Crea una tabla cruzada y gráfico de proporciones (`pd.crosstab`) entre los clústeres descubiertos y la variable real `stroke`.
  3. **Pregunta final de análisis:** *¿Cuál de los clústeres formados concentra el mayor porcentaje de casos reales de derrame cerebral? ¿El algoritmo de clustering fue capaz de aislar un grupo de alto riesgo médico sin haber visto nunca la etiqueta "stroke" durante su entrenamiento?*

---

## Resumen de Resultados Obtenidos ($k = 4$, `SEED = 42`)

| Clúster Descubierto | Edad Media (años) | Glucosa Media (mg/dL) | IMC Medio ($\text{kg/m}^2$) | Pacientes ($n$) | Casos `stroke=1` | Prevalencia Intra-Clúster | % del Total de Derrames (249) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clúster 0** *(Adulto Mayor Normoglucémico)* | `60.03` | `89.54` | `27.74` | 1,877 | 141 | `7.51%` (`1.54×` basal) | **`56.63%`** |
| **Clúster 1** *(Adulto Obesidad Normoglucémica)* | `41.52` | `90.32` | **`39.03`** | 922 | 15 | `1.63%` | `6.02%` |
| **Clúster 2** *(Joven / Pediátrico Saludable)* | `17.89` | `92.67` | `22.81` | 1,639 | 4 | `0.24%` | `1.61%` |
| **Clúster 3** *(Adulto Mayor Hiperglucémico — Alto Riesgo)* | **`60.54`** | **`207.30`** | **`32.75`** | 672 | 89 | **`13.24%` (`2.72×` basal)** | **`35.74%`** |
| **Total / Basal** | `43.23` | `106.15` | `28.86` | **5,110** | **249** | **`4.87%`** | **`100.00%`** |

> **Hallazgo clave:** Sin haber visto jamás la columna `stroke`, K-Means aisló en el **Clúster 3** una prevalencia de derrame cerebral de **`13.24%`** (`2.72×` el riesgo basal y `54.3×` el riesgo del Clúster 2), y entre los dos perfiles de adultos mayores (**Clúster 3 + Clúster 0**) capturó el **`92.37%` (`230/249`)** de todos los derrames cerebrales del dataset.
