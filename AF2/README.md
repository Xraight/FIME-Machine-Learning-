# AF 2: Introducción al Aprendizaje Máquina — Modelos de Aprendizaje Supervisado

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/AF2/AF2.ipynb)

- **Libreta de la actividad:** [`AF2.ipynb`](./AF2.ipynb)
- **Reporte en PDF:** [`Actividad fundamental 2.pdf`](./Actividad%20fundamental%202.pdf)
- **Enlace directo al Notion de la Fundamental 2:** [AF 2: Introducción al Aprendizaje Máquina](https://geode-cloud-e9e.notion.site/AF-2-Introducci-n-al-Aprendizaje-M-quina-4000ddbf18788024976dcd0570ccc18f)

---

## Descripción
Utilizando el dataset limpio de la Actividad Fundamental 1, desarrollar los siguientes ejercicios en un notebook de Python.

---

### Sección 1. División de datos y escalado (*Preparación para Deep Learning*)

**Instrucciones:**
1. Separar el dataset en variables predictoras (`X`) y la variable objetivo (`y = stroke`).
2. Dividir los datos en conjuntos de entrenamiento y prueba mediante `train_test_split`, utilizando una proporción de 80/20 y el parámetro `stratify=y` debido al desbalance de clases.
   - El parámetro `stratify=y` permite mantener la proporción de las clases tanto en el conjunto de entrenamiento (`train`) como en el conjunto de prueba (`test`), conservando la distribución original del dataset.
3. **Investigación:** ¿Por qué es necesario aplicar un escalador como `StandardScaler` o normalizar las variables continuas antes de ingresarlas a una red neuronal, a diferencia de otros algoritmos como los árboles de decisión?

---

### Sección 2. Diseño de la arquitectura y selección de funciones (*Keras / TensorFlow*)

**Instrucciones:**
1. Construir un modelo secuencial (`keras.Sequential`) con al menos dos capas ocultas.
2. Definir la capa de salida.
3. **Investigación y diseño:**
   - ¿Qué función de activación elegiste para las capas ocultas (ej. `ReLU`, `Tanh`) y por qué?
   - ¿Qué función de activación se debe utilizar en la capa de salida para un problema de clasificación binaria (`stroke` = 0 o 1), y cuál es su rango de salida?
   - Si el dataset está fuertemente desbalanceado, ¿por qué la función de pérdida (`loss`) debe ser `binary_crossentropy` en lugar de Error Cuadrático Medio (MSE)?

---

### Sección 3. Entrenamiento, métricas y optimizadores

**Instrucciones:**
1. Compilar el modelo seleccionando un optimizador (ej. `Adam`, `SGD`) y una métrica adecuada para clases desbalanceadas, como `AUC` o `Recall`, en lugar de utilizar únicamente `Accuracy`.
2. Entrenar el modelo mediante `model.fit`, utilizando un conjunto de validación (`validation_split`) y callbacks opcionales como `EarlyStopping`.
3. **Investigación:**
   - ¿Qué hace la función `EarlyStopping`?
   - ¿Cuál es la diferencia fundamental entre los optimizadores `SGD` (Descenso de Gradiente Estocástico) y `Adam`? ¿Por qué `Adam` suele converger más rápido durante las primeras épocas?

---

## Entregables de la Actividad Fundamental

**Documento de reporte que incluya:**
1. **Portada:** datos de los integrantes que realizaron la actividad (nombre completo, matrícula, hora y día de clase).
2. **Introducción.**
3. **Desarrollo:** descripción de los procedimientos realizados e inclusión del enlace al repositorio de GitHub donde se encuentra el notebook con la solución de los ejercicios.
4. **Conclusión.**
5. **Referencias consultadas.**

- **Modalidad:** Trabajo en equipo.
- **Entrega:** Todos los integrantes deberán entregar la actividad.
