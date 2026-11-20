# Documentación Técnica y Arquitectura del Proyecto (`PIA-ML`)

## 1. Descripción General del Sistema

Este repositorio implementa una solución reproducible, modular y documentada para el **Proyecto Final: Clasificación de Gestos (Piedra, Papel o Tijera) mediante Redes Neuronales Convolucionales (CNN)** de la Facultad de Ingeniería Mecánica y Eléctrica (FIME – UANL).

### Función General
- **Pipeline de Aprendizaje Profundo desde Cero**: Descargar el dataset público *Rock, Paper, Scissors* de Kaggle (`drgfreeman/rockpaperscissors`), auditar integridad de imágenes, dividir estrictamente en `Train / Validation / Test (70% / 15% / 15%)` antes del preprocesamiento, entrenar y comparar dos CNN construidas desde cero en TensorFlow/Keras (sin modelos preentrenados), y generar artefactos analíticos completos.
- **Reproducibilidad sin Dependencias en el Host**: Todo el entorno de ejecución y validación corre dentro de contenedores Docker (`docker/Dockerfile`) o directamente en **Google Colab**, manteniendo limpio el sistema operativo anfitrión.
- **Generación de Reporte Académico en LaTeX**: Sincronización entre los resultados cuantitativos (`outputs/experiment_results.json`), las figuras de alta resolución (`report/Imágenes/`) y el documento fuente LaTeX (`report/main.tex` + `report/contenido.tex`) compatible con **Murfy AI** y TeXLive.

---

## 2. Descripción Específica de Componentes y Funciones

| Componente / Archivo | Función Específica |
| :--- | :--- |
| [`notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb`](../notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb) | Libreta ejecutable en Google Colab y Jupyter con las 4 secciones del proyecto, gráficas `matplotlib`/`seaborn`, celdas Markdown de análisis tras cada bloque de código y exportación automática de métricas. |
| [`scripts/build_notebook.py`](../scripts/build_notebook.py) | Generador determinista de la libreta `.ipynb` usando únicamente la biblioteca estándar de Python (`json`), asegurando trazabilidad y estructura homogénea. |
| [`docker/Dockerfile`](../docker/Dockerfile) & [`docker/requirements.txt`](../docker/requirements.txt) | Definición del contenedor reproducible (`python:3.12-slim` + `tensorflow-cpu`, `kagglehub`, `scikit-learn`, `matplotlib`, `seaborn`, `nbconvert`, `ruff`) para ejecución y pruebas locales aisladas. |
| [`docker/docker-compose.yml`](../docker/docker-compose.yml) | Orquestación declarativa del servicio `ml-runner` para ejecutar la libreta o scripts dentro del contenedor con montaje del repositorio. |
| [`report/main.tex`](../report/main.tex) | Archivo maestro LaTeX basado en la plantilla institucional FIME–UANL (`article`, `babel[spanish,mexico]`, `natbib`, `apalike`, encabezados `fancyhdr` y marca de agua). |
| [`report/datos.tex`](../report/datos.tex) & [`report/portada.tex`](../report/portada.tex) | Metadatos institucionales y estructura de la portada (escudos UANL/FIME, Unidad de Aprendizaje, título, día/hora de clase, docente y tabla de integrantes). |
| [`report/contenido.tex`](../report/contenido.tex) | Cuerpo técnico del reporte: Introducción, Antecedentes (teoría de CNN), Metodología, Resultados experimentales reales, Discusión Crítica y Conclusiones. |
| [`report/bibliografia.bib`](../report/bibliografia.bib) | Base bibliográfica en formato BibTeX con referencias académicas sobre CNN, optimización, regularización, TensorFlow/Keras y el dataset de Kaggle. |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Pipeline de Integración Continua (GitHub Actions) que valida el esquema JSON y compilación de celdas de la libreta, aplica análisis estático con `ruff` y verifica la compilación del reporte LaTeX a PDF. |

---

## 3. Funciones Clave dentro de la Libreta (`PIA_CNN_Piedra_Papel_Tijera.ipynb`)

1. **`resolve_class_dirs(base_dir, class_names)`**: Localiza de forma robusta los directorios de clases (`paper`, `rock`, `scissors`) dentro de la caché descargada por `kagglehub`.
2. **`load_and_preprocess_split(df_split)`**: Carga imágenes RGB desde disco, aplica redimensionamiento uniforme bilineal a `150 × 150 × 3`, normaliza intensidades dividiendo entre `255.0` hacia `[0, 1]` y codifica las etiquetas en formato *one-hot*.
3. **`build_baseline_cnn(input_shape, num_classes)`**: Construye la arquitectura base `CNN_v1_Base` desde cero con 3 bloques `Conv2D → MaxPooling2D` (32, 64 y 128 filtros), `Flatten` y clasificador `Dense(128, relu) → Dense(3, softmax)`.
4. **`build_regularized_cnn(input_shape, num_classes)`**: Construye la arquitectura regularizada `CNN_v2_Regularizada` desde cero añadiendo aumento de datos solo en entrenamiento (`RandomFlip`, `RandomRotation`, `RandomZoom`), 4 bloques `Conv2D → BatchNormalization → ReLU → MaxPooling2D` y `Dropout(0.4)`.
5. **`make_callbacks(model_name)`**: Configura `EarlyStopping` (restauración de mejores pesos según `val_loss`) y `ReduceLROnPlateau` para estabilizar la convergencia.
6. **`plot_learning_curves(history, title_prefix, save_path)`**: Genera y guarda en alta resolución (`220 DPI`) las curvas paralelas de *Loss* y *Accuracy* para entrenamiento y validación.

---

## 4. Registro de Cambios Realizados

- Creación de la infraestructura contenerizada en `docker/` sin instalar paquetes en el sistema host.
- Generación y ejecución integral de `notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb` sobre el dataset real de Kaggle.
- Adaptación de la plantilla institucional LaTeX en `report/` integrando figuras reales, métricas exactas del conjunto de prueba y referencias académicas.
- Configuración del pipeline CI/CD en `.github/workflows/ci.yml`.
