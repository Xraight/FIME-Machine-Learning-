# Introducción al Machine Learning — Actividades Fundamentales (AF) y Proyecto Integrador (PIA)

**Universidad Autónoma de Nuevo León (UANL) — Facultad de Ingeniería Mecánica y Eléctrica (FIME)**  
**Unidad de Aprendizaje:** Introducción a la Inteligencia Artificial / Introducción al Aprendizaje Máquina  
**Repositorio Oficial:** [`https://github.com/Xraight/FIME-Machine-Learning-`](https://github.com/Xraight/FIME-Machine-Learning-)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb)

---

## Índice de Entregas en el Repositorio

| Actividad | Carpeta / Archivo Principal | Descripción |
| :--- | :--- | :--- |
| **AF1** | [`nombredelacarpeta/AF1.ipynb`](./nombredelacarpeta/AF1.ipynb) | Actividad Fundamental 1 ([PDF](./nombredelacarpeta/Actividad%20fundamental%201.pdf)) |
| **AF2** | [`AF2 Modelo-de-aprendizaje-supervisado/AF2.ipynb`](./AF2%20Modelo-de-aprendizaje-supervisado/AF2.ipynb) | Actividad Fundamental 2 — Modelos de Aprendizaje Supervisado ([PDF](./AF2%20Modelo-de-aprendizaje-supervisado/Actividad%20fundamental%202.pdf)) |
| **PIA (Proyecto Final)** | [`notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb`](./notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb) / [`PIA-Clasificacion-de-Gestos-CNN/`](./PIA-Clasificacion-de-Gestos-CNN/) | Clasificación de Gestos (Piedra, Papel o Tijera) mediante Redes Neuronales Convolucionales (CNN) desde Cero + Reporte LaTeX ([`report/`](./report/)) |

---

## Proyecto Final (PIA): Clasificación de Gestos (Piedra, Papel o Tijera) con CNN

**Dataset:** [Rock, Paper, Scissors (`drgfreeman/rockpaperscissors` en Kaggle)](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)

Este repositorio contiene la solución integral y reproducible del Proyecto Final de clasificación de gestos de mano (**Piedra, Papel o Tijera**) utilizando **Redes Neuronales Convolucionales (CNN) diseñadas 100% desde cero** en TensorFlow/Keras (sin arquitecturas preentrenadas):

1. **Libreta Ejecutable de Google Colab ([`notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb`](./notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb))**:
   - **Sección 1**: Descarga automática vía `kagglehub`, auditoría de integridad (`2,188` imágenes RGB, `0` corruptas), análisis exploratorio visual, división estratificada **Train (70% = 1,531) / Validation (15% = 328) / Test (15% = 329)** *antes* del preprocesamiento, redimensionamiento a `150×150×3`, normalización a `[0, 1]` y respuesta técnica a la Pregunta 1.
   - **Sección 2**: Diseño desde cero y comparación de **`CNN_v1_Base`** (`5.40 M` parámetros, 3 bloques `Conv2D → MaxPooling2D`) y **`CNN_v2_Regularizada`** (`1.57 M` parámetros, 4 bloques `Conv2D → BatchNormalization → ReLU → MaxPooling2D` + `Dropout(0.35)` + *Data Augmentation* en entrenamiento).
   - **Sección 3**: Entrenamiento monitoreado con `EarlyStopping` y `ReduceLROnPlateau`, curvas de aprendizaje (*Loss* y *Accuracy*) y evaluación en el conjunto independiente de prueba (*Test*).
   - **Sección 4**: Matriz de confusión (conteos y normalizada), reporte de clasificación (*Precision*, *Recall*, *F1-score*), inspección visual de aciertos y errores, y discusión crítica final.
2. **Reporte Académico en LaTeX ([`report/`](./report/))**:
   - Basado en la plantilla institucional FIME–UANL ([`main.tex`](./report/main.tex), [`portada.tex`](./report/portada.tex), [`datos.tex`](./report/datos.tex), [`contenido.tex`](./report/contenido.tex), [`bibliografia.bib`](./report/bibliografia.bib) y figuras reales en [`report/Imágenes/`](./report/Imágenes/)).
   - Incluye el paquete listo para importar en **Murfy AI**: [`outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip`](./outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip).
3. **Entorno Reproducible en Docker ([`docker/`](./docker/)) y CI/CD ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml))**:
   - Ejecución contenerizada sin instalar dependencias en el sistema anfitrión.

---

## Resumen de Resultados Experimentales (`SEED = 42`)

| Modelo | Parámetros | Épocas | Val Loss | Val Accuracy | Test Loss | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CNN_v1_Base` | `5,402,179` | 14 | `0.0646` | `98.17%` | `0.0469` | **`99.39%`** (`327/329`) |
| `CNN_v2_Regularizada` *(Seleccionado por Val)* | **`1,569,507`** | 20 | **`0.0065`** | **`99.70%`** | **`0.0180`** | **`98.78%`** (`325/329`) |

### Métricas por Clase en Test (`CNN_v2_Regularizada`, $N = 329$)

| Clase | Precision | Recall | F1-Score | Soporte |
| :--- | :---: | :---: | :---: | :---: |
| **Papel** | `0.9727` | `1.0000` | `0.9862` | 107 |
| **Piedra** | `0.9909` | `1.0000` | `0.9954` | 109 |
| **Tijera** | `1.0000` | `0.9646` | `0.9820` | 113 |
| **Macro Avg** | `0.9879` | `0.9882` | `0.9879` | 329 |
| **Weighted Avg** | `0.9881` | `0.9878` | `0.9878` | 329 |

---

## Estructura del Repositorio

```text
PIA-ML/
├── notebooks/
│   └── PIA_CNN_Piedra_Papel_Tijera.ipynb   # Libreta ejecutada lista para Google Colab
├── report/
│   ├── main.tex                             # Documento maestro LaTeX (plantilla FIME-UANL)
│   ├── portada.tex                          # Portada institucional
│   ├── datos.tex                            # Metadatos de integrantes, materia, hora y docente
│   ├── contenido.tex                        # Desarrollo completo del reporte
│   ├── bibliografia.bib                     # Referencias en formato BibTeX
│   └── Imágenes/                            # Escudos institucionales + gráficas reales del experimento
├── outputs/
│   ├── experiment_results.json              # Métricas cuantitativas exportadas por la libreta
│   └── PIA_ML_Reporte_LaTeX_MurfyAI.zip     # Paquete LaTeX listo para importar en Murfy AI
├── scripts/
│   └── build_notebook.py                    # Generador determinista de la libreta .ipynb
├── docker/
│   ├── Dockerfile                           # Imagen reproducible con TensorFlow y herramientas
│   ├── docker-compose.yml                   # Servicio contenerizado ml-runner
│   └── requirements.txt                     # Dependencias fijadas del contenedor
├── docs/
│   └── ARCHITECTURE.md                      # Documentación general y específica del sistema
└── .github/workflows/
    └── ci.yml                               # Pipeline CI/CD (validación de libreta, lint y LaTeX)
```

---

## Ejecución

### 1. En Google Colab
Abre directamente [`notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb`](./notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb) en **Google Colab** (selecciona entorno de ejecución con **GPU T4**) y ejecuta todas las celdas (`Runtime > Run all`). La libreta descarga automáticamente el dataset de Kaggle mediante `kagglehub`.

### 2. En Docker Local (Sin instalar paquetes en el sistema host)
```bash
# Construir la imagen
docker build -t pia-ml-cnn:latest -f docker/Dockerfile .

# Ejecutar y validar la libreta completa dentro del contenedor
docker run --rm -v "$PWD:/workspace" -w /workspace pia-ml-cnn:latest \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 notebooks/PIA_CNN_Piedra_Papel_Tijera.ipynb
```
