# Introducción al Machine Learning — Actividades Fundamentales (AF) y Proyecto Integrador (PIA)

**Universidad Autónoma de Nuevo León (UANL) — Facultad de Ingeniería Mecánica y Eléctrica (FIME)**  
**Unidad de Aprendizaje:** Introducción a la Inteligencia Artificial / Introducción al Aprendizaje Máquina  
**Repositorio Oficial:** [`https://github.com/Xraight/FIME-Machine-Learning-`](https://github.com/Xraight/FIME-Machine-Learning-)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb)

---

## Estructura del Repositorio

```text
FIME-Machine-Learning-/
├── AF1/                                      # Actividad Fundamental 1 (EDA y Preprocesamiento)
│   ├── AF1.ipynb
│   ├── Actividad fundamental 1.pdf
│   └── README.md
├── AF2/                                      # Actividad Fundamental 2 (Aprendizaje Supervisado)
│   ├── AF2.ipynb
│   ├── Actividad fundamental 2.pdf
│   └── README.md
├── AF3/                                      # Actividad Fundamental 3 (Clustering K-Means)
│   ├── AF3.ipynb
│   ├── README.md
│   ├── report/
│   ├── outputs/
│   └── scripts/
├── PIA/                                      # Proyecto Integrador de Aprendizaje (Proyecto Final)
│   ├── PIA_CNN_Piedra_Papel_Tijera.ipynb     # Libreta ejecutada lista para Google Colab
│   ├── README.md                             # Descripción e instrucciones del PIA
│   ├── report/                               # Reporte académico en LaTeX (plantilla FIME–UANL) + figuras
│   ├── outputs/                              # Métricas JSON + paquete ZIP listo para Murfy AI
│   ├── docker/                               # Entorno reproducible en Docker
│   ├── scripts/                              # Script generador de la libreta
│   └── docs/                                 # Documentación técnica y de arquitectura
└── README.md
```

| Entrega | Carpeta | Libreta Ejecutable | Documento / Reporte |
| :--- | :--- | :--- | :--- |
| **AF1** | [`AF1/`](./AF1/) | [`AF1/AF1.ipynb`](./AF1/AF1.ipynb) | [`AF1/Actividad fundamental 1.pdf`](./AF1/Actividad%20fundamental%201.pdf) |
| **AF2** | [`AF2/`](./AF2/) | [`AF2/AF2.ipynb`](./AF2/AF2.ipynb) | [`AF2/Actividad fundamental 2.pdf`](./AF2/Actividad%20fundamental%202.pdf) |
| **AF3** | [`AF3/`](./AF3/) | [`AF3/AF3.ipynb`](./AF3/AF3.ipynb) | [`AF3/report/main.tex`](./AF3/report/main.tex) / [`AF3/outputs/AF3_Reporte_LaTeX_MurfyAI.zip`](./AF3/outputs/AF3_Reporte_LaTeX_MurfyAI.zip) |
| **PIA (Proyecto Final)** | [`PIA/`](./PIA/) | [`PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb`](./PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb) | [`PIA/report/main.tex`](./PIA/report/main.tex) / [`PIA/outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip`](./PIA/outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip) |

---

## Proyecto Final (PIA): Clasificación de Gestos (Piedra, Papel o Tijera) mediante CNN

- **Abrir directamente en Google Colab:** [`PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb`](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb)
- **Dataset:** [Rock, Paper, Scissors (`drgfreeman/rockpaperscissors` en Kaggle)](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) (`2,188` imágenes RGB de `300×200`, `0` corruptas)
- **Partición Estratificada (antes del preprocesamiento):** Train `70%` (`1,531`), Validation `15%` (`328`), Test `15%` (`329`)

### Resumen de Resultados Experimentales (`SEED = 42`)

| Modelo (Construido desde Cero) | Parámetros | Épocas | Val Loss | Val Accuracy | Test Loss | Test Accuracy |
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
