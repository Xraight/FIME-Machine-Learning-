# Proyecto Final (PIA): Clasificación de Gestos (Piedra, Papel o Tijera) mediante Redes Neuronales Convolucionales (CNN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xraight/FIME-Machine-Learning-/blob/main/PIA/PIA_CNN_Piedra_Papel_Tijera.ipynb)

- **Libreta ejecutable (Google Colab / Jupyter):** [`PIA_CNN_Piedra_Papel_Tijera.ipynb`](./PIA_CNN_Piedra_Papel_Tijera.ipynb)
- **Reporte en LaTeX (Plantilla FIME–UANL):** [`report/main.tex`](./report/main.tex) y [`report/contenido.tex`](./report/contenido.tex)
- **Paquete ZIP listo para Murfy AI:** [`outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip`](./outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip)
- **Dataset:** [Rock, Paper, Scissors (Kaggle — `drgfreeman/rockpaperscissors`)](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)

---

## Contenido de la Carpeta `PIA/`

| Archivo / Carpeta | Descripción |
| :--- | :--- |
| [`PIA_CNN_Piedra_Papel_Tijera.ipynb`](./PIA_CNN_Piedra_Papel_Tijera.ipynb) | Libreta completa con las 4 secciones del proyecto, ejecutada de principio a fin y con las 7 figuras embebidas. |
| [`report/`](./report/) | Código fuente LaTeX del reporte ([`main.tex`](./report/main.tex), [`portada.tex`](./report/portada.tex), [`datos.tex`](./report/datos.tex), [`contenido.tex`](./report/contenido.tex), [`bibliografia.bib`](./report/bibliografia.bib) e [`Imágenes/`](./report/Imágenes/)). |
| [`outputs/`](./outputs/) | Resumen cuantitativo [`experiment_results.json`](./outputs/experiment_results.json) y paquete [`PIA_ML_Reporte_LaTeX_MurfyAI.zip`](./outputs/PIA_ML_Reporte_LaTeX_MurfyAI.zip). |
| [`docker/`](./docker/) | Entorno reproducible en Docker (`Dockerfile`, `docker-compose.yml`, `requirements.txt`). |
| [`scripts/build_notebook.py`](./scripts/build_notebook.py) | Script generador determinista de la libreta `.ipynb`. |
| [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) | Documentación técnica general y específica del sistema. |

---

## Resumen de Resultados (`SEED = 42`)

| Modelo (Construido desde Cero) | Parámetros | Épocas | Val Loss | Val Accuracy | Test Loss | Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CNN_v1_Base` | `5,402,179` | 14 | `0.0646` | `98.17%` | `0.0469` | **`99.39%`** (`327/329`) |
| `CNN_v2_Regularizada` *(Seleccionado por Val)* | **`1,569,507`** | 20 | **`0.0065`** | **`99.70%`** | **`0.0180`** | **`98.78%`** (`325/329`) |
