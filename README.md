# Detección de lavado de activos con modelos no supervisados

Este proyecto implementa un sistema de detección de anomalías en transacciones financieras
orientado a casos de lavado de activos (AML), utilizando modelos **no supervisados**
( Isolation Forest y Local Outlier Factor ) combinados con **features de comportamiento**
y **features de red (grafos)**.

## Objetivos

- Identificar transacciones inusuales sin depender exclusivamente de etiquetas históricas.
- Enriquecer cada transacción con:
  - Agregados históricos por cuenta (volumen, frecuencia, montos).
  - Métricas de red (in/out degree) sobre el grafo de transacciones.
- Exponer los resultados en un dashboard interactivo con Streamlit para analistas AML.

## Estructura del proyecto

```bash
aml_aml_unsupervised/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ feature_engineering.py
│  ├─ models.py
│  └─ pipeline.py
├─ app/
│  └─ streamlit_app.py
├─ tests/
│  └─ test_feature_engineering.py
├─ config.yaml
├─ requirements.txt
└─ README.md
