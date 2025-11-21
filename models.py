
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, classification_report


def detect_label_column(df: pd.DataFrame, explicit_name: str | None) -> str | None:
    if explicit_name is not None and explicit_name in df.columns:
        logger.info(f"Usando columna de etiqueta explícita: {explicit_name}")
        return explicit_name

    candidates = ["label", "is_sar", "is_fraud", "fraud", "alert", "alert_flag"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            col = lower_cols[cand]
            logger.info(f"Columna de etiqueta detectada: {col}")
            return col

    logger.info("No se detectó columna de etiqueta.")
    return None


def build_feature_matrix(df: pd.DataFrame, label_col: str | None) -> Tuple[np.ndarray, list[str], StandardScaler]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    logger.info(f"Columnas numéricas detectadas: {len(numeric_cols)}")

    exclude_keywords = ["id", "flag", "label", "sar", "alert", "account"]
    cols_to_exclude = set()
    for col in numeric_cols:
        if any(k in col.lower() for k in exclude_keywords):
            cols_to_exclude.add(col)

    if label_col and label_col in numeric_cols:
        cols_to_exclude.add(label_col)

    feature_cols = [c for c in numeric_cols if c not in cols_to_exclude]
    if not feature_cols:
        raise ValueError("No quedaron columnas numéricas válidas para features.")

    logger.info(f"Features usadas ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols, scaler


def fit_isolation_forest(X_scaled: np.ndarray, contamination: float, random_state: int = 42):
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    decision_scores = iso.decision_function(X_scaled)
    anomaly_scores = -decision_scores
    return iso, anomaly_scores


def fit_lof(X_scaled: np.ndarray, contamination: float):
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
    )
    _ = lof.fit_predict(X_scaled)
    neg_factor = lof.negative_outlier_factor_
    lof_scores = -neg_factor
    return lof, lof_scores


def build_ensemble_scores(iso_scores: np.ndarray, lof_scores: np.ndarray) -> pd.DataFrame:
    scores_df = pd.DataFrame({
        "iso_score": iso_scores,
        "lof_score": lof_scores,
    })
    rank_iso = scores_df["iso_score"].rank(pct=True)
    rank_lof = scores_df["lof_score"].rank(pct=True)
    scores_df["ensemble_score"] = (rank_iso + rank_lof) / 2.0
    return scores_df


def flag_anomalies(scores_df: pd.DataFrame, threshold: float) -> pd.Series:
    anomalies = scores_df["ensemble_score"] >= threshold
    logger.info(
        f"Umbral de anomalía (ensemble_score >= {threshold:.3f}) "
        f"→ {anomalies.sum()} anomalías de {len(anomalies)} "
        f"({100 * anomalies.mean():.2f}%)"
    )
    return anomalies


def evaluate_with_labels(df: pd.DataFrame, label_col: str, anomalies: pd.Series):
    y_true = df[label_col]
    y_pred = anomalies.astype(int)

    logger.info("Evaluación vs etiqueta real:")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nMatriz de confusión:\n{cm}")
    report = classification_report(y_true, y_pred, digits=4)
    logger.info(f"\nReporte de clasificación:\n{report}")
