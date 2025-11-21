
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from .config import load_config, PROJECT_ROOT
from .data_loader import load_transactions, load_accounts, load_alerts, get_paths
from .feature_engineering import (
    detect_core_columns,
    add_basic_time_features,
    build_account_aggregates,
    build_graph_features,
    merge_all_features,
    merge_accounts_info,
    merge_alerts_info,
)
from .models import (
    detect_label_column,
    build_feature_matrix,
    fit_isolation_forest,   # modelo principal
    evaluate_with_labels,   # evaluación vs IS_FRAUD
)


def build_simple_explanation(row, amount_col: str) -> str:
    """
    Explicación heurística simple para cada transacción anómala.
    No afecta el modelo, solo sirve para interpretación en el dashboard.
    """
    reasons = []
    amt = row[amount_col]
    mean_sender = row.get("sender_amount_mean", np.nan)

    # Regla 1: monto muy alto vs histórico de la cuenta origen
    if not np.isnan(mean_sender) and amt > 3 * mean_sender:
        reasons.append("Monto > 3x promedio histórico cuenta origen")

    # Regla 2: cuenta con muchas conexiones salientes en el grafo
    out_deg = row.get("sender_graph_out_degree", np.nan)
    if not np.isnan(out_deg) and out_deg > 10:
        reasons.append("Cuenta origen con alto grado de salida en la red")

    # Regla 3: horario poco habitual
    hour = row.get("tx_hour", None)
    if hour is not None and not np.isnan(hour) and (hour < 6 or hour > 22):
        reasons.append("Transacción en horario poco habitual")

    if not reasons:
        return "Score alto según Isolation Forest (modelo no supervisado)"
    return "; ".join(reasons)


def main():
    # Logger
    logger.add("aml_pipeline.log", rotation="1 MB")

    cfg = load_config()
    logger.info("Iniciando pipeline AML no supervisado (solo Isolation Forest)")

    # 1. Cargar datos
    df_tx = load_transactions(cfg)
    df_acc = load_accounts(cfg)
    df_alerts = load_alerts(cfg)

    # 2. Detectar columnas clave en transacciones
    core_cols = detect_core_columns(df_tx, cfg)

    # 3. Features de tiempo
    df_tx = add_basic_time_features(df_tx, core_cols["timestamp"])

    # 4. Aggregates por cuenta (sender/receiver) a partir de transacciones
    sender_feats, receiver_feats = build_account_aggregates(df_tx, core_cols, cfg)

    # 5. Features de grafo (in/out degree de cuentas)
    graph_feats = build_graph_features(df_tx, core_cols)

    # 6. Merge de features de comportamiento + grafo
    df_base = merge_all_features(df_tx, core_cols, sender_feats, receiver_feats, graph_feats)

    # 7. Merge con accounts.csv (info estática de la cuenta)
    df_with_acc = merge_accounts_info(df_base, df_acc, core_cols, cfg)

    # 8. Merge con alerts.csv (tipo de alerta, ALERT_IS_FRAUD)
    df_full = merge_alerts_info(df_with_acc, df_alerts, cfg)

    # 9. Detectar columna de etiqueta (para evaluación, NO para entrenar)
    label_col = detect_label_column(df_full, explicit_name=core_cols["label"])

    # 10. Construir matriz de features X
    X_scaled, feature_cols, scaler = build_feature_matrix(df_full, label_col)

    # 11. Entrenar SOLO Isolation Forest
    contamination = cfg["model"]["contamination"]
    threshold = cfg["model"]["ensemble_threshold"]

    logger.info("Entrenando Isolation Forest (modelo principal de anomalías)...")
    iso_model, iso_scores = fit_isolation_forest(X_scaled, contamination=contamination)

    # 12. Construir DataFrame de scores y ensemble_score basado solo en IF
    scores_df = pd.DataFrame({"iso_score": iso_scores})
    scores_df["ensemble_score"] = scores_df["iso_score"].rank(pct=True)

    # 13. Marcar anomalías según el umbral (por defecto top 1%)
    anomalies = scores_df["ensemble_score"] >= threshold
    logger.info(
        f"Umbral de anomalía (ensemble_score >= {threshold:.3f}) "
        f"→ {anomalies.sum()} anomalías de {len(anomalies)} "
        f"({100 * anomalies.mean():.2f}%)"
    )

    # 14. Unir resultados al DataFrame original
    df_result = df_full.copy()
    df_result["iso_score"] = scores_df["iso_score"]
    df_result["ensemble_score"] = scores_df["ensemble_score"]
    df_result["is_anomaly_model"] = anomalies.astype(int)

    # 15. Explicación heurística por transacción (columna 'explanation')
    amount_col = core_cols["amount"]
    if amount_col is not None:
        df_result["explanation"] = df_result.apply(
            lambda row: build_simple_explanation(row, amount_col),
            axis=1,
        )

    # 16. Evaluar contra etiqueta real si existe (IS_FRAUD)
    if label_col is not None:
        evaluate_with_labels(df_result, label_col, anomalies)

    # 17. Guardar salida
    _, processed_dir = get_paths(cfg)
    output_path = processed_dir / "transactions_with_scores.csv"
    df_result.to_csv(output_path, index=False)
    logger.info(f"Archivo de salida guardado en: {output_path}")
    logger.info("Pipeline AML finalizado correctamente.")


if __name__ == "__main__":
    main()