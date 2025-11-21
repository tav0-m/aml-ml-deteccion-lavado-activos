from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger

from .config import PROJECT_ROOT


def detect_column(df: pd.DataFrame, explicit_name: str | None, keywords: list[str]) -> str | None:
    """
    Si explicit_name no es None y existe, se usa.
    Si no, intenta detectar columna cuyo nombre contenga alguno de keywords.
    """
    if explicit_name is not None and explicit_name in df.columns:
        logger.info(f"Usando columna explícita: {explicit_name}")
        return explicit_name

    lower_cols = {c.lower(): c for c in df.columns}
    for col_lower, col in lower_cols.items():
        if any(k in col_lower for k in keywords):
            logger.info(f"Columna detectada por keywords {keywords}: {col}")
            return col

    logger.warning(f"No se encontró columna que matchee con {keywords}")
    return None


def detect_core_columns(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Detecta columnas clave en transacciones.
    """
    col_cfg = cfg.get("columns", {})

    timestamp_col = detect_column(df, col_cfg.get("timestamp"), ["time", "date"])
    amount_col = detect_column(df, col_cfg.get("amount"), ["amount", "amt", "value"])
    sender_col = detect_column(df, col_cfg.get("sender_account"), ["sender", "src"])
    receiver_col = detect_column(df, col_cfg.get("receiver_account"), ["receiver", "dest"])
    label_col = detect_column(df, col_cfg.get("label"), ["label", "is_fraud", "fraud", "alert"])

    core_cols = {
        "timestamp": timestamp_col,
        "amount": amount_col,
        "sender_account": sender_col,
        "receiver_account": receiver_col,
        "label": label_col,
    }
    logger.info(f"Columnas clave detectadas transacciones: {core_cols}")
    return core_cols


def add_basic_time_features(df: pd.DataFrame, timestamp_col: str | None) -> pd.DataFrame:
    """
    En AMLSim, TIMESTAMP es un paso de simulación (0,1,2,...).
    Aquí lo transformamos en:
    - tx_step: paso de simulación
    - tx_hour: hora simulada (mod 24)
    - tx_dow: día de la semana simulado (mod 7)
    """
    if timestamp_col is None:
        return df

    df = df.copy()
    # Convertimos a numérico por seguridad
    df["tx_step"] = pd.to_numeric(df[timestamp_col], errors="coerce")

    # Hora y día de la semana simulados
    df["tx_hour"] = df["tx_step"] % 24
    df["tx_dow"] = df["tx_step"] % 7

    return df


def build_account_aggregates(
    df: pd.DataFrame,
    core_cols: dict,
    cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Features agregadas por cuenta origen y destino (a partir de transacciones).
    """
    amount_col = core_cols["amount"]
    sender_col = core_cols["sender_account"]
    receiver_col = core_cols["receiver_account"]

    if amount_col is None or (sender_col is None and receiver_col is None):
        logger.warning("No se pueden construir agregados de cuenta (falta amount o sender/receiver).")
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()

    sender_feats = pd.DataFrame()
    if sender_col is not None:
        grp = df.groupby(sender_col)[amount_col]
        sender_feats = grp.agg(
            sender_tx_count="count",
            sender_amount_sum="sum",
            sender_amount_mean="mean",
            sender_amount_max="max",
        ).reset_index()

    receiver_feats = pd.DataFrame()
    if receiver_col is not None:
        grp_r = df.groupby(receiver_col)[amount_col]
        receiver_feats = grp_r.agg(
            receiver_tx_count="count",
            receiver_amount_sum="sum",
            receiver_amount_mean="mean",
            receiver_amount_max="max",
        ).reset_index()

    return sender_feats, receiver_feats


def build_graph_features(df: pd.DataFrame, core_cols: dict) -> pd.DataFrame:
    """
    Grafo dirigido cuenta_origen -> cuenta_destino y grados (in/out).
    """
    sender_col = core_cols["sender_account"]
    receiver_col = core_cols["receiver_account"]

    if sender_col is None or receiver_col is None:
        logger.warning("No se pueden construir features de grafo (falta sender o receiver).")
        return pd.DataFrame()

    edges = df[[sender_col, receiver_col]].dropna()
    G = nx.from_pandas_edgelist(
        edges,
        source=sender_col,
        target=receiver_col,
        create_using=nx.DiGraph(),
    )

    accounts = list(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    graph_df = pd.DataFrame({
        "account_id": accounts,
        "graph_in_degree": [in_deg[a] for a in accounts],
        "graph_out_degree": [out_deg[a] for a in accounts],
    })
    return graph_df


def merge_all_features(
    df_tx: pd.DataFrame,
    core_cols: dict,
    sender_feats: pd.DataFrame,
    receiver_feats: pd.DataFrame,
    graph_feats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une al DataFrame de transacciones:
    - features de cuenta origen
    - features de cuenta destino
    - features de grafo (para origen y destino)
    """
    df = df_tx.copy()
    sender_col = core_cols["sender_account"]
    receiver_col = core_cols["receiver_account"]

    if sender_col is not None and not sender_feats.empty:
        df = df.merge(sender_feats, how="left", on=sender_col)

    if receiver_col is not None and not receiver_feats.empty:
        df = df.merge(receiver_feats, how="left", on=receiver_col)

    if not graph_feats.empty:
        graph_feats = graph_feats.rename(columns={"account_id": "graph_account"})
        if sender_col is not None:
            df = df.merge(
                graph_feats.add_prefix("sender_"),
                how="left",
                left_on=sender_col,
                right_on="sender_graph_account",
            ).drop(columns=["sender_graph_account"])
        if receiver_col is not None:
            df = df.merge(
                graph_feats.add_prefix("receiver_"),
                how="left",
                left_on=receiver_col,
                right_on="receiver_graph_account",
            ).drop(columns=["receiver_graph_account"])

    return df


# ---------- MERGE CON accounts.csv ----------

def merge_accounts_info(
    df_tx: pd.DataFrame,
    df_acc: pd.DataFrame,
    core_cols: dict,
    cfg: dict,
) -> pd.DataFrame:
    """
    Enlaza información estática de accounts.csv a la cuenta origen y destino.
    """
    if df_acc.empty:
        logger.warning("Accounts vacío. Se omiten features de cuenta.")
        return df_tx

    df = df_tx.copy()

    col_cfg = cfg.get("columns", {})
    account_id_cfg = col_cfg.get("account_id")

    # Detectar columna de id de cuenta en accounts.csv
    account_id_col = account_id_cfg
    if account_id_col is None or account_id_col not in df_acc.columns:
        candidates = [c for c in df_acc.columns if "account_id" in c.lower()]
        account_id_col = candidates[0] if candidates else None

    if account_id_col is None:
        logger.warning("No se detectó columna ACCOUNT_ID en accounts.csv.")
        return df

    acc = df_acc.copy()

    # Renombrar IS_FRAUD de accounts para no confundirlo con label de transacciones
    if "IS_FRAUD" in acc.columns:
        acc = acc.rename(columns={"IS_FRAUD": "ACCOUNT_IS_FRAUD"})

    sender_col = core_cols["sender_account"]
    receiver_col = core_cols["receiver_account"]

    # Merge a cuenta origen
    if sender_col is not None:
        df = df.merge(
            acc.add_prefix("sender_acc_"),
            how="left",
            left_on=sender_col,
            right_on=f"sender_acc_{account_id_col}",
        ).drop(columns=[f"sender_acc_{account_id_col}"])

    # Merge a cuenta destino
    if receiver_col is not None:
        df = df.merge(
            acc.add_prefix("receiver_acc_"),
            how="left",
            left_on=receiver_col,
            right_on=f"receiver_acc_{account_id_col}",
        ).drop(columns=[f"receiver_acc_{account_id_col}"])

    return df


# ---------- MERGE CON alerts.csv ----------

def merge_alerts_info(
    df_tx: pd.DataFrame,
    df_alerts: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Enlaza información de alerts.csv a las transacciones.
    Se usa TX_ID como llave.
    """
    if df_alerts.empty:
        logger.warning("Alerts vacío. Se omiten features de alertas.")
        return df_tx

    df = df_tx.copy()

    # Detectar TX_ID
    tx_id_col = next((c for c in df.columns if "tx_id" in c.lower()), None)
    alerts_tx_id_col = next((c for c in df_alerts.columns if "tx_id" in c.lower()), None)

    if tx_id_col is None or alerts_tx_id_col is None:
        logger.warning("No se detectó TX_ID en df_tx o alerts. No se hace merge de alertas.")
        return df

    alerts = df_alerts.copy()
    # Renombrar IS_FRAUD de alerts para no confundirlo
    if "IS_FRAUD" in alerts.columns:
        alerts = alerts.rename(columns={"IS_FRAUD": "ALERT_IS_FRAUD"})

    # Nos quedamos con TX_ID, ALERT_TYPE y ALERT_IS_FRAUD
    cols_keep = [alerts_tx_id_col]
    for c in ["ALERT_TYPE", "ALERT_IS_FRAUD"]:
        if c in alerts.columns:
            cols_keep.append(c)

    alerts_small = alerts[cols_keep].drop_duplicates(subset=[alerts_tx_id_col])

    df = df.merge(
        alerts_small,
        how="left",
        left_on=tx_id_col,
        right_on=alerts_tx_id_col,
    )

    if alerts_tx_id_col != tx_id_col:
        df = df.drop(columns=[alerts_tx_id_col])

    return df
