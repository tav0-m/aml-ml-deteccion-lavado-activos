
from pathlib import Path
import pandas as pd
from loguru import logger

from .config import PROJECT_ROOT, load_config


def get_paths(cfg: dict):
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    processed_dir = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir


def load_transactions(cfg: dict) -> pd.DataFrame:
    raw_dir, _ = get_paths(cfg)
    tx_file = raw_dir / cfg["paths"]["transactions_file"]
    if not tx_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de transacciones: {tx_file}")
    df = pd.read_csv(tx_file)
    logger.info(f"Transacciones cargadas: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def load_accounts(cfg: dict) -> pd.DataFrame:
    raw_dir, _ = get_paths(cfg)
    acc_file = raw_dir / cfg["paths"].get("accounts_file", "accounts.csv")
    if not acc_file.exists():
        logger.warning(f"No se encontró accounts.csv en {acc_file}. Se continúa sin features de cuenta.")
        return pd.DataFrame()
    df = pd.read_csv(acc_file)
    logger.info(f"Accounts cargadas: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def load_alerts(cfg: dict) -> pd.DataFrame:
    raw_dir, _ = get_paths(cfg)
    alerts_file = raw_dir / cfg["paths"].get("alerts_file", "alerts.csv")
    if not alerts_file.exists():
        logger.warning(f"No se encontró alerts.csv en {alerts_file}. Se continúa sin info de alertas.")
        return pd.DataFrame()
    df = pd.read_csv(alerts_file)
    logger.info(f"Alerts cargadas: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df
