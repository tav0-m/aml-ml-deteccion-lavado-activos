import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULT_FILE = DATA_PROCESSED / "transactions_with_scores.csv"


@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)
    # Solo intentamos convertir a datetime por si hubiera fechas reales en otros datasets
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def main():
    st.set_page_config(page_title="Detección AML - Anomalías", layout="wide")
    st.title("Detección de lavado de activos con modelos no supervisados")

    if not RESULT_FILE.exists():
        st.error(f"No se encontró el archivo de resultados: {RESULT_FILE}")
        st.stop()

    df = load_data(RESULT_FILE)

    # Detectar columna de monto
    amount_col = None
    for col in df.columns:
        if amount_col is None and "amount" in col.lower():
            amount_col = col

    # Detectar columna de paso de simulación (tiempo)
    step_col = "tx_step" if "tx_step" in df.columns else None

    # Detectar columna de fecha real (por si en otro dataset la hubiera)
    time_col = None
    for col in df.columns:
        if ("time" in col.lower() or "date" in col.lower()) and np.issubdtype(df[col].dtype, np.datetime64):
            time_col = col
            break

    if "is_anomaly_model" not in df.columns:
        st.error("El archivo no contiene 'is_anomaly_model'. Revisa el pipeline.")
        st.stop()

    account_cols = [c for c in df.columns if "account" in c.lower()]

    # ---- SIDEBAR: FILTROS ----
    st.sidebar.header("Filtros")

    # Filtro por montos
    if amount_col is not None:
        min_amt = float(df[amount_col].min())
        max_amt = float(df[amount_col].max())
        amount_range = st.sidebar.slider(
            "Rango de montos",
            min_value=round(min_amt, 2),
            max_value=round(max_amt, 2),
            value=(round(min_amt, 2), round(max_amt, 2)),
        )
    else:
        amount_range = None

    # Filtro por tiempo simulado (tx_step) o por fecha real
    step_range = None
    date_range = None

    if step_col is not None:
        min_step = int(df[step_col].min())
        max_step = int(df[step_col].max())
        step_range = st.sidebar.slider(
            "Rango de pasos (tiempo simulado)",
            min_value=min_step,
            max_value=max_step,
            value=(min_step, max_step),
        )
    elif time_col is not None:
        min_date = df[time_col].min().date()
        max_date = df[time_col].max().date()
        date_range = st.sidebar.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
        )

    # Filtro por cuenta
    selected_account = None
    if account_cols:
        account_col = st.sidebar.selectbox("Columna de cuenta", options=account_cols)
        unique_accounts = df[account_col].dropna().unique()
        selected_account = st.sidebar.multiselect(
            "Cuenta(s) específicas",
            options=unique_accounts,
        )
    else:
        account_col = None

    # ---- APLICAR FILTROS ----
    filtered_df = df.copy()

    if amount_range and amount_col is not None:
        filtered_df = filtered_df[
            (filtered_df[amount_col] >= amount_range[0]) &
            (filtered_df[amount_col] <= amount_range[1])
        ]

    if step_col is not None and step_range is not None:
        filtered_df = filtered_df[
            (filtered_df[step_col] >= step_range[0]) &
            (filtered_df[step_col] <= step_range[1])
        ]
    elif time_col is not None and date_range is not None:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df[time_col].dt.date >= start_date) &
            (filtered_df[time_col].dt.date <= end_date)
        ]

    if selected_account and account_col is not None and len(selected_account) > 0:
        filtered_df = filtered_df[filtered_df[account_col].isin(selected_account)]

    # ---- KPIs ----
    total_tx = len(filtered_df)
    total_anomalies = int(filtered_df["is_anomaly_model"].sum())
    anomaly_rate = (total_anomalies / total_tx * 100) if total_tx > 0 else 0.0

    if amount_col is not None:
        total_amount = filtered_df[amount_col].sum()
        anomalous_amount = filtered_df.loc[filtered_df["is_anomaly_model"] == 1, amount_col].sum()
        anomalous_amount_share = (anomalous_amount / total_amount * 100) if total_amount > 0 else 0.0
    else:
        total_amount = anomalous_amount = anomalous_amount_share = None

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Transacciones", f"{total_tx:,}")
    kpi_cols[1].metric("Anomalías detectadas", f"{total_anomalies:,}", f"{anomaly_rate:.2f}%")

    if amount_col is not None:
        kpi_cols[2].metric("Monto total", f"{total_amount:,.2f}")
        kpi_cols[3].metric(
            "Monto en anomalías",
            f"{anomalous_amount:,.2f}",
            f"{anomalous_amount_share:.2f}%",
        )
    else:
        kpi_cols[2].write("Sin columna de monto detectada")
        kpi_cols[3].write("")

    st.markdown("---")

    # ---- DISTRIBUCIÓN DE MONTOS ----
    if amount_col is not None:
        st.subheader("Distribución de montos (normales vs anómalos)")
        fig_amount = px.histogram(
            filtered_df,
            x=amount_col,
            color="is_anomaly_model",
            nbins=50,
            barmode="overlay",
            marginal="box",
            labels={"is_anomaly_model": "Anomalía (1 = sí)"},
        )
        fig_amount.update_layout(hovermode="x unified")
        st.plotly_chart(fig_amount, use_container_width=True)

    # ---- ANOMALÍAS EN EL TIEMPO ----
    if step_col is not None:
        st.subheader("Cantidad de anomalías en el tiempo (pasos de simulación)")
        tmp = filtered_df.copy()
        grouped = tmp.groupby(step_col)["is_anomaly_model"].agg(
            anomalies="sum",
            total="count",
        ).reset_index()
        grouped["anomaly_rate"] = grouped["anomalies"] / grouped["total"] * 100

        fig_time = px.line(
            grouped,
            x=step_col,
            y="anomalies",
            labels={
                step_col: "Paso de simulación",
                "anomalies": "Número de anomalías",
            },
        )
        st.plotly_chart(fig_time, use_container_width=True)

    elif time_col is not None:
        st.subheader("Cantidad de anomalías en el tiempo")
        tmp = filtered_df.copy()
        tmp["date"] = tmp[time_col].dt.date
        grouped = tmp.groupby("date")["is_anomaly_model"].agg(
            anomalies="sum",
            total="count",
        ).reset_index()
        grouped["anomaly_rate"] = grouped["anomalies"] / grouped["total"] * 100

        fig_time = px.line(
            grouped,
            x="date",
            y="anomalies",
            labels={"anomalies": "Número de anomalías", "date": "Fecha"},
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ---- TOP CUENTAS ----
    if account_col is not None:
        st.subheader(f"Top cuentas con más anomalías ({account_col})")
        top_accounts = (
            filtered_df[filtered_df["is_anomaly_model"] == 1]
            .groupby(account_col)
            .size()
            .reset_index(name="num_anomalies")
            .sort_values("num_anomalies", ascending=False)
            .head(10)
        )
        if not top_accounts.empty:
            fig_accounts = px.bar(
                top_accounts,
                x=account_col,
                y="num_anomalies",
                labels={account_col: "Cuenta", "num_anomalies": "Anomalías"},
            )
            st.plotly_chart(fig_accounts, use_container_width=True)
        else:
            st.info("No hay anomalías en el subconjunto filtrado.")

    # ---- TABLA DETALLE ----
    st.subheader("Detalle de transacciones (top anómalas)")
    df_sorted = filtered_df.sort_values("ensemble_score", ascending=False)
    cols_to_show = df_sorted.columns.tolist()
    if "explanation" in cols_to_show:
        cols_to_show = (
            ["is_anomaly_model", "ensemble_score", "explanation"]
            + [c for c in cols_to_show if c not in ["is_anomaly_model", "ensemble_score", "explanation"]]
        )
    st.dataframe(df_sorted[cols_to_show].head(200))


if __name__ == "__main__":
    main()
