
import pandas as pd

from src.feature_engineering import detect_core_columns, build_account_aggregates


def test_build_account_aggregates_basic():
    data = {
        "sender_account_id": ["A", "A", "B"],
        "receiver_account_id": ["C", "D", "C"],
        "amount": [100.0, 200.0, 300.0]
    }
    df = pd.DataFrame(data)

    cfg = {
        "columns": {
            "timestamp": None,
            "amount": "amount",
            "sender_account": "sender_account_id",
            "receiver_account": "receiver_account_id",
            "label": None
        },
        "feature_engineering": {"time_windows_days": [30]}
    }

    core_cols = detect_core_columns(df, cfg)
    sender_feats, receiver_feats = build_account_aggregates(df, core_cols, cfg)

    assert not sender_feats.empty
    assert "sender_amount_sum" in sender_feats.columns
    assert sender_feats.loc[sender_feats["sender_account_id"] == "A", "sender_tx_count"].iloc[0] == 2
