"""LSTM sequence construction that only uses past information."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences_from_block(
    df_block: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df_block[feature_cols].to_numpy(dtype=np.float32)
    y = df_block[target_col].to_numpy(dtype=np.float32)
    dates = df_block[date_col].to_numpy()

    X_seq, y_seq, d_seq = [], [], []
    for end_idx in range(seq_len - 1, len(df_block)):
        start_idx = end_idx - seq_len + 1
        X_seq.append(X[start_idx : end_idx + 1])
        y_seq.append(y[end_idx])
        d_seq.append(dates[end_idx])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32), np.asarray(d_seq)


def make_sequences_with_context(
    history_df: pd.DataFrame,
    target_block_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(target_block_df) == 0:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    context = history_df.iloc[-(seq_len - 1) :].copy() if seq_len > 1 else history_df.iloc[0:0].copy()
    combined = pd.concat([context, target_block_df], axis=0).reset_index(drop=True)
    X_all, y_all, d_all = make_sequences_from_block(combined, feature_cols, target_col, date_col, seq_len)

    target_dates = set(pd.to_datetime(target_block_df[date_col]).values)
    mask = np.array([pd.to_datetime(d) in target_dates for d in d_all])
    return X_all[mask], y_all[mask], d_all[mask]


def latest_inference_sequence(df_scaled: pd.DataFrame, feature_cols: list[str], seq_len: int) -> np.ndarray:
    if len(df_scaled) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows for inference sequence.")
    X = df_scaled[feature_cols].tail(seq_len).to_numpy(dtype=np.float32)
    return X.reshape(1, seq_len, len(feature_cols))
