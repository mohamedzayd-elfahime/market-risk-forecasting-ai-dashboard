"""Reusable functions extracted from the research notebooks.

The notebooks keep the narrative and result cells; this module keeps shared code.
"""

import random

from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

from scipy.stats import chi2, ttest_1samp

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.linear_model import Ridge

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    date_col: str = "date"
    dataset_col: Optional[str] = "dataset"
    dataset_value: str = "MASI"

    return_col: str = "log_return"
    target_col: str = "target"

    feature_cols: Tuple[str, ...] = (
        "log_return",
        "log_return_std_5",
        "log_return_std_21",
        "log_return_mean_21",
    )

    rolling_std_5: int = 5
    rolling_std_21: int = 21
    rolling_mean_21: int = 21

    alpha: float = 0.05

    dev_size: int = 3998
    test_size: int = 764
    val_fraction_within_dev: float = 0.15

    seq_len: int = 20

    # VaR LSTM
    lstm_hidden_1: int = 128
    lstm_hidden_2: int = 64
    dense_hidden: int = 32
    dropout: float = 0.20

    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 10

    # ES Ridge
    es_min_violations: int = 20
    es_shortfall_floor: float = 1e-6
    es_ridge_alpha: float = 1.0

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def build_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    data = df.copy()
    data[config.date_col] = pd.to_datetime(data[config.date_col])
    data = data.sort_values(config.date_col).reset_index(drop=True)

    if config.dataset_col is not None and config.dataset_col in data.columns:
        data = data[data[config.dataset_col] == config.dataset_value].copy()

    out = pd.DataFrame()
    out[config.date_col] = data[config.date_col]
    out["log_return"] = data[config.return_col]
    out["log_return_std_5"] = data[config.return_col].rolling(config.rolling_std_5).std()
    out["log_return_std_21"] = data[config.return_col].rolling(config.rolling_std_21).std()
    out["log_return_mean_21"] = data[config.return_col].rolling(config.rolling_mean_21).mean()

    out[config.target_col] = data[config.return_col].shift(-1)

    out = out.dropna().reset_index(drop=True)
    return out

def imposed_dev_test_split(
    df_model: pd.DataFrame,
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = config.dev_size + config.test_size
    if len(df_model) < required:
        raise ValueError(
            f"Pas assez de donnees valides : {len(df_model)} disponibles, "
            f"mais {required} requises."
        )

    selected = df_model.iloc[-required:].copy().reset_index(drop=True)
    dev_df = selected.iloc[:config.dev_size].copy().reset_index(drop=True)
    test_df = selected.iloc[config.dev_size:].copy().reset_index(drop=True)

    print("===== Controle split global =====")
    print(f"total selected observations : {len(selected)}")
    print(f"dev observations            : {len(dev_df)}")
    print(f"test observations           : {len(test_df)}")
    print(f"first dev date              : {dev_df[config.date_col].iloc[0]}")
    print(f"last dev date               : {dev_df[config.date_col].iloc[-1]}")
    print(f"first test date             : {test_df[config.date_col].iloc[0]}")
    print(f"last test date              : {test_df[config.date_col].iloc[-1]}")
    print("================================")

    return dev_df, test_df

def split_train_val_within_dev(
    dev_df: pd.DataFrame,
    config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_dev = len(dev_df)
    n_val = max(int(round(n_dev * config.val_fraction_within_dev)), config.seq_len + 5)

    if n_val >= n_dev:
        raise ValueError("Validation trop grande par rapport au bloc developpement.")

    train_df = dev_df.iloc[: n_dev - n_val].copy().reset_index(drop=True)
    val_df = dev_df.iloc[n_dev - n_val :].copy().reset_index(drop=True)

    print("===== Controle split train/val dans dev =====")
    print(f"train observations         : {len(train_df)}")
    print(f"validation observations    : {len(val_df)}")
    print(f"first train date           : {train_df[config.date_col].iloc[0]}")
    print(f"last train date            : {train_df[config.date_col].iloc[-1]}")
    print(f"first validation date      : {val_df[config.date_col].iloc[0]}")
    print(f"last validation date       : {val_df[config.date_col].iloc[-1]}")
    print("=============================================")

    return train_df, val_df

def fit_feature_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def transform_features(
    df_part: pd.DataFrame,
    scaler,
    feature_cols: List[str]
) -> pd.DataFrame:
    out = df_part.copy()
    if len(out) == 0:
        return out
    out.loc[:, feature_cols] = scaler.transform(out[feature_cols])
    return out

def make_sequences_from_block(
    df_block: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: str,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df_block[feature_cols].to_numpy(dtype=np.float32)
    y = df_block[target_col].to_numpy(dtype=np.float32)
    dates = df_block[date_col].to_numpy()

    X_seq, y_seq, d_seq = [], [], []

    for end_idx in range(seq_len - 1, len(df_block)):
        start_idx = end_idx - seq_len + 1
        X_seq.append(X[start_idx:end_idx + 1])
        y_seq.append(y[end_idx])
        d_seq.append(dates[end_idx])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(d_seq),
    )

def make_sequences_with_context(
    history_df: pd.DataFrame,
    target_block_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    date_col: str,
    seq_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(target_block_df) == 0:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    if seq_len == 1:
        combined = target_block_df.copy().reset_index(drop=True)
    else:
        context = history_df.iloc[-(seq_len - 1):].copy()
        combined = pd.concat([context, target_block_df], axis=0).reset_index(drop=True)

    X_all, y_all, d_all = make_sequences_from_block(
        combined, feature_cols, target_col, date_col, seq_len
    )

    target_dates = set(pd.to_datetime(target_block_df[date_col]).values)
    mask = np.array([pd.to_datetime(d) in target_dates for d in d_all])

    return X_all[mask], y_all[mask], d_all[mask]

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class PinballLoss(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        err = y_true - y_pred
        return torch.maximum(self.alpha * err, (self.alpha - 1.0) * err).mean()

class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_1: int = 128,
        lstm_hidden_2: int = 64,
        dense_hidden: int = 32,
        dropout: float = 0.20
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_1,
            num_layers=1,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_1,
            hidden_size=lstm_hidden_2,
            num_layers=1,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(lstm_hidden_2, dense_hidden)
        self.act = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout2(last_hidden)

        out = self.fc1(last_hidden)
        out = self.act(out)
        out = self.dropout3(out)
        out = self.fc_out(out)
        return out

def train_var_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    train_ds = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    val_ds = SequenceDataset(X_val, y_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = QuantileLSTM(
        input_size=X_train.shape[2],
        lstm_hidden_1=config.lstm_hidden_1,
        lstm_hidden_2=config.lstm_hidden_2,
        dense_hidden=config.dense_hidden,
        dropout=config.dropout,
    ).to(config.device)

    criterion = PinballLoss(config.alpha)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = {"train_loss": [], "val_loss": []}
    best_metric = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses))
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def predict_var(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_tensor).squeeze(-1).cpu().numpy()
    return preds

def extract_sequence_aligned_rows_no_context(
    df_block: pd.DataFrame,
    seq_len: int
) -> pd.DataFrame:
    return df_block.iloc[seq_len - 1:].copy().reset_index(drop=True)

def extract_sequence_aligned_rows_with_context(
    df_block: pd.DataFrame
) -> pd.DataFrame:
    return df_block.copy().reset_index(drop=True)

def build_es_training_frame(
    seq_raw_df: pd.DataFrame,
    y_true: np.ndarray,
    var_pred: np.ndarray
) -> pd.DataFrame:
    out = seq_raw_df.copy().reset_index(drop=True)
    out["y_true"] = y_true
    out["var_pred"] = var_pred
    out["abs_var_pred"] = np.abs(var_pred)
    out["violation"] = (out["y_true"] < out["var_pred"]).astype(int)
    out["shortfall"] = np.maximum(out["var_pred"] - out["y_true"], 0.0)
    return out

def fit_es_ridge_model(
    es_train_df: pd.DataFrame,
    ridge_alpha: float = 2.0,
    shortfall_floor: float = 1e-6,
    min_violations: int = 20,
) -> Dict[str, object]:
    viol_df = es_train_df[es_train_df["violation"] == 1].copy()

    if len(viol_df) < min_violations:
        raise ValueError(
            f"Violations insuffisantes pour le modele ES Ridge : {len(viol_df)}."
        )

    es_feature_names = [
        "abs_var_pred",
        "log_return",
        "log_return_std_5",
        "log_return_std_21",
        "log_return_mean_21",
    ]

    X = viol_df[es_feature_names].copy()
    y_shortfall = viol_df["shortfall"].clip(lower=shortfall_floor).to_numpy()
    y_log = np.log1p(y_shortfall)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=ridge_alpha)
    model.fit(X_scaled, y_log)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": es_feature_names,
        "shortfall_floor": shortfall_floor,
        "n_violations_used": len(viol_df),
    }

def predict_es_ridge(
    es_model_bundle: Dict[str, object],
    seq_raw_df: pd.DataFrame,
    var_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    X = pd.DataFrame({
        "abs_var_pred": np.abs(var_pred),
        "log_return": seq_raw_df["log_return"].to_numpy(),
        "log_return_std_5": seq_raw_df["log_return_std_5"].to_numpy(),
        "log_return_std_21": seq_raw_df["log_return_std_21"].to_numpy(),
        "log_return_mean_21": seq_raw_df["log_return_mean_21"].to_numpy(),
    })

    X_scaled = es_model_bundle["scaler"].transform(X)
    pred_log_shortfall = es_model_bundle["model"].predict(X_scaled)

    predicted_shortfall = np.expm1(pred_log_shortfall)
    predicted_shortfall = np.maximum(predicted_shortfall, es_model_bundle["shortfall_floor"])

    es_pred = var_pred - predicted_shortfall
    es_pred = np.minimum(es_pred, var_pred)

    return es_pred, predicted_shortfall

def run_two_stage_var_es_pipeline(
    df: pd.DataFrame,
    config: Config
) -> Dict[str, object]:
    set_seed(config.seed)

    feature_cols = list(config.feature_cols)

    df_model = build_features(df, config)

    dev_raw, test_raw = imposed_dev_test_split(df_model, config)
    train_raw, val_raw = split_train_val_within_dev(dev_raw, config)

    feature_scaler = fit_feature_scaler(train_raw, feature_cols)

    train_scaled = transform_features(train_raw, feature_scaler, feature_cols)
    val_scaled = transform_features(val_raw, feature_scaler, feature_cols)
    dev_scaled = transform_features(dev_raw, feature_scaler, feature_cols)
    test_scaled = transform_features(test_raw, feature_scaler, feature_cols)

    X_train, y_train, d_train = make_sequences_from_block(
        train_scaled, feature_cols, config.target_col, config.date_col, config.seq_len
    )

    X_val, y_val, d_val = make_sequences_with_context(
        history_df=train_scaled,
        target_block_df=val_scaled,
        feature_cols=feature_cols,
        target_col=config.target_col,
        date_col=config.date_col,
        seq_len=config.seq_len
    )

    var_model, var_history = train_var_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config
    )

    X_dev, y_dev, d_dev = make_sequences_from_block(
        dev_scaled, feature_cols, config.target_col, config.date_col, config.seq_len
    )

    X_test, y_test, d_test = make_sequences_with_context(
        history_df=dev_scaled,
        target_block_df=test_scaled,
        feature_cols=feature_cols,
        target_col=config.target_col,
        date_col=config.date_col,
        seq_len=config.seq_len
    )

    var_dev_pred = predict_var(var_model, X_dev, config.device)
    var_test_pred = predict_var(var_model, X_test, config.device)

    dev_seq_raw = extract_sequence_aligned_rows_no_context(dev_raw, config.seq_len)
    test_seq_raw = extract_sequence_aligned_rows_with_context(test_raw)

    if not (len(dev_seq_raw) == len(y_dev) == len(var_dev_pred)):
        raise ValueError("Mauvais alignement sur le bloc dev pour ES.")

    if not (len(test_seq_raw) == len(y_test) == len(var_test_pred)):
        raise ValueError("Mauvais alignement sur le bloc test.")

    es_dev_df = build_es_training_frame(
        seq_raw_df=dev_seq_raw,
        y_true=y_dev,
        var_pred=var_dev_pred
    )

    es_ridge_bundle = fit_es_ridge_model(
        es_train_df=es_dev_df,
        ridge_alpha=config.es_ridge_alpha,
        shortfall_floor=config.es_shortfall_floor,
        min_violations=config.es_min_violations
    )
    es_test_pred_ridge, shortfall_test_pred_ridge = predict_es_ridge(
        es_model_bundle=es_ridge_bundle,
        seq_raw_df=test_seq_raw,
        var_pred=var_test_pred
    )

    test_predictions = pd.DataFrame({
        "date": pd.to_datetime(d_test),
        "realized_return": y_test,
        "var_pred": var_test_pred,
        "es_pred_ridge": es_test_pred_ridge,
        "predicted_shortfall_ridge": shortfall_test_pred_ridge,
        "violation": (y_test < var_test_pred).astype(int),
    })

    return {
        "df_model": df_model,
        "dev_raw": dev_raw,
        "test_raw": test_raw,
        "train_raw": train_raw,
        "val_raw": val_raw,
        "feature_scaler": feature_scaler,
        "var_model": var_model,
        "var_history": var_history,
        "var_dev_pred": var_dev_pred,
        "var_test_pred": var_test_pred,
        "es_ridge_model": es_ridge_bundle,
        "test_predictions": test_predictions,
    }

ALPHA = 0.05

EPS = 1e-12

def clip_probability(p, eps=EPS):
    return np.clip(p, eps, 1.0 - eps)

def safe_loglik_bernoulli(successes, trials, prob):
    prob = clip_probability(prob)
    return successes * np.log(prob) + (trials - successes) * np.log(1.0 - prob)

def validate_inputs(realized_returns, var_forecasts, es_forecasts=None):
    data = pd.DataFrame({
        "realized_return": pd.Series(realized_returns),
        "var_forecast": pd.Series(var_forecasts),
    })

    if es_forecasts is not None:
        data["es_forecast"] = pd.Series(es_forecasts)

    data = data.dropna().copy()

    if data.empty:
        raise ValueError("Aucune observation valide apres suppression des NaN.")

    if es_forecasts is not None:
        invalid_es = (data["es_forecast"] > data["var_forecast"]).sum()
        if invalid_es > 0:
            raise ValueError(
                f"Incoherence detectee : {invalid_es} observations ont ES > VaR. "
                "Sur l'echelle des rendements, on attend ES <= VaR."
            )

    return data

def kupiec_pof_test(violations, alpha=ALPHA):
    hits = np.asarray(violations, dtype=int)
    n = len(hits)

    if n == 0:
        return np.nan, np.nan

    x = int(hits.sum())
    p_hat = x / n

    loglik_null = safe_loglik_bernoulli(x, n, alpha)
    loglik_alt = safe_loglik_bernoulli(x, n, p_hat)

    lr_pof = -2.0 * (loglik_null - loglik_alt)
    p_value = 1.0 - chi2.cdf(lr_pof, df=1)

    return lr_pof, p_value

def christoffersen_independence_test(violations):
    hits = np.asarray(violations, dtype=int)
    n = len(hits)

    if n < 2:
        return np.nan, np.nan

    prev_hits = hits[:-1]
    curr_hits = hits[1:]

    n00 = int(((prev_hits == 0) & (curr_hits == 0)).sum())
    n01 = int(((prev_hits == 0) & (curr_hits == 1)).sum())
    n10 = int(((prev_hits == 1) & (curr_hits == 0)).sum())
    n11 = int(((prev_hits == 1) & (curr_hits == 1)).sum())

    total_0 = n00 + n01
    total_1 = n10 + n11
    total_all = n00 + n01 + n10 + n11

    if total_all == 0 or total_0 == 0 or total_1 == 0:
        return np.nan, np.nan

    pi0 = n01 / total_0
    pi1 = n11 / total_1
    pi = (n01 + n11) / total_all

    loglik_null = (
        n00 * np.log(clip_probability(1.0 - pi))
        + n01 * np.log(clip_probability(pi))
        + n10 * np.log(clip_probability(1.0 - pi))
        + n11 * np.log(clip_probability(pi))
    )

    loglik_alt = (
        n00 * np.log(clip_probability(1.0 - pi0))
        + n01 * np.log(clip_probability(pi0))
        + n10 * np.log(clip_probability(1.0 - pi1))
        + n11 * np.log(clip_probability(pi1))
    )

    lr_ind = -2.0 * (loglik_null - loglik_alt)
    p_value = 1.0 - chi2.cdf(lr_ind, df=1)

    return lr_ind, p_value

def christoffersen_conditional_coverage_test(violations, alpha=ALPHA):
    lr_pof, _ = kupiec_pof_test(violations, alpha=alpha)
    lr_ind, _ = christoffersen_independence_test(violations)

    if np.isnan(lr_pof) or np.isnan(lr_ind):
        return np.nan, np.nan

    lr_cc = lr_pof + lr_ind
    p_value = 1.0 - chi2.cdf(lr_cc, df=2)

    return lr_cc, p_value

def es_tail_calibration_test(realized_returns, var_forecasts, es_forecasts):
    rr = np.asarray(realized_returns, dtype=float)
    vf = np.asarray(var_forecasts, dtype=float)
    ef = np.asarray(es_forecasts, dtype=float)

    tail_mask = rr < vf
    tail_realized = rr[tail_mask]
    tail_es = ef[tail_mask]

    n_tail = int(tail_mask.sum())

    if n_tail < 2:
        return n_tail, np.nan, np.nan, np.nan

    tail_residuals = tail_realized - tail_es

    test_stat, p_value = ttest_1samp(tail_residuals, popmean=0.0, nan_policy="omit")
    tail_mean = float(np.nanmean(tail_residuals))

    return n_tail, test_stat, p_value, tail_mean

def backtest_var_es_model(model_data, alpha=ALPHA):
    required_columns = ["realized_return", "var_5pct", "es_5pct"]
    missing_cols = [c for c in required_columns if c not in model_data.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    clean = validate_inputs(
        realized_returns=model_data["realized_return"],
        var_forecasts=model_data["var_5pct"],
        es_forecasts=model_data["es_5pct"],
    )

    realized_returns = clean["realized_return"]
    var_forecasts = clean["var_forecast"]
    es_forecasts = clean["es_forecast"]

    violations = (realized_returns < var_forecasts).astype(int)

    n_obs = int(len(violations))
    n_var_violations = int(violations.sum())
    violation_rate = n_var_violations / n_obs if n_obs > 0 else np.nan

    kupiec_stat, kupiec_p_value = kupiec_pof_test(violations, alpha=alpha)
    ind_stat, ind_p_value = christoffersen_independence_test(violations)
    cc_stat, cc_p_value = christoffersen_conditional_coverage_test(violations, alpha=alpha)

    n_es_tail, es_stat, es_p_value, es_tail_mean = es_tail_calibration_test(
        realized_returns=realized_returns,
        var_forecasts=var_forecasts,
        es_forecasts=es_forecasts,
    )

    return {
        "n_observations": n_obs,
        "n_var_violations": n_var_violations,
        "violation_rate": violation_rate,
        "expected_violation_rate": alpha,
        "kupiec_pof_stat": kupiec_stat,
        "kupiec_pof_p_value": kupiec_p_value,
        "christoffersen_independence_stat": ind_stat,
        "christoffersen_independence_p_value": ind_p_value,
        "christoffersen_cc_stat": cc_stat,
        "christoffersen_cc_p_value": cc_p_value,
        "n_es_tail_observations": n_es_tail,
        "es_tail_calibration_stat": es_stat,
        "es_tail_calibration_p_value": es_p_value,
        "es_tail_residual_mean": es_tail_mean,
    }

def backtest_forecast_results(forecast_results, alpha=ALPHA):
    required_columns = ["model_family", "realized_return", "var_5pct", "es_5pct"]
    missing_cols = [c for c in required_columns if c not in forecast_results.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans forecast_results : {missing_cols}")

    backtest_rows = []

    for model_family, model_data in forecast_results.groupby("model_family"):
        summary = backtest_var_es_model(model_data=model_data, alpha=alpha)
        summary["model_family"] = model_family
        backtest_rows.append(summary)

    result = pd.DataFrame(backtest_rows)

    ordered_cols = [
        "model_family",
        "n_observations",
        "n_var_violations",
        "violation_rate",
        "expected_violation_rate",
        "kupiec_pof_stat",
        "kupiec_pof_p_value",
        "christoffersen_independence_stat",
        "christoffersen_independence_p_value",
        "christoffersen_cc_stat",
        "christoffersen_cc_p_value",
        "n_es_tail_observations",
        "es_tail_calibration_stat",
        "es_tail_calibration_p_value",
        "es_tail_residual_mean",
    ]

    return result[ordered_cols].sort_values("model_family").reset_index(drop=True)

def build_forecast_results_from_pipeline(results: Dict[str, object]) -> pd.DataFrame:
    test_preds = results["test_predictions"].copy()

    forecast_results = pd.DataFrame({
            "model_family": "deep_lstm_es_ridge",
            "realized_return": test_preds["realized_return"],
            "var_5pct": test_preds["var_pred"],
            "es_5pct": test_preds["es_pred_ridge"],
    })

    return forecast_results
