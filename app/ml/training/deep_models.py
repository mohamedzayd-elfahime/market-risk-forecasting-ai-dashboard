"""Deep models reused from the validated MASI notebook."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.core.config import ForecastConfig


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class ReactiveMSELoss(nn.Module):
    """Loss that forces predictions to match the variance of the target and maximizes correlation."""
    def __init__(self, corr_weight: float = 1.0, var_weight: float = 2.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.corr_weight = corr_weight
        self.var_weight = var_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(y_pred, y_true)
        
        if y_pred.shape[0] > 1:
            var_pred = torch.var(y_pred)
            var_true = torch.var(y_true)
            
            # Force the model to output predictions with the exact same variance as the target
            var_penalty = torch.abs(var_true - var_pred)
            
            # Maximize Pearson correlation to focus on shape/direction
            y_pred_c = y_pred - torch.mean(y_pred)
            y_true_c = y_true - torch.mean(y_true)
            cov = torch.mean(y_pred_c * y_true_c)
            corr = cov / (torch.sqrt(var_pred * var_true) + 1e-8)
            corr_penalty = 1.0 - corr
        else:
            var_penalty = 0.0
            corr_penalty = 0.0
            
        return mse_loss + self.var_weight * var_penalty + self.corr_weight * corr_penalty


class QuantileLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_1: int = 128,
        lstm_hidden_2: int = 64,
        dense_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_1, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_1, hidden_size=lstm_hidden_2, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden_2, dense_hidden)
        self.act = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(dense_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        last_hidden = self.dropout2(out[:, -1, :])
        out = self.fc1(last_hidden)
        out = self.act(out)
        out = self.dropout3(out)
        return self.fc_out(out)


class ReturnPatchTransformer(nn.Module):
    """Compact PatchTST-style encoder for standardized return innovations."""

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        patch_len: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = max(1, min(int(patch_len), int(seq_len)))
        self.n_patches = max(1, self.seq_len // self.patch_len)
        used_len = self.n_patches * self.patch_len
        self.start_idx = self.seq_len - used_len

        self.patch_proj = nn.Linear(input_size * self.patch_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.start_idx :, :]
        batch_size, _, n_features = x.shape
        x = x.reshape(batch_size, self.n_patches, self.patch_len * n_features)
        x = self.patch_proj(x) + self.pos_embedding
        x = self.encoder(x)
        pooled = self.norm(x.mean(dim=1))
        return self.head(pooled)


class ReturnCnnLstm(nn.Module):
    """CNN-LSTM hybrid for EGARCH-standardized innovation forecasting."""

    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 32,
        kernel_size: int = 3,
        lstm_hidden_1: int = 64,
        lstm_hidden_2: int = 32,
        dense_hidden: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = max(0, int(kernel_size) // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, cnn_filters, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )
        self.lstm1 = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden_1, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_1, hidden_size=lstm_hidden_2, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_2, dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        return self.head(x[:, -1, :])


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, list[float]]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _model_hyperparameters(config: ForecastConfig, model_role: str = "var") -> dict[str, float | int]:
    """Return role-specific LSTM/training hyperparameters."""

    if model_role == "return":
        return {
            "lstm_hidden_1": config.return_lstm_hidden_1 or config.lstm_hidden_1,
            "lstm_hidden_2": config.return_lstm_hidden_2 or config.lstm_hidden_2,
            "dense_hidden": config.return_dense_hidden or config.dense_hidden,
            "dropout": config.return_dropout if config.return_dropout is not None else config.dropout,
            "batch_size": config.return_batch_size or config.batch_size,
            "lr": config.return_lr if config.return_lr is not None else config.lr,
            "weight_decay": (
                config.return_weight_decay if config.return_weight_decay is not None else config.weight_decay
            ),
        }

    return {
        "lstm_hidden_1": config.lstm_hidden_1,
        "lstm_hidden_2": config.lstm_hidden_2,
        "dense_hidden": config.dense_hidden,
        "dropout": config.dropout,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ForecastConfig,
    loss: nn.Module,
    model_role: str = "var",
) -> TrainResult:
    device = _device()
    hyperparams = _model_hyperparameters(config, model_role=model_role)
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=int(hyperparams["batch_size"]),
        shuffle=False,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=int(hyperparams["batch_size"]),
        shuffle=False,
    )

    model = QuantileLSTM(
        input_size=X_train.shape[2],
        lstm_hidden_1=int(hyperparams["lstm_hidden_1"]),
        lstm_hidden_2=int(hyperparams["lstm_hidden_2"]),
        dense_hidden=int(hyperparams["dense_hidden"]),
        dropout=float(hyperparams["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hyperparams["lr"]),
        weight_decay=float(hyperparams["weight_decay"]),
    )
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_metric = float("inf")
    patience_counter = 0

    for _epoch in range(config.epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            batch_loss = loss(y_pred, y_batch)
            batch_loss.backward()
            optimizer.step()
            train_losses.append(batch_loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(loss(model(X_batch), y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history)


def _train_model_with_loaders(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: ForecastConfig,
    loss: nn.Module,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> TrainResult:
    device = _device()
    model = model.to(device)
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_metric = float("inf")
    patience_counter = 0

    for _epoch in range(config.epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            batch_loss = loss(model(X_batch), y_batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(batch_loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                val_losses.append(loss(model(X_batch), y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history)


def train_var_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: ForecastConfig):
    set_seed(config.seed)
    return train_lstm_model(X_train, y_train, X_val, y_val, config, PinballLoss(config.alpha), model_role="var")


def train_return_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, config: ForecastConfig):
    set_seed(config.seed)
    model_type = config.return_model_type.strip().lower()
    if model_type in {"patch_transformer", "transformer", "patchtst"}:
        hyperparams = _model_hyperparameters(config, model_role="return")
        n_heads = int(config.return_transformer_heads)
        d_model = int(config.return_transformer_d_model)
        if d_model % n_heads != 0:
            raise ValueError("return_transformer_d_model must be divisible by return_transformer_heads.")
        model = ReturnPatchTransformer(
            input_size=X_train.shape[2],
            seq_len=X_train.shape[1],
            patch_len=config.return_transformer_patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=config.return_transformer_layers,
            ff_mult=config.return_transformer_ff_mult,
            dropout=float(hyperparams["dropout"]),
        )
        return _train_model_with_loaders(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            ReactiveMSELoss(corr_weight=config.return_corr_weight, var_weight=config.return_var_weight),
            batch_size=int(hyperparams["batch_size"]),
            lr=float(hyperparams["lr"]),
            weight_decay=float(hyperparams["weight_decay"]),
        )

    if model_type in {"cnn_lstm", "cnnlstm", "egarch_cnn_lstm"}:
        hyperparams = _model_hyperparameters(config, model_role="return")
        model = ReturnCnnLstm(
            input_size=X_train.shape[2],
            cnn_filters=config.return_cnn_filters,
            kernel_size=config.return_cnn_kernel_size,
            lstm_hidden_1=int(hyperparams["lstm_hidden_1"]),
            lstm_hidden_2=int(hyperparams["lstm_hidden_2"]),
            dense_hidden=int(hyperparams["dense_hidden"]),
            dropout=float(hyperparams["dropout"]),
        )
        return _train_model_with_loaders(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            config,
            ReactiveMSELoss(corr_weight=config.return_corr_weight, var_weight=config.return_var_weight),
            batch_size=int(hyperparams["batch_size"]),
            lr=float(hyperparams["lr"]),
            weight_decay=float(hyperparams["weight_decay"]),
        )

    return train_lstm_model(
        X_train,
        y_train,
        X_val,
        y_val,
        config,
        ReactiveMSELoss(corr_weight=config.return_corr_weight, var_weight=config.return_var_weight),
        model_role="return",
    )


def predict_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = _device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X_tensor).squeeze(-1).cpu().numpy()
