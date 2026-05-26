"""Run an Optuna chronological hyperparameter search for the MASI risk models.

The search evaluates sampled configurations on the same out-of-sample test
protocol as the production training pipeline. By default, existing production
artifacts are restored after the search finishes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.core.config import DEFAULT_CONFIG
from backend.core.paths import ARTIFACT_DIR, MASTER_DATASET_PATH, REPORT_DIR
from ml.training.train_pipeline import train_forecast_models
from ml.utils.backtesting import summarize_economic_backtest, summarize_test_predictions
from ml.utils.data_loading import dataset_version, load_master_dataset
from ml.utils.diagnostics import summarize_loss_history
from jobs.hyperparameter_config import save_best_hyperparameters


SEARCH_REPORT_DIR = REPORT_DIR / "hyperparameter_search"
MIN_ACCEPTABLE_P_VALUE = 0.05
MAX_ACCEPTABLE_VIOLATION_GAP = 0.02


def _suggest_params(trial: Any, quick: bool) -> dict[str, Any]:
    """Sample a conservative Optuna search space for the current project."""

    if quick:
        return {
            "seq_len": trial.suggest_categorical("seq_len", [30, 50]),
            "lstm_hidden_1": trial.suggest_categorical("lstm_hidden_1", [64, 128]),
            "lstm_hidden_2": trial.suggest_categorical("lstm_hidden_2", [32, 64]),
            "dense_hidden": trial.suggest_categorical("dense_hidden", [16, 32]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3, step=0.1),
            "lr": trial.suggest_categorical("lr", [5e-4, 1e-3]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
            "es_ridge_alpha": trial.suggest_categorical("es_ridge_alpha", [0.001, 0.01, 0.1]),
        }

    return {
        "seq_len": trial.suggest_categorical("seq_len", [20, 30, 50, 75]),
        "lstm_hidden_1": trial.suggest_categorical("lstm_hidden_1", [64, 96, 128, 192]),
        "lstm_hidden_2": trial.suggest_categorical("lstm_hidden_2", [32, 48, 64, 96]),
        "dense_hidden": trial.suggest_categorical("dense_hidden", [16, 32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "es_ridge_alpha": trial.suggest_float("es_ridge_alpha", 1e-4, 1.0, log=True),
    }


def _artifact_files() -> list[Path]:
    if not ARTIFACT_DIR.exists():
        return []
    return [path for path in ARTIFACT_DIR.iterdir() if path.is_file()]


def _backup_artifacts(backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in _artifact_files():
        shutil.copy2(path, backup_dir / path.name)


def _restore_artifacts(backup_dir: Path) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for path in _artifact_files():
        path.unlink()
    for path in backup_dir.glob("*"):
        if path.is_file():
            shutil.copy2(path, ARTIFACT_DIR / path.name)


def _score_row(row: dict[str, Any], alpha: float) -> float:
    """Lower is better; Kupiec and Christoffersen dominate selection.

    The first objective is to maximize the VaR validation p-values. Secondary
    terms only break ties between statistically acceptable candidates.
    """

    violation_rate = row.get("violation_rate")
    pinball = row.get("mean_pinball_loss")
    es_abs = abs(row.get("es_tail_residual_mean") or 0.0)

    violation_gap = abs(float(violation_rate) - alpha) if pd.notna(violation_rate) else 1.0
    pinball_penalty = float(pinball) if pd.notna(pinball) else 1.0

    kupiec_p = _safe_pvalue(row.get("kupiec_pof_p_value"))
    christoffersen_p = _safe_pvalue(row.get("christoffersen_cc_p_value"))
    es_p = _safe_pvalue(row.get("es_tail_calibration_p_value"))

    primary_pvalue_loss = (
        (1.0 - kupiec_p)
        + (1.0 - christoffersen_p)
        + 0.25 * abs(kupiec_p - christoffersen_p)
    )
    secondary_quality_loss = (
        0.05 * violation_gap
        + 0.5 * pinball_penalty
        + 0.05 * es_abs
        + 0.10 * (1.0 - es_p)
        + _economic_penalty(row)
    )
    return 100.0 * primary_pvalue_loss + secondary_quality_loss


def _safe_pvalue(value: Any) -> float:
    try:
        pvalue = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(pvalue):
        return 0.0
    return min(max(pvalue, 0.0), 1.0)


def _economic_penalty(row: dict[str, Any]) -> float:
    """Small tie-breaker for economically dominated risk-managed simulations."""

    penalty = 0.0
    try:
        if float(row.get("annualized_sharpe")) < float(row.get("buy_hold_sharpe")):
            penalty += 0.03
    except (TypeError, ValueError):
        pass
    try:
        if abs(float(row.get("max_drawdown"))) > abs(float(row.get("buy_hold_max_drawdown"))):
            penalty += 0.03
    except (TypeError, ValueError):
        pass
    return penalty


def _acceptance_rejection_reason(row: dict[str, Any] | None, alpha: float) -> str | None:
    """Return None only when a trial is acceptable for production retraining."""

    if row is None:
        return "no successful Optuna trial"

    checks = [
        ("Kupiec p-value", row.get("kupiec_pof_p_value"), MIN_ACCEPTABLE_P_VALUE),
        ("Christoffersen conditional coverage p-value", row.get("christoffersen_cc_p_value"), MIN_ACCEPTABLE_P_VALUE),
        ("ES calibration p-value", row.get("es_tail_calibration_p_value"), MIN_ACCEPTABLE_P_VALUE),
    ]
    failed = []
    for label, value, threshold in checks:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            failed.append(f"{label} unavailable")
            continue
        if pd.isna(numeric) or numeric < threshold:
            failed.append(f"{label}={numeric:.4g} < {threshold:.2f}")

    try:
        violation_gap = abs(float(row.get("violation_rate")) - alpha)
    except (TypeError, ValueError):
        violation_gap = float("inf")
    if pd.isna(violation_gap) or violation_gap > MAX_ACCEPTABLE_VIOLATION_GAP:
        failed.append(f"violation gap={violation_gap:.4g} > {MAX_ACCEPTABLE_VIOLATION_GAP:.2f}")

    return "; ".join(failed) if failed else None


def _evaluate_config(
    master_df: pd.DataFrame,
    data_version_value: str,
    config: ForecastConfig,
    trial_id: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    outputs = train_forecast_models(master_df, data_version_value, config)
    test_predictions = outputs["test_predictions"]

    stats = summarize_test_predictions(test_predictions, alpha=config.alpha)
    econ = summarize_economic_backtest(test_predictions)
    var_diag = summarize_loss_history(outputs["var_history"])
    return_diag = summarize_loss_history(outputs["return_history"])

    row = {
        "trial_id": trial_id,
        "status": "ok",
        **params,
        "epochs": config.epochs,
        "patience": config.patience,
        "n_observations": stats.get("n_observations"),
        "n_var_breaches": stats.get("n_var_breaches"),
        "violation_rate": stats.get("violation_rate"),
        "expected_violation_rate": stats.get("expected_violation_rate"),
        "kupiec_pof_p_value": stats.get("kupiec_pof_p_value"),
        "christoffersen_cc_p_value": stats.get("christoffersen_cc_p_value"),
        "n_es_tail_observations": stats.get("n_es_tail_observations"),
        "es_tail_calibration_p_value": stats.get("es_tail_calibration_p_value"),
        "es_tail_residual_mean": stats.get("es_tail_residual_mean"),
        "mean_pinball_loss": stats.get("mean_pinball_loss"),
        "rmse_return_prediction": stats.get("rmse_return_prediction"),
        "final_wealth": econ.get("final_wealth"),
        "buy_hold_final_wealth": econ.get("buy_hold_final_wealth"),
        "max_drawdown": econ.get("max_drawdown"),
        "buy_hold_max_drawdown": econ.get("buy_hold_max_drawdown"),
        "annualized_sharpe": econ.get("annualized_sharpe"),
        "buy_hold_sharpe": econ.get("buy_hold_sharpe"),
        "var_best_val_loss": var_diag.get("best_val_loss"),
        "var_epochs_ran": var_diag.get("epochs_ran"),
        "return_best_val_loss": return_diag.get("best_val_loss"),
        "return_epochs_ran": return_diag.get("epochs_ran"),
    }
    kupiec_p = _safe_pvalue(row.get("kupiec_pof_p_value"))
    christoffersen_p = _safe_pvalue(row.get("christoffersen_cc_p_value"))
    row["validation_pvalue_mean"] = (kupiec_p + christoffersen_p) / 2.0
    row["validation_pvalue_min"] = min(kupiec_p, christoffersen_p)
    row["selection_score"] = _score_row(row, config.alpha)
    return row


def run_search(
    max_trials: int,
    epochs: int,
    patience: int,
    quick: bool,
    keep_best: bool,
    study_name: str | None = None,
    storage: str | None = None,
) -> pd.DataFrame:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna n'est pas installe. Lance `pip install optuna` ou reinstalle les dependances du projet."
        ) from exc

    SEARCH_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    master_df = load_master_dataset(MASTER_DATASET_PATH)
    data_version_value = dataset_version(MASTER_DATASET_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_study_name = study_name or f"masi_hyperparameter_search_{timestamp}"
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_params: dict[str, Any] | None = None

    with TemporaryDirectory(prefix="masi_artifact_backup_") as tmp:
        backup_dir = Path(tmp)
        _backup_artifacts(backup_dir)

        try:
            sampler = optuna.samplers.TPESampler(seed=DEFAULT_CONFIG.seed)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=3)
            study = optuna.create_study(
                study_name=resolved_study_name,
                storage=storage,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=bool(storage),
            )

            def objective(trial: Any) -> float:
                nonlocal best_row

                params = _suggest_params(trial, quick=quick)
                print(f"\nTrial {trial.number + 1}/{max_trials}: {params}")
                config = replace(
                    DEFAULT_CONFIG,
                    **params,
                    epochs=epochs,
                    patience=patience,
                )
                try:
                    row = _evaluate_config(
                        master_df,
                        data_version_value,
                        config,
                        trial.number + 1,
                        params,
                    )
                    print(
                        "  ok | violation_rate="
                        f"{row['violation_rate']:.4f} | kupiec_p={row['kupiec_pof_p_value']:.4f} "
                        f"| christoffersen_p={row['christoffersen_cc_p_value']:.4f} "
                        f"| score={row['selection_score']:.6f}"
                    )
                    if best_row is None or row["selection_score"] < best_row["selection_score"]:
                        best_row = row
                    trial.set_user_attr("metrics", row)
                    rows.append(row)
                    trial.report(row["selection_score"], step=0)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    return float(row["selection_score"])
                except optuna.TrialPruned:
                    raise
                except Exception as exc:
                    row = {
                        "trial_id": trial.number + 1,
                        "status": "failed",
                        **params,
                        "selection_score": float("nan"),
                        "error": str(exc),
                    }
                    print(f"  failed: {exc}")
                    rows.append(row)
                    trial.set_user_attr("metrics", row)
                    return float("inf")

            study.optimize(objective, n_trials=max_trials, gc_after_trial=True)
            best_params = dict(study.best_trial.params) if best_row is not None else None
        finally:
            if keep_best and best_params is not None:
                print("\nRetraining and keeping best configuration as production artifacts...")
                best_config = replace(
                    DEFAULT_CONFIG,
                    **best_params,
                    epochs=DEFAULT_CONFIG.epochs,
                    patience=DEFAULT_CONFIG.patience,
                )
                train_forecast_models(master_df, data_version_value, best_config)
            else:
                print("\nRestoring previous production artifacts...")
                _restore_artifacts(backup_dir)

    results = pd.DataFrame(rows)
    results_path = SEARCH_REPORT_DIR / f"hyperparameter_search_{timestamp}.csv"
    summary_path = SEARCH_REPORT_DIR / f"hyperparameter_search_{timestamp}.json"

    sort_columns = [col for col in ("status", "selection_score") if col in results.columns]
    if sort_columns:
        results = results.sort_values(sort_columns, na_position="last")
    results.to_csv(results_path, index=False)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "master_dataset": str(MASTER_DATASET_PATH),
        "data_version": data_version_value,
        "max_trials": max_trials,
        "epochs_per_trial": epochs,
        "patience_per_trial": patience,
        "optimizer": "optuna",
        "selection_objective": (
            "maximize Kupiec POF and Christoffersen conditional coverage p-values first; "
            "use violation gap, pinball loss, ES calibration and economic metrics only as tie-breakers"
        ),
        "study_name": resolved_study_name,
        "storage": storage,
        "keep_best": keep_best,
        "best_trial": best_row,
        "best_params": best_params,
        "results_csv": str(results_path),
    }
    rejection_reason = _acceptance_rejection_reason(best_row, DEFAULT_CONFIG.alpha)
    payload["best_trial_accepted_for_production"] = rejection_reason is None
    payload["best_trial_rejection_reason"] = rejection_reason
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    best_params_path = None
    if best_params is not None:
        best_params_path = save_best_hyperparameters(
            best_params,
            summary_path,
            best_row,
            accepted=rejection_reason is None,
            rejection_reason=rejection_reason,
        )
        payload["best_params_path"] = str(best_params_path)
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\nHyperparameter search completed.")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")
    if best_params_path is not None:
        print(f"Best hyperparameters: {best_params_path}")
    if best_row is not None:
        print("\nBest trial:")
        print(pd.DataFrame([best_row]).to_string(index=False))
        if rejection_reason is not None:
            print("\nBest trial NOT accepted for production:")
            print(rejection_reason)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MASI LSTM hyperparameter search with Optuna.")
    parser.add_argument("--max-trials", type=int, default=20, help="Maximum number of Optuna trials to evaluate.")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs per trial during search.")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience per trial.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller Optuna search space for smoke tests.")
    parser.add_argument("--study-name", default=None, help="Optional Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///app/data/reports/hyperparameter_search/optuna.db.",
    )
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Retrain and keep the best configuration as production artifacts. Default restores current artifacts.",
    )
    args = parser.parse_args()

    run_search(
        max_trials=args.max_trials,
        epochs=args.epochs,
        patience=args.patience,
        quick=args.quick,
        keep_best=args.keep_best,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
