"""Training diagnostics for model convergence checks."""

from __future__ import annotations

import math


def summarize_loss_history(history: dict[str, list[float]], min_relative_improvement: float = 0.01) -> dict[str, float | int | bool | str]:
    train_loss = [float(x) for x in history.get("train_loss", [])]
    val_loss = [float(x) for x in history.get("val_loss", [])]

    if not train_loss or not val_loss:
        return {"converged": False, "reason": "empty_loss_history"}

    finite = all(math.isfinite(x) for x in train_loss + val_loss)
    first_val = val_loss[0]
    best_val = min(val_loss)
    best_epoch = val_loss.index(best_val) + 1
    final_val = val_loss[-1]
    final_train = train_loss[-1]
    improvement = (first_val - best_val) / abs(first_val) if first_val != 0 else 0.0
    generalization_gap = final_val - final_train

    converged = bool(
        finite
        and improvement >= min_relative_improvement
        and best_epoch < len(val_loss)
        and final_val <= first_val
    )
    if not finite:
        reason = "non_finite_loss"
    elif improvement < min_relative_improvement:
        reason = "weak_validation_improvement"
    elif best_epoch >= len(val_loss):
        reason = "best_epoch_is_last_epoch"
    elif final_val > first_val:
        reason = "validation_loss_finished_above_initial"
    else:
        reason = "ok"

    return {
        "converged": converged,
        "reason": reason,
        "epochs_ran": len(val_loss),
        "best_epoch": best_epoch,
        "first_train_loss": train_loss[0],
        "final_train_loss": final_train,
        "first_val_loss": first_val,
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "relative_val_improvement": improvement,
        "final_generalization_gap": generalization_gap,
    }
