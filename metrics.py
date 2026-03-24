"""Shared metrics for comparing true vs predicted counts."""

import numpy as np


def compute_prediction_metrics(true, pred) -> dict:
    """Compute prediction quality metrics.

    Args:
        true: Array-like of true values.
        pred: Array-like of predicted values.

    Returns:
        Dict with keys: corr, mae, r2, mse, n
    """
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    n = len(true)
    if n < 2:
        return {"corr": 0.0, "mae": float("inf"), "r2": 0.0, "mse": float("inf"), "n": n}

    mae = np.mean(np.abs(true - pred))
    mse = np.mean((true - pred) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    corr = np.corrcoef(true, pred)[0, 1]

    return {"corr": corr, "mae": mae, "r2": r2, "mse": mse, "n": n}
