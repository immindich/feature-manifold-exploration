"""Shared plotting utilities for true vs predicted count visualizations."""

import numpy as np
from metrics import compute_prediction_metrics


def scatter_true_vs_predicted(ax, true, pred, title=None):
    """Draw a scatter plot of true vs predicted values with a y=x reference line.

    Annotates MAE and R² in the title. Assumes ax is a matplotlib Axes.

    Args:
        ax: Matplotlib Axes to draw on.
        true: Array-like of true values.
        pred: Array-like of predicted values.
        title: Optional title prefix; metrics are appended.
    """
    true, pred = np.asarray(true, dtype=float), np.asarray(pred, dtype=float)
    metrics = compute_prediction_metrics(true, pred)

    ax.scatter(true, pred, s=8, alpha=0.3)

    lo, hi = true.min(), true.max()
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, linestyle="--", label="y=x")

    label = title or ""
    ax.set_title(f"{label} (MAE={metrics['mae']:.1f}, R²={metrics['r2']:.3f})")
    ax.set_xlabel("Actual count")
    ax.set_ylabel("Predicted count")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.set_xlim(lo - 5, hi + 5)
    ax.set_ylim(lo - 5, hi + 5)
