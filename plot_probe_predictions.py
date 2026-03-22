#!/usr/bin/env python3
"""Load trained probe weights and plot predicted vs actual count on the test set."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_probes import LinearProbe, MLPProbe, load_data, split_train_test

DATA_PATH = "activations-27b.pt"
WEIGHTS_PATH = "probe-mlp-27b.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load data and split (same seed as training)
activations, metadata, layers = load_data(DATA_PATH)
_, _, test_acts, test_counts = split_train_test(activations, metadata)

# Load saved probe weights
saved = torch.load(WEIGHTS_PATH, weights_only=False)
hidden_dim = saved["hidden_dim"]
probe_type = saved["probe_type"]
mlp_hidden = saved["mlp_hidden"]

# Get predictions per layer
predictions = {}
for layer in saved["layers"]:
    if probe_type == "linear":
        probe = LinearProbe(hidden_dim)
    else:
        probe = MLPProbe(hidden_dim, mlp_hidden)
    probe.load_state_dict(saved["probes"][layer])
    probe.eval().to(DEVICE)

    li = saved["layers"].index(layer)
    test_x = test_acts[:, li, :].to(DEVICE)
    with torch.no_grad():
        pred = probe(test_x).cpu().numpy()
    predictions[layer] = pred

# Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (layer, pred) in enumerate(predictions.items()):
    ax = axes[i]
    ax.scatter(test_counts, pred, s=8, alpha=0.3)
    lo, hi = test_counts.min(), test_counts.max()
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, linestyle="--", label="y=x")
    ax.set_xlabel("Actual count")
    ax.set_ylabel("Predicted count")
    mae = np.abs(pred - test_counts).mean()
    r2 = 1 - np.sum((pred - test_counts) ** 2) / np.sum((test_counts - test_counts.mean()) ** 2)
    ax.set_title(f"Layer {layer} (MAE={mae:.1f}, R²={r2:.3f})")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.set_xlim(lo - 5, hi + 5)
    ax.set_ylim(lo - 5, hi + 5)

fig.suptitle(f"{probe_type.upper()} Probe: Predicted vs Actual Count (Test Set)", fontsize=14)
plt.tight_layout()
plt.savefig("probe_mlp_predictions_27b.png", dpi=150)
plt.show()
print("Saved to probe_mlp_predictions_27b.png")
