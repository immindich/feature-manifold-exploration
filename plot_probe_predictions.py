#!/usr/bin/env python3
"""Load trained probe weights and plot predicted vs actual count on the test set."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from plotting import scatter_true_vs_predicted
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
    scatter_true_vs_predicted(axes[i], test_counts, pred, title=f"Layer {layer}")

fig.suptitle(f"{probe_type.upper()} Probe: Predicted vs Actual Count (Test Set)", fontsize=14)
plt.tight_layout()
plt.savefig("probe_mlp_predictions_27b.png", dpi=150)
plt.show()
print("Saved to probe_mlp_predictions_27b.png")
