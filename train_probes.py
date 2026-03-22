#!/usr/bin/env python3
"""
Train linear and MLP probes to predict count from residual stream activations.

Trains one probe per layer. Supports linear (hidden_dim -> 1) and single-hidden-layer
MLP (hidden_dim -> mlp_hidden -> 1) architectures.

Usage:
    python train_probes.py --data activations-27b.pt --model linear
    python train_probes.py --data activations-27b.pt --model mlp --mlp-hidden 128
    python train_probes.py --test  # run on fake data to verify correctness
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_data(path: str) -> tuple[list[torch.Tensor], list[dict], list[int]]:
    """Load activations file and return activations, metadata, layers."""
    data = torch.load(path, weights_only=False)
    return data["activations"], data["metadata"], data["layers"]


def make_fake_data(
    n_layers: int = 6,
    hidden_dim: int = 4608,
    min_count: int = 1,
    max_count: int = 150,
    seqs_per_count: int = 80,
) -> tuple[list[torch.Tensor], list[dict], list[int]]:
    """Generate fake data with the same structure as real activations for testing."""
    layers = list(range(0, n_layers * 10, 10))
    activations = []
    metadata = []
    for count in range(min_count, max_count + 1):
        for _ in range(seqs_per_count):
            act = torch.randn(n_layers, hidden_dim, dtype=torch.bfloat16)
            # Inject a weak linear signal so probes have something to learn
            act[:, 0] += count / max_count
            activations.append(act)
            metadata.append({"true_count": count})
    return activations, metadata, layers


def split_train_val_test(
    activations: list[torch.Tensor],
    metadata: list[dict],
    test_per_count: int = 10,
    val_per_count: int = 10,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Split data into train/val/test.

    Test is held out for final evaluation, val is used for early stopping.
    Returns (train_acts, train_counts, val_acts, val_counts, test_acts, test_counts).
    Activations are tensors of shape (n_samples, n_layers, hidden_dim).
    """
    count_to_indices = defaultdict(list)
    for i, meta in enumerate(metadata):
        count_to_indices[meta["true_count"]].append(i)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for count in sorted(count_to_indices.keys()):
        indices = np.array(count_to_indices[count])
        rng.shuffle(indices)
        test_idx.extend(indices[:test_per_count])
        val_idx.extend(indices[test_per_count:test_per_count + val_per_count])
        train_idx.extend(indices[test_per_count + val_per_count:])

    train_acts = torch.stack([activations[i] for i in train_idx]).float()
    val_acts = torch.stack([activations[i] for i in val_idx]).float()
    test_acts = torch.stack([activations[i] for i in test_idx]).float()
    train_counts = np.array([metadata[i]["true_count"] for i in train_idx])
    val_counts = np.array([metadata[i]["true_count"] for i in val_idx])
    test_counts = np.array([metadata[i]["true_count"] for i in test_idx])

    return train_acts, train_counts, val_acts, val_counts, test_acts, test_counts


def train_probe(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    max_epochs: int = 500,
    patience: int = 20,
    device: str = "cpu",
) -> dict:
    """Train a probe with early stopping on validation MAE. Returns metrics dict."""
    model = model.to(device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True
    )

    best_val_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_mae = (val_pred - val_y).abs().mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Evaluate best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_pred = model(train_x).cpu().numpy()
        val_pred = model(val_x).cpu().numpy()

    train_y_np = train_y.cpu().numpy()
    val_y_np = val_y.cpu().numpy()

    metrics = {
        "train_mae": float(np.abs(train_pred - train_y_np).mean()),
        "val_mae": float(np.abs(val_pred - val_y_np).mean()),
        "train_r2": float(1 - np.sum((train_pred - train_y_np) ** 2) / np.sum((train_y_np - train_y_np.mean()) ** 2)),
        "val_r2": float(1 - np.sum((val_pred - val_y_np) ** 2) / np.sum((val_y_np - val_y_np.mean()) ** 2)),
        "epochs": epoch + 1,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train probes to predict count from activations")
    parser.add_argument("--data", type=str, help="Path to activations .pt file")
    parser.add_argument("--test", action="store_true", help="Run on fake data to verify correctness")
    parser.add_argument("--model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mlp-hidden", type=int, default=128, help="Hidden dim for MLP probe")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.test and not args.data:
        parser.error("Either --data or --test is required")

    # Load or generate data
    if args.test:
        print("Running on fake data...")
        activations, metadata, layers = make_fake_data()
    else:
        print(f"Loading data from {args.data}...")
        activations, metadata, layers = load_data(args.data)

    # Split
    train_acts, train_counts, val_acts, val_counts, test_acts, test_counts = split_train_val_test(activations, metadata)
    n_layers = train_acts.shape[1]
    hidden_dim = train_acts.shape[2]
    print(f"Train: {len(train_counts)}, Val: {len(val_counts)}, Test: {len(test_counts)} samples")
    print(f"Layers: {layers}, Hidden dim: {hidden_dim}")
    print(f"Probe type: {args.model}" + (f" (hidden={args.mlp_hidden})" if args.model == "mlp" else ""))

    # Train one probe per layer (early stopping on val, final metrics on test)
    results = {}
    trained_probes = {}
    for li, layer in enumerate(layers):
        train_x = train_acts[:, li, :]
        train_y = torch.tensor(train_counts, dtype=torch.float32)
        val_x = val_acts[:, li, :]
        val_y = torch.tensor(val_counts, dtype=torch.float32)
        test_x = test_acts[:, li, :]
        test_y = torch.tensor(test_counts, dtype=torch.float32)

        if args.model == "linear":
            probe = LinearProbe(hidden_dim)
        else:
            probe = MLPProbe(hidden_dim, args.mlp_hidden)

        metrics = train_probe(
            probe, train_x, train_y, val_x, val_y,
            lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, max_epochs=args.max_epochs,
            patience=args.patience, device=args.device,
        )

        # Compute final test metrics with best model
        probe.eval()
        with torch.no_grad():
            test_pred = probe.to(args.device)(test_x.to(args.device)).cpu().numpy()
        test_y_np = test_y.numpy()
        metrics["test_mae"] = float(np.abs(test_pred - test_y_np).mean())
        metrics["test_r2"] = float(1 - np.sum((test_pred - test_y_np) ** 2) / np.sum((test_y_np - test_y_np.mean()) ** 2))

        results[layer] = metrics
        trained_probes[layer] = probe.cpu().state_dict()
        print(f"  Layer {layer:3d}: val_mae={metrics['val_mae']:.2f}, test_mae={metrics['test_mae']:.2f}, test_r2={metrics['test_r2']:.3f} ({metrics['epochs']} epochs)")

    # Save results and weights
    if args.output:
        output_path = Path(args.output)
        save_data = {
            "probe_type": args.model,
            "mlp_hidden": args.mlp_hidden if args.model == "mlp" else None,
            "layers": layers,
            "results": results,
            "args": vars(args),
        }
        output_path.write_text(json.dumps(save_data, indent=2))
        print(f"Results saved to {output_path}")

        weights_path = output_path.with_suffix(".pt")
        torch.save({
            "probe_type": args.model,
            "mlp_hidden": args.mlp_hidden if args.model == "mlp" else None,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "probes": trained_probes,
        }, weights_path)
        print(f"Weights saved to {weights_path}")


if __name__ == "__main__":
    main()
