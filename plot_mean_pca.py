#!/usr/bin/env python3
"""Plot mean activations per count in 2D PCA space, per layer.

Mirrors the analysis in analyze_activations.py but parameterized and
non-interactive (saves a PNG).
"""

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def load_means(path: str, test_per_count: int = 10):
    """Load activations and return (mean_acts_per_count, unique_counts, layers).

    mean_acts_per_count has shape (n_layers, n_counts, hidden_dim).
    """
    data = torch.load(path, weights_only=False)
    layers = data["layers"]
    activations = data["activations"]
    metadata = data["metadata"]

    count_to_indices = defaultdict(list)
    for i, m in enumerate(metadata):
        count_to_indices[m["true_count"]].append(i)

    unique_counts = np.array(sorted(count_to_indices.keys()))
    n_layers = len(layers)
    hidden_dim = activations[0].shape[-1]
    means = np.zeros((n_layers, len(unique_counts), hidden_dim), dtype=np.float32)

    rng = np.random.default_rng(42)
    for ci, count in enumerate(unique_counts):
        idx = np.array(count_to_indices[count])
        rng.shuffle(idx)
        train_idx = idx[test_per_count:]
        train_acts = torch.stack([activations[i] for i in train_idx]).float()
        means[:, ci, :] = train_acts.mean(dim=0).numpy()

    return means, unique_counts, layers, data.get("model_name", "model")


def fit_pca_per_layer(means: np.ndarray, n_components: int = 4):
    """Fit PCA per layer and align PC signs to a reference layer.

    Returns (projected_means, cum_var, pcas, signs). The `pcas` and `signs`
    can be reused to project additional activation matrices through the same
    per-layer transforms (e.g. individual sequences on top of mean means).
    """
    n_layers = means.shape[0]
    pcas = [PCA(n_components=n_components).fit(means[li]) for li in range(n_layers)]

    # Align by flipping PCs whose dot product with the last layer's is negative.
    ref = pcas[-1].components_
    signs = np.ones((n_layers, n_components))
    for li in range(n_layers - 1):
        for ci in range(n_components):
            if np.dot(ref[ci], pcas[li].components_[ci]) < 0:
                signs[li, ci] = -1

    projected = np.zeros((n_layers, means.shape[1], n_components))
    for li in range(n_layers):
        projected[li] = pcas[li].transform(means[li]) * signs[li]

    cum_var = np.stack([np.cumsum(p.explained_variance_ratio_) for p in pcas])
    return projected, cum_var, pcas, signs


def project_individuals(acts: np.ndarray, pcas, signs) -> np.ndarray:
    """Project per-sequence activations through the per-layer mean-fitted PCAs.

    acts: (n_seqs, n_layers, hidden_dim)
    Returns: (n_layers, n_seqs, n_components)
    """
    n_seqs, n_layers, _ = acts.shape
    n_components = pcas[0].n_components_
    out = np.zeros((n_layers, n_seqs, n_components))
    for li in range(n_layers):
        out[li] = pcas[li].transform(acts[:, li, :]) * signs[li]
    return out


def plot_grid(projected, cum_var, unique_counts, layers, model_name, output: str, dims: int = 2):
    """Render one PCA scatter per layer + a cumulative-variance heatmap.

    dims=2 uses PC1/PC2; dims=3 uses PC1/PC2/PC3 with mpl 3D axes.
    """
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")

    n_layers = projected.shape[0]
    ncols = 3
    nrows = (n_layers + ncols - 1) // ncols + 1  # extra row for variance heatmap
    fig = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for li in range(n_layers):
        if dims == 2:
            ax = fig.add_subplot(nrows, ncols, li + 1)
            sc = ax.scatter(
                projected[li, :, 0],
                projected[li, :, 1],
                c=unique_counts,
                cmap="viridis",
                s=40,
                edgecolors="black",
                linewidths=0.4,
            )
            ax.plot(projected[li, :, 0], projected[li, :, 1], color="gray", alpha=0.4, linewidth=1)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        else:
            ax = fig.add_subplot(nrows, ncols, li + 1, projection="3d")
            sc = ax.scatter(
                projected[li, :, 0],
                projected[li, :, 1],
                projected[li, :, 2],
                c=unique_counts,
                cmap="viridis",
                s=30,
                edgecolors="black",
                linewidths=0.3,
            )
            ax.plot(
                projected[li, :, 0],
                projected[li, :, 1],
                projected[li, :, 2],
                color="gray",
                alpha=0.4,
                linewidth=1,
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
        ax.set_title(f"Layer {layers[li]}")
        fig.colorbar(sc, ax=ax, label="Count", shrink=0.7)
        if dims == 2:
            ax.grid(True, alpha=0.3)

    # Cumulative variance heatmap spanning the full bottom row.
    ax_var = fig.add_subplot(nrows, 1, nrows)
    im = ax_var.imshow(cum_var.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_var.set_xticks(range(n_layers))
    ax_var.set_xticklabels([f"L{l}" for l in layers])
    ax_var.set_yticks(range(cum_var.shape[1]))
    ax_var.set_yticklabels([f"PC1-{i+1}" for i in range(cum_var.shape[1])])
    ax_var.set_title("Cumulative variance explained by first n PCs")
    fig.colorbar(im, ax=ax_var, label="Variance explained")

    fig.suptitle(
        f"{model_name}: mean activations per count in {dims}D PCA space "
        f"(counts {unique_counts.min()}–{unique_counts.max()})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Activations .pt file")
    parser.add_argument("--output", default="mean_pca.png", help="Output PNG path")
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--dims", type=int, choices=[2, 3], default=2,
                        help="2D or 3D PCA scatter (default: 2)")
    parser.add_argument("--test-per-count", type=int, default=10,
                        help="Reserved per-count test slice (mean computed on train remainder)")
    args = parser.parse_args()

    # Ensure we fit enough components for the requested view.
    n_components = max(args.n_components, args.dims)
    means, unique_counts, layers, model_name = load_means(args.data, args.test_per_count)
    projected, cum_var, _pcas, _signs = fit_pca_per_layer(means, n_components)
    plot_grid(projected, cum_var, unique_counts, layers, model_name, args.output, dims=args.dims)


if __name__ == "__main__":
    main()
