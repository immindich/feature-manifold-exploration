# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# %% Load activations
# Update this path to point to your saved activations file
ACTIVATIONS_PATH = "activations-27b.pt"

data = torch.load(ACTIVATIONS_PATH, weights_only=False)

# %% Inspect saved data structure
print("Keys in saved data:", list(data.keys()))
print(f"Model: {data['model_name']}")
print(f"Layers extracted: {data['layers']}")
print(f"Collection args: {data['args']}")

# %% Group activations by count, split train/test, compute means
layers = data["layers"]
all_activations = data["activations"]  # list of (n_layers, hidden_dim) tensors
all_metadata = data["metadata"]  # list of dicts with 'true_count' etc.

# Group indices by true count
from collections import defaultdict
count_to_indices = defaultdict(list)
for i, meta in enumerate(all_metadata):
    count_to_indices[meta["true_count"]].append(i)

unique_counts = np.array(sorted(count_to_indices.keys()))
n_counts = len(unique_counts)
TEST_PER_COUNT = 10

# Split into train/test and compute means
n_layers = len(layers)
hidden_dim = all_activations[0].shape[-1]
mean_acts_per_count = np.zeros((n_layers, n_counts, hidden_dim), dtype=np.float32)
test_acts = []  # list of (n_layers, hidden_dim) arrays
test_counts = []  # corresponding true counts

rng = np.random.default_rng(42)
for count_idx, count in enumerate(unique_counts):
    indices = np.array(count_to_indices[count])
    rng.shuffle(indices)
    test_idx = indices[:TEST_PER_COUNT]
    train_idx = indices[TEST_PER_COUNT:]

    # Compute mean from train set
    train_acts = torch.stack([all_activations[i] for i in train_idx]).float()  # (n_train, n_layers, hidden_dim)
    mean_acts_per_count[:, count_idx, :] = train_acts.mean(dim=0).numpy()

    # Store test activations
    for i in test_idx:
        test_acts.append(all_activations[i].float().numpy())
        test_counts.append(count)

test_acts = np.stack(test_acts)  # (n_test_total, n_layers, hidden_dim)
test_counts = np.array(test_counts)

print(f"Mean activations shape: {mean_acts_per_count.shape}")
print(f"Test set: {len(test_counts)} samples ({TEST_PER_COUNT} per count)")
print(f"Count range: {unique_counts.min()} - {unique_counts.max()}")
print(f"Sequences per count: {len(count_to_indices[unique_counts[0]])}")
print(f"Unique counts: {n_counts}")

# %% PCA on mean activations per layer
from sklearn.decomposition import PCA

n_components = 10
pca_per_layer = []

for layer_idx in range(n_layers):
    pca = PCA(n_components=n_components).fit(mean_acts_per_count[layer_idx])
    pca_per_layer.append(pca)

print(f"Fitted PCA with {n_components} components on mean activations for each of {n_layers} layers")

# %% Align PC directions across layers
# Use the final layer as the reference. For each other layer, flip PCs to point
# in the same direction as the reference (based on sign of dot product).
pc_signs = np.ones((n_layers, n_components))
ref_layer = n_layers - 1

for layer_idx in range(n_layers - 1):
    for comp_idx in range(n_components):
        ref_pc = pca_per_layer[ref_layer].components_[comp_idx]
        cur_pc = pca_per_layer[layer_idx].components_[comp_idx]
        if np.dot(ref_pc, cur_pc) < 0:
            pc_signs[layer_idx, comp_idx] = -1

print(f"Aligned PC directions across layers (using layer {ref_layer} as reference)")

# %% Project mean activations onto principal components
# projected_means[layer, count_idx, component] = projection of mean activation onto PC
projected_means = np.zeros((n_layers, len(unique_counts), n_components))

for layer_idx in range(n_layers):
    pca = pca_per_layer[layer_idx]
    projected_means[layer_idx] = pca.transform(mean_acts_per_count[layer_idx]) * pc_signs[layer_idx]

print(f"Projected means shape: {projected_means.shape}")

# %% Project test activations onto principal components
# projected_test[layer, sample, component] = projection of test activation onto PC
n_test = len(test_counts)
projected_test = np.zeros((n_layers, n_test, n_components))

for layer_idx in range(n_layers):
    pca = pca_per_layer[layer_idx]
    projected_test[layer_idx] = pca.transform(test_acts[:, layer_idx, :]) * pc_signs[layer_idx]

print(f"Projected test shape: {projected_test.shape}")

# %% Compute cumulative variance explained by first n PCs for each layer
# cumulative_var_explained[layer, n] = variance explained by first n+1 PCs
cumulative_var_explained = np.zeros((n_layers, n_components))

for layer_idx in range(n_layers):
    pca = pca_per_layer[layer_idx]
    cumulative_var_explained[layer_idx] = np.cumsum(pca.explained_variance_ratio_)

print(f"Cumulative variance explained shape: {cumulative_var_explained.shape}")

# %% Plot cumulative variance explained
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(cumulative_var_explained.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xlabel("Layer")
ax.set_ylabel("Number of PCs")
ax.set_yticks(range(n_components))
ax.set_yticklabels([f"{i+1}" for i in range(n_components)])
ax.set_title("Cumulative Variance Explained by First n PCs")
plt.colorbar(im, ax=ax, label="Variance Explained")
plt.tight_layout()
plt.show()

# %% Print cumulative variance explained as a table
import pandas as pd

df = pd.DataFrame(
    cumulative_var_explained.T,
    index=[f"PC 1-{i+1}" for i in range(n_components)],
    columns=[f"L{i}" for i in range(n_layers)],
)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df.round(3).to_string())

# %% Function to plot projection of mean activations onto a principal component
def plot_projection(layer_idx: int, component_idx: int, ax=None, show_test: bool = True):
    """Plot the projection of mean activations onto a principal component vs count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    projections = projected_means[layer_idx, :, component_idx]

    if show_test:
        test_proj = projected_test[layer_idx, :, component_idx]
        ax.scatter(test_counts, test_proj, alpha=0.15, s=10, c="tab:blue", label="Test samples")

    ax.plot(unique_counts, projections, marker="o", markersize=3, color="tab:orange", label="Train mean", zorder=5)
    ax.set_xlabel("Count")
    ax.set_ylabel(f"PC{component_idx + 1} projection")
    ax.set_title(f"Layer {layer_idx}: Activation projection onto PC{component_idx + 1}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax

def plot_2d_pca(layer_idx: int, components: tuple[int, int] = (0, 1), ax=None, draw_line: bool = False, show_test: bool = True):
    """Plot mean activations for a layer in 2D PCA space, colored by count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    c1, c2 = components
    x = projected_means[layer_idx, :, c1]
    y = projected_means[layer_idx, :, c2]

    if show_test:
        tx = projected_test[layer_idx, :, c1]
        ty = projected_test[layer_idx, :, c2]
        ax.scatter(tx, ty, c=test_counts, cmap="viridis", s=10, alpha=0.2, marker="x")

    if draw_line:
        ax.plot(x, y, color="gray", alpha=0.5, linewidth=1)

    scatter = ax.scatter(x, y, c=unique_counts, cmap="viridis", s=30, edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_xlabel(f"PC{c1 + 1}")
    ax.set_ylabel(f"PC{c2 + 1}")
    ax.set_title(f"Layer {layer_idx}: Activations in 2D PCA space")
    plt.colorbar(scatter, ax=ax, label="Count")

    return ax


def plot_3d_pca_interactive(layer_idx: int, components: tuple[int, int, int] = (0, 1, 2), draw_line: bool = False, show_test: bool = False):
    """Plot mean activations for a layer in 3D PCA space using Plotly (interactive)."""
    c1, c2, c3 = components
    x = projected_means[layer_idx, :, c1]
    y = projected_means[layer_idx, :, c2]
    z = projected_means[layer_idx, :, c3]

    fig = go.Figure()

    if show_test:
        tx = projected_test[layer_idx, :, c1]
        ty = projected_test[layer_idx, :, c2]
        tz = projected_test[layer_idx, :, c3]
        fig.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode='markers',
            marker=dict(size=2, color=test_counts, colorscale='Viridis', opacity=0.2),
            name='Test samples', showlegend=True
        ))

    if draw_line:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color='gray', width=2), opacity=0.5,
            showlegend=False
        ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=5, color=unique_counts, colorscale='Viridis',
                    colorbar=dict(title='Count'), showscale=True),
        name='Train means', showlegend=True
    ))

    fig.update_layout(
        title=f"Layer {layer_idx}: Activations in 3D PCA space",
        scene=dict(
            xaxis_title=f"PC{c1 + 1}",
            yaxis_title=f"PC{c2 + 1}",
            zaxis_title=f"PC{c3 + 1}"
        )
    )
    fig.show()
    return fig

# %%

plot_projection(3, 0,)
plt.show()
# %%
plot_projection(8, 1)
plt.show()

# %%
plot_projection(8, 2)
plt.show()
# %%
plot_projection(8, 3)
plt.show()
# %%
plot_projection(8, 4)
plt.show()
# %%
plot_projection(8, 5)
plt.show()
# %%
plot_projection(2, 6)
plt.show()

# %%
plot_3d_pca_interactive(4, (0, 1, 2), draw_line=True, show_test=True)
# %%
plot_3d_pca_interactive(2, (1, 3, 4), draw_line=True)
# %%
plot_3d_pca_interactive(5, (2, 3, 4), draw_line=True)
# %%
plot_3d_pca_interactive(1, (0, 1, 5))

# %% Pairwise cosine similarity of mean activation vectors
from sklearn.metrics.pairwise import cosine_similarity

def plot_cosine_similarity(layer_idx: int, ax=None):
    """Plot pairwise cosine similarity matrix of mean activations, ordered by count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # mean_acts_per_count[layer_idx] has shape (n_counts, hidden_dim)
    cos_sim = cosine_similarity(mean_acts_per_count[layer_idx])

    # Use actual data range for color scale
    vmin, vmax = cos_sim.min(), cos_sim.max()
    im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Count")
    ax.set_title(f"Layer {layer_idx}: Pairwise cosine similarity of mean activations")
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    # Set tick labels to actual count values
    tick_step = max(1, len(unique_counts) // 10)
    tick_positions = list(range(0, len(unique_counts), tick_step))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([unique_counts[i] for i in tick_positions])
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([unique_counts[i] for i in tick_positions])

    return ax

# %%
plot_cosine_similarity(n_layers - 9)
plt.show()