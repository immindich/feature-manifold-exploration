# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# %% Load activations
# Update this path to point to your saved activations file
ACTIVATIONS_PATH = "mean-activations-27b.pt"

data = torch.load(ACTIVATIONS_PATH, weights_only=False)

# %% Inspect saved data structure
print("Keys in saved data:", list(data.keys()))
print(f"Model: {data['model_name']}")
print(f"Layers extracted: {data['layers']}")
print(f"Collection args: {data['args']}")

# %% Load mean activations
layers = data["layers"]

mean_acts_per_count = data["mean_activations"].float().numpy()  # (n_layers, n_counts, hidden_dim)
unique_counts = data["counts"]
n_layers, n_counts, hidden_dim = mean_acts_per_count.shape
print(f"Mean activations shape: {mean_acts_per_count.shape}")
print(f"Count range: {unique_counts.min()} - {unique_counts.max()}")
print(f"Sequences per count: {data.get('sequences_per_count', 'unknown')}")

n_layers = mean_acts_per_count.shape[0]
print(f"Unique counts: {len(unique_counts)}")

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
def plot_projection(layer_idx: int, component_idx: int, ax=None):
    """Plot the projection of mean activations onto a principal component vs count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    projections = projected_means[layer_idx, :, component_idx]
    ax.plot(unique_counts, projections, marker="o", markersize=3)
    ax.set_xlabel("Count")
    ax.set_ylabel(f"PC{component_idx + 1} projection")
    ax.set_title(f"Layer {layer_idx}: Mean activation projection onto PC{component_idx + 1}")
    ax.grid(True, alpha=0.3)

    return ax

def plot_2d_pca(layer_idx: int, components: tuple[int, int] = (0, 1), ax=None, draw_line: bool = False):
    """Plot mean activations for a layer in 2D PCA space, colored by count."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    c1, c2 = components
    x = projected_means[layer_idx, :, c1]
    y = projected_means[layer_idx, :, c2]

    if draw_line:
        ax.plot(x, y, color="gray", alpha=0.5, linewidth=1)

    scatter = ax.scatter(x, y, c=unique_counts, cmap="viridis", s=30)
    ax.set_xlabel(f"PC{c1 + 1}")
    ax.set_ylabel(f"PC{c2 + 1}")
    ax.set_title(f"Layer {layer_idx}: Mean activations in 2D PCA space")
    plt.colorbar(scatter, ax=ax, label="Count")

    return ax


def plot_3d_pca_interactive(layer_idx: int, components: tuple[int, int, int] = (0, 1, 2), draw_line: bool = False):
    """Plot mean activations for a layer in 3D PCA space using Plotly (interactive)."""
    c1, c2, c3 = components
    x = projected_means[layer_idx, :, c1]
    y = projected_means[layer_idx, :, c2]
    z = projected_means[layer_idx, :, c3]

    fig = go.Figure()

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
        showlegend=False
    ))

    fig.update_layout(
        title=f"Layer {layer_idx}: Mean activations in 3D PCA space",
        scene=dict(
            xaxis_title=f"PC{c1 + 1}",
            yaxis_title=f"PC{c2 + 1}",
            zaxis_title=f"PC{c3 + 1}"
        )
    )
    fig.show()
    return fig

# %%

plot_projection(8, 0,)
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
plot_3d_pca_interactive(9, (0, 1, 2), draw_line=True)
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