import torch as th
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from einops import rearrange

from src.configs import *
from src.plotting_utils import savefig
from exp.pred_umap import ExperimentConfig


def plot_pred_structure(results, cfg):
    """
    Plot UMAP embeddings from saved results, colored by position index.

    Args:
        results: Dictionary containing experiment results with 'embedding', 'pos_labels', 'hover_texts', and 'config' keys
        cfg: Experiment configuration (including num_sequences and connect_sequences)
    """
    # Extract first result (assuming single config for now)
    result_dict = results[next(iter(results.keys()))]

    # Load embedding, position labels, and hover texts
    embedding = th.tensor(result_dict["embedding"])  # (B, L, n_components)
    pos_labels = np.array(result_dict["pos_labels"])  # (B, L)
    pos_indices = result_dict["pos_indices"]
    hover_texts_saved = result_dict.get("hover_texts", None)  # List of hover texts

    # Subsample sequences if requested
    if cfg.num_sequences is not None:
        embedding = embedding[:cfg.num_sequences]
        pos_labels = pos_labels[:cfg.num_sequences]
        if hover_texts_saved is not None:
            # hover_texts is a flat list of (B*L) elements
            # Need to reshape and subsample
            seq_len = embedding.shape[1]
            hover_texts_saved = hover_texts_saved[:cfg.num_sequences * seq_len]

    # Store original shape for sequence connections
    batch_size, seq_len, n_dims = embedding.shape

    # Convert to numpy for plotting
    if isinstance(embedding, th.Tensor):
        embedding_np = embedding.cpu().numpy()  # (B, L, n_components)
    else:
        embedding_np = embedding


    if cfg.act_path == "activations":
        title_prefix = "UMAP of LLM activations"
    elif cfg.act_path == "codes":
        title_prefix = f"UMAP of {cfg.sae.name} codes"
    elif cfg.act_path == "pred_codes":
        title_prefix = f"UMAP of {cfg.sae.name} pred codes"
    elif cfg.act_path == "novel_codes":
        title_prefix = f"UMAP of {cfg.sae.name} novel codes"
    else:
        raise ValueError(f"Unknown act_path: {cfg.act_path}")
    title_prefix += f" {cfg.data.name}"

    # Create figure based on number of components
    if cfg.n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Draw sequence connection lines first (so they appear behind points)
        if cfg.connect_sequences:
            for b in range(batch_size):
                ax.plot(
                    embedding_np[b, :, 0],
                    embedding_np[b, :, 1],
                    'k-',
                    linewidth=0.5,
                    alpha=0.3,
                    zorder=1
                )

        # Flatten for scatter plot
        embedding_flat = rearrange(embedding_np, 'B L D -> (B L) D')
        pos_labels_flat = pos_labels.flatten()

        # Use viridis colormap for gradient from low to high position
        scatter = ax.scatter(
            embedding_flat[:, 0],
            embedding_flat[:, 1],
            c=pos_labels_flat,
            cmap="viridis",
            s=20,
            alpha=0.6,
            edgecolors="none",
            zorder=2
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Position Index", rotation=270, labelpad=20)

        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(f"{title_prefix}\n(colored by position)", fontsize=14)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    elif cfg.n_components == 3:
        # Flatten for plotting
        embedding_flat = rearrange(embedding_np, 'B L D -> (B L) D')
        pos_labels_flat = pos_labels.flatten()

        # Use saved hover texts if available, otherwise fallback to position only
        if hover_texts_saved is not None:
            hover_texts = hover_texts_saved
        else:
            # Fallback: just show position
            hover_texts = [f"Position: {p}" for p in pos_labels_flat]

        # Create trace data
        traces = []

        # Add scatter points
        traces.append(
            go.Scatter3d(
                x=embedding_flat[:, 0],
                y=embedding_flat[:, 1],
                z=embedding_flat[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=pos_labels_flat,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Position Index"),
                    opacity=0.6,
                ),
                text=hover_texts,
                hovertemplate="%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>",
            )
        )

        # Add sequence connection lines
        if cfg.connect_sequences:
            for b in range(batch_size):
                traces.append(
                    go.Scatter3d(
                        x=embedding_np[b, :, 0],
                        y=embedding_np[b, :, 1],
                        z=embedding_np[b, :, 2],
                        mode="lines",
                        line=dict(color="black", width=2),
                        opacity=0.3,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        # Create interactive 3D plot with plotly
        fig = go.Figure(data=traces)

        fig.update_layout(
            title=f"{title_prefix} (colored by position)",
            scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
            width=1000,
            height=800,
        )

        # Save as interactive HTML
        html_path = os.path.join(
            cfg.env.plots_dir,
            f"pred_structure_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_path}.html",
        )
        fig.write_html(html_path)
        print(f"Saved interactive 3D plot to: {html_path}")

        return fig

    else:
        raise ValueError(f"n_components must be 2 or 3, got {cfg.n_components}")

    # Save figure (for 2D only)
    savefig(f"pred_structure_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_path}")

    return fig


def main():
    # Create config matching the experiment setup
    cfg = ExperimentConfig(
        sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        act_path="pred_codes",
        # sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        # act_path="codes",
        # sae=None,
        # act_path="activations",
        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
        data=SIMPLESTORIES_DS_CFG,
        # data=WEBTEXT_DS_CFG,
        # data=CODE_DS_CFG,
        # Position subsampling
        min_p=1,
        max_p=499,
        num_p=20,
        do_log_scale=False,
        # UMAP parameters
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        #Plot config
        connect_sequences=True,
        num_sequences=200,
        hover_window=50,  # Show 50 tokens around current position
    )

    # Load results
    results = load_results_multiple_configs(
        exp_name="pred_structure_",
        source_cfgs=[cfg],
        target_folder=cfg.env.results_dir,
        recency_rank=0,
        compared_attributes=["llm", "data", "sae", "act_path"],
        verbose=True,
    )

    # Plot results
    plot_pred_structure(results, cfg)


if __name__ == "__main__":
    main()
