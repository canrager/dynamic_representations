import torch as th
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from src.configs import *
from src.plotting_utils import savefig
from exp.pred_umap import ExperimentConfig


def plot_pred_structure(results, cfg):
    """
    Plot UMAP embeddings from saved results, colored by position index.

    Args:
        results: Dictionary containing experiment results with 'embedding', 'pos_labels', and 'config' keys
    """
    # Extract first result (assuming single config for now)
    result_dict = results[next(iter(results.keys()))]

    # Load embedding and position labels
    embedding = th.tensor(result_dict["embedding"])
    pos_labels = np.array(result_dict["pos_labels"])
    pos_indices = result_dict["pos_indices"]

    # Convert to numpy for plotting
    if isinstance(embedding, th.Tensor):
        embedding_np = embedding.cpu().numpy()
    else:
        embedding_np = embedding

    if cfg.act_type == "activations":
        title_prefix = "UMAP of LLM activations"
    elif cfg.act_type == "codes":
        title_prefix = f"UMAP of {cfg.sae.name} codes"
    elif cfg.act_type == "pred_codes":
        title_prefix = f"UMAP of {cfg.sae.name} pred codes"
    elif cfg.act_type == "novel_codes":
        title_prefix = f"UMAP of {cfg.sae.name} novel codes"
    else:
        raise ValueError(f"Unknown act_type: {cfg.act_type}")
    title_prefix += f" {cfg.data.name}"

    # Create figure based on number of components
    if cfg.n_components == 2:
        fig, ax = plt.subplots(figsize=(12, 9))

        # Use viridis colormap for gradient from low to high position
        scatter = ax.scatter(
            embedding_np[:, 0],
            embedding_np[:, 1],
            c=pos_labels,
            cmap="viridis",
            s=20,
            alpha=0.6,
            edgecolors="none",
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
        # Create interactive 3D plot with plotly
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=embedding_np[:, 0],
                    y=embedding_np[:, 1],
                    z=embedding_np[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=pos_labels,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Position Index"),
                        opacity=0.6,
                    ),
                    text=[f"Position: {p}" for p in pos_labels],
                    hovertemplate="<b>Position: %{text}</b><br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}<br>UMAP3: %{z:.2f}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=f"{title_prefix} (colored by position)",
            scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
            width=1000,
            height=800,
        )

        # Save as interactive HTML
        html_path = os.path.join(
            cfg.env.plots_dir,
            f"pred_structure_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_type}.html",
        )
        fig.write_html(html_path)
        print(f"Saved interactive 3D plot to: {html_path}")

        return fig

    else:
        raise ValueError(f"n_components must be 2 or 3, got {cfg.n_components}")

    # Save figure (for 2D only)
    savefig(f"pred_structure_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_type}")

    return fig


def main():
    # Create config matching the experiment setup
    cfg = ExperimentConfig(
        sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        act_type="pred_codes",
        # sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        # act_type="codes",
        # sae=None,
        # act_type="activations",
        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
        # data=SIMPLESTORIES_DS_CFG,
        # data=WEBTEXT_DS_CFG,
        data=CODE_DS_CFG,
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
    )

    # Load results
    results = load_results_multiple_configs(
        exp_name="pred_structure_",
        source_cfgs=[cfg],
        target_folder=cfg.env.results_dir,
        recency_rank=0,
        compared_attributes=["llm", "data", "sae", "act_type"],
        verbose=True,
    )

    # Plot results
    plot_pred_structure(results, cfg)


if __name__ == "__main__":
    main()
