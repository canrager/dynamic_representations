import torch as th
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.configs import *
from src.plotting_utils import savefig
from src.model_utils import load_hf_model
from src.exp_utils import load_tokens_of_story
from exp.pred_pca import ExperimentConfig


def plot_pred_structure_pca(results, cfg, tokens_BL=None, tokenizer=None):
    """
    Plot PCA embeddings from saved results, colored by position index.
    Shows n_components subplots with PC1 on x-axis and each component on y-axis.
    For 3D, creates an interactive plotly plot with story hover text.

    Args:
        results: Dictionary containing experiment results with 'embedding', 'pos_labels', and 'config' keys
        cfg: ExperimentConfig object
        tokens_BL: Optional tokens tensor of shape (B, L) for hover text
        tokenizer: Optional tokenizer for decoding tokens
    """
    # Extract first result (assuming single config for now)
    result_dict = results[next(iter(results.keys()))]

    # Load embedding and position labels
    embedding = th.tensor(result_dict["embedding"])  # (B, L, n_components)
    pos_labels = np.array(result_dict["pos_labels"])  # (B, L)
    pos_indices = result_dict["pos_indices"]
    explained_variance_ratio = result_dict["explained_variance_ratio"]

    # Subsample sequences if requested
    if cfg.num_sequences is not None:
        embedding = embedding[:cfg.num_sequences]
        pos_labels = pos_labels[:cfg.num_sequences]

    # Store original shape for sequence connections
    batch_size, seq_len, n_dims = embedding.shape

    # Convert to numpy for plotting
    if isinstance(embedding, th.Tensor):
        embedding_np = embedding.cpu().numpy()  # (B, L, n_components)
    else:
        embedding_np = embedding

    if cfg.act_path == "activations":
        title_prefix = "PCA of LLM activations"
    elif cfg.act_path == "codes":
        title_prefix = f"PCA of {cfg.sae.name} codes"
    elif cfg.act_path == "pred_codes":
        title_prefix = f"PCA of {cfg.sae.name} pred codes"
    elif cfg.act_path == "novel_codes":
        title_prefix = f"PCA of {cfg.sae.name} novel codes"
    else:
        raise ValueError(f"Unknown act_path: {cfg.act_path}")
    title_prefix += f" {cfg.data.name}"

    # Create figure based on number of components
    n_components = cfg.n_components

    if n_components == 3:
        # Create interactive 3D plotly plot
        from einops import rearrange

        # Flatten for plotting
        embedding_flat = rearrange(embedding_np, 'B L D -> (B L) D')
        pos_labels_flat = pos_labels.flatten()

        # Generate hover text with story context
        hover_texts = []
        if tokens_BL is not None and tokenizer is not None:
            for b_idx in range(batch_size):
                # Load tokens for this story once
                token_strs_L = load_tokens_of_story(
                    tokens_BP=tokens_BL,
                    story_idx=b_idx,
                    model_name=cfg.llm.hf_name,
                    omit_BOS_token=False,
                    seq_length=None,
                    tokenizer=tokenizer,
                )

                for l_idx_in_seq, pos_idx in enumerate(pos_indices):
                    # pos_idx is the actual position in the full sequence
                    # Show previous hover_window tokens + current token (bolded last)
                    start_idx = max(0, pos_idx - cfg.hover_window)
                    end_idx = min(len(token_strs_L), pos_idx + 1, cfg.max_p)

                    story_str = ""
                    token_count = 0
                    for i in range(start_idx, end_idx):
                        t = token_strs_L[i]

                        # Add linebreak every 15 tokens
                        if token_count > 0 and token_count % 15 == 0:
                            story_str += "<br>"

                        if i == pos_idx:
                            story_str += f"<b>{t}</b>"  # Bold the current token (last one)
                        else:
                            story_str += t

                        token_count += 1

                    hover_texts.append(f"Position: {pos_idx}<br>Story: {story_str}")
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
                hovertemplate="%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
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
            scene=dict(
                xaxis_title=f"PC1 ({explained_variance_ratio[0]:.1%} var)",
                yaxis_title=f"PC2 ({explained_variance_ratio[1]:.1%} var)",
                zaxis_title=f"PC3 ({explained_variance_ratio[2]:.1%} var)",
            ),
            width=1000,
            height=800,
        )

        # Save as interactive HTML
        html_path = os.path.join(
            cfg.env.plots_dir,
            f"pred_structure_pca_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_path}.html",
        )
        fig.write_html(html_path)
        print(f"Saved interactive 3D plot to: {html_path}")

        return fig

    else:
        # 2D or multi-component matplotlib plots
        from einops import rearrange

        # Flatten for 2D plotting
        embedding_flat = rearrange(embedding_np, 'B L D -> (B L) D')
        pos_labels_flat = pos_labels.flatten()

        n_plots = n_components - 1  # Plot PC0 vs PC1, PC0 vs PC2, ..., PC0 vs PC(n-1)

        if n_plots <= 1:
            n_rows, n_cols = 1, n_plots
        elif n_plots <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_rows = (n_plots + 2) // 3
            n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

        # Flatten axes array for easier indexing
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_plots > 1 else [axes]

        # Plot PC0 vs PC1, PC0 vs PC2, PC0 vs PC3, etc.
        for i in range(n_plots):
            ax = axes[i]

            # Plot PC0 vs PC(i+1)
            scatter = ax.scatter(
                embedding_flat[:, 0],
                embedding_flat[:, i + 1],
                c=pos_labels_flat,
                cmap="viridis",
                s=20,
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_ylabel(f"PC{i+2} ({explained_variance_ratio[i+1]:.1%} var)", fontsize=10)
            ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} var)", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Add colorbar to the last subplot
        if n_plots > 0:
            cbar = fig.colorbar(scatter, ax=axes[:n_plots], orientation="vertical", pad=0.02)
            cbar.set_label("Position Index", rotation=270, labelpad=20)

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)

        # Add overall title
        fig.suptitle(
            f"{title_prefix}\n(colored by position, {n_components} components)", fontsize=14, y=1.00
        )

        plt.tight_layout()

        # Save figure
        savefig(f"pred_structure_pca_by_position_{cfg.llm.name}_{cfg.data.name}_{cfg.act_path}.png")

    return fig


def main():
    # Create config matching the experiment setup
    cfg = ExperimentConfig(
        # sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        # act_path="pred_codes",
        # sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        # act_type="codes",
        sae=None,
        act_path="activations",
        # llm=GEMMA2_LLM_CFG,
        llm=IT_GEMMA2_LLM_CFG,
        env=ENV_CFG,
        # data=SIMPLESTORIES_DS_CFG,
        data=CHAT_DS_CFG,
        # data=WEBTEXT_DS_CFG,
        # data=CODE_DS_CFG,
        # Position subsampling
        min_p=10,
        max_p=499,
        num_p=20,
        do_log_scale=False,
        # PCA parameters
        n_components=3,
        center=True,
        # Plot config
        connect_sequences=True,
        num_sequences=100,
        hover_window=50,  # Show ±50 tokens around current position
    )

    # Load results
    results = load_results_multiple_configs(
        exp_name="pred_structure_pca_",
        source_cfgs=[cfg],
        target_folder=cfg.env.results_dir,
        recency_rank=0,
        compared_attributes=["llm", "data", "sae", "act_path"],
        verbose=True,
    )

    # Load tokens and tokenizer for hover text
    artifacts, _ = load_matching_activations(
        cfg,
        ["tokens"],
        cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )
    tokens_BL = artifacts["tokens"]
    tokenizer = load_hf_model(cfg, tokenizer_only=True)

    # Plot results
    plot_pred_structure_pca(results, cfg, tokens_BL=tokens_BL, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
