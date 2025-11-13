from dataclasses import dataclass
from collections import defaultdict
import os

import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.configs import *
from src.exp_utils import load_tokens_of_story


@dataclass
class PlotConfig:
    figsize: tuple
    cmap: str
    selected_sequence_idx: int
    story_indices: list
    seq_start_idx: int
    seq_end_idx: int
    omit_bos_token: bool
    use_log_scale: bool

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def plot_attn_graphs(graphs_AHLL, tokens, cfg):
    # Create subplot layout
    num_layers, num_heads = graphs_AHLL.shape[:2]
    dynamic_figsize = (num_heads * 4, 3.5)
    fig, axes = plt.subplots(num_layers, num_heads, figsize=dynamic_figsize, squeeze=False)

    graphs_AHLL = graphs_AHLL.to(dtype=th.float32, device="cpu")
    graphs_AHLL = graphs_AHLL[:, :, :, 1:]
    if cfg.omit_bos_token:
        graphs_AHLL = graphs_AHLL[:, :, 1:, 1:]
    vmax = 1

    if cfg.use_log_scale:
        from matplotlib.colors import LogNorm

        # Set up log scale parameters
        log_vmin = 1e-6  # Avoid log(0)
        norm = LogNorm(vmin=log_vmin, vmax=vmax)
    else:
        vmin = 0
        norm = None

    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer, head]

            graph_LL = graphs_AHLL[layer, head]

            if cfg.use_log_scale:
                # Replace zeros with small value for log scale
                graph_data = th.where(graph_LL == 0, log_vmin, graph_LL)
            else:
                graph_data = graph_LL

            if cfg.use_log_scale:
                im = ax.imshow(graph_data.numpy(), cmap=cfg.cmap, norm=norm, aspect="auto")
            else:
                im = ax.imshow(
                    graph_data.numpy(), cmap=cfg.cmap, vmin=vmin, vmax=vmax, aspect="auto"
                )

            # Set tick labels
            ax.set_xticks(range(len(tokens) - 1))
            ax.set_xticklabels(tokens[:-1], rotation=90, ha="right")

            if head == 0:
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)
            else:
                ax.set_yticks([])

            # Set title
            # ax.set_title(f"L{layer} H{head}")

            # Adjust tick label size for readability
            ax.tick_params(labelsize=7)

    # Add shared colorbar (log scale already applied to images)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()

    # Save plot
    os.makedirs(cfg.env.plots_dir, exist_ok=True)
    save_name = f"attn_maps_{cfg.llm.name}_{cfg.sae.name}_{cfg.data.name}.pdf"
    plot_path = os.path.join(cfg.env.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"saved figure to: {plot_path}")
    plt.close()


def plot_attn_sum_across_heads(graphs_BAHLL, tokens_BP, cfg):
    num_stories = len(cfg.story_indices)
    # Get number of layers from the first story
    num_layers = graphs_BAHLL.shape[1]
    summed_figsize = (5 * num_stories, 5 * num_layers)
    fig, axes = plt.subplots(num_layers, num_stories, figsize=summed_figsize, squeeze=False)

    # Find global vmax across all stories and layers for consistent scaling
    all_vmax = 0
    story_layer_data = []

    for story_idx in cfg.story_indices:
        # Extract data for this story
        graphs_AHLL = graphs_BAHLL[
            story_idx,
            :,
            :,
            cfg.seq_start_idx : cfg.seq_end_idx,
            cfg.seq_start_idx : cfg.seq_end_idx,
        ]

        # Sum across heads only (keep layers separate)
        graphs_ALL = th.sum(
            graphs_AHLL, dim=1
        )  # Sum across heads, shape: (num_layers, seq_len, seq_len)
        graphs_ALL = graphs_ALL.to(dtype=th.float32, device="cpu")
        story_layer_data.append(graphs_ALL)

        story_vmax = graphs_ALL.max().item()
        all_vmax = max(all_vmax, story_vmax)

    if cfg.use_log_scale:
        from matplotlib.colors import LogNorm

        log_vmin = 1e-6
        norm = LogNorm(vmin=log_vmin, vmax=all_vmax)
    else:
        vmin = 0
        norm = None

    for story_i, (story_idx, graphs_ALL) in enumerate(zip(cfg.story_indices, story_layer_data)):
        # Load tokens for this story
        tokens = load_tokens_of_story(
            tokens_BP=tokens_BP,
            story_idx=story_idx,
            model_name=cfg.llm.hf_name,
            omit_BOS_token=cfg.omit_bos_token,
            seq_length=cfg.seq_end_idx,
        )
        tokens = tokens[cfg.seq_start_idx : cfg.seq_end_idx]

        for layer_i in range(num_layers):
            ax = axes[layer_i, story_i]

            graph_LL = graphs_ALL[layer_i]

            if cfg.use_log_scale:
                graph_data = th.where(graph_LL == 0, log_vmin, graph_LL)
            else:
                graph_data = graph_LL

            if cfg.use_log_scale:
                im = ax.imshow(graph_data.numpy(), cmap=cfg.cmap, norm=norm, aspect="auto")
            else:
                im = ax.imshow(
                    graph_data.numpy(), cmap=cfg.cmap, vmin=vmin, vmax=all_vmax, aspect="auto"
                )

            # Set tick labels
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right")
            ax.set_yticklabels(tokens)

            ax.set_title(f"Story {story_idx} Layer {layer_i} (Sum across heads)")
            ax.tick_params(labelsize=8)

    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, cmap=cfg.cmap)

    # Save plot
    os.makedirs(cfg.env.plots_dir, exist_ok=True)
    save_name = f"attn_maps_sum_heads_stories_{cfg.llm.name}_{cfg.sae.name}_{cfg.data.name}.png"
    plot_path = os.path.join(cfg.env.plots_dir, save_name)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"saved figure to: {plot_path}")
    plt.close()


def main():
    cfg = PlotConfig(
        figsize=(10, 10),
        cmap="magma",
        selected_sequence_idx=2,  # for the per-head plot
        story_indices=[0],  # for the sum across heads plot
        seq_start_idx=0,
        seq_end_idx=40,
        omit_bos_token=True,
        use_log_scale=False,
        # Artifacts
        env=ENV_CFG,
        data=CODE_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=GEMMA2_TEMPORAL_SAE_CFG,
    )
    # cfg.data.num_sequences = 10

    art, _ = load_matching_activations(
        cfg,
        [
            "tokens",
            f"{cfg.sae.name}/attn_graphs",
        ],
        target_folder=cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )

    # Detokenize dataset tokens
    offset = 1 if cfg.omit_bos_token else 0
    tokens = load_tokens_of_story(
        tokens_BP=art["tokens"],
        story_idx=cfg.selected_sequence_idx,
        model_name=cfg.llm.hf_name,
        omit_BOS_token=cfg.omit_bos_token,
        seq_length=cfg.seq_end_idx - offset,
    )
    # Truncate tokens to selected range
    tokens = tokens[cfg.seq_start_idx : cfg.seq_end_idx]

    graphs_BAHLL = art[
        f"{cfg.sae.name}/attn_graphs"
    ]  # shape (num_seq, num_attn_layers, num_heads, context_length, context_length)
    graphs_AHLL = graphs_BAHLL[
        cfg.selected_sequence_idx,
        :,
        :,
        cfg.seq_start_idx : cfg.seq_end_idx,
        cfg.seq_start_idx : cfg.seq_end_idx,
    ]

    # Create and save plot
    plot_attn_graphs(graphs_AHLL, tokens, cfg)

    # # Create and save sum across heads plot for multiple stories
    # plot_attn_sum_across_heads(graphs_BAHLL, art["tokens"], cfg)


if __name__ == "__main__":
    main()
