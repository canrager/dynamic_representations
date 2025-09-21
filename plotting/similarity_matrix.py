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
    seq_start_idx: int
    seq_end_idx: int
    omit_bos_token: bool

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def pairwise_cosine_similarity(x_LD):
    eps = 1e-8
    x_norm_LD = x_LD / (th.norm(x_LD, dim=-1, keepdim=True) + eps)
    return th.matmul(x_norm_LD, x_norm_LD.T)


def plot_similarity_comparison(sims, tokens, cfg):
    # Create subplot layout
    fig, axes = plt.subplots(2, 3, figsize=cfg.figsize)

    # Define plot order and titles
    plot_keys = [
        ("activations", "LLM Activations"),
        (f"{cfg.sae.name}/pred_codes", "Predicted Codes"),
        (f"{cfg.sae.name}/novel_codes", "Novel Codes"),
        (f"{cfg.sae.name}/total_recons", "Total Reconstructions"),
        (f"{cfg.sae.name}/pred_recons", "Predicted Reconstructions"),
        (f"{cfg.sae.name}/novel_recons", "Novel Reconstructions"),
    ]

    # Create subplots
    for idx, (key, title) in enumerate(plot_keys):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        im = ax.imshow(sims[key], cmap=cfg.cmap)
        ax.set_title(title)

        # Set tick labels to tokens
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        if row == 1:  # Bottom row
            ax.set_xlabel("Tokens")
        if col == 0:  # Left column
            ax.set_ylabel("Tokens")

        # Add individual colorbar for each subplot
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    # Save plot
    os.makedirs(cfg.env.plots_dir, exist_ok=True)
    save_name = f"similarity_comparison.pdf"
    plot_path = os.path.join(cfg.env.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"saved figure to: {plot_path}")
    plt.close()


def main():
    cfg = PlotConfig(
        figsize=(18, 12),
        cmap="magma",
        selected_sequence_idx=0,
        seq_start_idx=3,
        seq_end_idx=25,
        omit_bos_token=True,
        # Artifacts
        env=ENV_CFG,
        data=SIMPLESTORIES_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=TEMPORAL_GEMMA2_SAE_CFG,
    )
    cfg.data.num_sequences = 10

    art, _ = load_matching_activations(
        cfg,
        [
            "tokens",
            "activations",
            f"{cfg.sae.name}/novel_codes",
            f"{cfg.sae.name}/novel_recons",
            f"{cfg.sae.name}/pred_codes",
            f"{cfg.sae.name}/pred_recons",
            f"{cfg.sae.name}/total_recons",
        ],
        target_folder=cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )

    # Detokenize dataset tokens
    tokens = load_tokens_of_story(
        tokens_BP=art["tokens"],
        story_idx=cfg.selected_sequence_idx,
        model_name=cfg.llm.hf_name,
        omit_BOS_token=cfg.omit_bos_token,
        seq_length=cfg.seq_end_idx,
    )
    # Truncate tokens to selected range
    tokens = tokens[cfg.seq_start_idx : cfg.seq_end_idx]

    # compute similarities
    sims = {}
    for key in art:
        if key == "tokens":
            continue
        selected_LD = art[key][cfg.selected_sequence_idx]
        if cfg.omit_bos_token:
            selected_LD = selected_LD[1:]
        selected_LD = selected_LD[cfg.seq_start_idx : cfg.seq_end_idx]
        sims[key] = pairwise_cosine_similarity(selected_LD).float()

    # Create and save plot
    plot_similarity_comparison(sims, tokens, cfg)


if __name__ == "__main__":
    main()
