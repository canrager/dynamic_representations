from dataclasses import dataclass
import os

import torch as th
import matplotlib.pyplot as plt

from src.configs import *
from src.exp_utils import load_tokens_of_story


@dataclass
class SurprisalSimilarityMatrixConfig:
    figsize: tuple
    cmap: str
    selected_sequence_indices: list[int]
    seq_start_idx: int
    seq_end_idx: int
    omit_bos_token: bool

    llm: LLMConfig
    env: EnvironmentConfig

@dataclass
class SurprisalActivationConfig:
    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig
    act_path: str



def pairwise_cosine_similarity(x_LD):
    eps = 1e-8
    x_norm_LD = x_LD / (th.norm(x_LD, dim=-1, keepdim=True) + eps)
    return th.matmul(x_norm_LD, x_norm_LD.T)


def plot_surprisal_similarity_comparison(sims_by_seq, tokens_by_seq, masks_by_seq, plot_cfg):
    # Calculate layout: rows = sequences, cols = act_path modes
    num_sequences = len(sims_by_seq)
    num_cols = len(next(iter(sims_by_seq.values())))  # Number of act_path modes

    # Create subplot layout
    fig, axes = plt.subplots(num_sequences, num_cols, figsize=plot_cfg.figsize)

    # Handle case where we have only one sequence or one mode
    if num_sequences == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_sequences == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    # Create subplots
    for seq_idx, (seq_key, sims) in enumerate(sims_by_seq.items()):
        tokens = tokens_by_seq[seq_key]
        mask = masks_by_seq[seq_key]

        # Get valid (non-masked) token indices
        valid_indices = th.where(mask)[0].tolist()
        valid_tokens = [tokens[i] for i in valid_indices]

        for col_idx, (act_key, sim_matrix) in enumerate(sims.items()):
            ax = axes[seq_idx][col_idx]

            im = ax.imshow(sim_matrix, cmap=plot_cfg.cmap)
            title = f"Seq {seq_key} - {act_key}"
            ax.set_title(title, fontsize=10)

            # Set tick labels to valid tokens only
            ax.set_xticks(range(len(valid_tokens)))
            ax.set_yticks(range(len(valid_tokens)))
            ax.set_xticklabels(valid_tokens, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(valid_tokens, fontsize=8)

            if seq_idx == num_sequences - 1:  # Bottom row
                ax.set_xlabel("Tokens")
            if col_idx == 0:  # Left column
                ax.set_ylabel("Tokens")

            # Add individual colorbar for each subplot
            fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    # Save plot
    os.makedirs(plot_cfg.env.plots_dir, exist_ok=True)
    save_name = f"surprisal_similarity_matrix.png"
    plot_path = os.path.join(plot_cfg.env.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"saved figure to: {plot_path}")
    plt.close()


def main():
    plot_cfg = SurprisalSimilarityMatrixConfig(
        figsize=(30, 50),
        cmap="magma",
        selected_sequence_indices=[0, 1, 2, 3, 4, 5, 6],
        seq_start_idx=1,
        seq_end_idx=50,
        omit_bos_token=True,

        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
    )

    configs = get_gemma_act_configs(
        cfg_class=SurprisalActivationConfig,
        # Artifacts
        env=ENV_CFG,
        data=DatasetConfig(
            name="Surprisal",
            hf_name="surprisal.json",
            num_sequences=59,
            context_length=None,
        ),
        llm=GEMMA2_LLM_CFG,
        sae=None,  # overwritten
        # SAE to act_path map
        act_paths=(
            (
                [None],
                [
                    "sentences",
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    "pred_codes",
                    "novel_codes",
                ]
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    "codes",
                ]
            ),
        ),
    )

    print("loading tokens and masks")
    art, _ = load_matching_activations(
        configs[0],
        ["tokens", "masks"],
        configs[0].env.activations_dir,
        compared_attributes=["llm", "data"],
    )

    # Detokenize dataset tokens and load masks for each sequence
    tokens_by_seq = {}
    masks_by_seq = {}
    for seq_idx in plot_cfg.selected_sequence_indices:
        tokens = load_tokens_of_story(
            tokens_BP=art["tokens"],
            story_idx=seq_idx,
            model_name=plot_cfg.llm.hf_name,
            omit_BOS_token=plot_cfg.omit_bos_token,
            seq_length=plot_cfg.seq_end_idx,
        )
        # Get attention mask for this sequence
        mask = art["masks"][seq_idx]
        if plot_cfg.omit_bos_token:
            mask = mask[1:]

        # Truncate tokens and mask to selected range
        tokens_by_seq[seq_idx] = tokens[plot_cfg.seq_start_idx : plot_cfg.seq_end_idx]
        masks_by_seq[seq_idx] = mask[plot_cfg.seq_start_idx : plot_cfg.seq_end_idx]

    print("loading activations")

    sims_by_seq = {}
    for seq_idx in plot_cfg.selected_sequence_indices:
        sims_by_seq[seq_idx] = {}

        for cfg in configs:
            act_path = cfg.act_path
            if cfg.sae is not None:
                act_path = os.path.join(cfg.sae.name, act_path)
            art, _ = load_matching_activations(
                cfg, [act_path], cfg.env.activations_dir, compared_attributes=["llm", "data"], verbose=False
            )
            act_BLD = art[act_path]
            selected_LD = act_BLD[seq_idx]
            if plot_cfg.omit_bos_token:
                selected_LD = selected_LD[1:]
            selected_LD = selected_LD[plot_cfg.seq_start_idx : plot_cfg.seq_end_idx]
            selected_LD = selected_LD - selected_LD.mean(dim=0, keepdim=True)

            # Apply mask to only keep non-masked tokens
            mask = masks_by_seq[seq_idx]
            valid_indices = th.where(mask)[0]
            masked_LD = selected_LD[valid_indices]

            sims_by_seq[seq_idx][act_path] = pairwise_cosine_similarity(masked_LD).float()

    # Create and save plot
    plot_surprisal_similarity_comparison(sims_by_seq, tokens_by_seq, masks_by_seq, plot_cfg)


if __name__ == "__main__":
    main()