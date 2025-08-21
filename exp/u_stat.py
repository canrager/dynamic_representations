"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import math
import json
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm, trange

from src.project_config import (
    PLOTS_DIR,
    DEVICE,
    LLMConfig,
    DatasetConfig,
)
from src.exp_utils import (
    compute_or_load_llm_artifacts,
)
from src.model_utils import load_tokenizer, load_nnsight_model, load_sae


@dataclass
class Config:
    verbose: bool = True
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.float32
    save_artifacts: bool = True
    force_recompute: bool = False

    loaded_llm: Tuple = None

    # llm: LLMConfig = LLMConfig("openai-community/gpt2", 6, 100, None, force_recompute)
    llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 10, None, force_recompute)

    dataset: DatasetConfig = DatasetConfig("monology/pile-uncopyrighted", "text")

    num_total_stories: int = 500
    num_tokens_per_story: int = 250
    omit_BOS_token: bool = True
    selected_story_idxs: Tuple[int] = None

    exp_name: str = f"u-stat_{llm.name.split("/")[-1]}_{num_total_stories}N_{num_tokens_per_story}T"


def u_statistic(acts_BPD: th.Tensor, cfg):
    acts_BPD = acts_BPD.to(cfg.device)
    B, P, D = acts_BPD.shape
    id_P = th.zeros(P)

    # mean_D = acts_BPD.mean(dim=(0, 1))
    # acts_BPD = acts_BPD - mean_D
    acts_normalized_BPD = acts_BPD / acts_BPD.norm(dim=-1, keepdim=True)

    for p in range(P):
        X = acts_normalized_BPD[:, p, :]
        gram = X @ X.T
        fro2 = (gram**2).sum()

        id_P[p] = (B**2 - B) / (fro2 - B)

    return id_P


def phase_randomized_surrogate(X_BPD: th.Tensor) -> th.Tensor:
    """
    Phase-randomized surrogate per (B, D) series along time P.
    Preserves power spectrum per dim, randomizes phases -> stationary.

    Args:
        X_BPD: Tensor of shape (B, P, D)

    Returns:
        Phase-randomized surrogate with same shape
    """
    B, P, D = X_BPD.shape
    X_sur = th.empty_like(X_BPD)
    X_np = X_BPD.detach().cpu().numpy()

    for b in range(B):
        for d in range(D):
            x = X_np[b, :, d]
            fft_x = np.fft.rfft(x)
            mag = np.abs(fft_x)
            # random phases in [0, 2π), keep DC/Nyquist magnitudes
            rand_phase = np.exp(1j * np.random.uniform(0.0, 2 * np.pi, size=fft_x.shape))
            # ensure DC (0-freq) has zero phase
            rand_phase[0] = 1.0 + 0.0j
            fft_new = mag * rand_phase
            x_new = np.fft.irfft(fft_new, n=P)
            X_sur[b, :, d] = th.from_numpy(x_new).to(X_BPD)
    return X_sur


def plot_u_statistic(id_P: th.Tensor, cfg: Config):
    """
    Plot u-statistic intrinsic dimensionality across token positions.

    Args:
        id_P: U-statistic values of shape (P,)
        cfg: Configuration object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    positions = th.arange(len(id_P))
    ax.plot(positions, id_P.cpu(), linewidth=2, color="C0", marker="o", markersize=3)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    ax.set_title(f"U-Statistic Intrinsic Dimensionality\n{cfg.llm.name} Layer {cfg.llm.layer_idx}")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"u_statistic_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic plot saved to: {plot_path}")
    plt.show()


def plot_u_statistic_comparison(id_original_P: th.Tensor, id_surrogate_P: th.Tensor, cfg: Config):
    """
    Plot comparison of u-statistic intrinsic dimensionality for original vs surrogate data.
    Creates two subplots side by side.

    Args:
        id_original_P: U-statistic values for original data of shape (P,)
        id_surrogate_P: U-statistic values for surrogate data of shape (P,)
        cfg: Configuration object
    """
    fig, (ax_orig, ax_surr) = plt.subplots(1, 2, figsize=(12, 5))

    positions = th.arange(len(id_original_P))

    # Plot original data
    ax_orig.plot(positions, id_original_P.cpu(), linewidth=2, color="C0", marker="o", markersize=3)
    ax_orig.set_xlabel("Token Position")
    ax_orig.set_ylabel("U-Statistic")
    ax_orig.set_title("Original")
    ax_orig.grid(True, alpha=0.3)
    ax_orig.set_xscale("log")
    ax_orig.set_yscale("log")

    # Plot surrogate data
    ax_surr.plot(positions, id_surrogate_P.cpu(), linewidth=2, color="C1", marker="o", markersize=3)
    ax_surr.set_xlabel("Token Position")
    ax_surr.set_ylabel("U-Statistic")
    ax_surr.set_title("Surrogate")
    ax_surr.grid(True, alpha=0.3)
    ax_surr.set_xscale("log")
    ax_surr.set_yscale("log")

    # Overall title
    fig.suptitle(
        f"U-Statistic Comparison: Original vs Surrogate\n{cfg.llm.name} Layer {cfg.llm.layer_idx}",
        fontsize=14,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"u_statistic_comparison_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic comparison plot saved to: {plot_path}")
    plt.show()



if __name__ == "__main__":
    cfg = Config()

    # No need to load the model if working with precomputed activations
    # cfg.loaded_llm = load_nnsight_model(cfg.llm)

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(cfg)
    )
    llm_act_BPD = llm_act_LBPD[cfg.llm.layer_idx]

    print(f"Analyzing intrinsic dimensionality for {cfg.llm.name} layer {cfg.llm.layer_idx}")
    print(f"Activation tensor shape: {llm_act_BPD.shape}")

    # Generate surrogate data
    print("\nGenerating phase-randomized surrogate data...")
    llm_act_surrogate_BPD = phase_randomized_surrogate(llm_act_BPD)

    # Compute u-statistics for both original and surrogate data
    id_original_P = u_statistic(llm_act_BPD, cfg)
    id_surrogate_P = u_statistic(llm_act_surrogate_BPD, cfg)

    # Plot individual u-statistic
    plot_u_statistic(id_original_P, cfg)

    # Plot comparison (side by side)
    plot_u_statistic_comparison(id_original_P, id_surrogate_P, cfg)

    # Plot overlay (single subplot)
    plot_u_statistic_overlay(id_original_P, id_surrogate_P, cfg)
