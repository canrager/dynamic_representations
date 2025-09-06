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
    compute_or_load_surrogate_artifacts
)
from src.model_utils import load_tokenizer, load_nnsight_model, load_sae


@dataclass
class Config:
    verbose: bool = True
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.bfloat16
    save_artifacts: bool = True
    force_recompute: bool = False
    do_normalize: bool = False

    loaded_llm: Tuple = None

    # llm: LLMConfig = LLMConfig("openai-community/gpt2", 6, 100, None, force_recompute)
    llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 10, None, force_recompute)

    dataset: DatasetConfig = DatasetConfig("monology/pile-uncopyrighted", "text")
    # dataset: DatasetConfig = None
    datasets: Tuple[DatasetConfig] = (
        DatasetConfig("SimpleStories/SimpleStories", "story"),
        DatasetConfig("monology/pile-uncopyrighted", "text"),
        DatasetConfig("NeelNanda/code-10k", "text"),
    )

    num_total_stories: int = 500
    num_tokens_per_story: int = 500
    p_start: int = 10
    p_end: int = num_tokens_per_story
    num_p: int = 10
    do_log_p: int = True
    omit_BOS_token: bool = False

    selected_story_idxs: Tuple[int] = None

    @property
    def p_indices(self):
        """Compute p_indices for subsampling based on configuration."""
        if self.p_end is None:
            p_end = self.num_tokens_per_story
        else:
            p_end = self.p_end
            
        if self.do_log_p:
            p_indices = th.logspace(
                th.log10(th.tensor(float(max(1, self.p_start)))),
                th.log10(th.tensor(float(p_end - 1))),
                steps=self.num_p,
            ).long()
        else:
            p_indices = th.linspace(self.p_start, p_end - 1, self.num_p).long()
        
        return p_indices

    @property
    def exp_name(self):
        return f"{self.llm.name.split('/')[-1]}" \
               f"_{self.dataset.name.split('/')[-1]}" \
               f"_{self.num_total_stories}N" \
               f"_{self.num_tokens_per_story}T" \
               f"_{self.do_normalize}"


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
        f"u_statistic_{model_name_clean}_layer_{cfg.llm.layer_idx}.pdf",
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
        f"u_statistic_comparison_{model_name_clean}_layer_{cfg.llm.layer_idx}.pdf",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic comparison plot saved to: {plot_path}")
    plt.show()


def plot_u_statistic_overlay(id_original_P: th.Tensor, id_surrogate_P: th.Tensor, cfg: Config, p_indices: th.Tensor = None):
    """
    Plot comparison of u-statistic intrinsic dimensionality for original vs surrogate data.
    Shows both curves in a single subplot with different colors.

    Args:
        id_original_P: U-statistic values for original data of shape (P,)
        id_surrogate_P: U-statistic values for surrogate data of shape (P,)
        cfg: Configuration object
        p_indices: Position indices used for subsampling (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if p_indices is not None:
        # Use provided p_indices for x-axis positions
        positions_plot = p_indices
        id_original_plot = id_original_P
        id_surrogate_plot = id_surrogate_P
    else:
        # Fallback to original logic for backward compatibility
        positions = th.arange(len(id_original_P))
        max_datapoints = len(id_original_P)

        # Filter data to xlim boundaries [30:max_datapoints]
        mask = (positions >= 30) & (positions < max_datapoints)
        positions_filtered = positions[mask]
        id_original_filtered = id_original_P[mask]
        id_surrogate_filtered = id_surrogate_P[mask]

        # Create log-spaced indices within the filtered range
        if len(positions_filtered) > 0:
            log_indices = th.logspace(
                th.log10(th.tensor(30.0)),
                th.log10(th.tensor(float(max_datapoints - 1))),
                steps=min(50, len(positions_filtered)),
            ).long()

            # Ensure indices are within bounds and filter to available positions
            valid_mask = log_indices < len(id_original_P)
            log_indices = log_indices[valid_mask]
            boundary_mask = log_indices >= 30
            log_indices = log_indices[boundary_mask]

            positions_plot = log_indices
            id_original_plot = id_original_P[log_indices]
            id_surrogate_plot = id_surrogate_P[log_indices]

    # Plot both curves on same axis
    ax.plot(
        positions_plot,
        id_original_plot.cpu(),
        linewidth=5,
        color="#F63642",
        label="Original",
    )
    ax.plot(
        positions_plot,
        id_surrogate_plot.cpu(),
        linewidth=5,
        color="#000000",
        label="Surrogate",
    )

    # Set xlim and ylim based on plotted data
    if p_indices is not None:
        ax.set_xlim(positions_plot.min().item(), positions_plot.max().item())
    else:
        ax.set_xlim(30, max_datapoints)

    # Adapt ylim to plotted data range
    all_values = th.cat([id_original_plot, id_surrogate_plot])
    y_min, y_max = all_values.min().item(), all_values.max().item()
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    ax.set_title(f"U-Statistic: Original vs Surrogate\n{cfg.llm.name} Layer {cfg.llm.layer_idx}")
    # ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"u_statistic_overlay_{model_name_clean}_layer_{cfg.llm.layer_idx}.pdf",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic overlay plot saved to: {plot_path}")
    plt.show()

def main_single_dataset():
    cfg = Config()

    # No need to load the model if working with precomputed activations
    cfg.loaded_llm = load_nnsight_model(cfg.llm)

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs, p_indices = (
        compute_or_load_llm_artifacts(cfg, p_indices=cfg.p_indices)
    )
    llm_act_BPD = llm_act_LBPD[cfg.llm.layer_idx]

    print(f"Analyzing intrinsic dimensionality for {cfg.llm.name} layer {cfg.llm.layer_idx}")
    print(f"Activation tensor shape: {llm_act_BPD.shape}")

    # Generate surrogate data
    print("\nGenerating phase-randomized surrogate data...")
    llm_act_surrogate_BPD = compute_or_load_surrogate_artifacts(cfg, llm_act_BPD)

    # Compute u-statistics for both original and surrogate data
    id_original_P = u_statistic(llm_act_BPD, cfg)
    id_surrogate_P = u_statistic(llm_act_surrogate_BPD, cfg)

    # Plot individual u-statistic
    plot_u_statistic(id_original_P, cfg)

    # Plot comparison (side by side)
    plot_u_statistic_comparison(id_original_P, id_surrogate_P, cfg)

    # Plot overlay (single subplot)
    plot_u_statistic_overlay(id_original_P, id_surrogate_P, cfg, p_indices)

def main_multiple_datasets():
    global_cfg = Config()

    # No need to load the model if working with precomputed activations
    loaded_llm = load_nnsight_model(global_cfg.llm)

    results = {}
    for dataset_cfg in global_cfg.datasets:
        current_cfg = Config(dataset=dataset_cfg, loaded_llm=loaded_llm)

        llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs, p_indices = (
            compute_or_load_llm_artifacts(current_cfg, p_indices=current_cfg.p_indices)
        )
        llm_act_BPD = llm_act_LBPD[current_cfg.llm.layer_idx]

        print(f"Analyzing intrinsic dimensionality for {current_cfg.llm.name} layer {current_cfg.llm.layer_idx}")
        print(f"Activation tensor shape: {llm_act_BPD.shape}")

        # Generate surrogate data
        print("\nGenerating phase-randomized surrogate data...")
        llm_act_surrogate_BPD = compute

        # Compute u-statistics for both original and surrogate data
        id_original_P = u_statistic(llm_act_BPD, current_cfg)
        id_surrogate_P = u_statistic(llm_act_surrogate_BPD, current_cfg)

        results[current_cfg.dataset.name] = {
            "cfg": current_cfg,
            "id_orig_P": id_original_P,
            "id_surr_P": id_surrogate_P,
            "p_indices": p_indices
        }
    
    # Use p_indices from the first dataset (they should be the same across datasets)
    first_p_indices = next(iter(results.values()))["p_indices"]
    plot_u_statistic_overlay_multiple_datasets(results, first_p_indices)


def plot_u_statistic_overlay_multiple_datasets(results: dict, p_indices: th.Tensor = None):
    """
    Plot comparison of u-statistic intrinsic dimensionality for multiple datasets.
    Creates n_dataset subplots horizontally aligned, each showing original vs surrogate overlay.

    Args:
        results: Dictionary with dataset configs as keys and dict containing 
                'id_orig_P' and 'id_surr_P' tensors as values
        p_indices: Position indices used for subsampling (optional)
    """
    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (dataset_name, data) in enumerate(results.items()):
        ax = axes[idx]
        cfg = data["cfg"]
        id_original_P = data["id_orig_P"]
        id_surrogate_P = data["id_surr_P"]
        
        if p_indices is not None:
            # Use provided p_indices for x-axis positions
            positions_plot = p_indices
            id_original_plot = id_original_P
            id_surrogate_plot = id_surrogate_P
        else:
            # Fallback to original logic for backward compatibility
            positions = th.arange(len(id_original_P))
            max_datapoints = len(id_original_P)

            # Filter data to xlim boundaries [30:max_datapoints]
            mask = (positions >= 30) & (positions < max_datapoints)
            positions_filtered = positions[mask]
            
            if len(positions_filtered) > 0:
                # Create log-spaced indices within the filtered range
                log_indices = th.logspace(
                    th.log10(th.tensor(30.0)),
                    th.log10(th.tensor(float(max_datapoints - 1))),
                    steps=min(50, len(positions_filtered)),
                ).long()

                # Ensure indices are within bounds and filter to available positions
                valid_mask = log_indices < len(id_original_P)
                log_indices = log_indices[valid_mask]
                boundary_mask = log_indices >= 30
                log_indices = log_indices[boundary_mask]

                positions_plot = log_indices
                id_original_plot = id_original_P[log_indices]
                id_surrogate_plot = id_surrogate_P[log_indices]

        # Plot both curves on same axis
        ax.plot(
            positions_plot,
            id_original_plot.cpu(),
            linewidth=2,
            color="C0",
            marker="o",
            markersize=3,
            label="Original",
        )
        ax.plot(
            positions_plot,
            id_surrogate_plot.cpu(),
            linewidth=2,
            color="C1",
            marker="s",
            markersize=3,
            label="Surrogate",
        )

        # Set xlim and ylim based on plotted data
        if p_indices is not None:
            ax.set_xlim(positions_plot.min().item(), positions_plot.max().item())
        else:
            ax.set_xlim(30, max_datapoints)

        # Adapt ylim to plotted data range
        all_values = th.cat([id_original_plot, id_surrogate_plot])
        y_min, y_max = all_values.min().item(), all_values.max().item()
        ax.set_ylim(y_min * 0.9, y_max * 1.1)

        ax.set_xlabel("Token Position")
        if idx == 0:  # Only leftmost subplot gets y-label
            ax.set_ylabel("U-Statistic")
        
        # Use dataset name for subplot title
        dataset_title = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        ax.set_title(f"{dataset_title}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()

    # Overall title
    first_data = next(iter(results.values()))
    first_cfg = first_data["cfg"]
    fig.suptitle(
        f"U-Statistic: Original vs Surrogate (Multiple Datasets)\\n{first_cfg.llm.name} Layer {first_cfg.llm.layer_idx}",
        fontsize=14,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = first_cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"u_statistic_overlay_multiple_datasets_{model_name_clean}_layer_{first_cfg.llm.layer_idx}.pdf",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic multiple datasets overlay plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    # main_multiple_datasets()
    main_single_dataset()
 