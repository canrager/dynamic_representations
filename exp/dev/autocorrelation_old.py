"""
Plot magnitude of activations for syntactic complexity phrasal verb variations
"""

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import json
from typing import Optional, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
import hashlib
import einops

from src.project_config import (
    INPUTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    DEVICE,
    LLMConfig,
    DatasetConfig,
    SAEConfig,
)
from src.exp_utils import (
    compute_or_load_llm_artifacts,
    compute_or_load_sae_artifacts,
)
from src.model_utils import load_tokenizer, load_nnsight_model

from dictionary_learning.dictionary import IdentityDict, AutoEncoder


@dataclass
class Config:
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.float32
    save_artifacts: bool = False
    force_recompute: bool = True # has to be true if plotting for multiple
    exp_name: str = "autocorr"

    loaded_llm: Tuple = None

    # llm: LLMConfig = LLMConfig("openai-community/gpt2", 6, 100, None, force_recompute)
    # llm: LLMConfig = LLMConfig("google/gemma-2-2b", 12, 100, None, force_recompute)
    llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 100, None, force_recompute)
    # sae: SAEConfig = None
    # saes = (
    #     SAEConfig(
    #         IdentityDict, 4096, 100, "Residual stream neurons", None, force_recompute=False
    #     ),  # This is the LLM residual stream baseline
    #     SAEConfig(
    #         AutoEncoder,
    #         4096,
    #         100,
    #         "L1 ReLU saebench",
    #         "artifacts/trained_saes/Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_2/ae.pt",
    #         force_recompute=True,
    #     ),
    # )
    # dataset = DatasetConfig("SimpleStories/SimpleStories", "story")
    # dataset = DatasetConfig("monology/pile-uncopyrighted", "text")
    dataset: DatasetConfig = None
    datasets: Tuple[DatasetConfig] = (
        DatasetConfig("monology/pile-uncopyrighted", "text"),
        DatasetConfig("SimpleStories/SimpleStories", "story"),
        DatasetConfig("NeelNanda/code-10k", "text"),
    )

    num_total_stories: int = 50
    selected_story_idxs: Optional[List[int]] = None
    omit_BOS_token: bool = True
    num_tokens_per_story: int = 200
    do_train_test_split: bool = False
    num_train_stories: int = 75


# ==== Surrogates & Whitening Utilities =====================================

import numpy as np


def phase_randomized_surrogate(X_BPD: th.Tensor) -> th.Tensor:
    """
    Phase-randomized surrogate per (B, D) series along time P.
    Preserves power spectrum per dim, randomizes phases -> stationary.
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


def compute_autocorr_heatmap(Y_BPD: th.Tensor) -> th.Tensor:
    """
    Y_BPD: [B, P, D] already whitened (and optionally unit-normalized per row).
    Returns mean cosine-similarity heatmap for specific (x,w) pairs where x in range(20, 201, 20)
    and w in range(x-20, x+1). Returns as 21x10 matrix (flipped axes).
    """
    _, P, _ = Y_BPD.shape
    x_values = list(range(20, min(201, P + 1), 20))

    # Create heatmap matrix: rows are w offsets from x-20 to x, columns are x values
    heatmap = th.full((15, len(x_values)), th.nan)  # 15 rows for w offsets 0 to 14

    for i, x in enumerate(x_values):
        if x >= P:
            break
        w_values = list(range(max(0, x - 20), min(P - 5, x - 5)))
        for w in w_values:
            # Compute cosine similarity between position x and position w
            cos_sim = th.cosine_similarity(Y_BPD[:, x, :], Y_BPD[:, w, :], dim=-1)
            mean_cos_sim = cos_sim.mean()
            # w offset from x-20
            w_offset = w - (x - 20)
            if 0 <= w_offset < 15:  # Updated bound check for 15 rows
                heatmap[w_offset, i] = mean_cos_sim

    return heatmap, x_values


def compute_heatmaps_for_dataset(cfg: Config, loaded_llm, dataset: DatasetConfig) -> tuple:
    """Compute autocorrelation heatmaps for a specific dataset."""
    # Create a temporary config with the specific dataset
    temp_cfg = Config(dataset=dataset, loaded_llm=loaded_llm)

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(temp_cfg)
    )
    llm_act_BPD = llm_act_LBPD[temp_cfg.llm.layer_idx]

    # Center and normalize
    llm_act_BPD -= llm_act_BPD.mean(dim=(0, 1), keepdim=True)
    llm_act_BPD /= llm_act_BPD.norm(dim=-1, keepdim=True)

    # Compute heatmaps
    heatmap_orig, x_values = compute_autocorr_heatmap(llm_act_BPD)
    X_phase_BPD = phase_randomized_surrogate(llm_act_BPD)
    heatmap_phase, _ = compute_autocorr_heatmap(X_phase_BPD)

    return heatmap_orig, heatmap_phase, x_values, dataset.name


def plot_single_dataset_heatmaps(heatmap_orig, heatmap_phase, x_values, dataset_name, cfg: Config):
    """Plot autocorrelation heatmaps for a single dataset (original old plotting code)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes = axes.ravel()

    titles = [
        "Original",
        "Phase-randomized surrogate",
    ]
    heatmaps = [heatmap_orig, heatmap_phase]

    # Find common vmin/vmax for all heatmaps
    all_values = []
    for heatmap in heatmaps:
        if heatmap is not None:
            heatmap_np = heatmap.detach().cpu().numpy()
            valid_values = heatmap_np[~np.isnan(heatmap_np)]
            all_values.extend(valid_values)

    if all_values:
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin, vmax = 0, 1

    ims = []
    for i, (ax, title, heatmap) in enumerate(zip(axes, titles, heatmaps)):
        if heatmap is not None:
            # Convert to numpy for plotting
            heatmap_np = heatmap.detach().cpu().numpy()

            # Create heatmap with reversed colormap and masked NaN values
            masked_heatmap = np.ma.masked_where(np.isnan(heatmap_np), heatmap_np)
            im = ax.imshow(
                masked_heatmap, cmap="Blues_r", aspect="auto", origin="lower", vmin=vmin, vmax=vmax
            )
            ims.append(im)
            ax.set_title(title, fontsize=13)

            # Set x-axis labels (x values)
            ax.set_xlabel("x (Position)")
            x_ticks = list(range(len(x_values)))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_values)

            # Set y-axis labels (w offset from x-20 to x)
            ax.set_ylabel("w offset from (x-20)")
            y_ticks = [0, 5, 10, 14]  # Adjusted for 15-row heatmap
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f"{w-20}" for w in y_ticks])
        else:
            ax.text(
                0.5,
                0.5,
                f"{title}\n(Not computed)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # Add single shared colorbar
    if ims:
        cbar = fig.colorbar(ims[0], ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Cosine Similarity")

    # plt.tight_layout()

    save_fname = f"autocorr_heatmaps_single_{cfg.exp_name}"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.pdf")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved single dataset autocorrelation heatmaps to {fig_path}")
    plt.close()


def plot_autocorr_comparison(all_results: list, cfg: Config):
    """Plot autocorrelation heatmaps for all datasets in a 3x2 grid."""
    n_datasets = len(all_results)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(10, 4 * n_datasets))

    if n_datasets == 1:
        axes = axes.reshape(1, -1)

    for row, (heatmap_orig, heatmap_phase, x_values, dataset_name) in enumerate(all_results):
        heatmaps = [heatmap_orig, heatmap_phase]
        titles = ["Original", "Phase-randomized surrogate"]

        # Find vmin/vmax for this row
        all_values = []
        for heatmap in heatmaps:
            if heatmap is not None:
                heatmap_np = heatmap.detach().cpu().numpy()
                valid_values = heatmap_np[~np.isnan(heatmap_np)]
                all_values.extend(valid_values)

        vmin, vmax = (np.min(all_values), np.max(all_values)) if all_values else (0, 1)

        ims = []
        for col, (heatmap, title) in enumerate(zip(heatmaps, titles)):
            ax = axes[row, col]

            if heatmap is not None:
                heatmap_np = heatmap.detach().cpu().numpy()
                masked_heatmap = np.ma.masked_where(np.isnan(heatmap_np), heatmap_np)
                im = ax.imshow(
                    masked_heatmap,
                    cmap="Blues_r",
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
                ims.append(im)

                if row == 0:  # Only add title to top row
                    ax.set_title(title, fontsize=13)

                # Set x-axis labels
                ax.set_xlabel("x (Position)")
                x_ticks = list(range(len(x_values)))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_values)

                # Set y-axis labels
                ax.set_ylabel("w offset from (x-20)")
                y_ticks = [0, 5, 10, 14]
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f"{w-20}" for w in y_ticks])

            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}\n(Not computed)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

        # Add row-specific colorbar
        if ims:
            cbar = fig.colorbar(ims[0], ax=axes[row, :].tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Cosine Similarity")

        # Add dataset label on the left
        axes[row, 0].text(
            -0.3,
            0.5,
            dataset_name,
            rotation=90,
            va="center",
            ha="center",
            transform=axes[row, 0].transAxes,
            fontsize=12,
            fontweight="bold",
        )

    save_fname = f"autocorr_all_datasets_{cfg.exp_name}_{cfg.llm.name.split("/")[-1]}"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.pdf")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved autocorrelation comparison to {fig_path}")
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    loaded_llm = load_nnsight_model(cfg.llm)

    # Compute heatmaps for all datasets
    all_results = []
    for i, dataset in enumerate(cfg.datasets):

        result = compute_heatmaps_for_dataset(cfg, loaded_llm, dataset)
        all_results.append(result)

        # additionally plot fist dataset in isolation
        if i==0:
            plot_single_dataset_heatmaps(*result, cfg)
            


    # Plot all results in a comparison figure

    plot_autocorr_comparison(all_results, cfg)
