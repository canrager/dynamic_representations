"""
Plot magnitude of activations for syntactic complexity phrasal verb variations
"""

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import json
from typing import Optional, List, Any
from collections import defaultdict
from dataclasses import dataclass
import hashlib

from src.project_config import INPUTS_DIR, PLOTS_DIR, MODELS_DIR, DEVICE
from src.exp_utils import (
    compute_or_load_llm_artifacts,
    compute_or_load_sae_artifacts,
)
from src.model_utils import load_tokenizer

from dictionary_learning.dictionary import IdentityDict, AutoEncoder


@dataclass(frozen=True)
class LLMConfig:
    name: str
    layer_idx: int
    batch_size: int
    revision: str | None
    force_recompute: bool


@dataclass(frozen=True)
class SAEConfig:
    dict_class: Any
    dict_size: int
    batch_size: int
    name: str
    local_filename: str | None
    force_recompute: bool


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    hf_text_identifier: str


@dataclass(frozen=True)
class Config:
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.float32
    save_artifacts: bool = False

    llm: LLMConfig = LLMConfig("google/gemma-2-2b", 12, 100, None, force_recompute=False)
    # llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 100, None, force_recompute=False)
    sae: SAEConfig = None
    saes = (
        SAEConfig(
            IdentityDict, 4096, 100, "Residual stream neurons", None, force_recompute=False
        ),  # This is the LLM residual stream baseline
        SAEConfig(
            AutoEncoder,
            4096,
            100,
            "L1 ReLU saebench",
            "artifacts/trained_saes/Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_2/ae.pt",
            force_recompute=True,
        ),
    )
    dataset = DatasetConfig("SimpleStories/SimpleStories", "story")
    # dataset = DatasetConfig("monology/pile-uncopyrighted", "text")
    # dataset = DatasetConfig("NeelNanda/code-10k", "text")

    num_total_stories: int = 100
    selected_story_idxs: Optional[List[int]] = None
    omit_BOS_token: bool = True
    num_tokens_per_story: int = 100
    do_train_test_split: bool = False
    num_train_stories: int = 75
    # force_recompute: bool = (
    #     False  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
    # )


def plot_sorted_distribution_per_sae(sae_act_results):
    """
    Plot mean and CI of sorted activation distributions per SAE architecture
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    color_idx = 0
    for result in sae_act_results:
        sorted_activations = result["sorted_sae_act_BPD"]  # Shape: [B, P, D]
        print(sorted_activations.shape)
        sae_name = result["cfg"].sae.name

        # Flatten over batch and position dimensions
        flattened_activations = sorted_activations.flatten(0, 1)  # Shape: [B*P, D]

        # Compute mean and std across samples
        mean_activations = flattened_activations.mean(dim=0)  # Shape: [D]
        std_activations = flattened_activations.std(dim=0)  # Shape: [D]

        # Compute 95% CI
        n_samples = flattened_activations.shape[0]
        ci_activations = std_activations # * 1.96 / (n_samples**0.5)

        # Create x-axis (component indices)
        x_indices = th.arange(len(mean_activations))

        # Get color
        color = f"C{color_idx}"

        # Plot mean line and CI band
        ax.plot(x_indices, mean_activations, linewidth=2, color=color, label=sae_name)
        ax.fill_between(
            x_indices,
            mean_activations - ci_activations,
            mean_activations + ci_activations,
            alpha=0.3,
            color=color,
            label="std dev",
        )

        color_idx += 1

    ax.set_xlim((-5, 200))
    # ax.set_yscale("log")

    ax.set_xlabel("Sorted Component Index", fontsize=14)
    ax.set_ylabel("Mean Activation Magnitude", fontsize=14)
    ax.set_title(
        f"Sorted Activation Magnitude Distribution - {result["cfg"].llm.name.split("/")[-1]} Layer {result["cfg"].llm.layer_idx}",
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.tight_layout()

    save_fname = "sorted_activation_magnitude_per_sae"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved sorted activation magnitude plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    global_cfg = Config()

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(global_cfg)
    )
    llm_act_BPD = llm_act_LBPD[global_cfg.llm.layer_idx]

    # Only Iterating over saes
    sae_act_results = []
    for sae_cfg in global_cfg.saes:
        current_config = Config(sae=sae_cfg)

        sae_artifact = compute_or_load_sae_artifacts(llm_act_BPD, current_config)
        sorted_sae_act_BPS, _ = sae_artifact.sae_act_BPS.sort(dim=-1, descending=True)
        # sorted_sae_act_BPS = sae_artifact.sae_act_BPS

        sae_act_results.append({"sorted_sae_act_BPD": sorted_sae_act_BPS, "cfg": current_config})

    # get mean, ci of sorted distribution
    plot_sorted_distribution_per_sae(sae_act_results)
