"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import math
import json
from typing import Optional, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import dataclasses
import hashlib
import einops
from tqdm import tqdm, trange

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
    compute_centered_svd,
)
from src.model_utils import load_tokenizer, load_nnsight_model, load_sae

from dictionary_learning.dictionary import IdentityDict, AutoEncoder

try:
    import skdim
except ImportError:
    print("Warning: skdim not installed. Install with: pip install scikit-dimension")
    skdim = None


@dataclass
class Config:
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.float32
    save_artifacts: bool = True
    force_recompute: bool = False
    exp_name: str = "intrinsic_dimensionality2000"

    loaded_llm: Tuple = None

    llm: LLMConfig = LLMConfig("openai-community/gpt2", 6, 100, None, force_recompute)
    # llm: LLMConfig = LLMConfig("google/gemma-2-2b", 12, 100, None, force_recompute)
    # llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 10, None, force_recompute)

    dataset: DatasetConfig = DatasetConfig("monology/pile-uncopyrighted", "text")
    datasets: Tuple[DatasetConfig] = (
        DatasetConfig("monology/pile-uncopyrighted", "text"),
        DatasetConfig("SimpleStories/SimpleStories", "story"),
        DatasetConfig("NeelNanda/code-10k", "text"),
    )

    sae: SAEConfig = (
        SAEConfig(
            AutoEncoder,
            4096,
            100,
            "L1 ReLU saebench",
            "artifacts/trained_saes/Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_2/ae.pt",
            force_recompute=True,
        ),
    )

    num_total_stories: int = 2000
    selected_story_idxs: Optional[List[int]] = None
    omit_BOS_token: bool = True
    num_tokens_per_story: int = 500
    do_train_test_split: bool = False
    num_train_stories: int = 75

    # Intrinsic dimensionality specific parameters
    max_position_to_analyze: int = 100
    # Available methods: "fisher", "lpca", "pca"
    # Note: Fisher needs ≥20 samples, lPCA needs ≥10 samples per analysis
    # For trajectory-wise analysis, only PCA works reliably with small sample sizes
    id_methods: List[str] = dataclasses.field(default_factory=lambda: ["fisher", "lpca", "pca"])
    normalize: bool = True


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


def compute_intrinsic_dimensionality_fisher(act_train_BPD: th.Tensor) -> float:
    """
    Compute intrinsic dimensionality using Fisher Information Matrix.

    Args:
        activations_nD: Tensor of shape (n_samples, n_features)

    Returns:
        Estimated intrinsic dimensionality
    """

    # Convert to numpy for skdim
    B, P, D = act_train_BPD.shape

    # Select a subset of batches to make it faster
    # b = max_num_samples // P + 1
    # if b < B:
    #     print(f"only running Fisher ID evaluation on first {b} batches!")
    # act_train_BPD = act_train_BPD[:b]

    act_train_nD = act_train_BPD.reshape(-1, D)
    act_train_nD = act_train_nD.detach().cpu().numpy()

    # Compute intrinsic dimension using Fisher Information
    estimator = skdim.id.FisherS()
    id_test_n = estimator.fit_transform_pw(act_train_nD)
    id_test_BP = th.tensor(id_test_n).reshape(B, P)

    return id_test_BP


def compute_id_windowed_fisher(act_train_BPD: th.Tensor):
    B, P, D = act_train_BPD.shape
    id_test_BP = th.zeros(B, P)

    for p in trange(P, desc="iterating over windows"):
        act_train_BpD = act_train_BPD[:, : p + 1, :]
        windowed_id_BP = compute_intrinsic_dimensionality_fisher(act_train_BpD)
        id_test_BP[:, p] = windowed_id_BP[:, p]

    return id_test_BP


def compute_intrinsic_dimensionality_lpca(act_train_BPD: th.Tensor) -> float:
    """
    Compute intrinsic dimensionality using local PCA.

    Args:
        activations_nD: Tensor of shape (n_samples, n_features)

    Returns:
        Estimated intrinsic dimensionality
    """
    # Convert to numpy for skdim
    B, P, D = act_train_BPD.shape
    act_train_nD = act_train_BPD.reshape(-1, D)
    act_train_nD = act_train_nD.detach().cpu().numpy()

    # Compute intrinsic dimension using local PCA
    estimator = skdim.id.lPCA(ver="ratio", alphaRatio=0.9)
    id_test_n = estimator.fit_transform_pw(act_train_nD)
    id_test_BP = th.tensor(id_test_n).reshape(B, P)

    return id_test_BP


def compute_id_windowed_lpca(act_train_BPD: th.Tensor):
    B, P, D = act_train_BPD.shape
    id_test_BP = th.zeros(B, P)

    for p in trange(P, desc="iterating over windows"):
        act_train_BpD = act_train_BPD[:, : p + 1, :]
        windowed_id_BP = compute_intrinsic_dimensionality_lpca(act_train_BpD)
        id_test_BP[:, p] = windowed_id_BP[:, p]

    return id_test_BP


def compute_intrinsic_dimensionality_knn(act_train_BPD: th.Tensor, k: int = 10) -> th.Tensor:
    """
    Compute intrinsic dimensionality using k-nearest neighbors.

    Args:
        act_train_BPD: Tensor of shape (B, P, D)
        k: Number of nearest neighbors to use

    Returns:
        Estimated intrinsic dimensionality tensor of shape (B, P)
    """
    # Convert to numpy for skdim
    B, P, D = act_train_BPD.shape
    act_train_nD = act_train_BPD.reshape(-1, D)
    act_train_nD = act_train_nD.detach().cpu().numpy()

    # Compute intrinsic dimension using k-NN
    estimator = skdim.id.KNN(k=k)
    id_test_n = estimator.fit_transform_pw(act_train_nD)
    id_test_BP = th.tensor(id_test_n).reshape(B, P)

    return id_test_BP


def compute_id_windowed_knn(act_train_BPD: th.Tensor, k: int = 10):
    B, P, D = act_train_BPD.shape
    id_test_BP = th.zeros(B, P)

    for p in trange(P, desc="iterating over windows"):
        act_train_BpD = act_train_BPD[:, : p + 1, :]
        windowed_id_BP = compute_intrinsic_dimensionality_knn(act_train_BpD, k=k)
        id_test_BP[:, p] = windowed_id_BP[:, p]

    return id_test_BP


def knn_k_sweep(
    act_train_BPD: th.Tensor, k_values: List[int] = None, mode: str = "single"
) -> th.Tensor:
    """
    Wrapper around KNN that sweeps over k parameter values.

    Args:
        act_train_BPD: Tensor of shape (B, P, D)
        k_values: List of k values to sweep over. If None, uses [5, 10, 15, 20, 25]
        mode: Either "single" for single estimation or "windowed" for windowed estimation

    Returns:
        id_test_kBP: Tensor of shape (K, B, P) where K is the number of k values
    """
    if k_values is None:
        k_values = [5, 10, 15, 20, 25]

    B, P, D = act_train_BPD.shape
    K = len(k_values)
    id_test_kBP = th.zeros(K, B, P)

    for i, k in enumerate(tqdm(k_values, desc="Sweeping over k values")):
        if mode == "single":
            id_test_BP = compute_intrinsic_dimensionality_knn(act_train_BPD, k=k)
        elif mode == "windowed":
            id_test_BP = compute_id_windowed_knn(act_train_BPD, k=k)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single' or 'windowed'")

        id_test_kBP[i] = id_test_BP

    return id_test_kBP


def get_dictionary(mode: str, act_BPD: th.Tensor, cfg):
    match mode:
        case "sae_encoder":
            sae = load_sae(cfg)
            mean = None
            return sae.encoder, mean

        case "pca":
            _, _, D = act_BPD.shape
            act_nD = act_BPD.reshape(-1, D)
            n, D = act_nD.shape
            assert n > D, "We need at least D+1 samples to fit a PCA Basis"

            # Center the full dataset
            train_mean_D = th.mean(act_nD, dim=0, keepdim=True)
            X_train_centered = act_nD - train_mean_D

            # Compute SVD over the full dataset
            _, _, V = th.linalg.svd(X_train_centered, full_matrices=False)
            # Project each sample onto the PCA components
            # V is (D, min(B*P, D)) - transpose to get (min(B*P, D), D)
            V_T = V.T  # (D, n_components)
            return V_T, train_mean_D

        case _:
            raise ValueError("Unrecognized mode passed.")


def compute_intrinsic_dimensionality_pca(
    act_train_BPD: th.Tensor, act_test_BPD: th.Tensor, variance_threshold: float = 0.95
) -> th.Tensor:
    """
    Compute intrinsic dimensionality using PCA variance explained.

    Computes PCA over the full dataset and returns the minimum number of
    components needed to achieve variance threshold for each sample.

    Args:
        act_train_BPD: Tensor of shape (B, P, D) where B=batch, P=position, D=features
        variance_threshold: Threshold for cumulative variance explained

    Returns:
        Tensor of shape (B, P) with intrinsic dimensionalities for each sample
    """

    # Reshape to (B*P, D) for PCA computation
    _, _, D = act_train_BPD.shape
    act_nD = act_train_BPD.reshape(-1, D)
    n, D = act_nD.shape
    assert n > D, "We need at least D+1 samples to fit a PCA Basis"

    # Center the full dataset
    train_mean_D = th.mean(act_nD, dim=0, keepdim=True)
    X_train_centered = act_nD - train_mean_D

    # Compute SVD over the full dataset
    _, _, V = th.linalg.svd(X_train_centered, full_matrices=False)
    # Project each sample onto the PCA components
    # V is (D, min(B*P, D)) - transpose to get (min(B*P, D), D)
    V_T = V.T  # (D, n_components)

    B_test, P_test, D = act_test_BPD.shape
    X_test_centered = act_test_BPD - train_mean_D
    projected = X_test_centered @ V_T  # (B*P, n_components)

    # For each sample, compute variance in each PCA dimension
    sample_variances = projected**2  # (B*P, n_components)

    # Normalize by total variance for each sample
    total_variance_original = th.sum(X_test_centered**2, dim=-1, keepdim=True)
    # Avoid division by zero
    sample_variance_ratios = sample_variances / total_variance_original  # (B*P, n_components)
    sample_variance_ratios, _ = th.sort(sample_variance_ratios, dim=-1, descending=True)

    # Compute cumulative variance for each sample
    cumulative_sample_variance = th.cumsum(sample_variance_ratios, dim=-1)  # (B*P, n_components)

    # Find number of components needed for each sample
    # For each sample, find the first component where cumulative variance >= threshold
    n_components_per_sample = th.sum(cumulative_sample_variance < variance_threshold, dim=-1) + 1

    # Reshape back to (B, P)
    id_BP = n_components_per_sample.float().reshape(B_test, P_test)

    return id_BP


def compute_id_pca(act_BPD, mode: str):
    B, P, D = act_BPD.shape

    match mode:
        case "full":
            return compute_intrinsic_dimensionality_pca(act_train_BPD=act_BPD, act_test_BPD=act_BPD)

        case "t":
            id_test_BP = th.zeros(B, P)
            for p in range(P):
                id_out_B1 = compute_intrinsic_dimensionality_pca(
                    act_train_BPD=act_BPD[:, : p + 1], act_test_BPD=act_BPD[:, [p], :]
                )
                print(f"id_out_B1 {id_out_B1.shape}")
                id_test_BP[:, p] = id_out_B1[:, -1]
            return id_test_BP

        case "t-1":
            id_test_BP = th.zeros(B, P)
            for p in range(1, P):
                id_out_B1 = compute_intrinsic_dimensionality_pca(
                    act_train_BPD=act_BPD[:, :p], act_test_BPD=act_BPD[:, [p], :]
                )
                id_test_BP[:, p] = id_out_B1[:, -1]
            return id_test_BP

        case _:
            raise ValueError(f"Unknown mode: {mode}")


def plot_fisher_alpha_analysis(act_train_BPD: th.Tensor, cfg: Config, batch_idx: int = 0):
    """
    Plot Fisher ID estimates for different alpha values for a single trajectory.
    Computes Fisher ID pointwise over the entire dataset, then extracts one batch's trajectory.

    Args:
        act_train_BPD: Activation tensor of shape (B, P, D)
        cfg: Configuration object
        batch_idx: Which batch (trajectory) to analyze
    """
    B, P, D = act_train_BPD.shape

    # Define alpha values to test
    alphas = np.arange(0.6, 1.0, 0.2)

    # Reshape to (B*P, D) for pointwise computation
    act_train_nD = act_train_BPD.reshape(-1, D).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Generate colors for different alphas
    colors = cm.viridis(np.linspace(0, 1, len(alphas)))

    position_range = th.arange(P)

    for i, alpha in enumerate(alphas):
        try:
            # Create FisherS estimator with specific alpha
            estimator = skdim.id.FisherS(alphas=np.array([[alpha]]))

            # Compute pointwise Fisher ID over entire dataset
            fisher_ids_pw = estimator.fit_transform_pw(act_train_nD)

            # Reshape back to (B, P) and extract the specified batch
            fisher_ids_BP = th.tensor(fisher_ids_pw).reshape(B, P)
            trajectory = fisher_ids_BP[batch_idx].numpy()  # Shape: (P,)

            # Filter out NaN values for plotting
            valid_indices = [j for j, v in enumerate(trajectory) if not math.isnan(v)]
            valid_positions = [position_range[j] for j in valid_indices]
            valid_fisher_ids = [trajectory[j] for j in valid_indices]

            if valid_fisher_ids:
                ax.plot(
                    valid_positions,
                    valid_fisher_ids,
                    color=colors[i],
                    alpha=0.8,
                    linewidth=2,
                    marker="o",
                    markersize=3,
                    label=f"α={alpha:.2f}",
                )

        except Exception as e:
            print(f"Error with alpha {alpha}: {e}")
            continue

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Fisher Intrinsic Dimensionality")
    ax.set_title(
        f"Fisher ID vs Alpha for Trajectory {batch_idx}\n{cfg.llm.name} Layer {cfg.llm.layer_idx}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"fisher_alpha_analysis_{model_name_clean}_layer_{cfg.llm.layer_idx}_batch_{batch_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Fisher alpha analysis plot saved to: {plot_path}")
    plt.show()


def plot_trajectory_comparison(id_original_BP: th.Tensor, id_surrogate_BP: th.Tensor, cfg: Config):
    """
    Plot comparison of trajectories along position P for original vs surrogate data.
    Two subplots: left for original, right for surrogate.
    Each trajectory shown in a single color, mean trajectory in black.

    Args:
        id_original_BP: Original intrinsic dimensionality tensor of shape (B, P)
        id_surrogate_BP: Surrogate intrinsic dimensionality tensor of shape (B, P)
        cfg: Configuration object
    """
    B, P = id_original_BP.shape
    positions = th.arange(P)

    fig, (ax_orig, ax_surr) = plt.subplots(1, 2, figsize=(12, 5))

    # Generate colors for each trajectory
    colors = cm.tab10(np.linspace(0, 1, min(B, 10)))  # Use up to 10 colors

    # Plot original trajectories
    for b in range(B):
        traj = id_original_BP[b].detach().cpu().numpy()
        color = colors[b % len(colors)]
        ax_orig.plot(positions, traj, color=color, alpha=0.6, linewidth=1)

    # Plot original mean in black
    mean_orig = th.mean(id_original_BP.float(), dim=0).detach().cpu().numpy()
    ax_orig.plot(positions, mean_orig, color="black", linewidth=3, label="Mean")
    ax_orig.set_title("Original")
    ax_orig.set_xlabel("Token Position")
    ax_orig.set_ylabel("Intrinsic Dimensionality")
    ax_orig.grid(True, alpha=0.3)
    ax_orig.legend()

    # Plot surrogate trajectories
    for b in range(B):
        traj = id_surrogate_BP[b].detach().cpu().numpy()
        color = colors[b % len(colors)]
        ax_surr.plot(positions, traj, color=color, alpha=0.6, linewidth=1)

    # Plot surrogate mean in black
    mean_surr = th.mean(id_surrogate_BP.float(), dim=0).detach().cpu().numpy()
    ax_surr.plot(positions, mean_surr, color="black", linewidth=3, label="Mean")
    ax_surr.set_title("Surrogate")
    ax_surr.set_xlabel("Token Position")
    ax_surr.set_ylabel("Intrinsic Dimensionality")
    ax_surr.grid(True, alpha=0.3)
    ax_surr.legend()

    # Overall title
    fig.suptitle(
        f"Trajectory Comparison: Original vs Surrogate\n{cfg.llm.name} Layer {cfg.llm.layer_idx}",
        fontsize=14,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"trajectory_comparison_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Trajectory comparison plot saved to: {plot_path}")
    plt.show()


def plot_pca_full_original(act_original_BPD: th.Tensor, cfg: Config, step_size: int = 25):
    """
    Simple plotting function for just the PCA "full" case on original data.
    Shows mean trajectory with 95% confidence interval.

    Args:
        act_original_BPD: Original activation tensor of shape (B, P, D)
        cfg: Configuration object
        step_size: Only compute and plot every k-th position (default: 25)
    """
    B, P, _ = act_original_BPD.shape

    # Select positions to analyze (every step_size positions)
    positions_to_analyze = th.arange(0, P, step_size)
    if positions_to_analyze[-1] != P - 1:  # Always include the last position
        positions_to_analyze = th.cat([positions_to_analyze, th.tensor([P - 1])])

    # Create single subplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Compute intrinsic dimensionality for "full" case only at selected positions
    print(
        f"Computing PCA ID for case: full at {len(positions_to_analyze)} positions (step_size={step_size})"
    )

    # Subsample the test data to only include selected positions
    act_test_subsampled = act_original_BPD[:, positions_to_analyze, :]

    # Compute PCA ID only on subsampled positions
    id_selected = compute_intrinsic_dimensionality_pca(
        act_train_BPD=act_original_BPD, act_test_BPD=act_test_subsampled
    )

    positions_selected = positions_to_analyze

    # Compute mean and confidence interval
    mean_values = th.mean(id_selected.float(), dim=0).detach().cpu().numpy()
    std_values = th.std(id_selected.float(), dim=0).detach().cpu().numpy()
    ci_values = 1.96 * std_values / (B**0.5)  # 95% confidence interval

    # Plot mean line with markers
    ax.plot(
        positions_selected,
        mean_values,
        linewidth=2,
        color="C0",
        marker="o",
        markersize=4,
        label="Mean",
    )

    # Plot 95% CI band
    ax.fill_between(
        positions_selected,
        mean_values - ci_values,
        mean_values + ci_values,
        alpha=0.2,
        color="C0",
        label="95% CI",
    )

    ax.set_title(f"PCA Full - Original Data (step_size={step_size})")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Intrinsic Dimensionality")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Overall title
    fig.suptitle(
        f"PCA Intrinsic Dimensionality (Full)\n{cfg.llm.name} Layer {cfg.llm.layer_idx}",
        fontsize=14,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"pca_full_original_{model_name_clean}_layer_{cfg.llm.layer_idx}_step{step_size}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"PCA full original plot saved to: {plot_path}")
    plt.show()


def plot_pca_case_comparison(
    act_original_BPD: th.Tensor, act_surrogate_BPD: th.Tensor, cfg: Config
):
    """
    Plot comparison of PCA-based intrinsic dimensionality for all three cases
    ("full", "t", "t-1") for both original and surrogate data.
    Creates a 2x3 grid: rows for original/surrogate, columns for the three PCA cases.
    Each subplot shows individual trajectories and mean as in plot_trajectory_comparison.

    Args:
        act_original_BPD: Original activation tensor of shape (B, P, D)
        act_surrogate_BPD: Surrogate activation tensor of shape (B, P, D)
        cfg: Configuration object
    """
    B, P, D = act_original_BPD.shape
    positions = th.arange(P)

    # Define the three PCA cases
    pca_cases = ["full", "t", "t-1"]

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Generate colors for each trajectory
    colors = cm.tab10(np.linspace(0, 1, min(B, 10)))  # Use up to 10 colors

    # Compute intrinsic dimensionality for each case
    for col, case in enumerate(pca_cases):
        print(f"Computing PCA ID for case: {case}")

        # Compute for original data
        id_original_BP = compute_id_pca(act_original_BPD, case)

        # Compute for surrogate data
        id_surrogate_BP = compute_id_pca(act_surrogate_BPD, case)

        # Plot original data (top row)
        ax_orig = axes[0, col]
        for b in range(B):
            traj = id_original_BP[b].detach().cpu().numpy()
            color = colors[b % len(colors)]
            ax_orig.plot(positions, traj, color=color, alpha=0.6, linewidth=1)

        # Plot original mean in black
        mean_orig = th.mean(id_original_BP.float(), dim=0).detach().cpu().numpy()
        ax_orig.plot(positions, mean_orig, color="black", linewidth=3, label="Mean")
        ax_orig.set_title(f"Original - {case}")
        ax_orig.set_ylabel("Intrinsic Dimensionality")
        ax_orig.grid(True, alpha=0.3)
        if col == 0:  # Add legend only to first subplot
            ax_orig.legend()

        # Plot surrogate data (bottom row)
        ax_surr = axes[1, col]
        for b in range(B):
            traj = id_surrogate_BP[b].detach().cpu().numpy()
            color = colors[b % len(colors)]
            ax_surr.plot(positions, traj, color=color, alpha=0.6, linewidth=1)

        # Plot surrogate mean in black
        mean_surr = th.mean(id_surrogate_BP.float(), dim=0).detach().cpu().numpy()
        ax_surr.plot(positions, mean_surr, color="black", linewidth=3, label="Mean")
        ax_surr.set_title(f"Surrogate - {case}")
        ax_surr.set_xlabel("Token Position")
        ax_surr.set_ylabel("Intrinsic Dimensionality")
        ax_surr.grid(True, alpha=0.3)
        if col == 0:  # Add legend only to first subplot
            ax_surr.legend()

    # Overall title
    fig.suptitle(
        f"PCA Intrinsic Dimensionality Comparison: All Cases\n{cfg.llm.name} Layer {cfg.llm.layer_idx}",
        fontsize=16,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"pca_case_comparison_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"PCA case comparison plot saved to: {plot_path}")
    plt.show()


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


def plot_u_statistic_overlay(id_original_P: th.Tensor, id_surrogate_P: th.Tensor, cfg: Config):
    """
    Plot comparison of u-statistic intrinsic dimensionality for original vs surrogate data.
    Shows both curves in a single subplot with different colors.

    Args:
        id_original_P: U-statistic values for original data of shape (P,)
        id_surrogate_P: U-statistic values for surrogate data of shape (P,)
        cfg: Configuration object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    positions = th.arange(len(id_original_P))

    # Plot both curves on same axis
    ax.plot(positions, id_original_P.cpu(), linewidth=2, color="C0", marker="o", markersize=3, label="Original")
    ax.plot(positions, id_surrogate_P.cpu(), linewidth=2, color="C1", marker="s", markersize=3, label="Surrogate")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    ax.set_title(f"U-Statistic: Original vs Surrogate\n{cfg.llm.name} Layer {cfg.llm.layer_idx}")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"u_statistic_overlay_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"U-statistic overlay plot saved to: {plot_path}")
    plt.show()


def plot_intrinsic_dimensionality_results(
    results_original: dict, results_surrogate: dict, cfg: Config, analysis_type: str = "cumulative"
):
    """
    Plot intrinsic dimensionality results across positions with subplots for each method.
    Shows both original and surrogate data in 2 rows.

    Args:
        results_original: Dictionary containing ID results for original data
        results_surrogate: Dictionary containing ID results for surrogate data
        cfg: Configuration object
        analysis_type: Type of analysis ("cumulative" or "position_specific")
    """
    n_methods = len(results_original)
    fig, axes = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))

    # If only one method, need to reshape axes
    if n_methods == 1:
        axes = axes.reshape(2, 1)

    row_labels = ["Original", "Surrogate"]
    results_list = [results_original, results_surrogate]

    for row, (results, row_label) in enumerate(zip(results_list, row_labels)):
        for col, (method, data) in enumerate(results.items()):
            ax = axes[row, col]
            positions = data["positions"]
            individual_trajectories = data["individual_trajectories"]
            mean_values = data["mean_values"]

            n_trajectories = len(individual_trajectories)
            colors = cm.tab10(np.linspace(0, 1, n_trajectories))

            # Plot individual trajectories in different colors (only if they exist)
            if individual_trajectories:  # Check if we have individual trajectories
                for traj_idx, trajectory in enumerate(individual_trajectories):
                    # Filter out NaN values for this trajectory
                    valid_indices = [j for j, v in enumerate(trajectory) if not math.isnan(v)]
                    valid_positions = [positions[j] for j in valid_indices]
                    valid_values = [trajectory[j] for j in valid_indices]

                    if cfg.normalize and valid_values:
                        max_val = max(valid_values) if max(valid_values) > 0 else 1
                        valid_values = [v / max_val for v in valid_values]

                    if valid_values:
                        ax.plot(
                            valid_positions,
                            valid_values,
                            color=colors[traj_idx],
                            alpha=0.6,
                            linewidth=1,
                            label=(
                                f"Story {traj_idx+1}"
                                if traj_idx < 5 and row == 0 and col == 0
                                else None
                            ),
                        )
            else:
                # No individual trajectories available (e.g., for Fisher/lPCA methods)
                # Add a text note about this
                if row == 0 and col == 0:
                    ax.text(
                        0.02,
                        0.98,
                        "Mean only\n(insufficient samples\nfor trajectories)",
                        transform=ax.transAxes,
                        fontsize=8,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    )

            # Plot mean trajectory in black
            valid_mean_indices = [j for j, v in enumerate(mean_values) if not math.isnan(v)]
            valid_mean_positions = [positions[j] for j in valid_mean_indices]
            valid_mean_values = [mean_values[j] for j in valid_mean_indices]

            if cfg.normalize and valid_mean_values:
                max_val = max(valid_mean_values) if max(valid_mean_values) > 0 else 1
                valid_mean_values = [v / max_val for v in valid_mean_values]

            if valid_mean_values:
                ax.plot(
                    valid_mean_positions,
                    valid_mean_values,
                    color="black",
                    linewidth=3,
                    label="Mean" if row == 0 and col == 0 else None,
                    marker="o",
                    markersize=4,
                )

            # Customize subplot
            if row == 1:  # Only add x-label to bottom row
                ax.set_xlabel("Token Position")
            if col == 0:  # Only add y-label to leftmost column
                ylabel = (
                    "Normalized Intrinsic Dimensionality"
                    if cfg.normalize
                    else "Intrinsic Dimensionality"
                )
                ax.set_ylabel(ylabel)

            # Add method title only to top row
            if row == 0:
                ax.set_title(f"{method.upper()}")

            ax.grid(True, alpha=0.3)

            # Add row label on the left
            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    row_label,
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )

            # Add legend only to top-left subplot
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Keep only the mean and first few trajectories
                filtered_handles = []
                filtered_labels = []
                for h, l in zip(handles, labels):
                    if l == "Mean" or (l.startswith("Story") and len(filtered_labels) < 6):
                        filtered_handles.append(h)
                        filtered_labels.append(l)
                ax.legend(filtered_handles, filtered_labels, fontsize=8)

    # Overall title
    title_suffix = "Cumulative" if analysis_type == "cumulative" else "Position-Specific"
    fig.suptitle(
        f"Intrinsic Dimensionality vs Token Position ({title_suffix})\n{cfg.llm.name} Layer {cfg.llm.layer_idx}",
        fontsize=14,
        y=0.96,
    )

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_name_clean = cfg.llm.name.replace("/", "_")
    plot_path = os.path.join(
        PLOTS_DIR,
        f"intrinsic_dimensionality_{analysis_type}_{model_name_clean}_layer_{cfg.llm.layer_idx}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    cfg = Config()

    cfg.loaded_llm = load_nnsight_model(cfg.llm)

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(cfg)
    )
    llm_act_BPD = llm_act_LBPD[cfg.llm.layer_idx]

    print(f"Analyzing intrinsic dimensionality for {cfg.llm.name} layer {cfg.llm.layer_idx}")
    print(f"Activation tensor shape: {llm_act_BPD.shape}")

    # Generate surrogate data
    print("\nGenerating phase-randomized surrogate data...")
    llm_act_surrogate_BPD = phase_randomized_surrogate(llm_act_BPD)

    # plot_pca_full_original(llm_act_BPD, cfg)

    # Compute u-statistics for both original and surrogate data
    id_original_P = u_statistic(llm_act_BPD, cfg)
    id_surrogate_P = u_statistic(llm_act_surrogate_BPD, cfg)
    
    # Plot individual u-statistic
    plot_u_statistic(id_original_P, cfg)
    
    # Plot comparison (side by side)
    plot_u_statistic_comparison(id_original_P, id_surrogate_P, cfg)
    
    # Plot overlay (single subplot)
    plot_u_statistic_overlay(id_original_P, id_surrogate_P, cfg)
