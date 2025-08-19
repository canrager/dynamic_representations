"""
Compare the t-1 case from intrinsic_dimensionality.py with the PCA variance explained implementation.
"""

import torch as th
import matplotlib.pyplot as plt
import os
from src.project_config import PLOTS_DIR, DEVICE, LLMConfig, DatasetConfig
from src.exp_utils import compute_or_load_llm_artifacts, load_activation_split
from exp.pca_variance_explained import Config as PCAConfig, compute_intrinsic_dimension
from exp.intrinsic_dimensionality import Config as IDConfig, compute_id_pca
from src.model_utils import load_nnsight_model


def main():
    # Use configuration similar to pca_variance_explained.py
    pca_cfg = PCAConfig()
    pca_cfg.loaded_llm = load_nnsight_model(pca_cfg.llm)

    # Load activations using the pca_variance_explained approach
    (
        act_train_LBPD,
        act_test_LBPD,
        mask_train_BP,
        mask_test_BP,
        tokens_test_BP,
        num_test_stories,
        dataset_idxs_test,
    ) = load_activation_split(pca_cfg)

    # Compute intrinsic dimension using PCA variance explained approach (t-1 equivalent)
    # This uses training data up to position p-1 to predict position p
    id_mean_PT_pca, id_ci_PT_pca = compute_intrinsic_dimension(
        act_train_LBPD, act_train_LBPD, pca_cfg, window_size=None
    )

    # Compute intrinsic dimension using t-1 case from intrinsic_dimensionality.py
    act_BPD = act_train_LBPD[pca_cfg.llm.layer_idx]
    id_t_minus_1_BP = compute_id_pca(act_BPD, mode="t-1")

    # Compute mean across batches for comparison
    id_mean_P_t_minus_1 = th.mean(id_t_minus_1_BP, dim=0)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))

    P = min(len(id_mean_P_t_minus_1), id_mean_PT_pca.shape[0])
    positions = th.arange(P)

    # Plot PCA variance explained approach
    pca_values = id_mean_PT_pca[:P, 0].cpu()  # First threshold
    pca_ci = id_ci_PT_pca[:P, 0].cpu() if id_ci_PT_pca is not None else th.zeros_like(pca_values)

    # Filter out missing values for PCA approach
    valid_pca = pca_values >= 0
    pca_positions = positions[valid_pca]
    pca_values_valid = pca_values[valid_pca]
    pca_ci_valid = pca_ci[valid_pca]

    ax.plot(
        pca_positions,
        pca_values_valid,
        "b-",
        linewidth=2,
        label="PCA Variance Explained (train→test)",
        marker="o",
    )
    ax.fill_between(
        pca_positions,
        pca_values_valid - pca_ci_valid,
        pca_values_valid + pca_ci_valid,
        alpha=0.2,
        color="blue",
    )

    # Plot t-1 case approach
    t_minus_1_values = id_mean_P_t_minus_1[:P].cpu()

    # Filter out missing values for t-1 approach (position 0 is typically missing)
    valid_t_minus_1 = t_minus_1_values > 0
    t_minus_1_positions = positions[valid_t_minus_1]
    t_minus_1_values_valid = t_minus_1_values[valid_t_minus_1]

    ax.plot(
        t_minus_1_positions,
        t_minus_1_values_valid,
        "r-",
        linewidth=2,
        label="Intrinsic Dim t-1 case",
        marker="s",
    )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Number of PCA Components / Intrinsic Dimension")
    ax.set_title(
        f"Comparison: PCA Variance Explained vs t-1 Case\n{pca_cfg.llm.name} Layer {pca_cfg.llm.layer_idx}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_fname = f"compare_t_minus_1_pca"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved comparison plot to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
