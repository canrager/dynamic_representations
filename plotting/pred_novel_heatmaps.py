import json
import os
import torch as th
import matplotlib.pyplot as plt
import numpy as np

from src.plotting_utils import savefig


def load_data(act_dir, sae_type):
    """Load data from activation directory."""
    path = os.path.join(act_dir, sae_type, "pred_codes.pt")
    with open(path, "rb") as f:
        pred_BLD = th.load(f, weights_only=False)

    path = os.path.join(act_dir, sae_type, "novel_codes.pt")
    with open(path, "rb") as f:
        novel_BLD = th.load(f, weights_only=False)

    path = os.path.join(act_dir, "tokens.pt")
    with open(path, "rb") as f:
        tokens_BL = th.load(f, weights_only=False)

    path = os.path.join(act_dir, "masks.pt")
    with open(path, "rb") as f:
        masks_BL = th.load(f, weights_only=False)

    path = os.path.join(act_dir, "config.json")
    with open(path, "r") as f:
        cfg = json.load(f)

    return pred_BLD, novel_BLD, tokens_BL, masks_BL, cfg


def compute_metrics(data_LD):
    """Compute rank and L0 for data.

    Args:
        data_LD: torch tensor of shape (L, D)

    Returns:
        rank: effective rank of the data
        l0: L0 norm (number of non-zero entries)
        mean_l0_per_token: mean L0 per token position
    """
    # Compute rank using SVD
    U, S, V = th.linalg.svd(data_LD.float(), full_matrices=False)
    # Effective rank using entropy is preferred, since it minds the tail of the distribution
    S_normalized = S / S.sum()
    S_normalized = S_normalized[S_normalized > 0]
    rank = th.exp(-th.sum(S_normalized * th.log(S_normalized)))

    # S_normalized = S**2 / (S**2).sum()
    # S_cumsum = th.cumsum(S_normalized, dim=0)
    # rank = (S_cumsum < 0.99).sum().item() + 1

    # Compute L0 (number of non-zero entries)
    l0 = (data_LD != 0).sum().item()

    # Compute mean L0 per token position (average across L dimension)
    mean_l0_per_token = (data_LD != 0).sum(dim=1).float().mean().item()

    return rank, l0, mean_l0_per_token


def plot_heatmap_with_marginals(
    ax_main,
    ax_right,
    data_LD,
    sorted_indices,
    title,
    percentile_threshold,
    rank,
    l0,
    mean_l0,
    binarize=True,
    show_ylabel=True,
    show_xlabel=True,
):
    """Create heatmap with marginal density plot on y-axis.

    Args:
        data_LD: Data to plot (L, D)
        sorted_indices: Pre-computed sorting indices to apply
        percentile_threshold: The cumulative density threshold (e.g., 0.9 for top 90%)
        rank: Effective rank
        l0: Total L0 norm
        mean_l0: Mean L0 per token
        binarize: If True, plot binary (0/1), else plot actual activation values
        show_ylabel: If True, show y-axis label
        show_xlabel: If True, show x-axis label
    """
    # Apply sorting
    data_LD_sorted = data_LD[:, sorted_indices]

    if binarize:
        data_plot = (data_LD_sorted != 0).cpu().float().numpy()
        cmap = "binary_r"
    else:
        data_plot = data_LD_sorted.cpu().float().numpy()
        cmap = "viridis"

    # Calculate density over features (sum over time, normalize)
    if binarize:
        density_features = data_plot.sum(axis=0) / data_plot.sum()
    else:
        # For non-binary, use sum of activation values
        density_features = data_plot.sum(axis=0) / data_plot.sum()

    # Calculate cumulative density to find threshold line
    cumsum_density = np.cumsum(density_features)
    threshold_idx = np.argmax(cumsum_density >= percentile_threshold)

    # Main heatmap
    im = ax_main.imshow(data_plot.T, aspect="auto", cmap=cmap, interpolation="nearest")
    if show_xlabel:
        ax_main.set_xlabel("Sequence Position", fontsize=20)
    if show_ylabel:
        ax_main.set_ylabel("Sorted Dictionary Index", fontsize=20)
    ax_main.set_title(title, fontsize=24)
    ax_main.tick_params(axis='both', which='major', labelsize=16)
    if not show_xlabel:
        plt.setp(ax_main.get_xticklabels(), visible=False)

    # Draw threshold line on main heatmap
    ax_main.axhline(
        y=threshold_idx,
        color="red",
        linestyle="--",
        linewidth=2,
    )
    # Add invisible lines for stats in legend
    ax_main.plot([], [], " ", label=f"Effective Rank: {rank:.1f}")
    ax_main.plot([], [], " ", label=f"Mean Sparsity / Position: {mean_l0:.1f}")
    ax_main.legend(loc='lower right', fontsize=16)

    # Right marginal (density over features)
    ax_right.fill_betweenx(range(len(density_features)), density_features, alpha=0.5)
    ax_right.plot(density_features, range(len(density_features)), linewidth=1)
    ax_right.set_xlabel("Density", fontsize=18)
    ax_right.tick_params(axis='x', labelsize=16)
    # Remove y-axis tick labels for marginal plot only (not affecting main plot due to sharey)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Draw threshold line on marginal plot
    ax_right.axhline(y=threshold_idx, color="red", linestyle="--", linewidth=2)


def main():
    ACT_DIRS = [
        ("/home/can/dynamic_representations/artifacts/activations/20251030_013127", "SimpleStories"),
        ("/home/can/dynamic_representations/artifacts/activations/20251030_012850", "Webtext"),
        ("/home/can/dynamic_representations/artifacts/activations/20251030_013349", "Code"),
    ]
    # sae_type = "temporal"
    sae_type = "temporal_split"
    pred_top_density = 0.9
    novel_top_density = 0.1
    sort_by = "pred"  # "novel" or "pred"
    story_indices = list([4])  # Default: stories 1-10 (indices 1-10)
    binarize = True  # If True, plot binary (0/1), else plot actual activation values

    # Create figure with 3x2 subplots
    fig = plt.figure(figsize=(24, 24))

    # Process each dataset
    for row_idx, (ACT_DIR, dataset_name) in enumerate(ACT_DIRS):
        print(f"\nProcessing {dataset_name}...")

        # Load data
        pred_BLD, novel_BLD, tokens_BL, masks_BL, cfg = load_data(ACT_DIR, sae_type)

        # Select and aggregate multiple stories
        if isinstance(story_indices, int):
            story_indices_list = [story_indices]
        else:
            story_indices_list = story_indices

        pred_LD_list = []
        novel_LD_list = []

        for story_idx in story_indices_list:
            pred_LD_story = pred_BLD[story_idx]
            novel_LD_story = novel_BLD[story_idx]
            mask_L = masks_BL[story_idx].bool()

            # Apply mask
            pred_LD_list.append(pred_LD_story[mask_L])
            novel_LD_list.append(novel_LD_story[mask_L])

        # Concatenate along time dimension
        pred_LD = th.cat(pred_LD_list, dim=0)
        novel_LD = th.cat(novel_LD_list, dim=0)

        # Compute metrics
        pred_rank, pred_l0, pred_mean_l0 = compute_metrics(pred_LD)
        novel_rank, novel_l0, novel_mean_l0 = compute_metrics(novel_LD)

        print(f"pred_LD  - Rank: {pred_rank:.2f}, L0: {pred_l0}, Mean L0/token: {pred_mean_l0:.2f}")
        print(f"novel_LD - Rank: {novel_rank:.2f}, L0: {novel_l0}, Mean L0/token: {novel_mean_l0:.2f}")

        # Determine sorting based on sort_by parameter
        if sort_by == "novel":
            sum_over_L = novel_LD.sum(dim=0)
        elif sort_by == "pred":
            sum_over_L = pred_LD.sum(dim=0)
        else:
            raise ValueError(f"sort_by must be 'novel' or 'pred', got {sort_by}")

        sorted_indices = th.argsort(sum_over_L, descending=False)

        # Calculate vertical position for this row
        row_height = 0.30
        row_gap = 0.03
        row_bottom = 0.98 - (row_idx + 1) * row_height - row_idx * row_gap

        # Left subplot (pred_LD)
        gs_left = fig.add_gridspec(
            1, 2, left=0.05, right=0.46, bottom=row_bottom, top=row_bottom + row_height,
            width_ratios=[4, 1], hspace=0.05, wspace=0.05
        )
        ax_main_left = fig.add_subplot(gs_left[0, 0])
        ax_right_left = fig.add_subplot(gs_left[0, 1], sharey=ax_main_left)

        # Right subplot (novel_LD)
        gs_right = fig.add_gridspec(
            1, 2, left=0.54, right=0.95, bottom=row_bottom, top=row_bottom + row_height,
            width_ratios=[4, 1], hspace=0.05, wspace=0.05
        )
        ax_main_right = fig.add_subplot(gs_right[0, 0])
        ax_right_right = fig.add_subplot(gs_right[0, 1], sharey=ax_main_right)

        # Add dataset name to predictive code plot
        pred_title = f"Predictive Code - {dataset_name}"
        novel_title = f"Novel Code - {dataset_name}"

        # Only show x-axis labels for bottom row
        is_bottom_row = (row_idx == len(ACT_DIRS) - 1)

        # Plot pred_LD with shared sorting
        plot_heatmap_with_marginals(
            ax_main_left,
            ax_right_left,
            pred_LD,
            sorted_indices,
            pred_title,
            percentile_threshold=pred_top_density,
            rank=pred_rank,
            l0=pred_l0,
            mean_l0=pred_mean_l0,
            binarize=binarize,
            show_xlabel=is_bottom_row,
        )

        # Plot novel_LD with shared sorting (no ylabel for right subplot)
        plot_heatmap_with_marginals(
            ax_main_right,
            ax_right_right,
            novel_LD,
            sorted_indices,
            novel_title,
            percentile_threshold=novel_top_density,
            rank=novel_rank,
            l0=novel_l0,
            mean_l0=novel_mean_l0,
            binarize=binarize,
            show_ylabel=False,
            show_xlabel=is_bottom_row,
        )

    # Save figure
    savefig("pred_novel_heatmaps", suffix=".png")
    plt.show()


if __name__ == "__main__":
    main()
