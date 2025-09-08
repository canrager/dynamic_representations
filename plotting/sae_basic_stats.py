"""
Plot SAE basic stats for precomputed eval results for each trainer in artifacts/tranied_saes_selection
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class PlotConfig:
    metric_names: list
    plots_dir: str
    trained_sae_dir: str
    omit_bos: bool
    logscale_x: bool
    num_points: int | None

def load_sae_statistics(cfg: PlotConfig):
    all_paths = os.listdir(cfg.trained_sae_dir)
    all_results = {}
    for path in all_paths:
        name = path.split("/")[-1]
        eval_results_path = os.path.join(cfg.trained_sae_dir, path, "eval_results.json")
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

        all_results[name] = eval_results["mean_per_position"]

    return all_results

def plot_per_token_metrics(
    eval_results: dict,
    cfg: PlotConfig
):
    """
    Plot per-token metrics as a row of subplots with position on x-axis and metric on y-axis.
    Shows mean values connected with lines and confidence intervals as shaded areas for all trainers.

    Args:
        eval_results: Dictionary containing evaluation results for different trainers
        save_dir: Directory where the plot should be saved
        omit_bos: Whether to omit the first token (BOS) from the plots
        logscale_x: Whether to plot x-axis on logarithmic scale
        num_points: Maximum number of points to plot (selected equidistantly). If None, plot all points.
    """

    all_trainer_keys = list(eval_results.keys())

    # Create subplots
    n_metrics = len(cfg.metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    # Define colors for different trainers
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trainer_keys)))
    
    for i, metric_name in enumerate(cfg.metric_names):
        
        for trainer_idx, trainer_key in enumerate(all_trainer_keys):
            per_pos_data = eval_results[trainer_key]
            
            mean_key = f"{metric_name}_mean_pos"
            ci_key = f"{metric_name}_ci_pos"

            if mean_key not in per_pos_data or ci_key not in per_pos_data:
                print(f"Missing data for metric {metric_name} in trainer {trainer_key}")
                continue

            mean_values = per_pos_data[mean_key]
            ci_values = per_pos_data[ci_key]

            # Apply omit_bos if requested
            if cfg.omit_bos and len(mean_values) > 1:
                mean_values = mean_values[1:]
                ci_values = ci_values[1:]
                positions = list(range(1, len(mean_values) + 1))
            else:
                positions = list(range(len(mean_values)))

            # Select equidistant points if num_points is specified
            if cfg.num_points is not None and len(positions) > cfg.num_points:
                if cfg.logscale_x and len(positions) > 1:
                    # For logscale, select points geometrically spaced
                    log_positions = np.logspace(
                        np.log10(positions[0] if positions[0] > 0 else 1),
                        np.log10(positions[-1]),
                        cfg.num_points,
                    )
                    # Find closest actual indices
                    selected_indices = []
                    for log_pos in log_positions:
                        closest_idx = min(
                            range(len(positions)), key=lambda j: abs(positions[j] - log_pos)
                        )
                        if closest_idx not in selected_indices:
                            selected_indices.append(closest_idx)
                    selected_indices = sorted(selected_indices)
                else:
                    # For linear scale, select linearly spaced points
                    selected_indices = np.linspace(0, len(positions) - 1, cfg.num_points, dtype=int)
                    selected_indices = sorted(list(set(selected_indices)))  # Remove duplicates and sort

                positions = [positions[idx] for idx in selected_indices]
                mean_values = [mean_values[idx] for idx in selected_indices]
                ci_values = [ci_values[idx] for idx in selected_indices]

            # Plot mean line with dots
            axes[i].plot(positions, mean_values, "o-", linewidth=2, markersize=4, 
                        color=colors[trainer_idx], label=trainer_key)

            # Plot confidence interval as shaded area
            lower_bound = [m - c for m, c in zip(mean_values, ci_values)]
            upper_bound = [m + c for m, c in zip(mean_values, ci_values)]
            axes[i].fill_between(positions, lower_bound, upper_bound, alpha=0.3, 
                               color=colors[trainer_idx])

        # Set logscale if requested
        if cfg.logscale_x:
            axes[i].set_xscale("log")

        # Set labels and title
        axes[i].set_xlabel("Position")
        axes[i].set_ylabel(metric_name.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
        
        # Add legend to the first subplot
        if i == 0:
            axes[i].legend()

    # Add main title with trainer name and plotting settings

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(cfg.plots_dir, "selected_saes_basic_metrics_overlay.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Per-token metrics plot saved to: {plot_path}")
    return plot_path


if __name__ == "__main__":
    cfg = PlotConfig(
        metric_names = ["l0", "normalized_mse", "cosine_similarity"],
        plots_dir="artifacts/plots",
        trained_sae_dir = "artifacts/trained_saes_selection",
        omit_bos = True,
        logscale_x = True,
        num_points = 25,
    )

    all_results = load_sae_statistics(cfg)

    plot_per_token_metrics(all_results, cfg)
