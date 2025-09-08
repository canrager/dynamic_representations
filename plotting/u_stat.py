"""
Plot u-statistic intrinsic dimensionality results computed by exp/u_stat.py
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.project_config import PLOTS_DIR


@dataclass
class UStatPlotConfig:
    plots_dir: str
    results_dir: str
    figsize: Tuple[int, int] = (8, 6)
    linewidth: int = 2
    markersize: int = 3
    alpha: float = 0.3
    
    # Colors for original vs surrogate
    original_color: str = "#F63642"
    surrogate_color: str = "#000000"
    
    # Multiple dataset colors (fallback to tab10 if more datasets than colors)
    dataset_colors: List[str] = None
    
    def __post_init__(self):
        if self.dataset_colors is None:
            self.dataset_colors = ["C0", "C1", "C2", "C3", "C4"]


def load_u_statistic_results(results_dir: str) -> Dict:
    """
    Load u-statistic results from JSON files in the results directory.
    
    Args:
        results_dir: Directory containing u_stat_llm.json files
        
    Returns:
        Dictionary mapping config identifiers to result data
    """
    results = {}
    
    # Look for u_stat_llm.json files in subdirectories
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path):
                json_path = os.path.join(item_path, "u_stat_llm.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    results[item] = data
            elif item.endswith("u_stat_llm.json"):
                # Direct JSON file in results_dir
                with open(item_path, "r") as f:
                    data = json.load(f)
                # Use filename without extension as key
                key = os.path.splitext(item)[0]
                results[key] = data
    
    return results


def plot_u_statistic_overlay(
    results: Dict,
    cfg: UStatPlotConfig,
    save_name: str = "u_statistic_overlay.pdf"
) -> str:
    """
    Plot u-statistic overlay for original vs surrogate data from a single result.
    
    Args:
        results: Dictionary containing single result with u-statistic data
        cfg: Plot configuration
        save_name: Name for saved plot file
        
    Returns:
        Path to saved plot
    """
    # Get the first (and presumably only) result
    result_key = list(results.keys())[0]
    data = results[result_key]
    
    config = data["config"]
    result_data = data["results"]
    
    # Extract data
    p_indices = th.tensor(result_data["p_indices"])
    ustat_act_P = th.tensor(result_data["ustat_act_P"])
    ustat_surr_P = th.tensor(result_data["ustat_surr_P"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=cfg.figsize)
    
    # Plot both curves
    ax.plot(
        p_indices,
        ustat_act_P,
        linewidth=cfg.linewidth,
        color=cfg.original_color,
        marker="o",
        markersize=cfg.markersize,
        label="Original",
    )
    ax.plot(
        p_indices,
        ustat_surr_P,
        linewidth=cfg.linewidth,
        color=cfg.surrogate_color,
        marker="s",
        markersize=cfg.markersize,
        label="Surrogate",
    )
    
    # Set scales and limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(p_indices.min().item(), p_indices.max().item())
    
    # Adapt ylim to plotted data range
    all_values = th.cat([ustat_act_P, ustat_surr_P])
    y_min, y_max = all_values.min().item(), all_values.max().item()
    ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    # Labels and title
    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    
    llm_name = config["llm_name"].split("/")[-1] if "/" in config["llm_name"] else config["llm_name"]
    dataset_name = config["dataset_name"].split("/")[-1] if "/" in config["dataset_name"] else config["dataset_name"]
    
    ax.set_title(f"U-Statistic: Original vs Surrogate\n{llm_name} Layer {config['layer_idx']} | {dataset_name}")
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(cfg.plots_dir, exist_ok=True)
    plot_path = os.path.join(cfg.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"U-statistic overlay plot saved to: {plot_path}")
    return plot_path


def plot_u_statistic_overlay_multiple_datasets(
    results: Dict,
    cfg: UStatPlotConfig,
    save_name: str = "u_statistic_overlay_multiple_datasets.pdf"
) -> str:
    """
    Plot u-statistic overlay for multiple datasets showing original vs surrogate for each.
    Creates n_dataset subplots horizontally aligned.
    
    Args:
        results: Dictionary with dataset configs as keys and result data as values
        cfg: Plot configuration
        save_name: Name for saved plot file
        
    Returns:
        Path to saved plot
    """
    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))
    
    # Handle single dataset case
    if n_datasets == 1:
        axes = [axes]
    
    # Get colors for datasets
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets)) if n_datasets > len(cfg.dataset_colors) else cfg.dataset_colors
    
    for idx, (dataset_key, data) in enumerate(results.items()):
        ax = axes[idx]
        config = data["config"]
        result_data = data["results"]
        
        # Extract data
        p_indices = th.tensor(result_data["p_indices"])
        ustat_act_P = th.tensor(result_data["ustat_act_P"])
        ustat_surr_P = th.tensor(result_data["ustat_surr_P"])
        
        # Plot both curves on same axis
        ax.plot(
            p_indices,
            ustat_act_P,
            linewidth=cfg.linewidth,
            color=colors[0] if isinstance(colors[0], str) else colors[0],
            marker="o",
            markersize=cfg.markersize,
            label="Original",
        )
        ax.plot(
            p_indices,
            ustat_surr_P,
            linewidth=cfg.linewidth,
            color=colors[1] if isinstance(colors[1], str) else colors[0],
            marker="s", 
            markersize=cfg.markersize,
            label="Surrogate",
        )
        
        # Set scales and limits
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(p_indices.min().item(), p_indices.max().item())
        
        # Adapt ylim to plotted data range
        all_values = th.cat([ustat_act_P, ustat_surr_P])
        y_min, y_max = all_values.min().item(), all_values.max().item()
        ax.set_ylim(y_min * 0.9, y_max * 1.1)
        
        # Labels
        ax.set_xlabel("Token Position")
        if idx == 0:  # Only leftmost subplot gets y-label
            ax.set_ylabel("U-Statistic")
        
        # Use dataset name for subplot title
        dataset_name = config["dataset_name"].split("/")[-1] if "/" in config["dataset_name"] else config["dataset_name"]
        ax.set_title(f"{dataset_name}")
        ax.legend()
    
    # Overall title
    first_data = next(iter(results.values()))
    first_config = first_data["config"]
    llm_name = first_config["llm_name"].split("/")[-1] if "/" in first_config["llm_name"] else first_config["llm_name"]
    
    fig.suptitle(
        f"U-Statistic: Original vs Surrogate (Multiple Datasets)\n{llm_name} Layer {first_config['layer_idx']}",
        fontsize=14,
    )
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(cfg.plots_dir, exist_ok=True)
    plot_path = os.path.join(cfg.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"U-statistic multiple datasets overlay plot saved to: {plot_path}")
    return plot_path


def plot_u_statistic_comparison_datasets(
    results: Dict,
    cfg: UStatPlotConfig,
    save_name: str = "u_statistic_datasets_comparison.pdf"
) -> str:
    """
    Plot u-statistic comparison across multiple datasets on the same subplot.
    Shows original data for all datasets in different colors.
    
    Args:
        results: Dictionary with dataset configs as keys and result data as values
        cfg: Plot configuration
        save_name: Name for saved plot file
        
    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=cfg.figsize)
    
    # Get colors for datasets
    n_datasets = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    
    for idx, (dataset_key, data) in enumerate(results.items()):
        config = data["config"]
        result_data = data["results"]
        
        # Extract data
        p_indices = th.tensor(result_data["p_indices"])
        ustat_act_P = th.tensor(result_data["ustat_act_P"])
        
        # Get dataset name for label
        dataset_name = config["dataset_name"].split("/")[-1] if "/" in config["dataset_name"] else config["dataset_name"]
        
        # Plot original data curve
        ax.plot(
            p_indices,
            ustat_act_P,
            linewidth=cfg.linewidth,
            color=colors[idx],
            marker="o",
            markersize=cfg.markersize,
            label=dataset_name,
        )
    
    # Set scales and limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Labels and title
    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    
    # Get model info from first dataset
    first_data = next(iter(results.values()))
    first_config = first_data["config"]
    llm_name = first_config["llm_name"].split("/")[-1] if "/" in first_config["llm_name"] else first_config["llm_name"]
    
    ax.set_title(f"U-Statistic Comparison Across Datasets\n{llm_name} Layer {first_config['layer_idx']}")
    ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(cfg.plots_dir, exist_ok=True)
    plot_path = os.path.join(cfg.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"U-statistic datasets comparison plot saved to: {plot_path}")
    return plot_path


if __name__ == "__main__":
    cfg = UStatPlotConfig(
        plots_dir=PLOTS_DIR,
        results_dir="artifacts/interim",  # Where u_stat_llm.json files are stored
    )
    
    # Load results
    results = load_u_statistic_results(cfg.results_dir)
    
    if not results:
        print(f"No u-statistic results found in {cfg.results_dir}")
        print("Make sure to run exp/u_stat.py first to generate results.")
    else:
        print(f"Found {len(results)} u-statistic result(s)")
        
        if len(results) == 1:
            # Single result - plot overlay
            plot_u_statistic_overlay(results, cfg)
        else:
            # Multiple results - plot multiple dataset overlays and comparison
            plot_u_statistic_overlay_multiple_datasets(results, cfg)
            plot_u_statistic_comparison_datasets(results, cfg)