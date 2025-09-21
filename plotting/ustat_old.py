"""
Plot u-statistic intrinsic dimensionality results computed by exp/u_stat.py
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.project_config import BaseConfig
from src.preprocessing_utils import load_experiment_result, create_sweep_configs

@dataclass
class UStatPlotConfig:
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


def plot_u_statistic_overlay(
    results: Dict,
    plot_cfg: UStatPlotConfig,
    base_cfg: BaseConfig,
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
    fig, ax = plt.subplots(figsize=plot_cfg.figsize)
    
    # Plot both curves
    ax.plot(
        p_indices,
        ustat_act_P,
        linewidth=plot_cfg.linewidth,
        color=plot_cfg.original_color,
        marker="o",
        markersize=plot_cfg.markersize,
        label="Original",
    )
    ax.plot(
        p_indices,
        ustat_surr_P,
        linewidth=plot_cfg.linewidth,
        color=plot_cfg.surrogate_color,
        marker="s",
        markersize=plot_cfg.markersize,
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
    os.makedirs(base_cfg.plots_dir, exist_ok=True)
    plot_path = os.path.join(base_cfg.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"U-statistic overlay plot saved to: {plot_path}")
    return plot_path


def plot_u_statistic_multiple_layers(
    layer_results: Dict[int, Dict],
    plot_cfg: UStatPlotConfig,
    base_cfg: BaseConfig,
    save_name: str = "u_statistic_multiple_layers.pdf"
) -> str:
    """
    Plot u-statistic for multiple layers with original (solid) vs surrogate (dashed).
    
    Args:
        layer_results: Dictionary mapping layer indices to results
        plot_cfg: Plot configuration
        base_cfg: Base configuration
        save_name: Name for saved plot file
        
    Returns:
        Path to saved plot
    """
    # Create plot
    fig, ax = plt.subplots(figsize=plot_cfg.figsize)
    
    # Get unique colors for each layer
    layer_indices = sorted(layer_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_indices)))
    
    for i, layer_idx in enumerate(layer_indices):
        results = layer_results[layer_idx]
        
        # Get the first (and presumably only) result for this layer
        result_key = list(results.keys())[0]
        data = results[result_key]
        result_data = data["results"]
        
        # Extract data
        p_indices = th.tensor(result_data["p_indices"])
        ustat_act_P = th.tensor(result_data["ustat_act_P"])
        ustat_surr_P = th.tensor(result_data["ustat_surr_P"])
        
        color = colors[i]
        
        # Plot activation (solid line)
        ax.plot(
            p_indices,
            ustat_act_P,
            linewidth=plot_cfg.linewidth,
            color=color,
            linestyle='-',
            marker="o",
            markersize=plot_cfg.markersize,
            label=f"Layer {layer_idx} Original",
        )
        
        # Plot surrogate (dashed line)
        ax.plot(
            p_indices,
            ustat_surr_P,
            linewidth=plot_cfg.linewidth,
            color=color,
            linestyle='--',
            marker="s",
            markersize=plot_cfg.markersize,
            label=f"Layer {layer_idx} Surrogate",
        )
    
    # Set scales and limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Get data range for all layers
    all_p_indices = []
    all_values = []
    
    for layer_idx in layer_indices:
        results = layer_results[layer_idx]
        result_key = list(results.keys())[0]
        data = results[result_key]
        result_data = data["results"]
        
        p_indices = th.tensor(result_data["p_indices"])
        ustat_act_P = th.tensor(result_data["ustat_act_P"])
        ustat_surr_P = th.tensor(result_data["ustat_surr_P"])
        
        all_p_indices.append(p_indices)
        all_values.extend([ustat_act_P, ustat_surr_P])
    
    # Set limits based on all data
    all_p_cat = th.cat(all_p_indices)
    all_values_cat = th.cat(all_values)
    
    ax.set_xlim(all_p_cat.min().item(), all_p_cat.max().item())
    
    y_min, y_max = all_values_cat.min().item(), all_values_cat.max().item()
    ax.set_ylim(y_min * 0.9, y_max * 1.1)
    
    # Labels and title
    ax.set_xlabel("Token Position")
    ax.set_ylabel("U-Statistic")
    
    # Get model and dataset info from first result
    first_result = list(layer_results.values())[0]
    first_key = list(first_result.keys())[0]
    config = first_result[first_key]["config"]
    
    llm_name = config["llm_name"].split("/")[-1] if "/" in config["llm_name"] else config["llm_name"]
    dataset_name = config["dataset_name"].split("/")[-1] if "/" in config["dataset_name"] else config["dataset_name"]
    
    ax.set_title(f"U-Statistic: Multiple Layers\\n{llm_name} | {dataset_name}")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(base_cfg.plots_dir, exist_ok=True)
    plot_path = os.path.join(base_cfg.plots_dir, save_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"U-statistic multiple layers plot saved to: {plot_path}")
    return plot_path

def main_single_result():
    base_cfg = BaseConfig(
        experiment_name="u_stat",
        verbose=False,
        llm_name="meta-llama/Llama-3.1-8B",
        revision=None,
        layer_idx=16,
        llm_batch_size=100,
        llm_hidden_dim=4096,
        dtype="bfloat16",
        dataset_name="monology/pile-uncopyrighted",
        hf_text_identifier="text",
        num_sequences=1_000,
        context_length=500,
        p_start=10,
        p_end=500,
        num_p=10,
        do_log_p=True,
        omit_bos=True,
        normalize_activations=False,
    )
    
    # Load results
    results = load_experiment_result(base_cfg)

    plot_cfg = UStatPlotConfig()
    plot_u_statistic_overlay(results, plot_cfg, base_cfg)


def main_multiple_layers():
    """Plot u-statistic results for multiple layers."""
    base_cfg = BaseConfig(
        experiment_name="u_stat",
        verbose=False,
        llm_name="meta-llama/Llama-3.1-8B",
        revision=None,
        llm_batch_size=100,
        llm_hidden_dim=4096,
        dtype="bfloat16",
        dataset_name="monology/pile-uncopyrighted",
        hf_text_identifier="text",
        num_sequences=1_000,
        context_length=500,
        p_start=10,
        p_end=500,
        num_p=10,
        do_log_p=True,
        omit_bos=True,
        normalize_activations=False,
    )
    
    # Define sweep parameters for different layers
    sweep_params = {
        'layer_idx': [0, 8, 16, 24, 31]  # Multiple layers to analyze
    }
    
    # Create configs for each layer
    configs, param_combinations = create_sweep_configs(base_cfg, sweep_params)
    
    # Load results for each layer
    layer_results = {}
    for cfg, param_combo in zip(configs, param_combinations):
        layer_idx = param_combo['layer_idx']
        try:
            results = load_experiment_result(cfg)
            layer_results[layer_idx] = results
            print(f"Loaded results for layer {layer_idx}")
        except FileNotFoundError as e:
            print(f"No results found for layer {layer_idx}: {e}")
    
    if not layer_results:
        print("No results found for any layers. Make sure experiments have been run first.")
        return
    
    # Create plot
    plot_cfg = UStatPlotConfig(figsize=(12, 8))
    plot_u_statistic_multiple_layers(layer_results, plot_cfg, base_cfg)


if __name__ == "__main__":
    main_multiple_layers()