"""
Plotting functions for garden path entropy analysis results.

Creates plots showing entropy and magnitude trajectories across relative token positions
for different garden path sentence types (ambiguous, gp, post) and different activation types.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import torch as th

from exp.garden_path_entropy import GardenPathEntropyConfig
from src.configs import *
from src.model_utils import load_tokenizer


def load_garden_path_entropy_results():
    """Load all garden path entropy results using the config pattern."""
    configs = get_gemma_act_configs(
        cfg_class=GardenPathEntropyConfig,
        scaling_factor=0.00666666667,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    "surrogate"
                ]
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    "codes",
                    "recons"
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    "novel_codes",
                    "novel_recons",
                    "pred_codes",
                    "pred_recons",
                    "total_recons",
                ]
            ),
        ),
        num_final_tokens=5,
        data=DatasetConfig(
            name="GardenPath",
            hf_name="garden_path.csv",
            num_sequences=97,
            context_length=None,
        ),
        llm=LLMConfig(
            name="Gemma-2-2B",
            hf_name="google/gemma-2-2b",
            revision=None,
            layer_idx=12,
            hidden_dim=2304,
            batch_size=50,
        ),
        env=ENV_CFG,
    )

    results = load_results_multiple_configs(
        exp_name="garden_path_entropy_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,
        verbose=True,
    )

    return results, configs



def organize_results_by_type(results: List[Dict]) -> Dict[str, Dict]:
    """
    Organize results by SAE type and activation path.

    Returns:
        Dict with keys like "llm_activations", "batchtopk_codes", etc.
    """
    organized = {}

    for fname, result in results.items():
        config = result['config']
        sae_name = config['sae']['name'] if config['sae'] is not None else 'llm'
        act_path = config['act_path']

        key = f"{sae_name}_{act_path}"
        organized[key] = result

    return organized


def plot_individual_trajectories(results_dict: Dict[str, Dict], save_path: str = None, max_trajectories: int = 10):
    """
    Plot individual trajectories for each sequence.

    Args:
        results_dict: Organized results dictionary
        save_path: Optional path to save the figure
        max_trajectories: Maximum number of trajectories to plot per condition (None for all)
    """
    # Set up the plot
    n_conditions = len(results_dict)
    fig, axes = plt.subplots(n_conditions, 2, figsize=(12, 4 * n_conditions))

    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    # Color palette for the three modes
    colors = {'ambiguous': '#e74c3c', 'gp': '#3498db', 'post': '#2ecc71'}
    gp_modes = ['ambiguous', 'gp', 'post']

    row_idx = 0
    for condition_name, result in results_dict.items():
        pos_indices = result['pos_indices']

        # Plot entropy (left column)
        ax_entropy = axes[row_idx, 0]
        for mode in gp_modes:
            entropy_data = np.array(result['entropy'][mode])  # [batch, num_final_tokens]

            # Determine how many trajectories to plot
            num_trajectories = entropy_data.shape[0]
            if max_trajectories is not None:
                num_trajectories = min(num_trajectories, max_trajectories)

            # Plot each trajectory
            for i in range(num_trajectories):
                ax_entropy.plot(pos_indices, entropy_data[i],
                              color=colors[mode], alpha=0.3, linewidth=0.8)

        ax_entropy.set_xlabel('Relative Token Position')
        ax_entropy.set_ylabel('Entropy')
        ax_entropy.set_title(f'{condition_name} - Entropy')
        ax_entropy.grid(True, alpha=0.3)

        # Plot magnitude (right column)
        ax_magnitude = axes[row_idx, 1]
        for mode in gp_modes:
            magnitude_data = np.array(result['magnitude'][mode])  # [batch, num_final_tokens]

            # Determine how many trajectories to plot
            num_trajectories = magnitude_data.shape[0]
            if max_trajectories is not None:
                num_trajectories = min(num_trajectories, max_trajectories)

            # Plot each trajectory
            for i in range(num_trajectories):
                ax_magnitude.plot(pos_indices, magnitude_data[i],
                                color=colors[mode], alpha=0.3, linewidth=0.8)

        ax_magnitude.set_xlabel('Relative Token Position')
        ax_magnitude.set_ylabel('Magnitude')
        ax_magnitude.set_title(f'{condition_name} - Magnitude')
        ax_magnitude.grid(True, alpha=0.3)

        row_idx += 1

    # Create custom legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=colors[mode], label=mode.capitalize())
                     for mode in gp_modes]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual trajectories plot saved to: {save_path}")

    plt.show()


def plot_mean_and_confidence_intervals(results_dict: Dict[str, Dict], save_path: str = None):
    """
    Plot mean trajectories with 95% confidence intervals.

    Args:
        results_dict: Organized results dictionary
        save_path: Optional path to save the figure
    """
    # Set up the plot
    n_conditions = len(results_dict)
    fig, axes = plt.subplots(n_conditions, 2, figsize=(12, 4 * n_conditions))

    if n_conditions == 1:
        axes = axes.reshape(1, -1)

    # Color palette for the three modes
    colors = {'ambiguous': '#e74c3c', 'gp': '#3498db', 'post': '#2ecc71'}
    gp_modes = ['ambiguous', 'gp', 'post']

    row_idx = 0
    for condition_name, result in results_dict.items():
        pos_indices = result['pos_indices']

        # Plot entropy (left column)
        ax_entropy = axes[row_idx, 0]
        for mode in gp_modes:
            entropy_data = np.array(result['entropy'][mode])  # [batch, num_final_tokens]

            # Compute mean and 95% CI
            mean_entropy = np.mean(entropy_data, axis=0)
            std_entropy = np.std(entropy_data, axis=0)
            n_samples = entropy_data.shape[0]
            ci_entropy = 1.96 * std_entropy / np.sqrt(n_samples)  # 95% CI

            # Plot mean line
            ax_entropy.plot(pos_indices, mean_entropy, color=colors[mode],
                          linewidth=2, label=mode.capitalize())

            # Plot confidence interval
            ax_entropy.fill_between(pos_indices,
                                  mean_entropy - ci_entropy,
                                  mean_entropy + ci_entropy,
                                  color=colors[mode], alpha=0.2)

        ax_entropy.set_xlabel('Relative Token Position')
        ax_entropy.set_ylabel('Entropy')
        ax_entropy.set_title(f'{condition_name} - Entropy (Mean ± 95% CI)')
        ax_entropy.grid(True, alpha=0.3)
        ax_entropy.legend()

        # Plot magnitude (right column)
        ax_magnitude = axes[row_idx, 1]
        for mode in gp_modes:
            magnitude_data = np.array(result['magnitude'][mode])  # [batch, num_final_tokens]

            # Compute mean and 95% CI
            mean_magnitude = np.mean(magnitude_data, axis=0)
            std_magnitude = np.std(magnitude_data, axis=0)
            n_samples = magnitude_data.shape[0]
            ci_magnitude = 1.96 * std_magnitude / np.sqrt(n_samples)  # 95% CI

            # Plot mean line
            ax_magnitude.plot(pos_indices, mean_magnitude, color=colors[mode],
                            linewidth=2, label=mode.capitalize())

            # Plot confidence interval
            ax_magnitude.fill_between(pos_indices,
                                    mean_magnitude - ci_magnitude,
                                    mean_magnitude + ci_magnitude,
                                    color=colors[mode], alpha=0.2)

        ax_magnitude.set_xlabel('Relative Token Position')
        ax_magnitude.set_ylabel('Magnitude')
        ax_magnitude.set_title(f'{condition_name} - Magnitude (Mean ± 95% CI)')
        ax_magnitude.grid(True, alpha=0.3)
        ax_magnitude.legend()

        row_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mean + CI plot saved to: {save_path}")

    plt.show()


def plot_single_trajectory_with_tokens(results_dict: Dict[str, Dict], configs: List,
                                     condition_name: str = None, sequence_idx: int = 0,
                                     sentence_type: str = 'ambiguous', save_path: str = None):
    """
    Plot a single trajectory for one sequence with tokens aligned on the x-axis.

    Args:
        results_dict: Organized results dictionary
        configs: Configuration objects for tokenizer access
        condition_name: Which condition to plot (if None, uses first available)
        sequence_idx: Which sequence to plot (default 0)
        sentence_type: Which sentence type to plot ('ambiguous', 'gp', 'post')
        save_path: Optional path to save the figure
    """
    # Select condition to plot
    if condition_name is None:
        condition_name = list(results_dict.keys())[0]

    if condition_name not in results_dict:
        print(f"Condition '{condition_name}' not found in results")
        return

    result = results_dict[condition_name]

    # Get the configuration for this condition to access tokenizer
    cfg = None
    for config in configs:
        config_key = f"{config.sae.name if config.sae else 'llm'}_{config.act_path}"
        if config_key == condition_name:
            cfg = config
            break

    if cfg is None:
        print(f"Could not find configuration for condition '{condition_name}'")
        return

    # Check if tokens are saved in results
    if 'tokens' not in result or sentence_type not in result['tokens']:
        print(f"No tokens found for {sentence_type} in results. Please re-run the garden path entropy analysis.")
        return

    # Get data for this sentence type and sequence
    entropy_data = np.array(result['entropy'][sentence_type])  # [batch, num_final_tokens]
    magnitude_data = np.array(result['magnitude'][sentence_type])  # [batch, num_final_tokens]
    token_ids = result['tokens'][sentence_type]  # [batch, num_final_tokens]

    if sequence_idx >= entropy_data.shape[0]:
        print(f"Sequence index {sequence_idx} out of range (max: {entropy_data.shape[0]-1})")
        return

    # Extract data for the specific sequence
    entropy_seq = entropy_data[sequence_idx]  # [num_final_tokens]
    magnitude_seq = magnitude_data[sequence_idx]  # [num_final_tokens]
    token_ids_seq = token_ids[sequence_idx]  # [num_final_tokens]

    # Convert token IDs to text using tokenizer
    tokenizer = load_tokenizer(cfg.llm.hf_name, cache_dir=cfg.env.hf_cache_dir)
    final_tokens = []
    for token_id in token_ids_seq:
        if token_id >= 0:  # Valid token (not padded)
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            final_tokens.append(token_text.replace('\n', '<newline>'))
        else:
            final_tokens.append('[PAD]')  # Placeholder for padded positions

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(final_tokens) * 0.8), 8))

    # Plot entropy
    ax1.plot(range(len(final_tokens)), entropy_seq[:len(final_tokens)],
             'o-', linewidth=2, markersize=6, color='#e74c3c')
    ax1.set_ylabel('Entropy')
    ax1.set_title(f'{condition_name} - {sentence_type.capitalize()} (Sequence {sequence_idx}) - Entropy')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(final_tokens)))
    ax1.set_xticklabels(final_tokens, rotation=45, ha='right', fontsize=10)

    # Plot magnitude
    ax2.plot(range(len(final_tokens)), magnitude_seq[:len(final_tokens)],
             'o-', linewidth=2, markersize=6, color='#3498db')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlabel('Tokens')
    ax2.set_title(f'{condition_name} - {sentence_type.capitalize()} (Sequence {sequence_idx}) - Magnitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(final_tokens)))
    ax2.set_xticklabels(final_tokens, rotation=45, ha='right', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single trajectory plot saved to: {save_path}")

    plt.show()


def main():
    """Main function to create both types of plots."""
    print("Loading garden path entropy results...")
    results, configs = load_garden_path_entropy_results()

    print(f"Loaded {len(results)} result files")

    # Organize results by condition
    results_dict = organize_results_by_type(results)

    print(f"Organized into {len(results_dict)} conditions:")
    for key in results_dict.keys():
        print(f"  - {key}")

    # Create plots directory if it doesn't exist
    import os
    plots_dir = configs[0].env.plots_dir
    os.makedirs(plots_dir, exist_ok=True)

    # Plot individual trajectories
    print("\nCreating individual trajectories plot...")
    plot_individual_trajectories(
        results_dict,
        save_path=os.path.join(plots_dir, "garden_path_entropy_individual.png"),
        max_trajectories=None
    )

    # Plot mean and confidence intervals
    print("\nCreating mean + confidence intervals plot...")
    plot_mean_and_confidence_intervals(
        results_dict,
        save_path=os.path.join(plots_dir, "garden_path_entropy_mean_ci.png")
    )

    # Plot single trajectory with tokens
    print("\nCreating single trajectory with tokens plot...")
    plot_single_trajectory_with_tokens(
        results_dict,
        configs,
        condition_name=None,  # Uses first available condition
        sequence_idx=0,       # First sequence
        sentence_type='ambiguous',  # Can be changed to 'gp' or 'post'
        save_path=os.path.join(plots_dir, "garden_path_entropy_single_trajectory_tokens.png")
    )

    print("Plotting complete!")


if __name__ == "__main__":
    main()