"""
Plotting SAE Basic stats.

- L0 over sequence position t (mean + ci)
- L1 over t (mean + ci)
- MSE over t (mean)
- Normalized MSE over t (mean)
- Fraction variance explained over t (mean)
- cosine similarity (mean + ci)
- fraction alive (single bar plot, one bar per sae)

"""

import matplotlib.pyplot as plt

from exp.basic_stats import BasicStatsConfig
from src.configs import *
from src.plotting_utils import savefig


def plot_stats_across_saes(results):
    """
    Plotting function for comparing all SAEs.
    Plot with 2D grid of subplots
    rows: datasets (dynamically decided based on input dict, based on unique dataset values in the set)
    cols: metrics
         - L0 over sequence position t (mean + ci)
         - L1 over t (mean + ci)
         - MSE over t (mean)
         - Normalized MSE over t (mean)
         - Fraction variance explained over t (mean)
         - cosine similarity (mean + ci)
         - fraction alive (single bar plot, one bar per sae)

    lines in each plot: SAE_activation_type (labeled as SAE_architecture / activation type (without the .pt ending))
    legend outside the subplot grid
    """
    import numpy as np

    # Extract datasets and organize data
    datasets = {}
    for result in results.values():
        dataset_name = result['config']['data']['name']
        sae_name = result['config']['sae']['name']

        if dataset_name not in datasets:
            datasets[dataset_name] = {}

        # Store result by SAE name
        datasets[dataset_name][sae_name] = result

    # Get unique datasets and SAEs
    dataset_names = sorted(datasets.keys())
    all_saes = set()
    for dataset_data in datasets.values():
        all_saes.update(dataset_data.keys())

    # Define which activation types to show for each metric category
    shown_temporal_act_types = {
        "codes": [
            "codes", # standard
            # "pred_codes",
            "novel_codes",
        ],
        "recons": [
            "recons", # standard
            # "pred_recons",
            # "novel_recons",
            "total_recons",
        ]
    }

    # Option to sync colors across activation types from the same SAE architecture
    sync_sae_colors = True

    # Define metrics to plot based on actual computation in exp/basic_stats.py
    codes_metrics = [
        # ('l0', 'L0 over sequence position'),
        # ('l1', 'L1 over sequence position'),
        # ('fraction_alive', 'Fraction alive')  # Special case: bar plot
    ]

    recons_metrics = [
        # ('mse', 'MSE over sequence position'),
        ('normalized_mse', 'Normalized MSE over sequence position'),
        # ('fraction_variance_explained', 'Fraction variance explained over sequence position'),
        # ('cosine_similarity', 'Cosine similarity over sequence position')
    ]

    all_metrics = codes_metrics + recons_metrics

    # Create subplot grid
    factor = 5
    fig, axes = plt.subplots(len(dataset_names), len(all_metrics),
                            figsize=(len(all_metrics) * factor, len(dataset_names) * factor))

    # Handle axes indexing for different subplot configurations
    if len(dataset_names) == 1 and len(all_metrics) == 1:
        # Single subplot case - wrap in 2D structure for consistent indexing
        axes = np.array([[axes]])
    elif len(dataset_names) == 1:
        # Single row case
        axes = axes.reshape(1, -1)
    elif len(all_metrics) == 1:
        # Single column case
        axes = axes.reshape(-1, 1)

    # Define strong, dark colors manually for better visibility
    strong_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#ff0000',  # Bright red
        '#00ff00',  # Bright green
        '#0000ff',  # Bright blue
        '#ffff00',  # Yellow
        '#ff00ff',  # Magenta
        '#00ffff',  # Cyan
        '#800000',  # Maroon
        '#008000',  # Dark green
        '#000080',  # Navy
        '#800080',  # Purple
        '#008080',  # Teal
        '#c0c0c0',  # Silver
        '#808000',  # Olive
        '#ff6347',  # Tomato
        '#4682b4',  # Steel blue
        '#32cd32',  # Lime green
        '#ff1493',  # Deep pink
        '#00ced1',  # Dark turquoise
        '#ffd700',  # Gold
        '#dc143c',  # Crimson
        '#00fa9a',  # Medium spring green
        '#8a2be2',  # Blue violet
        '#a0522d',  # Sienna
        '#dda0dd',  # Plum
        '#98fb98',  # Pale green
        '#f0e68c',  # Khaki
        '#deb887',  # Burlywood
        '#5f9ea0',  # Cadet blue
        '#7fff00',  # Chartreuse
        '#d2691e',  # Chocolate
    ]

    # Define color mapping based on sync_sae_colors option
    combo_colors = {}

    if sync_sae_colors:
        # Same color for all activation types from the same SAE architecture
        all_saes = set()
        for dataset_data in datasets.values():
            all_saes.update(dataset_data.keys())

        num_saes = len(all_saes)
        if num_saes <= len(strong_colors):
            colors = [strong_colors[i] for i in range(num_saes)]
        else:
            base_colors = strong_colors * ((num_saes // len(strong_colors)) + 1)
            colors = base_colors[:num_saes]

        sae_colors = dict(zip(sorted(all_saes), colors))

        for dataset_data in datasets.values():
            for sae_name, result in dataset_data.items():
                activation_types = [k for k in result.keys() if k.endswith('.pt')]
                for act_type in activation_types:
                    combo_label = f"{sae_name}/{act_type[:-3]}"
                    combo_colors[combo_label] = sae_colors[sae_name]
    else:
        # Different color for each SAE/activation type combination
        all_sae_activation_combos = set()
        for dataset_data in datasets.values():
            for sae_name, result in dataset_data.items():
                activation_types = [k for k in result.keys() if k.endswith('.pt')]
                for act_type in activation_types:
                    all_sae_activation_combos.add(f"{sae_name}/{act_type[:-3]}")

        num_combos = len(all_sae_activation_combos)
        if num_combos <= len(strong_colors):
            colors = [strong_colors[i] for i in range(num_combos)]
        else:
            base_colors = strong_colors * ((num_combos // len(strong_colors)) + 1)
            colors = base_colors[:num_combos]

        combo_colors = dict(zip(sorted(all_sae_activation_combos), colors))

    # Track legend entries separately for codes and recons
    codes_legend_entries = set()
    recons_legend_entries = set()

    # Plot each metric for each dataset
    for row, dataset_name in enumerate(dataset_names):
        dataset_data = datasets[dataset_name]

        for col, (metric_key, metric_title) in enumerate(all_metrics):
            ax = axes[row, col]

            # Determine if this is a codes or recons metric
            is_codes_metric = (metric_key, metric_title) in codes_metrics

            # Collect data for this metric across all SAEs in this dataset
            for sae_name, result in dataset_data.items():
                # Get activation types for this SAE
                activation_types = [k for k in result.keys() if k.endswith('.pt')]

                if metric_key == 'fraction_alive':
                    # Special case: bar plot for fraction alive - collect data from all SAEs first
                    continue  # Handle fraction_alive separately after collecting all data

                else:
                    # Line plots for sequence-dependent metrics
                    for act_type in activation_types:
                        # Determine if this activation type is relevant for this metric
                        act_type_name = act_type[:-3]  # Remove .pt extension
                        if is_codes_metric:
                            if act_type_name not in shown_temporal_act_types["codes"]:
                                continue
                        else:
                            if act_type_name not in shown_temporal_act_types["recons"]:
                                continue

                        if metric_key in result[act_type]:
                            metric_data = result[act_type][metric_key]

                            if 'score' in metric_data:
                                scores = metric_data['score']
                                x_positions = result['sequence_pos_indices']

                                combo_label = f"{sae_name}/{act_type[:-3]}"
                                color = combo_colors[combo_label]

                                # Track for legend
                                if is_codes_metric:
                                    codes_legend_entries.add(combo_label)
                                else:
                                    recons_legend_entries.add(combo_label)

                                # Use dashed line for non-temporal SAEs
                                linestyle = '-' if 'temporal' in sae_name else '--'

                                # Plot main line
                                ax.plot(x_positions, scores, 'o',
                                       color=color, label=combo_label, alpha=0.8,
                                       linestyle=linestyle)

                                # Check if CI is available in the data and add if present
                                if 'ci' in metric_data and metric_data['ci'] is not None:
                                    ci = metric_data['ci']
                                    # Only plot CI if it's not all zeros/NaNs
                                    if not all(np.isnan(ci)) and not all(np.array(ci) == 0):
                                        scores_arr = np.array(scores)
                                        ci_arr = np.array(ci)
                                        ax.fill_between(x_positions,
                                                       scores_arr - ci_arr,
                                                       scores_arr + ci_arr,
                                                       color=color, alpha=0.2)

            # Set titles and labels
            if row == 0:
                ax.set_title(metric_title)
            if col == 0:
                ax.set_ylabel(dataset_name)
            if row == len(dataset_names) - 1:
                ax.set_xlabel('Sequence Position')

            # Set x-axis to log scale for better visualization
            if metric_key != 'fraction_alive':
                ax.set_xscale('log')

            ax.grid(True, alpha=0.3)

            # Special handling for fraction_alive after collecting all data
            if metric_key == 'fraction_alive':
                fraction_alive_values = []
                labels = []
                bar_colors = []

                # Collect all codes activation types across all SAEs for this dataset
                for sae_name, result in dataset_data.items():
                    activation_types = [k for k in result.keys() if k.endswith('.pt')]
                    for act_type in activation_types:
                        act_type_name = act_type[:-3]  # Remove .pt extension
                        if act_type_name in shown_temporal_act_types["codes"] and metric_key in result[act_type]:
                            fraction_alive_values.append(result[act_type][metric_key])
                            combo_label = f"{sae_name}/{act_type[:-3]}"
                            labels.append(combo_label)
                            bar_colors.append(combo_colors[combo_label])
                            codes_legend_entries.add(combo_label)

                if fraction_alive_values:
                    bar_positions = np.arange(len(fraction_alive_values))
                    ax.bar(bar_positions, fraction_alive_values, color=bar_colors, alpha=0.7)
                    ax.set_xticks(bar_positions)
                    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Create separate legends for codes and recons
    plt.tight_layout()
    # plt.subplots_adjust(right=0.75)  # Make more room for legends

    legend_y_top = 0.85
    legend_y_spacing = 0.05

    # Codes legend
    if codes_legend_entries:
        codes_handles = []
        codes_labels = []
        for combo_label in sorted(codes_legend_entries):
            sae_name = combo_label.split('/')[0]
            linestyle = '-' if 'temporal' in sae_name else '--'
            codes_handles.append(plt.Line2D([0], [0], color=combo_colors[combo_label],
                                          marker='o', linestyle=linestyle))
            codes_labels.append(combo_label)

        codes_legend_height = len(codes_labels) * 0.03
        codes_y_pos = legend_y_top - codes_legend_height / 2

        codes_legend = fig.legend(codes_handles, codes_labels,
                                title="Codes Metrics",
                                loc='center left', bbox_to_anchor=(1.02, codes_y_pos))

    # Recons legend
    if recons_legend_entries:
        recons_handles = []
        recons_labels = []
        for combo_label in sorted(recons_legend_entries):
            sae_name = combo_label.split('/')[0]
            linestyle = '-' if 'temporal' in sae_name else '--'
            recons_handles.append(plt.Line2D([0], [0], color=combo_colors[combo_label],
                                           marker='o', linestyle=linestyle))
            recons_labels.append(combo_label)

        # Position recons legend below codes legend
        recons_legend_height = len(recons_labels) * 0.03
        if codes_legend_entries:
            codes_legend_height = len(codes_legend_entries) * 0.03
            recons_y_pos = legend_y_top - codes_legend_height - legend_y_spacing - recons_legend_height / 2
        else:
            recons_y_pos = legend_y_top - recons_legend_height / 2

        recons_legend = fig.legend(recons_handles, recons_labels,
                                 title="Recons Metrics",
                                 loc='center left', bbox_to_anchor=(1.02, recons_y_pos))

    savefig("basic_stats_comparison", suffix=".pdf")
    plt.show()

def main():
    configs = get_configs(
        cfg_class=BasicStatsConfig,
        # config arguments
        min_p=1,
        max_p=499,
        num_p=10,
        # Artifacts
        env=ENV_CFG,
        # data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        # sae=GEMMA2_SELFTRAIN_SAE_CFGS,
        sae=GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
        # sae=GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS + GEMMA2_TEMPORAL_SELFTRAIN_SAE_CFGS,
        # sae=GEMMA2_TEMPORAL_SELFTRAIN_SAE_CFGS,
        # sae=TEMPORAL_SELFTRAIN_SAE_CFG
        # sae=[TEMPORAL_SELFTRAIN_SAE_CFG, BATCHTOPK_SELFTRAIN_SAE_CFG]
    )

    results = load_results_multiple_configs(
        exp_name="basic_stats_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=True,
    )
    plot_stats_across_saes(results)


if __name__ == "__main__":
    main()
