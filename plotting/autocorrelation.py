import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.configs import *
from exp.autocorrelation import AutocorrelationConfig
from src.plotting_utils import savefig

def plot_autocorrelation(results, global_colorbar=False):
    """
    Plot with 2D grid of subplots
    Each subplot is a heatmap displaying the values of autocorr_Wp
    cmap: blues
    colorbar: individual per subplot (default) or global if global_colorbar=True
    x_values: anchors_W
    y_values: relative_offsets_p

    Args:
        results: Dictionary of results from autocorrelation experiments
        global_colorbar: If True, use single colorbar with global scale. If False, individual colorbars per subplot.

    Num rows subplots: num_act_paths aka act types. find at runtime
    Num cols subplots: num datasets. Find unique datasets at runtime by looking through all results
    """
    # Group results by act_path and dataset
    grouped_results = defaultdict(lambda: defaultdict(dict))

    for result_name, result in results.items():
        act_path = result["config"]["act_path"]
        if result["config"]["sae"] is not None:
            act_path = f"{result["config"]["sae"]["name"]}/{act_path}"
        dataset_name = result["config"]["data"]["name"]
        grouped_results[act_path][dataset_name] = result

    # Get unique act_paths and datasets
    act_paths = list(grouped_results.keys())
    datasets = list(set(dataset for act_data in grouped_results.values() for dataset in act_data.keys()))

    num_rows = len(act_paths)
    num_cols = len(datasets)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 6*num_rows))

    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    # Collect all data to determine global vmin/vmax if using global colorbar
    if global_colorbar:
        all_data = []
        for act_path in act_paths:
            for dataset in datasets:
                if dataset in grouped_results[act_path]:
                    result = grouped_results[act_path][dataset]
                    autocorr_data = np.array(result["autocorr_Wp"])
                    all_data.append(autocorr_data)

        if all_data:
            vmin = np.min([np.min(data) for data in all_data])
            vmax = np.max([np.max(data) for data in all_data])
        else:
            vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None

    images = []
    for i, act_path in enumerate(act_paths):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]

            if dataset in grouped_results[act_path]:
                result = grouped_results[act_path][dataset]

                # Extract data
                autocorr_data = np.array(result["autocorr_Wp"])
                anchors = result["anchors_W"]
                offsets = result["relative_offsets_p"]

                # Create heatmap (transpose to flip axes)
                if global_colorbar:
                    im = ax.imshow(autocorr_data.T, cmap='Blues', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
                    images.append(im)
                else:
                    im = ax.imshow(autocorr_data.T, cmap='Blues', aspect='auto', origin='lower')
                    # Add individual colorbar for this subplot
                    fig.colorbar(im, ax=ax, shrink=0.6)

                # Set labels
                ax.set_xlabel('Anchors (W)')
                ax.set_ylabel('Relative Offsets (p)')
                ax.set_title(f'{act_path} - {dataset}')

                # Set ticks
                ax.set_xticks(np.arange(0, len(anchors), max(1, len(anchors)//5)))
                ax.set_xticklabels([anchors[idx] for idx in ax.get_xticks().astype(int) if idx < len(anchors)])

                ax.set_yticks(np.arange(0, len(offsets), max(1, len(offsets)//5)))
                ax.set_yticklabels([offsets[idx] for idx in ax.get_yticks().astype(int) if idx < len(offsets)])
            else:
                # Empty subplot if no data
                ax.set_title(f'{act_path} - {dataset} (No Data)')
                ax.axis('off')

    # Add row titles (SAE architecture / activation type)
    for i, act_path in enumerate(act_paths):
        # Add row title on the left
        title = f"{act_path}\n\n"
        axes[i][0].text(-0.1, 0.5, title, transform=axes[i][0].transAxes,
                       fontsize=14, fontweight='bold', ha='right', va='center', rotation=90)

    # Add column titles (dataset names)
    for j, dataset in enumerate(datasets):
        # Add column title at the top
        axes[0][j].text(0.5, 1.1, dataset, transform=axes[0][j].transAxes,
                       fontsize=14, fontweight='bold', ha='center', va='bottom')

    # Add global colorbar if requested
    if global_colorbar and images:
        fig.colorbar(images[0], ax=axes, shrink=0.6)
        # plt.subplots_adjust(right=0.9)
    # plt.tight_layout()
    savefig("autocorrelation_heatmaps", suffix=".png")


def main():
    configs = get_gemma_act_configs(
        cfg_class=AutocorrelationConfig,
        act_paths=(
            (
                [None], 
                [
                    "activations", 
                    # "surrogate"
                ]
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    # "codes",
                    "recons",
                    # "residuals",
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    # "novel_codes",
                    # "novel_recons",
                    # "pred_codes",
                    # "pred_recons",
                    "total_recons",
                    # "residuals",
                ]
            ),
        ),
        min_anchor=50,
        max_anchor=500,  # selected_context_length, inclusive
        num_anchors=10,
        min_offset=10,  # absolute value
        max_offset=30,  # absolute value
        # Artifacts
        env=ENV_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    results = load_results_multiple_configs(
        exp_name="autocorr_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=True,
    )
    plot_autocorrelation(results, global_colorbar=False)

if __name__ == "__main__":
    main()