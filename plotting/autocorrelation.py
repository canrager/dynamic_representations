import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.configs import *
from exp.autocorrelation import AutocorrelationConfig
from src.plotting_utils import savefig

def plot_autocorrelation(results):
    """
    Plot with 2D grid of subplots
    Each subplot is a heatmap displaying the values of autocorr_Wp
    cmap: blues
    colorbar: individual per subplot
    x_values: anchors_W
    y_values: relative_offsets_p

    Num rows subplots: num_act_paths aka act types. find at runtime
    Num cols subplots: num datasets. Find unique datasets at runtime by looking through all results
    """
    # Group results by act_path and dataset
    grouped_results = defaultdict(lambda: defaultdict(dict))

    for result_name, result in results.items():
        act_path = result["config"]["act_path"]
        dataset_name = result["config"]["data"]["name"]
        grouped_results[act_path][dataset_name] = result

    # Get unique act_paths and datasets
    act_paths = list(grouped_results.keys())
    datasets = list(set(dataset for act_data in grouped_results.values() for dataset in act_data.keys()))

    num_rows = len(act_paths)
    num_cols = len(datasets)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))

    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]


    for i, act_path in enumerate(act_paths):
        for j, dataset in enumerate(datasets):
            ax = axes[i][j]

            if dataset in grouped_results[act_path]:
                result = grouped_results[act_path][dataset]

                # Extract data
                autocorr_data = np.array(result["autocorr_Wp"])
                anchors = result["anchors_W"]
                offsets = result["relative_offsets_p"]

                # Create heatmap (transpose to flip axes) with individual scale
                im = ax.imshow(autocorr_data.T, cmap='Blues', aspect='auto', origin='lower')

                # Add individual colorbar for this subplot
                fig.colorbar(im, ax=ax, shrink=0.6)

                # Set labels
                ax.set_xlabel('Anchors (W)')
                ax.set_ylabel('Relative Offsets (p)')
                ax.set_title(f'{act_path} / {dataset}')

                # Set ticks
                ax.set_xticks(np.arange(0, len(anchors), max(1, len(anchors)//5)))
                ax.set_xticklabels([anchors[idx] for idx in ax.get_xticks().astype(int) if idx < len(anchors)])

                ax.set_yticks(np.arange(0, len(offsets), max(1, len(offsets)//5)))
                ax.set_yticklabels([offsets[idx] for idx in ax.get_yticks().astype(int) if idx < len(offsets)])
            else:
                # Empty subplot if no data
                ax.set_title(f'{act_path} / {dataset} (No Data)')
                ax.axis('off')


    plt.tight_layout()
    savefig("autocorrelation_heatmaps", suffix=".png")


def main():
    configs = get_gemma_act_configs(
        cfg_class=AutocorrelationConfig,
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
                    "recons",
                    "residuals",
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
                    "residuals",
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
        data=WEBTEXT_DS_CFG,
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
    plot_autocorrelation(results)

if __name__ == "__main__":
    main()