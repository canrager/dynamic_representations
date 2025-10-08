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
    cmap: Greys
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
                    im = ax.imshow(autocorr_data.T, cmap='Greys', aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
                    images.append(im)
                else:
                    im = ax.imshow(autocorr_data.T, cmap='Greys', aspect='equal', origin='lower')
                    # Add individual colorbar for this subplot
                    fig.colorbar(im, ax=ax, shrink=0.6)

                # Force square aspect ratio for the axes
                ax.set_aspect('equal', adjustable='box')

                # Set labels
                ax.set_xlabel('Sequence Position (W)')
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


def plot_autocorr_vs_surrogate(results, global_colorbar=False, row_colorbar=False, cfg=None):
    """
    Plot autocorrelation vs surrogate with datasets as columns and autocorr/surrogate as rows

    Args:
        results: Dictionary of results from autocorrelation experiments
        global_colorbar: If True, use single colorbar with global scale across all subplots.
        row_colorbar: If True, use shared colorbar per column (one per dataset spanning autocorr and surrogate).
        cfg: Configuration object (used for filename)
    """
    # Get unique datasets
    datasets = list(set(result["config"]["data"]["name"] for result in results.values()))
    num_datasets = len(datasets)

    fig, axes = plt.subplots(2, num_datasets, figsize=(5*num_datasets, 12))

    # Handle single dataset case
    if num_datasets == 1:
        axes = axes.reshape(2, 1)

    # Collect all data for global/row colorbar
    if global_colorbar:
        all_autocorr_data = []
        all_surrogate_data = []
        for result in results.values():
            all_autocorr_data.append(np.array(result["autocorr_Wp"]))
            all_surrogate_data.append(np.array(result["surrogate_Wp"]))

        if all_autocorr_data and all_surrogate_data:
            autocorr_vmin = np.min([np.min(data) for data in all_autocorr_data])
            autocorr_vmax = np.max([np.max(data) for data in all_autocorr_data])
            surrogate_vmin = np.min([np.min(data) for data in all_surrogate_data])
            surrogate_vmax = np.max([np.max(data) for data in all_surrogate_data])

            # Use same scale for both
            vmin = min(autocorr_vmin, surrogate_vmin)
            vmax = max(autocorr_vmax, surrogate_vmax)
        else:
            vmin, vmax = 0, 1
    elif row_colorbar:
        # For row colorbar, we'll compute per-column (dataset) scales during plotting
        vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    images = []

    for j, dataset in enumerate(datasets):
        # Find result for this dataset
        dataset_result = None
        for result in results.values():
            if result["config"]["data"]["name"] == dataset:
                dataset_result = result
                break

        if dataset_result is None:
            # No data for this dataset
            axes[0][j].set_title(f'{dataset} - Autocorr (No Data)')
            axes[0][j].axis('off')
            axes[1][j].set_title(f'{dataset} - Surrogate (No Data)')
            axes[1][j].axis('off')
            continue

        # Extract data
        autocorr_data = np.array(dataset_result["autocorr_Wp"])
        surrogate_data = np.array(dataset_result["surrogate_Wp"])
        anchors = dataset_result["anchors_W"]
        offsets = dataset_result["relative_offsets_p"]

        # For row colorbar, compute per-column (dataset) scale
        if row_colorbar:
            col_vmin = min(np.min(autocorr_data), np.min(surrogate_data))
            col_vmax = max(np.max(autocorr_data), np.max(surrogate_data))
        else:
            col_vmin, col_vmax = vmin, vmax

        # Plot autocorrelation (row 0)
        ax_auto = axes[0][j]
        if global_colorbar:
            im_auto = ax_auto.imshow(autocorr_data.T, cmap='Greys', aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
            images.append(im_auto)
        elif row_colorbar:
            im_auto = ax_auto.imshow(autocorr_data.T, cmap='Greys', aspect='equal', origin='lower', vmin=col_vmin, vmax=col_vmax)
        else:
            im_auto = ax_auto.imshow(autocorr_data.T, cmap='Greys', aspect='equal', origin='lower')
            fig.colorbar(im_auto, ax=ax_auto, shrink=0.6)

        ax_auto.set_aspect('equal', adjustable='box')
        ax_auto.set_xlabel('Sequence Position (W)')
        ax_auto.set_ylabel('Relative Offsets (p)')
        ax_auto.set_title(f'{dataset} - Autocorrelation')

        # Set ticks for autocorrelation
        ax_auto.set_xticks(np.arange(0, len(anchors), max(1, len(anchors)//5)))
        ax_auto.set_xticklabels([anchors[idx] for idx in ax_auto.get_xticks().astype(int) if idx < len(anchors)])
        ax_auto.set_yticks(np.arange(0, len(offsets), max(1, len(offsets)//5)))
        ax_auto.set_yticklabels([offsets[idx] for idx in ax_auto.get_yticks().astype(int) if idx < len(offsets)])

        # Plot surrogate (row 1)
        ax_surr = axes[1][j]
        if global_colorbar:
            im_surr = ax_surr.imshow(surrogate_data.T, cmap='Greys', vmin=vmin, vmax=vmax)
            if not images:  # Only append if we haven't added autocorr image
                images.append(im_surr)
        elif row_colorbar:
            im_surr = ax_surr.imshow(surrogate_data.T, cmap='Greys', vmin=col_vmin, vmax=col_vmax)
            # Add shared colorbar for this column (dataset)
            fig.colorbar(im_surr, ax=[ax_auto, ax_surr], shrink=0.6)
        else:
            im_surr = ax_surr.imshow(surrogate_data.T, cmap='Greys')
            fig.colorbar(im_surr, ax=ax_surr, shrink=0.6)

        ax_surr.set_aspect('equal', adjustable='box')
        ax_surr.set_xlabel('Sequence Position (W)')
        ax_surr.set_ylabel('Relative Offsets (p)')
        ax_surr.set_title(f'{dataset} - Surrogate')

        # Set ticks for surrogate
        ax_surr.set_xticks(np.arange(0, len(anchors), max(1, len(anchors)//5)))
        ax_surr.set_xticklabels([anchors[idx] for idx in ax_surr.get_xticks().astype(int) if idx < len(anchors)])
        ax_surr.set_yticks(np.arange(0, len(offsets), max(1, len(offsets)//5)))
        ax_surr.set_yticklabels([offsets[idx] for idx in ax_surr.get_yticks().astype(int) if idx < len(offsets)])


    # Add global colorbar if requested
    if global_colorbar and images:
        fig.colorbar(images[0], ax=axes, shrink=0.6)

    savefig(f"autocorr_vs_surrogate_heatmaps_{cfg.llm.name}", suffix=".png")


def plot_sae_autocorrelation(results, global_colorbar=False, row_colorbar=False, cfg=None):
    """
    Plot autocorrelation heatmaps for SAE results across different activation types and datasets

    Args:
        results: Dictionary of results from autocorrelation experiments
        global_colorbar: If True, use single colorbar with global scale across all subplots.
        row_colorbar: If True, use shared colorbar per column (one per dataset spanning all activation types).
        cfg: Configuration object (used for filename)
    """
    # Group results by SAE name, activation type, and dataset
    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for result_name, result in results.items():
        if result["config"]["sae"] is None:
            continue  # Skip non-SAE results

        sae_name = result["config"]["sae"]["name"]
        act_type = result["config"]["act_path"]
        dataset_name = result["config"]["data"]["name"]
        grouped_results[sae_name][act_type][dataset_name] = result

    if not grouped_results:
        print("No SAE results found")
        return

    # Get unique datasets across all SAEs
    all_datasets = set()
    for sae_data in grouped_results.values():
        for act_type, dataset_data in sae_data.items():
            all_datasets.update(dataset_data.keys())

    datasets = sorted(list(all_datasets))
    sae_names = sorted(list(grouped_results.keys()))

    num_datasets = len(datasets)

    # Build row structure: each SAE gets as many rows as it has activation types
    row_structure = []  # List of (sae_name, act_type) tuples
    for sae_name in sae_names:
        sae_data = grouped_results[sae_name]
        sae_act_types = sorted(sae_data.keys())
        for act_type in sae_act_types:
            row_structure.append((sae_name, act_type))

    total_rows = len(row_structure)
    fig, axes = plt.subplots(total_rows, num_datasets, figsize=(4*num_datasets, 3*total_rows))

    # Handle single subplot cases
    if total_rows == 1 and num_datasets == 1:
        axes = [[axes]]
    elif total_rows == 1:
        axes = [axes]
    elif num_datasets == 1:
        axes = [[ax] for ax in axes]

    # Collect all data for global colorbar
    if global_colorbar:
        all_data = []
        for sae_data in grouped_results.values():
            for act_type_data in sae_data.values():
                for result in act_type_data.values():
                    # Subsample the data for consistent color scaling
                    autocorr_data = np.array(result["autocorr_Wp"])
                    autocorr_data_subsampled = autocorr_data[:, ::4]
                    all_data.append(autocorr_data_subsampled)

        if all_data:
            vmin = np.min([np.min(data) for data in all_data])
            vmax = np.max([np.max(data) for data in all_data])
        else:
            vmin, vmax = 0, 1
    else:
        vmin, vmax = None, None

    images = []

    # Pre-compute column scales for row_colorbar mode
    col_scales = {}
    if row_colorbar:
        for j, dataset in enumerate(datasets):
            col_data = []
            for sae_name, act_type in row_structure:
                sae_data = grouped_results[sae_name]
                if act_type in sae_data and dataset in sae_data[act_type]:
                    # Subsample the data for consistent color scaling
                    autocorr_data = np.array(sae_data[act_type][dataset]["autocorr_Wp"])
                    autocorr_data_subsampled = autocorr_data[:, ::4]
                    col_data.append(autocorr_data_subsampled)

            if col_data:
                col_scales[j] = (np.min([np.min(data) for data in col_data]),
                               np.max([np.max(data) for data in col_data]))
            else:
                col_scales[j] = (0, 1)

    # Plot all SAE/activation type combinations
    for row_idx, (sae_name, act_type) in enumerate(row_structure):
        sae_data = grouped_results[sae_name]

        for j, dataset in enumerate(datasets):
            ax = axes[row_idx][j]

            if act_type in sae_data and dataset in sae_data[act_type]:
                result = sae_data[act_type][dataset]

                # Extract data
                autocorr_data = np.array(result["autocorr_Wp"])
                anchors = result["anchors_W"]
                offsets = result["relative_offsets_p"]

                # Subsample every 4th row (y-axis corresponds to offsets/lag values)
                autocorr_data_subsampled = autocorr_data[:, ::4]
                offsets_subsampled = [offsets[i] for i in range(0, len(offsets), 4)]

                # Set colorbar scale
                if row_colorbar:
                    col_vmin, col_vmax = col_scales[j]
                else:
                    col_vmin, col_vmax = vmin, vmax

                # Plot heatmap
                if global_colorbar:
                    im = ax.imshow(autocorr_data_subsampled.T, cmap='Greys', aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
                    images.append(im)
                elif row_colorbar:
                    im = ax.imshow(autocorr_data_subsampled.T, cmap='Greys', aspect='equal', origin='lower', vmin=col_vmin, vmax=col_vmax)
                    # Add column colorbar only for the last row
                    if row_idx == total_rows - 1:
                        fig.colorbar(im, ax=[axes[k][j] for k in range(total_rows)], shrink=0.6)
                else:
                    im = ax.imshow(autocorr_data_subsampled.T, cmap='Greys', aspect='equal', origin='lower')
                    fig.colorbar(im, ax=ax, shrink=0.6)

                ax.set_aspect('equal', adjustable='box')
                if row_idx == len(row_structure)-1:
                    ax.set_xlabel('Sequence Position (W)')
                if j == 0:
                    ax.set_ylabel('Relative Offsets (p)')
                # ax.set_title(f'{sae_name}/{act_type} - {dataset}')

                # Set ticks
                ax.set_xticks(np.arange(0, len(anchors), max(1, len(anchors)//5)))
                ax.set_xticklabels([anchors[idx] for idx in ax.get_xticks().astype(int) if idx < len(anchors)])
                ax.set_yticks(np.arange(0, len(offsets_subsampled), max(1, len(offsets_subsampled)//5)))
                ax.set_yticklabels([offsets_subsampled[idx] for idx in ax.get_yticks().astype(int) if idx < len(offsets_subsampled)])
            else:
                # Empty subplot if no data
                ax.set_title(f'{sae_name}/{act_type} - {dataset} (No Data)')
                ax.axis('off')

    # Add row titles (SAE/activation type combinations)
    for row_idx, (sae_name, act_type) in enumerate(row_structure):
        title = f"{sae_name}\n{act_type}\n\n"
        axes[row_idx][0].text(-0.2, 0.5, title, transform=axes[row_idx][0].transAxes,
                             fontsize=10, fontweight='bold', ha='right', va='center', rotation=90)

    # Add column titles (datasets)
    for j, dataset in enumerate(datasets):
        axes[0][j].text(0.5, 1.1, dataset, transform=axes[0][j].transAxes,
                       fontsize=12, fontweight='bold', ha='center', va='bottom')

    # Add global colorbar if requested
    if global_colorbar and images:
        fig.colorbar(images[0], ax=axes, shrink=0.6)

    # plt.tight_layout()
    savefig(f"sae_autocorr_heatmaps_all_{cfg.llm.name}", suffix=".png")


def main():
    configs = get_gemma_act_configs(
        cfg_class=AutocorrelationConfig,
        act_paths=(
            # (
            #     [None],
            #     [
            #         "activations",
            #         # "surrogate"
            #     ]
            # ),
            (
                GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
                [
                    "codes",
                    # "recons",
                    # "residuals"
                ]
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         "codes",
            #         # "recons",
            #         # "residuals",
            #     ]
            # ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    "novel_codes",
                    # "novel_recons",
                    "pred_codes",
                    # "pred_recons",
                    # "total_recons",
                    # "residuals",
                ]
            ),
        ),
        min_anchor=49,
        max_anchor=499,  # selected_context_length, inclusive
        num_anchors=10,
        min_offset=10,  # absolute value
        max_offset=30,  # absolute value
        # Artifacts
        env=ENV_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        # data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG],
        # data=WEBTEXT_DS_CFG,
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
        verbose=False,
    )
    # plot_autocorrelation(results, global_colorbar=False)
    # plot_autocorr_vs_surrogate(results, global_colorbar=True, cfg=configs[0])
    # plot_autocorr_vs_surrogate(results, row_colorbar=True, cfg=configs[0])
    # print([r["config"] for r in results.values()])
    plot_sae_autocorrelation(results, row_colorbar=False, cfg=configs[0])

if __name__ == "__main__":
    main()