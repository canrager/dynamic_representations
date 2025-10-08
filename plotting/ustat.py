"""
Plotting U statistic and rank.

- across layers
- across llm, snapshot, temporal
- across datasets

1. All reconstructions, ustat and rank separately
2. All codes, ustat and rank separately
3. All individual: ustat and rank on the same plot
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import numpy as np

from exp.ustat import IDConfig
from src.configs import *
from src.plotting_utils import savefig


def plot_allinone(results, show_hidden_dim=False, show_num_sequences=False):
    metrics = ["ustat", "rank"]
    for metric in metrics:
        for result_name, result in results.items():
            label = result["config"]["act_path"]
            linestyle = "-"  # default solid line
            if result["config"]["sae"]:
                sae_name = result["config"]["sae"]["name"]
                label = f"{sae_name} / {label}"
                # Use dashed line for architectures that don't contain "temporal"
                if "temporal" not in sae_name:
                    linestyle = "--"
            else:
                # Use dotted line for cases with no SAE
                linestyle = ":"

            if show_num_sequences:
                label += f"{result["config"]["num_sequences"]}seq"

            plt.plot(
                result["ps"],
                result[f"{metric}_p"],
                label=label,
                linestyle=linestyle,
                marker="o",
            )

        if show_hidden_dim:
            plt.axhline(
                y=int(result["config"]["llm"]["hidden_dim"]),
                color="red",
                linestyle="--",
                label="Hidden Dim",
            )

        plt.xlabel("Sequence Position")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.yscale("log")
        plt.xscale("log")

        # Set tick formatters to use absolute numbers instead of exponential notation
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        # plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Ensure at least 3 ticks on each axis
        plt.gca().xaxis.set_major_locator(ticker.LogLocator(numticks=5))
        # plt.gca().yaxis.set_major_locator(ticker.LogLocator(numticks=5))

        savefig(f"{metric}_across_num_sequences")


def plot_llm_figure(results, show_hidden_dim=False, show_num_sequences=False, cfg=None):
    metrics = ["rank"]

    def aggregate_data(x_data, y_data, bin_size=10):
        """Aggregate data into bins with mean and 95% CI"""
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        y_tensor = torch.tensor(y_data, dtype=torch.float32)

        # Create bins
        n_bins = len(x_data) // bin_size
        if len(x_data) % bin_size != 0:
            n_bins += 1

        x_binned = []
        y_mean = []
        y_lower = []
        y_upper = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(x_data))

            if end_idx > start_idx:
                x_bin = x_tensor[start_idx:end_idx]
                y_bin = y_tensor[start_idx:end_idx]

                # Calculate statistics
                x_binned.append(x_bin.mean().item())
                y_mean_val = y_bin.mean().item()
                y_std = y_bin.std().item()

                # 95% CI approximation (1.96 * std / sqrt(n))
                n_samples = len(y_bin)
                ci_margin = 1.96 * y_std / torch.sqrt(torch.tensor(float(n_samples)))

                y_mean.append(y_mean_val)
                y_lower.append(y_mean_val - ci_margin.item())
                y_upper.append(y_mean_val + ci_margin.item())

        return x_binned, y_mean, y_lower, y_upper

    for metric in metrics:
        plt.figure(figsize=(6,6))
        for result_name, result in results.items():
            # Only plot activations (not surrogate files)
            if result["config"]["act_path"] == "activations":
                if show_num_sequences:
                    label_suffix = f"{result['config']['num_sequences']}seq"
                else:
                    label_suffix = ""

                # Aggregate actual data
                x_actual, y_actual, y_actual_lower, y_actual_upper = aggregate_data(
                    result["ps"], result[f"{metric}_p"]
                )

                # Aggregate surrogate data
                x_surrogate, y_surrogate, y_surrogate_lower, y_surrogate_upper = aggregate_data(
                    result["ps"], result[f"surrogate_{metric}_p"]
                )

                # Plot actual data as line with shaded error
                plt.plot(x_actual, y_actual, label=f"Original {label_suffix}",
                        linewidth=5, color="#c65d3a")
                plt.fill_between(x_actual, y_actual_lower, y_actual_upper,
                               alpha=0.3, color="#c65d3a")

                # Plot surrogate data as line with shaded error
                plt.plot(x_surrogate, y_surrogate, label=f"Surrogate {label_suffix}", linestyle="--",
                        linewidth=5, color="#000000")
                plt.fill_between(x_surrogate, y_surrogate_lower, y_surrogate_upper,
                               alpha=0.3, color="#000000")

        if show_hidden_dim:
            plt.axhline(
                y=int(result["config"]["llm"]["hidden_dim"]),
                color="red",
                linestyle="--",
                label="Hidden Dim",
            )

        plt.xlabel("Sequence Position")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.yscale("log")
        plt.xscale("log")

        # Set tick formatters to use absolute numbers instead of exponential notation
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        # plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Ensure at least 3 ticks on each axis
        plt.gca().xaxis.set_major_locator(ticker.LogLocator(numticks=5))
        # plt.gca().yaxis.set_major_locator(ticker.LogLocator(numticks=5))

        savefig(f"{metric}_vs_surrogate_comparison_{cfg.llm.name}")


def plot_datasets_figure(results, show_hidden_dim=False, show_num_sequences=False, cfg=None):
    metrics = ["ustat"]

    def aggregate_data(x_data, y_data, bin_size=10):
        """Aggregate data into bins with mean and 95% CI"""
        device = cfg.env.device if cfg and hasattr(cfg, 'env') and hasattr(cfg.env, 'device') else 'cpu'
        x_tensor = torch.tensor(x_data, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)

        # Create bins
        n_bins = len(x_data) // bin_size
        if len(x_data) % bin_size != 0:
            n_bins += 1

        x_binned = []
        y_mean = []
        y_lower = []
        y_upper = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(x_data))

            if end_idx > start_idx:
                x_bin = x_tensor[start_idx:end_idx]
                y_bin = y_tensor[start_idx:end_idx]

                # Calculate statistics
                x_binned.append(x_bin.mean().item())
                y_mean_val = y_bin.mean().item()
                y_std = y_bin.std().item()

                # 95% CI approximation (1.96 * std / sqrt(n))
                n_samples = len(y_bin)
                ci_margin = 1.96 * y_std / torch.sqrt(torch.tensor(float(n_samples), device=device))

                y_mean.append(y_mean_val)
                y_lower.append(y_mean_val - ci_margin.item())
                y_upper.append(y_mean_val + ci_margin.item())

        return x_binned, y_mean, y_lower, y_upper

    # Group results by dataset
    datasets = {}
    for result_name, result in results.items():
        dataset_name = result["config"]["data"]["name"]
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][result_name] = result

    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (dataset_name, dataset_results) in enumerate(datasets.items()):
            ax = axes[idx]
            plt.sca(ax)

            for result_name, result in dataset_results.items():
                # Only plot activations (not surrogate files)
                if result["config"]["act_path"] == "activations":
                    if show_num_sequences:
                        label_suffix = f"{result['config']['num_sequences']}seq"
                    else:
                        label_suffix = ""

                    # Aggregate actual data
                    x_actual, y_actual, y_actual_lower, y_actual_upper = aggregate_data(
                        result["ps"], result[f"{metric}_p"]
                    )

                    # Aggregate surrogate data
                    x_surrogate, y_surrogate, y_surrogate_lower, y_surrogate_upper = aggregate_data(
                        result["ps"], result[f"surrogate_{metric}_p"]
                    )

                    # Plot actual data as line with shaded error
                    plt.plot(x_actual, y_actual, label=f"Original {label_suffix}",
                            linewidth=5, color="#c65d3a")
                    plt.fill_between(x_actual, y_actual_lower, y_actual_upper,
                                   alpha=0.3, color="#c65d3a")

                    # Plot surrogate data as line with shaded error
                    plt.plot(x_surrogate, y_surrogate, label=f"Surrogate {label_suffix}", linestyle="--",
                            linewidth=5, color="#000000")
                    plt.fill_between(x_surrogate, y_surrogate_lower, y_surrogate_upper,
                                   alpha=0.3, color="#000000")

            if show_hidden_dim:
                plt.axhline(
                    y=int(result["config"]["llm"]["hidden_dim"]),
                    color="red",
                    linestyle="--",
                    label="Hidden Dim",
                )

            plt.xlabel("Sequence Position")
            plt.ylabel(metric)
            plt.title(dataset_name)
            plt.legend()
            plt.xscale("log")

            # Set tick formatters to use absolute numbers instead of exponential notation
            plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

            # Ensure at least 3 ticks on each axis
            plt.gca().xaxis.set_major_locator(ticker.LogLocator(numticks=5))

        plt.tight_layout()
        savefig(f"{metric}_datasets_comparison_{cfg.llm.name}", suffix=".png")


def plot_sae_datasets_figure(results, show_hidden_dim=False, show_num_sequences=False, cfg=None):
    metrics = ["rank", "ustat"]

    def aggregate_data(x_data, y_data, bin_size=10):
        """Aggregate data into bins with mean and 95% CI"""
        device = cfg.env.device if cfg and hasattr(cfg, 'env') and hasattr(cfg.env, 'device') else 'cpu'
        x_tensor = torch.tensor(x_data, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_data, dtype=torch.float32, device=device)

        # Create bins
        n_bins = len(x_data) // bin_size
        if len(x_data) % bin_size != 0:
            n_bins += 1

        x_binned = []
        y_mean = []
        y_lower = []
        y_upper = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(x_data))

            if end_idx > start_idx:
                x_bin = x_tensor[start_idx:end_idx]
                y_bin = y_tensor[start_idx:end_idx]

                # Calculate statistics
                x_binned.append(x_bin.mean().item())
                y_mean_val = y_bin.mean().item()
                y_std = y_bin.std().item()

                # 95% CI approximation (1.96 * std / sqrt(n))
                n_samples = len(y_bin)
                ci_margin = 1.96 * y_std / torch.sqrt(torch.tensor(float(n_samples), device=device))

                y_mean.append(y_mean_val)
                y_lower.append(y_mean_val - ci_margin.item())
                y_upper.append(y_mean_val + ci_margin.item())

        return x_binned, y_mean, y_lower, y_upper

    # Group results by dataset
    datasets = {}
    for result_name, result in results.items():
        dataset_name = result["config"]["data"]["name"]
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][result_name] = result

    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (dataset_name, dataset_results) in enumerate(datasets.items()):
            ax = axes[idx]
            plt.sca(ax)

            for result_name, result in dataset_results.items():
                if show_num_sequences:
                    label_suffix = f" {result['config']['num_sequences']}seq"
                else:
                    label_suffix = ""

                # Plot LLM activations if available
                if result["config"]["act_path"] == "activations":
                    # Aggregate actual data
                    x_actual, y_actual, y_actual_lower, y_actual_upper = aggregate_data(
                        result["ps"], result[f"{metric}_p"]
                    )

                    # Plot actual data
                    plt.plot(x_actual, y_actual, label=f"Original{label_suffix}",
                            linewidth=3, color="#c65d3a")
                    plt.fill_between(x_actual, y_actual_lower, y_actual_upper,
                                   alpha=0.3, color="#c65d3a")

                    # Plot surrogate data if available
                    if f"surrogate_{metric}_p" in result:
                        x_surrogate, y_surrogate, y_surrogate_lower, y_surrogate_upper = aggregate_data(
                            result["ps"], result[f"surrogate_{metric}_p"]
                        )

                        plt.plot(x_surrogate, y_surrogate, label=f"Surrogate{label_suffix}", linestyle="--",
                                linewidth=3, color="#000000")
                        plt.fill_between(x_surrogate, y_surrogate_lower, y_surrogate_upper,
                                       alpha=0.3, color="#000000")

                # Plot SAE data
                else:
                    sae_name = result["config"]["sae"]["name"] if result["config"]["sae"] else "No SAE"
                    act_path = result["config"]["act_path"]
                    label = f"{sae_name} {act_path}{label_suffix}"

                    # Aggregate data
                    x_data, y_data, y_lower, y_upper = aggregate_data(
                        result["ps"], result[f"{metric}_p"]
                    )

                    # Plot data as line with shaded error
                    plt.plot(x_data, y_data, label=label, linewidth=3, marker='o', markersize=3)
                    plt.fill_between(x_data, y_lower, y_upper, alpha=0.2)

            if show_hidden_dim:
                plt.axhline(
                    y=int(result["config"]["llm"]["hidden_dim"]),
                    color="red",
                    linestyle="--",
                    label="Hidden Dim",
                )

            plt.xlabel("Sequence Position")
            plt.ylabel(metric)
            plt.yscale("log")
            plt.title(dataset_name)
            plt.legend()
            plt.xscale("log")

            # Set tick formatters to use absolute numbers instead of exponential notation
            plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

            # Ensure at least 3 ticks on each axis
            plt.gca().xaxis.set_major_locator(ticker.LogLocator(numticks=5))

        plt.tight_layout()
        savefig(f"{metric}_sae_datasets_comparison_{cfg.llm.name}", suffix=".png")


def main():
    configs = get_gemma_act_configs(
        cfg_class=IDConfig,
        reconstruction_threshold=0.9,
        min_p=20,
        max_p=499,  # 0-indexed
        num_p=200,
        do_log_spacing=True,
        # num_sequences=[10, 100, 1000, 10000],
        num_sequences=10000,
        # Artifacts
        env=ENV_CFG,
        # data=DatasetConfig(
        #     name="Webtext",
        #     hf_name="monology/pile-uncopyrighted",
        #     num_sequences=1000,
        #     context_length=500,
        # ),
        # data=WEBTEXT_DS_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        llm=GEMMA2_LLM_CFG,
        sae=None,  # overwritten
        act_path=None,  # overwritten
        # SAE to act_path map
        act_paths=(
            # ([None], ["activations"]),
            (
                GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
                [
                    # "codes",
                    # "recons"
                ]
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         # "codes",
            #         "recons"
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
                ]
            ),
        ),
    )

    results = load_results_multiple_configs(
        exp_name="id_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=True,
    )
    plot_sae_datasets_figure(results, cfg=configs[0])


if __name__ == "__main__":
    main()
