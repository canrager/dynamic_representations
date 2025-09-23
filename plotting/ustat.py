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
            plt.axhline(y=int(result["config"]["llm"]["hidden_dim"]), color='red', linestyle='--', label='Hidden Dim')

        plt.xlabel("Sequence Position")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.yscale("log")
        plt.xscale("log")

        # Set tick formatters to use absolute numbers instead of exponential notation
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        # plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Ensure at least 3 ticks on each axis
        plt.gca().xaxis.set_major_locator(ticker.LogLocator(numticks=5))
        # plt.gca().yaxis.set_major_locator(ticker.LogLocator(numticks=5))

        savefig(f"{metric}_across_num_sequences")


def main():
    configs = get_gemma_act_configs(
        cfg_class=IDConfig,
        reconstruction_threshold=0.9,
        min_p=20,
        max_p=499,  # 0-indexed
        num_p=7,
        do_log_spacing=True,
        # num_sequences=[10, 100, 1000, 10000],
        num_sequences=1000,
        # Artifacts
        env=ENV_CFG,
        # data=DatasetConfig(
        #     name="Webtext",
        #     hf_name="monology/pile-uncopyrighted",
        #     num_sequences=1000,
        #     context_length=500,
        # ),
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=None,  # overwritten
        act_path=None,  # overwritten
        # SAE to act_path map
        act_paths=(
            (
                [None], 
                [
                    "activations", 
                    "surrogate"
                ]
            ),
            (
                GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
                [
                    # "codes",
                    "recons"
                ]
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         # "codes",
            #         "recons"
            #     ]
            # ),
            # (
            #     [TEMPORAL_SELFTRAIN_SAE_CFG],
            #     [
            #         "novel_codes",
            #         "novel_recons",
            #         "pred_codes",
            #         "pred_recons",
            #         "total_recons",
            #     ]
            # ),
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
    plot_allinone(results)


if __name__ == "__main__":
    main()
