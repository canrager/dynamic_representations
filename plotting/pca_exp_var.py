import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap

from src.configs import *
from exp.pca_exp_var import PCAExpVarConfig
from src.plotting_utils import savefig


def plot_pca_exp_var(results):
    """
    Each result file is structured as
    {
        window_size1: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        window_size2: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        ...
    }

    we have two plotting modes:
    - frac var explained
    - num_components

    We want a plot with two subplots for the two metrics:
    x: window_end_pos
    y: frac_var_explained or num_components
    Multiple lines for multiple window sizes
    Plot a legend for the window sizes
    Magma colorbar for colors of window sizes

    """
    # Extract window sizes and sort them

    result_dict = results[next(iter(results.keys()))]

    window_sizes = []
    for key in result_dict.keys():
        if not key in ["config", "_filename"]:
            window_sizes.append(int(key))
    window_sizes.sort()

    # Set up colors using magma colormap
    colors = plt.cm.magma(np.linspace(0.1, 0.9, len(window_sizes)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot fraction of variance explained
    for i, ws in enumerate(window_sizes):
        ws_str = str(ws)
        data = result_dict[ws_str]

        # Convert to tensors if they're lists
        window_end_pos = (
            th.tensor(data["window_end_pos"])
            if isinstance(data["window_end_pos"], list)
            else data["window_end_pos"]
        )
        frac_var_exp = (
            th.tensor(data["frac_var_exp"])
            if isinstance(data["frac_var_exp"], list)
            else data["frac_var_exp"]
        )

        # Only plot the last point if window size is 490
        if ws == 490:
            ax1.plot(
                window_end_pos.cpu().numpy()[-1:],
                frac_var_exp.cpu().numpy()[-1:],
                color=colors[i],
                marker="o",
                label=f"Window size {ws}",
            )
        else:
            ax1.plot(
                window_end_pos.cpu().numpy(),
                frac_var_exp.cpu().numpy(),
                color=colors[i],
                marker="o",
                label=f"Window size {ws}",
            )

    ax1.set_ylim((0, 1))
    ax1.set_xlabel("Window End Position")
    ax1.set_ylabel("Fraction of Variance Explained")
    ax1.set_title("PCA Explained Variance vs Window Position")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot number of PCA components
    for i, ws in enumerate(window_sizes):
        ws_str = str(ws)
        data = result_dict[ws_str]

        # Convert to tensors if they're lists
        window_end_pos = (
            th.tensor(data["window_end_pos"])
            if isinstance(data["window_end_pos"], list)
            else data["window_end_pos"]
        )
        num_components = (
            th.tensor(data["num_pca_components"])
            if isinstance(data["num_pca_components"], list)
            else data["num_pca_components"]
        )

        # Only plot the last point if window size is 490
        if ws == 490:
            ax2.plot(
                window_end_pos.cpu().numpy()[-1:],
                num_components.cpu().numpy()[-1:],
                color=colors[i],
                marker="o",
                label=f"Window size {ws}",
            )
        else:
            ax2.plot(
                window_end_pos.cpu().numpy(),
                num_components.cpu().numpy(),
                color=colors[i],
                marker="o",
                label=f"Window size {ws}",
            )

    ax2.set_xlabel("Window End Position")
    ax2.set_ylabel("Number of PCA Components")
    ax2.set_title("PCA Components vs Window Position")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig("pca_exp_var", suffix=".png")


def plot_pca_paper(results):
    """
    Each result file is structured as
    {
        window_size1: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        window_size2: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        ...
    }

    we have two plotting modes:
    - frac var explained
    - num_components

    We want a plot with two subplots for the two metrics:
    x: window_end_pos
    y: frac_var_explained or num_components
    Multiple lines for multiple window sizes
    Plot a legend for the window sizes
    Magma colorbar for colors of window sizes

    """
    # Extract window sizes and sort them

    result_dict = results[next(iter(results.keys()))]

    window_sizes = []
    for key in result_dict.keys():
        if not key in ["config", "_filename"]:
            window_sizes.append(int(key))
    window_sizes.sort()

    # Set up colors using magma colormap

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Plot fraction of variance explained
    for i, ws in enumerate(window_sizes):
        ws_str = str(ws)
        data = result_dict[ws_str]

        # Convert to tensors if they're lists
        window_end_pos = (
            th.tensor(data["window_end_pos"])
            if isinstance(data["window_end_pos"], list)
            else data["window_end_pos"]
        )
        frac_var_exp = (
            th.tensor(data["frac_var_exp"])
            if isinstance(data["frac_var_exp"], list)
            else data["frac_var_exp"]
        )
        baseline_frac_var_exp = (
            th.tensor(data["baseline_frac_var_exp"])
            if isinstance(data["baseline_frac_var_exp"], list)
            else data["baseline_frac_var_exp"]
        )

        # Only plot the last point if window size is 490

        ax1.plot(
            window_end_pos.cpu().numpy(),
            frac_var_exp.cpu().numpy(),
            color=colors[i],
            label=f"Window Size {ws}",
            lw=5,
        )
        # if i == 0:
        #     ax1.plot(
        #         window_end_pos.cpu().numpy(),
        #         baseline_frac_var_exp.cpu().numpy(),
        #         color="#3d2f22",
        #         label=f"Baseline",
        #         lw=5,
        #         linestyle="--"
        #     )

    ax1.set_ylim(top=1, bottom=0)
    ax1.set_xlabel("Window End Position")
    ax1.set_ylabel("Fraction of Variance Explained")
    ax1.set_title("PCA Explained Variance vs Window Position")
    ax1.legend(loc="lower right")
    plt.tight_layout()
    savefig("pca_exp_var", suffix=".pdf")


def plot_context_projection_paper(results):
    """
    Each result file is structured as
    {
        window_size1: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        window_size2: {
            window_end_pos: Tensor,
            frac_var_explained: Tensor,
            num_components: Tensor
        },
        ...
    }

    we have two plotting modes:
    - frac var explained
    - num_components

    We want a plot with two subplots for the two metrics:
    x: window_end_pos
    y: frac_var_explained or num_components
    Multiple lines for multiple window sizes
    Plot a legend for the window sizes
    Magma colorbar for colors of window sizes

    """
    # Extract window sizes and sort them

    result_dict = results[next(iter(results.keys()))]

    window_sizes = []
    for key in result_dict.keys():
        if not key in ["config", "_filename"]:
            window_sizes.append(int(key))
    window_sizes.sort()

    # Set up colors using magma colormap
    # Your three key colors
    key_colors = [
        "#e7b40f",  # yellow/gold
        "#c65d3a",  # orange/red
        "#5a0e15",  # dark red/brown
    ]

    # Create a custom colormap from these three colors

    # Generate the desired number of colors
    n_colors = len(window_sizes)  # or whatever number you need
    if n_colors <= 3:
        colors = key_colors
    else:
        cmap = LinearSegmentedColormap.from_list("custom", key_colors)
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Plot fraction of variance explained
    for i, ws in enumerate(window_sizes):
        ws_str = str(ws)
        data = result_dict[ws_str]

        # Convert to tensors if they're lists
        window_end_pos = (
            th.tensor(data["window_end_pos"])
            if isinstance(data["window_end_pos"], list)
            else data["window_end_pos"]
        )
        frac_var_exp = (
            th.tensor(data["frac_var_exp"])
            if isinstance(data["frac_var_exp"], list)
            else data["frac_var_exp"]
        )

        # Only plot the last point if window size is 490

        ax1.plot(
            window_end_pos.cpu().numpy(),
            frac_var_exp.cpu().numpy(),
            color=colors[i],
            label=f"Window Size {ws}",
            lw=5,
        )

    ax1.set_ylim(top=1, bottom=0)
    ax1.set_xlabel("Window End Position")
    ax1.set_ylabel("Fraction of Variance Explained")
    ax1.set_title("Explained Variance by context vs Window Position")
    ax1.legend(loc="upper right")
    plt.tight_layout()
    savefig("pca_exp_var", suffix=".png")


def main():
    configs = get_gemma_act_configs(
        cfg_class=PCAExpVarConfig,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    # "surrogate"
                ],
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         # "codes",
            #         "recons",
            #         # "residuals",
            #     ],
            # ),
            # (
            #     [TEMPORAL_SELFTRAIN_SAE_CFG],
            #     [
            #         # "novel_codes",
            #         # "novel_recons",
            #         # "pred_codes",
            #         # "pred_recons",
            #         "total_recons",
            #         # "residuals",
            #     ],
            # ),
        ),
        # window_sizes=[
        #     [1, 3, 10, 32, 100, 316]
        # ],  # Choosing tuple so it will not be resulting in separate configs
        window_sizes=[[
            # 1,
            # 10,
            100,
        ]],
        # min_total_tokens_per_window=500_000,
        selected_context_length=500,
        # num_windows=50,
        num_windows=5,
        do_log_spacing=False,
        smallest_window_start=2,
        reconstruction_thresh=0.9,
        # Artifacts
        env=ENV_CFG,
        data=DatasetConfig(
            name="Webtext",
            hf_name="monology/pile-uncopyrighted",
            num_sequences=1000,
            context_length=500,
        ),
        # llm=LLAMA3_LLM_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    results = load_results_multiple_configs(
        exp_name="pca_exp_var_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=True,
    )
    plot_context_projection_paper(results)


if __name__ == "__main__":
    main()
