import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import SymmetricalLogLocator
from scipy.optimize import curve_fit

from src.configs import *
from exp.context_exp_var import ContextExpVarConfig
from src.plotting_utils import savefig


def get_colors(n_colors):
    # Set up colors using magma colormap
    key_colors = [
        "#e7b40f",  # yellow/gold
        "#c65d3a",  # orange/red
        "#5a0e15",  # dark red/brown
    ]

    # Generate the desired number of colors
    if n_colors <= 3:
        colors = key_colors
    else:
        cmap = LinearSegmentedColormap.from_list("custom", key_colors)
        colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors


def get_unique_sorted(l):
    return sorted(list(set(l)))


def power_law(x, a, b, c):
    return a * x**b + c


def fit_and_plot_power_law(ax, x, y, color="darkorange", label_prefix="Fit"):
    """Fit a power law y = a * x^b + c to data and plot the fit."""
    # Filter out any invalid values
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    # Fit power law with offset
    params, _ = curve_fit(power_law, x_valid, y_valid, p0=[1, -0.5, 0])
    a, b, c = params

    # Generate smooth fit line
    x_fit = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 100)
    y_fit = power_law(x_fit, a, b, c)

    # Plot fit
    ax.plot(
        x_fit,
        y_fit,
        color=color,
        linestyle="-",
        lw=5,
        label=f"{label_prefix}: y = {a:.2f} * x^{{{b:.2f}}} + {c:.2f}",
    )

    return a, b, c


def plot_context_projection_paper(results, cfg, fit_growing=False):
    """

    we have two plotting modes:
    - frac var explained
    - num_components

    We want a plot with two subplots for the two metrics:
    x: window_end_pos
    y: frac_var_explained or num_components
    Multiple lines for multiple window sizes
    Plot a legend for the window sizes
    Magma colorbar for colors of window sizes

    Args:
        results: Dictionary of results
        fit_growing: If True, fit power law to "growing" window size data
    """

    # Organize data by (window_size, num_sequences) across all results
    data_by_ws_ns = defaultdict(lambda: defaultdict(list))
    window_sizes, num_sequences = [], []
    growing_data = {"x": [], "y": []}

    for filename, result in results.items():
        window_size = result["config"]["window_size"]
        window_sizes.append(window_size)
        num_seq = result["config"]["num_sequences"]
        num_sequences.append(num_seq)

        # Convert to tensors if they're lists
        window_end_pos = (
            th.tensor(result["window_end_pos"])
            if isinstance(result["window_end_pos"], list)
            else result["window_end_pos"]
        )
        frac_var_exp = (
            th.tensor(result["frac_var_exp"])
            if isinstance(result["frac_var_exp"], list)
            else result["frac_var_exp"]
        )

        data_by_ws_ns[window_size][num_seq].append(
            {
                "window_end_pos": window_end_pos.cpu().numpy(),
                "frac_var_exp": frac_var_exp.cpu().numpy(),
            }
        )

    # unique_window_sizes = get_unique_sorted(window_sizes)
    # unique_num_sequences = get_unique_sorted(num_sequences)

    colors = get_colors(len(window_sizes))

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Define linestyles and markers for different num_sequences
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]

    # Get all unique num_sequences across all window sizes
    all_num_seqs = set()
    for ws_data in data_by_ws_ns.values():
        all_num_seqs.update(ws_data.keys())
    num_seq_to_linestyle = {
        ns: linestyles[i % len(linestyles)] for i, ns in enumerate(sorted(all_num_seqs))
    }
    num_seq_to_marker = {ns: markers[i % len(markers)] for i, ns in enumerate(sorted(all_num_seqs))}

    # Plot each combination of window size and num_sequences
    for i, ws in enumerate(window_sizes):
        if ws not in data_by_ws_ns:
            continue

        # Get all num_sequences for this window size and sort them
        num_seqs = sorted(data_by_ws_ns[ws].keys())

        for num_seq in num_seqs:
            data_list = data_by_ws_ns[ws][num_seq]

            # Plot each result separately (don't concatenate)
            for idx, data in enumerate(data_list):
                window_end_pos = data["window_end_pos"]
                frac_var_exp = data["frac_var_exp"]

                # Sort by window_end_pos for proper line plotting
                sort_idx = np.argsort(window_end_pos)
                window_end_pos = window_end_pos[sort_idx]
                frac_var_exp = frac_var_exp[sort_idx]

                # Only add label for the first line of each (ws, num_seq) combination
                if idx == 0:
                    label = f"WS {ws}, N={num_seq}" if num_seq is not None else f"Window Size {ws}"
                else:
                    label = None

                linestyle = num_seq_to_linestyle.get(num_seq, "-")
                marker = num_seq_to_marker.get(num_seq, "o")

                # Special styling for window_size="growing"
                if ws == "growing":
                    plot_color = "black"
                    plot_linestyle = "--"
                    markersize = 4
                    # Collect data for fitting
                    growing_data["x"].extend(window_end_pos)
                    growing_data["y"].extend(frac_var_exp)

                    if fit_growing and growing_data["x"]:
                        x_growing = np.array(growing_data["x"])
                        y_growing = np.array(growing_data["y"])
                        fit_and_plot_power_law(
                            ax1,
                            x_growing,
                            y_growing,
                            color="darkorange",
                            label_prefix="Growing fit",
                        )
                else:
                    plot_color = colors[i]
                    plot_linestyle = linestyle
                    markersize = 2

                ax1.scatter(
                    window_end_pos,
                    frac_var_exp,
                    color=plot_color,
                    # label=label,
                    # linestyle=plot_linestyle,
                    # lw=2,
                    # marker=marker,
                    lw=markersize,
                )

    ax1.set_ylim(top=1, bottom=0)
    ax1.set_xlabel("Window End Position")
    ax1.set_ylabel("Fraction of Variance Explained")
    ax1.set_title("Explained Variance by context vs Window Position")
    ax1.legend()
    # ax1.set_yscale("symlog", linthresh=0.1)
    # ax1.yaxis.set_minor_locator(
    #     SymmetricalLogLocator(linthresh=0.1, base=10, subs=np.arange(2, 10))
    # )
    ax1.set_xscale("log")
    # plt.tight_layout()
    savefig(f"context_exp_var_{cfg.data.name}_{cfg.llm.name.split(".")[0]}", suffix=".pdf")


def main():
    configs = get_gemma_act_configs(
        cfg_class=ContextExpVarConfig,
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
        # window_size=[1, 3, 10, 32, 100, 316, "growing"],
        window_size=[1, 9, 40, 161, "growing"],
        # window_size=[1, 10, 100, "growing"],
        # window_size=["growing"],
        # window_size=[490],
        selected_context_length=500,
        num_windows_across_context=100,
        do_log_spacing=True,
        smallest_window_start=10,
        num_sequences=1000,
        # Artifacts
        env=ENV_CFG,
        # data=DatasetConfig(
        #     name="Webtext",
        #     hf_name="monology/pile-uncopyrighted",
        #     num_sequences=50000,
        #     context_length=100,
        # ),
        # data=DatasetConfig(
        #     name="Webtext",
        #     hf_name="monology/pile-uncopyrighted",
        #     num_sequences=1000,
        #     context_length=500,
        # ),
        # data=SIMPLESTORIES_DS_CFG,
        # data=CODE_DS_CFG,
        data=WEBTEXT_DS_CFG,
        llm=LLAMA3_L15_LLM_CFG,
        # llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    results = load_results_multiple_configs(
        exp_name="context_exp_var_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=None,  # compare all attributes
        verbose=False,
    )
    plot_context_projection_paper(results, configs[0], fit_growing=True)


if __name__ == "__main__":
    main()
