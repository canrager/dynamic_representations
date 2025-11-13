import torch as th
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pickle as pkl
import os
import plotly.graph_objects as go

from src.configs import *
from exp.geometry_population import GeometryPopulationConfig, is_user
from src.plotting_utils import savefig
from src.geometry_utils import (
    make_fig_like_plotly,
    make_fig_like_plotly_with_user_labels,
    make_fig_like_plotly_flexible,
    make_fig_like_plotly_with_user_labels_flexible,
)


def plot_geometry_population_with_user_labels(results, tokens_BP, cfg):
    """
    Plot population geometry using 6 panels with user/model coloring.
    Black for user tokens, end color for model tokens.

    Args:
        results: Dictionary mapping filenames to result dictionaries
        tokens_BP: Token tensor of shape (B, P) for determining user/model labels
    """
    # Get start, end, and num_sequences from config to slice the labels correctly
    first_result = list(results.values())[0]
    start = first_result["config"]["start"]
    end = first_result["config"]["end"]
    num_sequences = first_result["config"]["num_sequences"]
    num_p = first_result["config"].get("num_p", None)
    do_log_scale = first_result["config"].get("do_log_scale", False)

    # Compute user labels for full sequence, then slice to match geometry experiment range
    is_user_BP = is_user(tokens_BP)
    is_user_sliced = is_user_BP[:num_sequences, start:end]  # Slice both B and P dimensions

    # Apply position subsampling if num_p is specified (matching exp logic)
    if num_p is not None:
        seq_len = is_user_sliced.shape[1]
        if do_log_scale:
            log_start = th.log10(th.tensor(0, dtype=th.float) + 1)
            log_end = th.log10(th.tensor(seq_len - 1, dtype=th.float) + 1)
            log_steps = th.linspace(log_start, log_end, num_p)
            ps = th.round(10**log_steps - 1).long().clamp(0, seq_len - 1)
        else:
            ps = th.linspace(0, seq_len - 1, num_p, dtype=th.long)

        is_user_sliced = is_user_sliced[:, ps]

    # Organize results by activation type
    data_by_type = {}

    for filename, result in results.items():
        act_path = result["config"]["act_path"]

        # Extract embeddings and labels
        embeddings = th.tensor(result["embeddings"])
        pos_labels = result["pos_labels"]

        data_by_type[act_path] = {
            "embeddings": embeddings[:cfg.num_sequences],
            "pos_labels": pos_labels[:cfg.num_sequences],
            "is_user": is_user_sliced[:cfg.num_sequences],
            "tortuosity": result["tortuosity"],
        }

    # Expected types for the 6 panels
    expected_types = [
        "activations",
        "relu/codes",
        "topk/codes",
        "batchtopk/codes",
        "temporal/novel_codes",
        "temporal/pred_codes",
    ]

    # Check if we have all expected types
    missing = [t for t in expected_types if t not in data_by_type]
    if missing:
        print(f"Warning: Missing activation types: {missing}")
        print(f"Available types: {list(data_by_type.keys())}")

    # Filter to only available activation types
    available_data = []
    for act_type in expected_types:
        if act_type in data_by_type:
            d = data_by_type[act_type]
            available_data.append(
                {
                    "type": act_type,
                    "embeddings": d["embeddings"],
                    "pos_labels": d["pos_labels"],
                    "is_user": d["is_user"],
                }
            )

    if not available_data:
        print("Error: No activation types available to plot.")
        return

    # Create the figure with user labels using only available data
    fig, axes = make_fig_like_plotly_with_user_labels_flexible(
        available_data,
        elev=11,
        azim=-15,
        base_marker=100,
        connect_idx=0,
        viz_title=True,
    )

    plt.tight_layout()

    # Get config info from any result
    result = list(results.values())[0]
    data_name = result["config"]["data"]["name"]
    llm_name = result["config"]["llm"]["name"]
    savefig(f"geometry_population_user_labeled_{data_name}_{llm_name}", suffix=".pdf")

    # Print tortuosity values
    print("\nTortuosity values:")
    for act_type in expected_types:
        if act_type in data_by_type:
            tort = data_by_type[act_type]["tortuosity"]
            print(f"  {act_type}: {tort:.3f}")


def plot_geometry_population(results):
    """
    Plot population geometry using 6 panels showing UMAP embeddings.

    Args:
        results: Dictionary mapping filenames to result dictionaries
    """
    # Organize results by activation type
    # Expected keys: 'activations', 'relu/codes', 'topk/codes', 'batchtopk/codes',
    #                'temporal/novel_codes', 'temporal/pred_codes'

    data_by_type = {}

    for filename, result in results.items():
        act_path = result["config"]["act_path"]

        # Extract embeddings and labels
        embeddings = th.tensor(result["embeddings"])
        pos_labels = result["pos_labels"]

        data_by_type[act_path] = {
            "embeddings": embeddings,
            "pos_labels": pos_labels,
            "tortuosity": result["tortuosity"],
        }

    # Expected types for the 6 panels
    expected_types = [
        "activations",
        "relu/codes",
        "topk/codes",
        "batchtopk/codes",
        "temporal/novel_codes",
        "temporal/pred_codes",
    ]

    # Check if we have all expected types
    missing = [t for t in expected_types if t not in data_by_type]
    if missing:
        print(f"Warning: Missing activation types: {missing}")
        print(f"Available types: {list(data_by_type.keys())}")

    # Filter to only available activation types
    available_data = []
    for act_type in expected_types:
        if act_type in data_by_type:
            d = data_by_type[act_type]
            available_data.append(
                {
                    "type": act_type,
                    "embeddings": d["embeddings"],
                    "pos_labels": d["pos_labels"],
                }
            )

    if not available_data:
        print("Error: No activation types available to plot.")
        return

    # Create the figure
    fig, axes = make_fig_like_plotly_flexible(
        available_data,
        elev=11,
        azim=-15,
        base_marker=20,  # 2x the original size (was 10)
        connect_idx=0,
        viz_title=True,
    )

    plt.tight_layout()
    data_name = result["config"]["data"]["name"]
    llm_name = result["config"]["llm"]["name"]
    savefig(f"geometry_population_{data_name}_{llm_name}", suffix=".png")

    # Print tortuosity values
    print("\nTortuosity values:")
    for act_type in expected_types:
        if act_type in data_by_type:
            tort = data_by_type[act_type]["tortuosity"]
            print(f"  {act_type}: {tort:.3f}")


def main():
    configs = get_gemma_act_configs(
        cfg_class=GeometryPopulationConfig,
        start=20,
        end=350,  # Adjust based on your needs
        num_sequences=100,
        centering=False,
        normalize=False,
        n_components=2,  # Use 2 for 2D UMAP, 3 for 3D UMAP
        # Position subsampling
        num_p=20,  # Number of positions to subsample (None = use all)
        do_log_scale=False,  # Use log scale for position subsampling
        # Artifacts
        env=ENV_CFG,
        # data=ALPACA_DS_CFG,
        data=CHAT_DS_CFG,
        llm=IT_GEMMA2_LLM_CFG,
        sae=None,
        act_paths=(
            (
                [None],
                [
                    "activations",
                ],
            ),
            (
                [GEMMA2_RELU_SAE_CFG, GEMMA2_TOPK_SAE_CFG, GEMMA2_BATCHTOPK_SAE_CFG],
                [
                    "codes",
                ],
            ),
            (
                [GEMMA2_TEMPORAL_SAE_CFG],
                [
                    "novel_codes",
                    "pred_codes",
                ],
            ),
        ),
    )

    results = load_results_multiple_configs(
        exp_name="geometry_population_",
        source_cfgs=configs,
        target_folder=configs[0].env.results_dir,
        recency_rank=0,
        compared_attributes=[
            "llm",
            "data",
            "start",
            "end",
            "centering",
            "normalize",
            "n_components",
            "num_p",
            "do_log_scale",
            "act_path",
        ],
        verbose=True,
    )

    # Load tokens from cached activations
    # Use the first config to determine the cache location
    cfg = configs[0]
    artifacts, _ = load_matching_activations(
        source_object=cfg,
        target_filenames=["tokens"],
        target_folder=cfg.env.activations_dir,
        recency_rank=0,
        compared_attributes=["llm", "data"],
    )
    tokens_BP = artifacts["tokens"]

    plot_geometry_population_with_user_labels(results, tokens_BP, cfg)


if __name__ == "__main__":
    main()
