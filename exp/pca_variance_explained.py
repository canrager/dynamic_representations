import os
import torch as th
from typing import Optional, List, Literal, Union, Tuple
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE, DatasetConfig, LLMConfig, SAEConfig
from src.exp_utils import (
    compute_centered_svd,
    load_activation_split,
    compute_or_load_llm_artifacts,
    compute_or_load_sae_artifacts,
)
from tqdm import trange
from dataclasses import dataclass
from dictionary_learning.dictionary import IdentityDict, AutoEncoder


@dataclass
class Config:
    debug: bool = False
    device: str = DEVICE
    dtype: th.dtype = th.float32
    save_artifacts: bool = False
    exp_name: str = "pca_variance_explained"
    loaded_llm: Tuple = None

    llm: LLMConfig = LLMConfig("openai-community/gpt2", 6, 100, None, force_recompute=False)
    # llm: LLMConfig = LLMConfig("google/gemma-2-2b", 12, 100, None, force_recompute=False)
    # llm: LLMConfig = LLMConfig("meta-llama/Llama-3.1-8B", 12, 100, None, force_recompute=False)
    # llm: LLMConfig | None = None

    # llms = [
    #     # LLMConfig("openai-community/gpt2", layer_idx=6, batch_size=100),
    #     # LLMConfig("google/gemma-2-2b", layer_idx=12, batch_size=100),
    #     LLMConfig("Qwen/Qwen2.5-7B", layer_idx=12, batch_size=100),
    #     # LLMConfig("meta-llama/Llama-3.1-8B", layer_idx=6, batch_size=100),
    #     # LLMConfig("meta-llama/Llama-3.1-8B", layer_idx=12, batch_size=100),
    #     # LLMConfig("meta-llama/Llama-3.1-8B", layer_idx=18, batch_size=100),
    #     # LLMConfig("meta-llama/Llama-3.1-8B", layer_idx=24, batch_size=100),
    #     # LLMConfig("meta-llama/Llama-3.1-8B", layer_idx=31, batch_size=100),
    #     # LLMConfig("allenai/OLMo-2-1124-7B", layer_idx=12, batch_size=100, revision="stage1-step150-tokens1B"),
    #     # LLMConfig("allenai/OLMo-2-1124-7B", layer_idx=12, batch_size=100, revision="stage1-step10000-tokens42B"),
    #     # LLMConfig("allenai/OLMo-2-1124-7B", layer_idx=12, batch_size=100, revision="stage1-step12000-tokens51B"),
    #     # LLMConfig("allenai/OLMo-2-1124-7B", layer_idx=12, batch_size=100, revision="stage1-step105000-tokens441B"),
    #     # LLMConfig("allenai/OLMo-2-1124-7B", layer_idx=12, batch_size=100),
    # ]

    sae: SAEConfig = None
    saes = (
        SAEConfig(
            IdentityDict, None, 100, "Residual stream neurons", None, force_recompute=False
        ),  # This is the LLM residual stream baseline
        # SAEConfig(
        #     AutoEncoder,
        #     4096,
        #     100,
        #     "L1 ReLU saebench",
        #     "artifacts/trained_saes/Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_2/ae.pt",
        #     force_recompute=True,
        # ),
    )
    dataset = DatasetConfig("SimpleStories/SimpleStories", "story")
    # dataset = DatasetConfig("monology/pile-uncopyrighted", "text")
    # dataset = DatasetConfig("NeelNanda/code-10k", "text")

    # dataset_name = "many_datasets"
    # datasets = [
    #     # DatasetConfig("SimpleStories/SimpleStories", "story"),
    #     DatasetConfig("monology/pile-uncopyrighted", "text"),
    #     # DatasetConfig("NeelNanda/code-10k", "text")
    # ]

    num_total_stories: int = 100
    selected_story_idxs: Optional[List[int]] = None
    omit_BOS_token: bool = True
    num_tokens_per_story: int = 25
    do_train_test_split: bool = False
    num_train_stories: int = 75
    # force_recompute: bool = (
    #     True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
    # )

    ### PCA

    # reconstruction_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    reconstruction_thresholds: Tuple[float] = (0.95,)
    # window_size needs to be an iterable, use [None] to disable
    # window_size = [1, 2, 3, 5, 10, 20]
    window_sizes: Tuple[Optional[int]] = (None, 1, 5, 10, 25, 50)
    # window_sizes: List[Optional[int]] = [1, None]
    include_test_token_t_in_train_pca: bool = False
    do_sort_pca_components_per_token: bool = True

    ### Dependent parameters
    num_test_stories = num_total_stories - num_train_stories

    ### Assertions
    if len(reconstruction_thresholds) > 1:
        assert num_test_stories == 1

    # ### String summarizing the parameters for saving results
    # model_str = llm_name.split("/")[-1]
    # dataset_str = dataset_name.split("/")[-1].split(".")[0]
    # story_idxs_str = (
    #     "_".join([str(i) for i in selected_story_idxs])
    #     if selected_story_idxs is not None
    #     else "all"
    # )
    # window_size_str = "_".join([str(w) for w in window_sizes])
    # threshold_str = "_".join([str(t) for t in reconstruction_thresholds])

    # input_file_str = (
    #     f"model_{model_str}"
    #     + f"_dataset_{dataset_str}"
    #     + f"_samples_{num_total_stories}"
    # )

    # output_file_str = (
    #     input_file_str
    #     + f"_lay_{layer_idx}"
    #     + f"_spl_{do_train_test_split}"
    #     + f"_ntr_{num_train_stories}"
    #     + f"_ntok_{num_tokens_per_story}"
    #     + f"_nobos_{omit_BOS_token}"
    #     + f"_didx_{story_idxs_str}"
    #     + f"_thr_{threshold_str}"
    #     + f"_ws_{window_size_str}"
    #     + f"_sort_{do_sort_pca_components_per_token}"
    #     + f"_incl_{include_test_token_t_in_train_pca}"
    # )


def compute_intrinsic_dimension(act_train_LBPD, act_test_LBPD, cfg, window_size=None):
    # over all tokens: simply set window_size to num_tokens_per_story
    # sliding window
    # over thresholds

    # For each token position p, compute the PCA basis for tokens before p
    # return the minimum number of PCA components required to reconstruct the variance of the test stories up to reconstruction threshold

    L, B_test, P, D = act_test_LBPD.shape
    C = D  # PCA basis spans the full model space
    T = len(cfg.reconstruction_thresholds)

    min_components_required_mean_PT = -th.ones(P, T)
    min_components_required_ci_PT = -th.ones(P, T)

    if window_size is not None:
        initial_p_idx = window_size
    else:
        initial_p_idx = 1

    # PCA basis changes at each token position p
    for p in trange(initial_p_idx, P, desc="Computing explained variance of PCA before token p"):
        # Determine window_start_idx and window_end_idx
        if window_size is not None:
            window_start_idx = p - window_size
        else:
            window_start_idx = 0

        window_end_idx = p

        if (
            cfg.include_test_token_t_in_train_pca
        ):  # move the current test token into the train window
            window_start_idx += 1
            window_end_idx += 1

        act_train_before_p_LBpD = act_train_LBPD[:, :, window_start_idx:window_end_idx, :]

        ##### Compute PCA (=centered SVD) results on full dataset of stories

        act_train_before_p_LbD = act_train_before_p_LBpD.flatten(1, 2)
        U_LbC, S_LC, Vt_LCD, mean_train_LD = compute_centered_svd(
            act_train_before_p_LbD,
            layer_idx=cfg.llm.layer_idx,
        )
        Vt_CD = Vt_LCD[0].to(DEVICE)
        mean_train_D = mean_train_LD[0]

        ##### Compute full variance of test representations, centered wrt. train set

        act_test_at_p_BD = act_test_LBPD[cfg.llm.layer_idx, :, p, :]

        act_test_at_p_centered_BD = act_test_at_p_BD - mean_train_D  # center wrt. train set

        act_test_at_p_centered_BD = act_test_at_p_centered_BD.to(DEVICE)
        total_variance_B = th.sum(act_test_at_p_centered_BD**2, dim=-1)

        ##### Compute variance explained by top-k PCA components

        # Cumulative variance of top-k PCA components, per token
        pca_coeffs_BC = einops.einsum(
            act_test_at_p_centered_BD,
            Vt_CD,
            "b d, c d -> b c",
        )
        pca_variance_BC = pca_coeffs_BC**2

        if cfg.do_sort_pca_components_per_token:
            pca_variance_BC, _ = th.sort(pca_variance_BC, dim=-1, descending=True)
        pca_cumulative_variance_BC = th.cumsum(pca_variance_BC, dim=-1)

        # Compute explained variance
        explained_variance_cumulative_BC = (
            pca_cumulative_variance_BC / total_variance_B[:, None]
        ).cpu()

        ##### Find top k components that meet reconstruction_thresholds for explained variance

        meets_threshold_BCT = (
            explained_variance_cumulative_BC[:, :, None]
            >= th.tensor(cfg.reconstruction_thresholds)[None, None, :]
        )
        min_components_required_BT = th.argmax(meets_threshold_BCT.int(), dim=-2) + 1  # 1-indexed

        # If no solution is found, set to max number of components
        has_solution_BT = th.any(meets_threshold_BCT, dim=-2)
        min_components_required_BT.masked_fill_(~has_solution_BT, C)

        ##### Format results

        # Compute mean and 95% confidence interval
        min_components_required_mean_BT = min_components_required_BT.float().mean(dim=0)
        min_components_required_std_BT = min_components_required_BT.float().std(dim=0)
        b = min_components_required_BT.shape[0]
        min_components_required_ci_BT = 1.96 * min_components_required_std_BT / (b**0.5)

        # Store results
        min_components_required_mean_PT[p, :] = min_components_required_mean_BT
        min_components_required_ci_PT[p, :] = min_components_required_ci_BT

    return min_components_required_mean_PT, min_components_required_ci_PT


def compute_intrinsic_dimension_multiple_ws(act_train_LBPD, act_test_LBPD, cfg):
    id_mean_WPT = []
    id_ci_WPT = []

    for w in cfg.window_sizes:
        print(f"window_size {w}")
        id_mean_PT, id_ci_PT = compute_intrinsic_dimension(
            act_train_LBPD, act_test_LBPD, cfg, window_size=w
        )
        id_mean_WPT.append(id_mean_PT)
        id_ci_WPT.append(id_ci_PT)

    id_mean_WPT = th.stack(id_mean_WPT)
    id_ci_WPT = th.stack(id_ci_WPT)

    return id_mean_WPT, id_ci_WPT


def plot_intrinsic_dimension(
    id_mean_WPT,
    id_ci_WPT,
    cfg,
):
    """
    Plot mean number of PCA components required across all stories with 95% confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    W, P, T = id_mean_WPT.shape

    # Plot for each threshold
    color_idx = 0
    for w_idx, w in enumerate(cfg.window_sizes):
        for t_idx, threshold in enumerate(cfg.reconstruction_thresholds):
            arange_P = th.arange(P)
            mean_components_P = id_mean_WPT[w_idx, :, t_idx].cpu()
            ci_components_P = id_ci_WPT[w_idx, :, t_idx].cpu()

            # Drop missing values
            is_valid_P = mean_components_P >= 0
            arange_P = arange_P[is_valid_P]
            mean_components_P = mean_components_P[is_valid_P]
            ci_components_P = ci_components_P[is_valid_P]

            # Get color from matplotlib's default color cycle
            color = f"C{color_idx}"

            # Get label
            if cfg.include_test_token_t_in_train_pca:
                end_str = "t"
                diff = 0
            else:
                end_str = "t-1"
                diff = 1
            if w is not None:
                start_str = f"t-{w - 1 + diff}"
            else:
                start_str = "0"
            label = f"{int(threshold*100)}% expvar, PCA on [{start_str}, {end_str}]"

            # Plot mean line
            ax.plot(arange_P, mean_components_P, linewidth=2, color=color)
            ax.scatter(arange_P, mean_components_P, label=label, color=color)

            # Plot 95% CI band
            ax.fill_between(
                arange_P,
                mean_components_P - ci_components_P,
                mean_components_P + ci_components_P,
                alpha=0.2,
                color=color,
            )

            color_idx += 1

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean Number of PCA Components Required")
    ax.set_title(
        f"Mean PCA Components Required Across Stories (95% CI) - Layer {cfg.llm.layer_idx}"
    )
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_fname = f"intrinsic_dimension_{hash(cfg)}"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved mean components across stories plot to {fig_path}")
    plt.close()


def plot_id_multiple_models(id_result_list, cfg):
    """
    Plot intrinsic dimension for multiple models at a single window size.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    color_idx = 0
    for results in id_result_list:
        id_mean_PT = results["id_mean_PT"]
        id_ci_PT = results["id_ci_PT"]

        P, T = id_mean_PT.shape

        # Plot for each threshold
        for t_idx, threshold in enumerate(cfg.reconstruction_thresholds):
            arange_P = th.arange(P)
            mean_components_P = id_mean_PT[:, t_idx].cpu()
            ci_components_P = id_ci_PT[:, t_idx].cpu()

            # Drop missing values
            is_valid_P = mean_components_P >= 0
            arange_P = arange_P[is_valid_P]
            mean_components_P = mean_components_P[is_valid_P]
            ci_components_P = ci_components_P[is_valid_P]

            # Get color from matplotlib's default color cycle
            color = f"C{color_idx}"

            # Get label - simplified model name
            model_short_name = results["llm_name"].split("/")[-1]
            dataset_short_name = results["dataset_name"].split("/")[-1].split(".")[0]
            label = f"{model_short_name} L{results["layer_idx"]}, {dataset_short_name}, {results["sae_name"]}"

            # Plot mean line
            ax.plot(arange_P, mean_components_P, linewidth=2, color=color)
            ax.scatter(arange_P, mean_components_P, label=label, color=color)

            # Plot 95% CI band
            ax.fill_between(
                arange_P,
                mean_components_P - ci_components_P,
                mean_components_P + ci_components_P,
                alpha=0.2,
                color=color,
            )

            color_idx += 1

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean Number of PCA Components Required")
    ax.set_title(
        f"Intrinsic Dimension Comparison Across Models (95% CI)\n {int(threshold*100)}% expvar"
    )
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_fname = f"intrinsic_dimension_multiple_models_{hash(cfg)}"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved multiple models comparison plot to {fig_path}")
    plt.close()


def main_id_multiple_ws():
    cfg = Config()

    # Load activations
    (
        act_train_LBPD,
        act_test_LBPD,
        mask_train_BP,
        mask_test_BP,
        tokens_test_BP,
        num_test_stories,
        dataset_idxs_test,
    ) = load_activation_split(cfg)

    # Compute intrinsic dimension
    id_mean_WPT, id_ci_WPT = compute_intrinsic_dimension_multiple_ws(
        act_train_LBPD=act_train_LBPD,
        act_test_LBPD=act_test_LBPD,
        cfg=cfg,
    )

    # Plot results
    plot_intrinsic_dimension(
        id_mean_WPT=id_mean_WPT,
        id_ci_WPT=id_ci_WPT,
        cfg=cfg,
    )


def main_id_multiple_models():
    global_cfg = Config()

    # Load activations and compute intrinsic dimension for each model
    id_result_list = []
    for llm_cfg in global_cfg.llms:
        for ds_cfg in global_cfg.datasets:
            # Update config for current model
            current_cfg = Config(dataset=ds_cfg, llm=llm_cfg)

            (
                act_train_LBPD,
                act_test_LBPD,
                mask_train_BP,
                mask_test_BP,
                tokens_test_BP,
                num_test_stories,
                dataset_idxs_test,
            ) = load_activation_split(current_cfg)

            # Compute intrinsic dimension for a single window (no window)
            id_mean_PT, id_ci_PT = compute_intrinsic_dimension(
                act_train_LBPD=act_train_LBPD,
                act_test_LBPD=act_test_LBPD,
                cfg=current_cfg,
                window_size=None,
            )

            id_result_list.append(
                {
                    "id_mean_PT": id_mean_PT,
                    "id_ci_PT": id_ci_PT,
                    "llm_name": current_cfg.llm.name,
                    "layer_idx": current_cfg.llm.layer_idx,
                    "dataset_name": current_cfg.dataset.name,
                }
            )

    # Plot results
    plot_id_multiple_models(id_result_list, current_cfg)


def main_id_multiple_saes():
    global_cfg = Config()

    llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(global_cfg)
    )
    llm_act_BPD = llm_act_LBPD[global_cfg.llm.layer_idx]

    # Load activations and compute intrinsic dimension for each model
    id_result_list = []
    for sae_cfg in global_cfg.saes:
        # Update config for current model
        current_cfg = Config(sae=sae_cfg)

        sae_artifact = compute_or_load_sae_artifacts(llm_act_BPD, current_cfg)
        sae_act_BPS = sae_artifact.sae_act_BPS

        ## TODO NEXT: Implement SAE caching here

        # Compute intrinsic dimension for a single window (no window)
        id_mean_PT, id_ci_PT = compute_intrinsic_dimension(
            act_train_LBPD=sae_act_BPS, act_test_LBPD=sae_act_BPS, cfg=current_cfg, window_size=None
        )

        id_result_list.append(
            {
                "id_mean_PT": id_mean_PT,
                "id_ci_PT": id_ci_PT,
                "llm_name": current_cfg.llm.name,
                "layer_idx": current_cfg.llm.layer_idx,
                "dataset_name": current_cfg.dataset.name,
                "sae_name": current_cfg.sae_cfg.name,
            }
        )

    # Plot results
    plot_id_multiple_models(id_result_list, global_cfg)


if __name__ == "__main__":
    # main_id_multiple_ws()
    main_id_multiple_saes()
    # main_id_multiple_models()
