import os
import torch
from typing import Optional, List, Literal, Union
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE
from src.exp_utils import (
    compute_or_load_svd,
    load_tokens_of_story,
    compute_or_load_llm_artifacts,
    compute_centered_svd,
)
from tqdm import trange
import skdim


def compute_intrinsic_dimension_skdim(
    act_train_LBPD,
    layer_idx,
    mode: Literal["fisher", "pca"] = "pca",
):
    L, B, P, D = act_train_LBPD.shape
    id_results_P = torch.zeros(P)
    act_train_before_p_BPD = act_train_LBPD[layer_idx, :, :, :]

    for p in trange(0, P, desc="Computing intrinsic dimension with skdim"):
        act_train_before_p_bD = act_train_before_p_BPD[:, : p + 1, :].flatten(0, 1)

        if mode == "fisher":
            estimator = skdim.id.FisherS().fit(act_train_before_p_bD)
        elif mode == "pca":
            estimator = skdim.id.lPCA(ver="ratio", alphaRatio=0.9).fit(
                act_train_before_p_bD
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        id_results_P[p] = estimator.dimension_

    return id_results_P, None


def compute_intrinsic_dimension_windowed_skdim(
    act_train_LBPD,
    layer_idx,
    window_size,
    mode: Literal["fisher", "pca"] = "pca",
):
    assert window_size is not None, "Window size must be provided"
    assert window_size > 0, "Window size must be positive"

    L, B, P, D = act_train_LBPD.shape
    id_results_P = torch.zeros(P)
    act_train_before_p_BPD = act_train_LBPD[layer_idx, :, :, :]

    for p in trange(window_size, P, desc="Computing intrinsic dimension with skdim"):
        act_train_before_p_bD = act_train_before_p_BPD[
            :, p - window_size : p, :
        ].flatten(0, 1)

        if mode == "fisher":
            estimator = skdim.id.FisherS().fit(act_train_before_p_bD)
        elif mode == "pca":
            estimator = skdim.id.lPCA(ver="ratio", alphaRatio=0.9).fit(
                act_train_before_p_bD
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        id_results_P[p] = estimator.dimension_

    return id_results_P, None


def compute_variance_explained_pca_over_all_tokens(
    act_train_LBPD,
    act_test_LBPD,
    layer_idx,
    reconstruction_thresholds,
    model_name,
):

    ##### Compute variance of full representations

    # Variance of full representations
    story_BPD = act_test_LBPD[layer_idx, :, :, :].to(DEVICE)
    mean_test_D = story_BPD.mean(dim=(0, 1))
    story_centered_BPD = (
        story_BPD - mean_test_D
    )  # Center the data wrt. test stories only
    total_variance_BP = torch.sum(story_centered_BPD**2, dim=-1)

    print(f"total_variance_BP.shape: {total_variance_BP.shape}")
    print(f"total_variance_BP max: {total_variance_BP.max()}")
    print(f"total_variance_BP min: {total_variance_BP.min()}")

    ##### Compute PCA (=centered SVD) results on full dataset of stories

    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(
        act_train_LBPD,
        model_name,
        dataset_name,
        num_total_stories,
        force_recompute,
        layer_idx=layer_idx,
    )

    Vt_CD = Vt_LCD[0].to(DEVICE)
    num_components, hidden_dim = Vt_CD.shape

    print(f"Vt_CD.shape: {Vt_CD.shape}")
    print(f"Vt_CD max: {Vt_CD.max()}")
    print(f"Vt_CD min: {Vt_CD.min()}")

    ##### Compute variance explained by top-k PCA components

    # Cumulative variance of top-k PCA components, per token
    pca_coeffs_BPC = einops.einsum(
        story_centered_BPD,
        Vt_CD,
        "b p d, c d -> b p c",
    )
    pca_variance_BPC = pca_coeffs_BPC**2
    pca_variance_sorted_BPC, _ = torch.sort(pca_variance_BPC, dim=-1, descending=True)
    pca_cumulative_variance_BPC = torch.cumsum(pca_variance_sorted_BPC, dim=-1)

    # Compute explained variance
    explained_variance_cumulative_BPC = (
        pca_cumulative_variance_BPC / total_variance_BP[:, :, None]
    ).cpu()
    print(
        f"explained_variance_cumulative_BPC.shape: {explained_variance_cumulative_BPC.shape}"
    )
    print(
        f"explained_variance_cumulative_BPC max: {explained_variance_cumulative_BPC.max()}"
    )
    print(
        f"explained_variance_cumulative_BPC min: {explained_variance_cumulative_BPC.min()}"
    )

    # Find first k components that meet reconstruction_thresholds for explained variance
    meets_threshold_BPCT = (
        explained_variance_cumulative_BPC[:, :, :, None]
        >= torch.tensor(reconstruction_thresholds)[None, None, None, :]
    )
    min_components_required_BPT = torch.argmax(meets_threshold_BPCT.int(), dim=-2)

    # If no solution is found, set to max number of components
    has_solution_BPT = torch.any(meets_threshold_BPCT, dim=-2)
    min_components_required_BPT.masked_fill_(~has_solution_BPT, num_components)

    print(f"min_components_required_BPT.shape: {min_components_required_BPT.shape}")
    print(f"min_components_required_BPT max: {min_components_required_BPT.max()}")
    print(f"min_components_required_BPT min: {min_components_required_BPT.min()}")

    return min_components_required_BPT


def compute_variance_explained_pca_before_token_p(
    act_train_LBPD,
    act_test_LBPD,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    do_sort_pca_components_per_token: bool = False,
    window_size: Optional[int] = 5,
):

    # For each token position p, compute the PCA basis for tokens before p
    # return the minimum number of PCA components required to reconstruct the variance of the test stories up to reconstruction threshold

    L, B_test, P, D = act_test_LBPD.shape
    C = D  # PCA basis spans the full model space
    T = len(reconstruction_thresholds)
    min_components_required_mean_PT = torch.zeros(P, T)
    min_components_required_std_PT = torch.zeros(P, T)

    if window_size is not None:
        start_token_idx = window_size
    else:
        start_token_idx = 1

    for p in trange(
        start_token_idx, P, desc="Computing explained variance of PCA before token p"
    ):
        if window_size is not None:
            act_train_before_p_LBpD = act_train_LBPD[
                :, :, p - window_size + 1 : p + 1, :
            ]
        else:
            act_train_before_p_LBpD = act_train_LBPD[:, :, : p + 1, :]

        ##### Compute PCA (=centered SVD) results on full dataset of stories
        act_train_before_p_LbD = act_train_before_p_LBpD.flatten(1, 2)
        U_LbC, S_LC, Vt_LCD, mean_train_LD = compute_centered_svd(
            act_train_before_p_LbD,
            layer_idx=layer_idx,
        )
        Vt_CD = Vt_LCD[0].to(DEVICE)
        mean_train_D = mean_train_LD[0]

        ##### Compute full variance of test representations
        act_test_at_p_BD = act_test_LBPD[layer_idx, :, p, :]

        ## Alternative: include all tokens before p
        # act_test_at_p_BPD = act_test_LBPD[layer_idx, :, :p+1, :]
        # act_test_at_p_BD = act_test_at_p_BPD.flatten(0, 1)

        # NOTE two options for centering the test set:
        act_test_at_p_centered_BD = (
            act_test_at_p_BD - mean_train_D
        )  # center wrt. train set
        # act_test_at_p_centered_BD = act_test_at_p_BD - act_test_at_p_BD.mean(dim=0) # center wrt. test set

        act_test_at_p_centered_BD = act_test_at_p_centered_BD.to(DEVICE)
        total_variance_B = torch.sum(act_test_at_p_centered_BD**2, dim=-1)

        ##### Compute variance explained by top-k PCA components

        # Cumulative variance of top-k PCA components, per token
        pca_coeffs_BC = einops.einsum(
            act_test_at_p_centered_BD,
            Vt_CD,
            "b d, c d -> b c",
        )
        pca_variance_BC = pca_coeffs_BC**2

        if do_sort_pca_components_per_token:
            pca_variance_BC, _ = torch.sort(pca_variance_BC, dim=-1, descending=True)
        pca_cumulative_variance_BC = torch.cumsum(pca_variance_BC, dim=-1)

        # Compute explained variance
        explained_variance_cumulative_BC = (
            pca_cumulative_variance_BC / total_variance_B[:, None]
        ).cpu()

        # Find first k components that meet reconstruction_thresholds for explained variance
        meets_threshold_BCT = (
            explained_variance_cumulative_BC[:, :, None]
            >= torch.tensor(reconstruction_thresholds)[None, None, :]
        )
        min_components_required_BT = (
            torch.argmax(meets_threshold_BCT.int(), dim=-2) + 1
        )  # 1-indexed

        # If no solution is found, set to max number of components
        has_solution_BT = torch.any(meets_threshold_BCT, dim=-2)
        min_components_required_BT.masked_fill_(~has_solution_BT, C)

        min_components_required_mean_PT[p, :] = min_components_required_BT.float().mean(
            dim=0
        )
        min_components_required_std_PT[p, :] = min_components_required_BT.float().std(
            dim=0
        )

    return min_components_required_mean_PT, min_components_required_std_PT


def compute_intrinsic_dimension(
    act_train_LBPD,
    act_test_LBPD,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    intrinsic_dimension_mode,
    window_size: Optional[Union[int, List[int]]] = None,
):
    if intrinsic_dimension_mode == "pca_before_p_manual":
        return compute_variance_explained_pca_before_token_p(
            act_train_LBPD,
            act_test_LBPD,
            layer_idx,
            reconstruction_thresholds,
            model_name,
        )
    elif intrinsic_dimension_mode == "pca_full_manual":
        return compute_variance_explained_pca_over_all_tokens(
            act_train_LBPD,
            act_test_LBPD,
            layer_idx,
            reconstruction_thresholds,
            model_name,
        )
    elif intrinsic_dimension_mode == "fisher_skdim":
        return compute_intrinsic_dimension_skdim(
            act_train_LBPD,
            layer_idx,
            "fisher",
        )
    elif intrinsic_dimension_mode == "pca_skdim":
        return compute_intrinsic_dimension_skdim(
            act_train_LBPD,
            layer_idx,
            "pca",
        )
    elif intrinsic_dimension_mode == "pca_windowed_skdim":
        if isinstance(window_size, list):
            results = []
            for ws in window_size:
                res = compute_intrinsic_dimension_windowed_skdim(
                    act_train_LBPD,
                    layer_idx,
                    window_size=ws,
                    mode="pca",
                )
                results.append(res)
            return torch.stack(results)
        else:
            return compute_intrinsic_dimension_windowed_skdim(
                act_train_LBPD,
                layer_idx,
                window_size=window_size,
                mode="pca",
            )
    elif intrinsic_dimension_mode == "fisher_windowed_skdim":
        if isinstance(window_size, list):
            results = []
            for ws in window_size:
                res = compute_intrinsic_dimension_windowed_skdim(
                    act_train_LBPD,
                    layer_idx,
                    window_size=ws,
                    mode="fisher",
                )
                results.append(res)
            return torch.stack(results)
        else:
            return compute_intrinsic_dimension_windowed_skdim(
                act_train_LBPD,
                layer_idx,
                window_size=window_size,
                mode="fisher",
            )
    else:
        raise ValueError(
            f"Invalid intrinsic dimension mode: {intrinsic_dimension_mode}"
        )


def plot_num_components_required_to_reconstruct(
    min_components_required_mean_PT,
    min_components_required_std_PT,
    story_idxs,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    dataset_name,
    num_total_stories,
):

    # Generate filename components
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    dataset_str = dataset_name.split("/")[-1].split(".")[0]
    save_fname = f"num_components_required_to_reconstruct_model_{model_str}_dataset_{dataset_str}_reconstruction_thresholds_{thresholds_str}_layer_{layer_idx}"

    P, T = min_components_required_mean_PT.shape

    fig, ax = plt.subplots(
        figsize=(P, 6)
    )  # Increase figure width for better readability
    for t in range(T):
        ax.plot(
            range(P),
            min_components_required_mean_PT[:, t],
            label=f"{reconstruction_thresholds[t]*100}% explained variance",
        )

    ax.legend()
    ax.set_xlabel("Token Position")
    ylabel = f"Minimum PCA Components required for variance threshold"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Number of PCA components required to reconstruct variance threshold for stories {story_idxs} in layer {layer_idx}"
    )

    # If only one story is provided, plot the tokens of the story
    if len(story_idxs) == 1:
        tokens_of_story = load_tokens_of_story(
            dataset_name,
            num_total_stories,
            story_idxs[0],
            model_name,
            omit_BOS_token,
            P,
        )
        print(f"tokens_of_story: {"".join(tokens_of_story)}")

        ax.set_xticks(range(P))
        ax.set_xticklabels(
            tokens_of_story, rotation=90, ha="right"
        )  # Rotate labels and align right
        ax.tick_params(
            axis="x",
            which="major",  # pad=20
        )  # Increase spacing between labels

        save_fname += f"_story_{story_idxs[0]}"

    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=80)
    print(f"Saved figure to {fig_path}")
    plt.close()


def plot_mean_components_across_stories(
    min_components_required_mean_PT,
    min_components_required_std_PT,
    layer_idx,
    reconstruction_thresholds,
    model_name,
    dataset_name,
    num_test_stories,
):
    """
    Plot mean number of PCA components required across all stories with 95% confidence intervals.

    Args:
        min_components_required_BPT: Tensor with shape (batch, position, thresholds)
        story_idxs: List of story indices
        layer_idx: Layer index for filename
        reconstruction_thresholds: List of threshold values
        model_name: Model name for filename
    """
    # Generate filename
    thresholds_str = "_".join([str(t) for t in reconstruction_thresholds])
    model_str = model_name.split("/")[-1]
    dataset_str = dataset_name.split("/")[-1].split(".")[0]
    save_fname = f"mean_components_across_stories_model_{model_str}_dataset_{dataset_str}_thresholds_{thresholds_str}_layer_{layer_idx}"

    fig, ax = plt.subplots(figsize=(12, 6))

    max_pos, _ = min_components_required_mean_PT.shape

    # Plot for each threshold
    for t_idx, threshold in enumerate(reconstruction_thresholds):
        # All stories have the same length now, so we can calculate mean and CI directly across the batch dimension.
        mean_components = min_components_required_mean_PT[:, t_idx].cpu()
        std_components = min_components_required_std_PT[:, t_idx].cpu()

        # 95% confidence interval: 1.96 * std / sqrt(n)
        positions_P = torch.arange(max_pos)
        ci_components_P = (
            1.96 * std_components / ((num_test_stories * (positions_P + 1)) ** 0.5)
        )  # NOTE the number of all samples increases with each token position

        # Plot mean line
        label = f"{threshold*100}% variance threshold"
        ax.plot(positions_P, mean_components, label=label, linewidth=2)

        # Plot 95% CI band
        ax.fill_between(
            positions_P,
            mean_components - ci_components_P,
            mean_components + ci_components_P,
            alpha=0.2,
        )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean Number of PCA Components Required")
    ax.set_title(
        f"Mean PCA Components Required Across Stories (95% CI) - Layer {layer_idx}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved mean components across stories plot to {fig_path}")
    plt.close()


def plot_intrinsic_dimension(
    intrinsic_dimension_P,
    layer_idx,
    model_name,
    dataset_name,
    intrinsic_dimension_mode,
    window_size: Optional[Union[int, List[int]]] = None,
):
    P = intrinsic_dimension_P.shape[0]
    fig, ax = plt.subplots(figsize=(12, 6))

    discrete_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "black",
        "cyan",
        "magenta",
        "yellow",
        "lime",
        "teal",
        "indigo",
        "violet",
        "maroon",
        "navy",
        "olive",
        "coral",
        "gold",
        "silver",
        "plum",
        "tan",
        "khaki",
        "lavender",
        "turquoise",
        "beige",
        "chocolate",
        "coral",
        "crimson",
        "fuchsia",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "maroon",
        "navy",
        "olive",
        "plum",
        "salmon",
        "tan",
        "teal",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "yellow",
    ]

    if "windowed" in intrinsic_dimension_mode and isinstance(window_size, list):
        for i, ws in enumerate(window_size):
            color = discrete_colors[i]
            ax.plot(intrinsic_dimension_P[i], label=f"window_size={ws}", color=color)
            ax.scatter(
                range(P),
                intrinsic_dimension_P[i],
                label=f"window_size={ws}",
                color=color,
            )
        ax.legend()
    else:
        ax.plot(intrinsic_dimension_P)
        ax.scatter(range(P), intrinsic_dimension_P)

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Intrinsic Dimension")
    ax.set_title(
        f"Intrinsic Dimension - {intrinsic_dimension_mode} - Layer {layer_idx}"
    )

    file_name_ws = ""
    if window_size is not None:
        if isinstance(window_size, list):
            file_name_ws = f"_ws_{'_'.join(map(str, window_size))}"
        else:
            file_name_ws = f"_ws_{window_size}"

    fig_path = os.path.join(
        PLOTS_DIR,
        f"intrinsic_dimension_{intrinsic_dimension_mode}_{model_name.split('/')[-1]}layer_{layer_idx}_dataset_{dataset_name.split('/')[-1].split('.')[0]}{file_name_ws}.png",
    )
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved intrinsic dimension plot to {fig_path}")


if __name__ == "__main__":
    ##### Parameters

    model_name = "openai-community/gpt2"
    # model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"
    layer_idx = 6

    dataset_name = "SimpleStories/SimpleStories"
    # dataset_name = "long_factual_sentences.json"
    # dataset_name = "simple_sentences.json"
    num_tokens_per_story = 25
    num_total_stories = 100
    num_total_stories_train = 80
    do_train_test_split = False
    omit_BOS_token = True

    # reconstruction_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    reconstruction_thresholds = [0.9]
    test_len = num_total_stories - num_total_stories_train
    if len(reconstruction_thresholds) > 1:
        assert test_len == 1

    available_intrinsic_dimension_modes = [
        "fisher_skdim",
        "pca_skdim",
        "pca_full_manual",
        "pca_before_p_manual",
        "pca_windowed_skdim",
    ]

    intrinsic_dimension_mode = "pca_before_p_manual"
    # intrinsic_dimension_mode = "pca_skdim"
    assert (
        intrinsic_dimension_mode in available_intrinsic_dimension_modes
    ), f"Invalid intrinsic dimension mode: {intrinsic_dimension_mode}"

    window_size = [1, 2, 3, 5, 10, 20]

    force_recompute = True

    ##### Load activations

    act_LBPD, dataset_story_idxs, tokens_BP = compute_or_load_llm_artifacts(
        model_name,
        num_total_stories,
        story_idxs=None,
        cfg.omit_BOS_token=omit_BOS_token,
        dataset_name=dataset_name,
    )
    if num_tokens_per_story is not None:
        act_LBPD = act_LBPD[:, :, :num_tokens_per_story, :]
        tokens_BP = tokens_BP[:, :num_tokens_per_story]

    # Do train-test split
    if do_train_test_split:
        rand_idxs = torch.randperm(num_total_stories)
        train_idxs = rand_idxs[:num_total_stories_train]
        test_idxs = rand_idxs[num_total_stories_train:]

        act_train_LBPD = act_LBPD[:, train_idxs, :, :]
        act_test_LBPD = act_LBPD[:, test_idxs, :, :]

        tokens_test_BP = [tokens_BP[i] for i in test_idxs]
        selected_dataset_idxs_test = [dataset_story_idxs[i] for i in test_idxs]
        num_test_stories = len(test_idxs)
    else:
        act_train_LBPD = act_LBPD
        act_test_LBPD = act_LBPD
        tokens_test_BP = tokens_BP
        selected_dataset_idxs_test = dataset_story_idxs
        num_test_stories = num_total_stories

    id_results_mean_PT, id_results_std_PT = compute_intrinsic_dimension(
        act_train_LBPD,
        act_test_LBPD,
        layer_idx,
        reconstruction_thresholds,
        model_name,
        intrinsic_dimension_mode,
        window_size=window_size,
    )

    ##### Plot results

    # if "manual" in intrinsic_dimension_mode:
    #     plot_num_components_required_to_reconstruct(
    #         min_components_required_mean_PT=id_results_mean_PT,
    #         min_components_required_std_PT=id_results_std_PT,
    #         story_idxs=selected_dataset_idxs_test,
    #         layer_idx=layer_idx,
    #         reconstruction_thresholds=reconstruction_thresholds,
    #         model_name=model_name,
    #         dataset_name=dataset_name,
    #         num_total_stories=num_total_stories,
    #     )

    # if "manual" in intrinsic_dimension_mode and len(selected_dataset_idxs_test) > 1:
    #     # Plot mean components across stories
    #     plot_mean_components_across_stories(
    #         min_components_required_mean_PT=id_results_mean_PT,
    #         min_components_required_std_PT=id_results_std_PT,
    #         layer_idx=layer_idx,
    #         reconstruction_thresholds=reconstruction_thresholds,
    #         model_name=model_name,
    #         dataset_name=dataset_name,
    #         num_test_stories=num_test_stories,
    #     )

    # if "skdim" in intrinsic_dimension_mode:
    plot_intrinsic_dimension(
        intrinsic_dimension_P=id_results_mean_PT,  # this is actually not PT but P
        layer_idx=layer_idx,
        model_name=model_name,
        dataset_name=dataset_name,
        intrinsic_dimension_mode=intrinsic_dimension_mode,
        window_size=window_size if "windowed" in intrinsic_dimension_mode else None,
    )

    ##### Alternative approaches  to compute cumulative variance

    # NOTE The above approach to use the variance of the PCA coefficients is numerically unstable, since PCA vectors are not exactly orthogonal.
    # However, this has no significant impact on min components required

    # reconstruction_BPD = torch.zeros_like(story_BPD).to(DEVICE)
    # pca_cumulative_variance_BPC = torch.zeros_like(pca_coeffs_BPC).to(DEVICE)

    # for c in range(num_components):
    #     reconstruction_BPD += einops.einsum(
    #         pca_coeffs_BPC[:, :, c], Vt_CD[c], "b p, d -> b p d"
    #     )
    #     pca_cumulative_variance_BPC[:, :, c] = torch.sum(reconstruction_BPD**2, dim=-1)

    # print(f"difference {(pca_cumulative_variance_BPC - pca_cumulative_variance1_BPC).max()}")
    # assert torch.allclose(
    #     pca_cumulative_variance_BPC, pca_cumulative_variance1_BPC, atol=1e-3
    # )

    # NOTE Computing pca_cumulative_variance_BPC in single batch exceeds GPU memory

    # pca_decomposition_BPCD = einops.einsum(
    #     pca_coeffs_BPC,
    #     Vt_CD,
    #     "b p c, c d -> b p c d"
    # )
    # pca_cumulative_reconstruction_BPCD = torch.cumsum(
    #     pca_decomposition_BPCD, dim=-2
    # )
    # pca_cumulative_variance_BPC = torch.sum(pca_cumulative_reconstruction_BPCD**2, dim=-1)
