import os
import torch as th
from typing import Optional, List, Literal, Union
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE
from src.exp_utils import (
    compute_centered_svd,
    load_activation_split,
)
from tqdm import trange


class Config:
    def __init__(self):
        ### Model
        self.model_name: str = "openai-community/gpt2"
        self.layer_idx: int = 6
        # self.model_name: str = "meta-llama/Llama-3.1-8B"
        # self.layer_idx: int = 12

        ### Dataset
        self.dataset_name: str = "SimpleStories/SimpleStories"
        self.num_total_stories: int = 100

        self.story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 75
        self.do_train_test_split: bool = False
        self.num_train_stories: int = 80
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        ### PCA
        self.available_intrinsic_dimension_modes: List[str] = [
            "fisher_skdim",
            "pca_skdim",
            "pca_full_manual",
            "pca_before_p_manual",
            "pca_windowed_skdim",
        ]
        self.intrinsic_dimension_mode: str = "pca_before_p_manual"

        # reconstruction_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.reconstruction_thresholds: List[float] = [0.9]
        # window_size needs to be an iterable, use [None] to disable
        # window_size = [1, 2, 3, 5, 10, 20]
        self.window_sizes: List[Optional[int]] = [None, 2, 5, 10, 25, 50] # window_size=1 needs include_test_token_t_in_train_pca=True
        self.include_test_token_t_in_train_pca: bool = True 
        self.do_sort_pca_components_per_token: bool = True

        ### Dependent parameters
        self.num_test_stories = self.num_total_stories - self.num_train_stories

        ### Assertions
        if len(self.reconstruction_thresholds) > 1:
            assert self.num_test_stories == 1

        assert (
            self.intrinsic_dimension_mode in self.available_intrinsic_dimension_modes
        ), f"Invalid intrinsic dimension mode: {self.intrinsic_dimension_mode}"

        ### String summarizing the parameters for saving results
        model_str = self.model_name.split("/")[-1]
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        story_idxs_str = (
            "_".join([str(i) for i in self.story_idxs])
            if self.story_idxs is not None
            else "all"
        )
        window_size_str = (
            "_".join([str(w) for w in self.window_sizes])
        )
        threshold_str = "_".join([str(t) for t in self.reconstruction_thresholds])

        self.input_str = (
            f"model_{model_str}"
            + f"_dataset_{dataset_str}"
            + f"_samples_{self.num_total_stories}"
        )

        self.output_str = (
            self.input_str
            + f"_mod_{self.intrinsic_dimension_mode}"
            + f"_lay_{self.layer_idx}"
            + f"_spl_{self.do_train_test_split}"
            + f"_ntr_{self.num_train_stories}"
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{story_idxs_str}"
            + f"_thr_{threshold_str}"
            + f"_ws_{window_size_str}"
            + f"_sort_{self.do_sort_pca_components_per_token}"
            + f"_incl_{self.include_test_token_t_in_train_pca}"
        )


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
    for p in trange(
        initial_p_idx, P, desc="Computing explained variance of PCA before token p"
    ):
        # Determine window_start_idx and window_end_idx
        if window_size is not None:
            window_start_idx = p - window_size
        else:
            window_start_idx = 0

        window_end_idx = p

        if cfg.include_test_token_t_in_train_pca: # move the current test token into the train window
            window_start_idx += 1
            window_end_idx += 1

        act_train_before_p_LBpD = act_train_LBPD[
            :, :, window_start_idx:window_end_idx, :
        ]

        ##### Compute PCA (=centered SVD) results on full dataset of stories

        act_train_before_p_LbD = act_train_before_p_LBpD.flatten(1, 2)
        U_LbC, S_LC, Vt_LCD, mean_train_LD = compute_centered_svd(
            act_train_before_p_LbD,
            layer_idx=cfg.layer_idx,
        )
        Vt_CD = Vt_LCD[0].to(DEVICE)
        mean_train_D = mean_train_LD[0]

        ##### Compute full variance of test representations, centered wrt. train set

        act_test_at_p_BD = act_test_LBPD[cfg.layer_idx, :, p, :]

        act_test_at_p_centered_BD = (
            act_test_at_p_BD - mean_train_D
        )  # center wrt. train set

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
        min_components_required_BT = (
            th.argmax(meets_threshold_BCT.int(), dim=-2) + 1
        )  # 1-indexed

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
        print(f'window_size {w}')
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
    fig, ax = plt.subplots(figsize=(12, 6))

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
        f"Mean PCA Components Required Across Stories (95% CI) - Layer {cfg.layer_idx}"
    )
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_fname = f"intrinsic_dimension_{cfg.output_str}"
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved mean components across stories plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    # Load activations
    (
        act_train_LBPD,
        act_test_LBPD,
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
