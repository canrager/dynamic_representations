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
from dataclasses import dataclass

@dataclass
class LLMConfig():
    name: str
    layer_idx: int
    batch_size: int
    revision: Optional[str] = None

@dataclass
class DatasetConfig():
    name: str
    hf_text_identifier: str


class Config:
    def __init__(self):
        self.debug = False

        ### Model
        # self.model_name: str = "openai-community/gpt2"
        # self.layer_idx: int = 6
        # self.llm_name: str = "meta-llama/Llama-3.1-8B"
        # self.llm_name: str = "Qwen/Qwen2.5-7B"
        self.llm_name: str = "mistralai/Mistral-7B-v0.1"
        # self.llm_name: str = "google/gemma-2-2b"
        self.layer_idx: int = 12
        self.llm_batch_size: str = 100
        self.llm_revision: Optional[str] = None

        # ## OR Multiple models
        # self.llm_name = "many_llms"
        # self.layer_idx = -1
        # self.llm_revision = None
        # self.llms = [
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

        ### Dataset
        # self.dataset_name: str = "SimpleStories/SimpleStories"
        self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        self.hf_text_identifier: str = "text"


        # self.dataset_name = "many_datasets"
        # self.datasets = [
        #     # DatasetConfig("SimpleStories/SimpleStories", "story"),
        #     DatasetConfig("monology/pile-uncopyrighted", "text"),
        #     # DatasetConfig("NeelNanda/code-10k", "text")
        # ]
        
        self.num_total_stories: int = 100
        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = False
        self.num_tokens_per_story: int = 25
        self.do_train_test_split: bool = True
        self.num_train_stories: int = 75
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        ### PCA

        # reconstruction_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.reconstruction_thresholds: List[float] = [0.9]
        # window_size needs to be an iterable, use [None] to disable
        # window_size = [1, 2, 3, 5, 10, 20]
        self.window_sizes: List[Optional[int]] = [None, 1, 5, 10, 25, 50]
        # self.window_sizes: List[Optional[int]] = [1, None] 
        self.include_test_token_t_in_train_pca: bool = False
        self.do_sort_pca_components_per_token: bool = True

        ### Dependent parameters
        self.num_test_stories = self.num_total_stories - self.num_train_stories

        ### Assertions
        if len(self.reconstruction_thresholds) > 1:
            assert self.num_test_stories == 1

        ### String summarizing the parameters for saving results
        model_str = self.llm_name.split("/")[-1]
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        story_idxs_str = (
            "_".join([str(i) for i in self.selected_story_idxs])
            if self.selected_story_idxs is not None
            else "all"
        )
        window_size_str = "_".join([str(w) for w in self.window_sizes])
        threshold_str = "_".join([str(t) for t in self.reconstruction_thresholds])

        self.input_file_str = (
            f"model_{model_str}"
            + f"_dataset_{dataset_str}"
            + f"_samples_{self.num_total_stories}"
        )

        self.output_file_str = (
            self.input_file_str
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

        if (
            cfg.include_test_token_t_in_train_pca
        ):  # move the current test token into the train window
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
        f"Mean PCA Components Required Across Stories (95% CI) - Layer {cfg.layer_idx}"
    )
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_fname = f"intrinsic_dimension_{cfg.output_file_str}"
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
            label = f"{model_short_name} L{results["layer_idx"]}, {dataset_short_name}"
            
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
    ax.set_title(f"Intrinsic Dimension Comparison Across Models (95% CI)\n {int(threshold*100)}% expvar")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_fname = f"intrinsic_dimension_multiple_models_{cfg.dataset_name.split('/')[-1]}"
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
    cfg = Config()

    # Load activations and compute intrinsic dimension for each model
    id_result_list = []
    for llm_cfg in cfg.llms:
        for ds_cfg in cfg.datasets:
            # Update config for current model
            cfg.llm_name = llm_cfg.name
            cfg.layer_idx = llm_cfg.layer_idx
            cfg.llm_batch_size = llm_cfg.batch_size
            cfg.dataset_name = ds_cfg.name
            cfg.hf_text_identifier = ds_cfg.hf_text_identifier
            cfg.llm_revision = llm_cfg.revision
            
            (
                act_train_LBPD,
                act_test_LBPD,
                mask_train_BP,
                mask_test_BP,
                tokens_test_BP,
                num_test_stories,
                dataset_idxs_test,
            ) = load_activation_split(cfg)

            # Compute intrinsic dimension for a single window (no window)
            id_mean_PT, id_ci_PT = compute_intrinsic_dimension(
                act_train_LBPD=act_train_LBPD,
                act_test_LBPD=act_test_LBPD,
                cfg=cfg,
                window_size=None
            )

            id_result_list.append({
                "id_mean_PT": id_mean_PT,
                "id_ci_PT": id_ci_PT, 
                "llm_name": llm_cfg.name,
                "layer_idx": llm_cfg.layer_idx,
                "dataset_name": ds_cfg.name
            })

    # Plot results
    plot_id_multiple_models(id_result_list, cfg)


if __name__ == "__main__":
    main_id_multiple_ws()
    # main_id_multiple_models()