"""
Plot number of active sae features over tokens
"""

import torch as th
from typing import Optional, List
from src.exp_utils import compute_or_load_sae_artifacts
from src.project_config import DEVICE
import matplotlib.pyplot as plt
import os
from src.project_config import PLOTS_DIR


class Config:
    def __init__(self):
        self.debug = False

        self.llm_name: str = "google/gemma-2-2b"
        # self.llm_name: str = "meta-llama/Llama-3.1-8B"
        # self.llm_name: str = "meta-llama/Meta-Llama-3-8B"
        self.layer_idx: int = 12
        self.llm_batch_size: int = 100

        self.dtype = th.float32

        # self.sae_architecture = "topk"
        # self.sae_repo_id = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
        # self.sae_filename = "gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2/ae.pt"
        # self.sae_name: str = "saebench_gemma-2-2b_topk-80_width-2pow14_layer_12_trainer_2"
        # self.d_sae: int = 192 # 16384

        self.sae_architecture = "relu"
        self.sae_repo_id = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
        self.sae_filename = (
            "gemma-2-2b_standard_new_width-2pow14_date-0107/resid_post_layer_12/trainer_4/ae.pt"
        )
        self.sae_name: str = "saebench_gemma-2-2b_relu_width-2pow14_layer_12_trainer_4"
        self.d_sae: int = 16384

        # self.sae_name: str = "EleutherAI/sae-llama-3-8b-32x"
        # self.d_sae: int = 80

        self.sae_batch_size: int = 100

        ### Dataset
        # self.dataset_name: str = "SimpleStories/SimpleStories"
        self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        # self.hf_text_identifier: str = "story"
        self.hf_text_identifier: str = "text"
        self.num_total_stories: int = 400

        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 100
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        # self.latent_active_threshs: bool = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        self.latent_active_threshs: bool = [0.1]
        self.sort_variance: bool = True
        self.reconstruction_thresholds: List[float] = [0.9]

        ### String summarizing the parameters for loading and saving artifacts
        self.llm_str = self.llm_name.split("/")[-1]
        self.sae_str = self.sae_name.split("/")[-1]
        self.dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.story_idxs_str = (
            "_".join([str(i) for i in self.story_idxs])
            if self.selected_story_idxs is not None
            else "all"
        )

        self.input_file_str = (
            f"{self.llm_str}" + f"_{self.dataset_str}" + f"_{self.num_total_stories}"
        )

        self.sae_file_str = self.input_file_str + f"_{self.sae_str}"

        self.output_file_str = (
            self.sae_file_str
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{self.story_idxs_str}"
        )


def plot_fvu(fvu_BP, cfg):
    fvu_mean_P = fvu_BP.float().mean(dim=0)
    fvu_std_P = fvu_BP.float().std(dim=0)
    B = fvu_BP.shape[0]
    fvu_ci_P = 1.96 * fvu_std_P / (B**0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fvu_mean_P, label="fvu")
    ax.fill_between(range(len(fvu_mean_P)), fvu_mean_P - fvu_ci_P, fvu_mean_P + fvu_ci_P, alpha=0.2)

    ax.set_xlabel("Token position")
    ax.set_ylabel("FVU")
    ax.set_title(f"FVU over tokens\nsae {cfg.sae_str}, dataset {cfg.dataset_str}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"fvu_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved FVU plot to {save_dir}")
    plt.close()


def plot_reconstruction_exp_var(llm_act_BPD, sae_out_BPD, cfg):
    llm_var_BP = (llm_act_BPD**2).sum(-1)
    out_var_BP = (sae_out_BPD**2).sum(-1)
    exp_var_BP = out_var_BP / llm_var_BP

    exp_var_mean_P = exp_var_BP.mean(0)
    exp_var_std_P = exp_var_BP.std(0)
    B = exp_var_BP.shape[0]
    exp_var_ci_P = 1.96 * exp_var_std_P / (B**0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(exp_var_mean_P, label="exp_var")
    ax.fill_between(
        range(len(exp_var_mean_P)),
        exp_var_mean_P - exp_var_ci_P,
        exp_var_mean_P + exp_var_ci_P,
        alpha=0.2,
    )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Explained variance")
    ax.set_title(f"Explained variance over tokens\nsae {cfg.sae_str}, dataset {cfg.dataset_str}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"exp_var_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved SAE explained variance over tokens plot to {save_dir}")
    plt.close()


def plot_activation_distribution(latent_act_BPS, cfg):
    latent_act_BPS = latent_act_BPS.float().detach().cpu()

    if cfg.sort_variance:
        latent_act_BPS, _ = th.sort(latent_act_BPS, dim=-1, descending=True)

    latent_act_mean_S = latent_act_BPS.mean(dim=(0, 1))
    latent_act_std_S = latent_act_BPS.std(dim=(0, 1))
    B, P, S = latent_act_BPS.shape
    latent_act_ci_S = 1.96 * latent_act_std_S / (B**0.5)  # only stories are independent

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(latent_act_mean_S, color="blue")
    ax.scatter(
        range(len(latent_act_mean_S)),
        latent_act_mean_S,
        label="activation value",
        color="blue",
    )
    ax.fill_between(
        range(len(latent_act_mean_S)),
        latent_act_mean_S - latent_act_ci_S,
        latent_act_mean_S + latent_act_ci_S,
        alpha=0.2,
        label=f"95% CI",
        color="blue",
    )

    ax.set_xlabel("rank")
    ax.set_ylabel("mean activation magnitude")
    ax.set_title(
        f"Activation, mean over batch and token pos, sorted by magnitude\nsae {cfg.sae_str}, dataset {cfg.dataset_str}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"activation_distribution_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved SAE latent var distribution over tokens plot to {save_dir}")
    plt.close()


def plot_num_active_latents(latent_acts_BPS, cfg):

    max_activation_S = latent_acts_BPS.flatten(0, 1).max(0).values

    # Find active latents
    latent_active_threshs_ST = (
        max_activation_S[:, None] * th.tensor(cfg.latent_active_threshs)[None, :]
    )
    is_active_BPST = latent_acts_BPS[:, :, :, None] > latent_active_threshs_ST[None, None, :, :]
    num_active_BPT = is_active_BPST.sum(dim=-2)

    # Mean, std, CI over batch
    num_active_mean_PT = num_active_BPT.float().mean(dim=0)
    num_active_std_PT = num_active_BPT.float().std(dim=0)
    B = num_active_BPT.shape[0]
    num_active_ci_PT = 1.96 * num_active_std_PT / (B**0.5)

    fig, ax = plt.subplots(figsize=(8, 6))
    for t_idx, t in enumerate(cfg.latent_active_threshs):
        num_active_mean_P = num_active_mean_PT[:, t_idx]
        num_active_ci_P = num_active_ci_PT[:, t_idx]
        ax.plot(num_active_mean_P, label=f"active above {t}*max_act")
        ax.fill_between(
            range(len(num_active_mean_P)),
            num_active_mean_P - num_active_ci_P,
            num_active_mean_P + num_active_ci_P,
            alpha=0.2,
        )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Number of active latents")
    ax.set_title(
        f"Number of active latents over tokens\nsae {cfg.sae_name}, dataset {cfg.dataset_name}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"num_active_latents_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved num active figure to {save_dir}")
    plt.close()


def compute_latent_over_latent_exp_var(latent_act_BPS, cfg):

    latent_var_BPS = latent_act_BPS**2
    print(f"latent_var_BPS {latent_var_BPS.shape}")
    latent_var_total_BP = latent_var_BPS.sum(-1)

    if cfg.sort_variance:
        latent_var_BPS, _ = th.sort(latent_var_BPS, dim=-1, descending=True)

    latent_var_cumulative_BPS = th.cumsum(latent_var_BPS, dim=-1)
    exp_var_cumulative_BPS = latent_var_cumulative_BPS / latent_var_total_BP[:, :, None]

    return exp_var_cumulative_BPS


def compute_recon_over_orig_exp_var(sae, llm_act_BPD, latent_act_BPS, cfg):

    llm_act_BPD = latent_act_BPS**2
    llm_var_total_BP = llm_act_BPD.sum(-1)

    raise NotImplementedError


def find_min_latents_required(exp_var_cumulative_BPS, cfg):
    exp_var_thresholds = th.tensor(cfg.reconstruction_thresholds)
    meets_threshold_BPST = (
        exp_var_cumulative_BPS[:, :, :, None] >= exp_var_thresholds[None, None, None, :]
    )

    min_latents_required_BPT = th.argmax(meets_threshold_BPST.int(), dim=-2) + 1

    has_solution_BPT = th.any(meets_threshold_BPST, dim=-2)
    min_latents_required_BPT.masked_fill_(~has_solution_BPT, cfg.d_sae)

    min_latents_required_mean_PT = min_latents_required_BPT.float().mean(0)
    min_latents_required_std_PT = min_latents_required_BPT.float().std(0)

    B = min_latents_required_BPT.shape[0]
    min_latents_required_ci_PT = 1.96 * min_latents_required_std_PT / (B**0.5)

    return min_latents_required_mean_PT, min_latents_required_ci_PT


def plot_min_components_exp_var(min_latents_required_mean_PT, min_latents_required_ci_PT):
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in range(len(cfg.reconstruction_thresholds)):
        min_latents_required_mean_P = min_latents_required_mean_PT[:, t]
        min_latents_required_ci_P = min_latents_required_ci_PT[:, t]
        ax.plot(
            min_latents_required_mean_P,
            label=f"exp_var > {cfg.reconstruction_thresholds[t]}",
        )
        ax.fill_between(
            range(len(min_latents_required_mean_P)),
            min_latents_required_mean_P - min_latents_required_ci_P,
            min_latents_required_mean_P + min_latents_required_ci_P,
            alpha=0.2,
        )

    ax.set_xlabel("Token position")
    ax.set_ylabel("Minimum number of latents required")
    ax.set_title(
        f"Number of sae latents required to reconstruct fixed fraction of total sae latent variance\nsae {cfg.sae_name}, dataset {cfg.dataset_name}"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"latent_over_latent_exp_var_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(
        f"\nSaved Latent SAE latent over latent explained variance over tokens plot to {save_dir}"
    )
    plt.close()


if __name__ == "__main__":
    cfg = Config()

    llm_act_BPD, masks_BP, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP = (
        compute_or_load_sae_artifacts(cfg)
    )

    plot_num_active_latents(latent_acts_BPS, cfg)

    exp_var_cumulative = compute_latent_over_latent_exp_var(latent_acts_BPS, cfg)
    print(exp_var_cumulative.shape)
    mean, ci = find_min_latents_required(exp_var_cumulative, cfg)
    print(mean.shape, ci.shape)
    plot_min_components_exp_var(mean, ci)
