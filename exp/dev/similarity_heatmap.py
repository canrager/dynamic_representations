import os
import torch as th
from typing import Optional, List, Literal, Union
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE
from src.exp_utils import compute_or_load_sae_artifacts
from tqdm import trange

from src.exp_utils import load_tokens_of_story


class Config:
    def __init__(self):
        self.debug: bool = False

        ### Model
        # self.model_name: str = "openai-community/gpt2"
        # self.layer_idx: int = 6
        # self.llm_name: str = "meta-llama/Meta-Llama-3-8B"
        # self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.llm_name: str = "google/gemma-2-2b"
        self.layer_idx: int = 12
        self.llm_batch_size: str = 100

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

        self.sae_batch_size: int = 100

        ### Dataset
        # self.dataset_name: str = "SimpleStories/SimpleStories"
        self.dataset_name: str = "twists_nopunctuation.json"
        # self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        self.hf_text_identifier: str = "story"
        self.num_total_stories: int = None
        self.selected_story_idxs = None  # Has to be none in this case

        self.story_to_plot: int = 7
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = None  # put 63 here for selecting story 1
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        ### String summarizing the parameters for saving results
        llm_str = self.llm_name.split("/")[-1]
        sae_str = self.sae_name.split("/")[-1]
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        story_idxs_str = (
            "_".join([str(i) for i in self.selected_story_idxs])
            if self.selected_story_idxs is not None
            else "all"
        )

        self.input_file_str = f"{llm_str}" + f"_{dataset_str}" + f"_{self.num_total_stories}"

        self.sae_file_str = self.input_file_str + f"_{sae_str}"

        self.output_file_str = (
            self.input_file_str
            + f"_{sae_str}"
            + f"_lay_{self.layer_idx}"
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{story_idxs_str}"
        )


if __name__ == "__main__":

    cfg = Config()

    llm_act_BPD, masks_BP, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP = (
        compute_or_load_sae_artifacts(cfg)
    )

    # llm_act_PD = th.cat([llm_act_BPD[0, 0:20, :], llm_act_BPD[1, 0:20, :]], dim=1)
    # llm_act_centered_PD = llm_act_PD - llm_act_PD.mean(dim=-1, keepdim=True)
    mask_P = masks_BP[cfg.story_to_plot].bool()
    llm_act_PD = llm_act_BPD[cfg.story_to_plot].squeeze()
    llm_act_PD = llm_act_PD[mask_P]

    print(f"llm_act_PD.shape {llm_act_PD.shape}")
    llm_act_norm_PD = llm_act_PD / llm_act_PD.norm(dim=-1, keepdim=True)
    llm_act_PP = llm_act_norm_PD @ llm_act_norm_PD.T
    print(f"max: {llm_act_PP.max()}, min: {llm_act_PP.min()}")
    llm_act_PP = llm_act_PP.float().cpu().numpy()

    # latent_acts_PS = th.cat([latent_acts_BPS[0, 0:20], latent_acts_BPS[1, 0:20]], dim=1)
    # latent_acts_centered_PS = latent_acts_PS - latent_acts_PS.mean(dim=-1, keepdim=True)

    latent_acts_PS = latent_acts_BPS[cfg.story_to_plot].squeeze()
    latent_acts_PS = latent_acts_PS[mask_P]
    print(f"latent_acts_PS {latent_acts_PS.shape}")
    latent_acts_norm_PS = latent_acts_PS / latent_acts_PS.norm(dim=-1, keepdim=True)
    latent_acts_PP = latent_acts_norm_PS @ latent_acts_norm_PS.T
    print(f"max: {latent_acts_PP.max()}, min: {latent_acts_PP.min()}")
    latent_acts_PP = latent_acts_PP.float().cpu().numpy()

    # Load token strings

    token_str = load_tokens_of_story(
        dataset_name=cfg.dataset_name,
        dataset_num_stories=cfg.num_total_stories,
        story_idx=cfg.story_to_plot,
        input_file_str=cfg.input_file_str,
        model_name=cfg.llm_name,
        omit_BOS_token=cfg.omit_BOS_token,
        seq_length=cfg.num_tokens_per_story,
    )
    token_str = [token_str[i] for i in range(len(token_str)) if mask_P[i]]
    token_ticks = th.arange(len(token_str))

    # Both are cosine similarity matrices, so they have the same scale [-1, 1]
    # vmin, vmax = -1, 1
    vmin, vmax = None, None

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot LLM activations
    im1 = ax[0].imshow(llm_act_PP, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax[0].set_title("LLM Activations")
    ax[0].set_xlabel("Token Index")
    ax[0].set_ylabel("Token Index")
    ax[0].set_yticks(token_ticks)
    ax[0].set_yticklabels(token_str)

    # Plot SAE outputs
    im2 = ax[1].imshow(latent_acts_PP, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax[1].set_title("SAE Latents")
    ax[1].set_xlabel("Token Index")
    ax[1].set_ylabel("Token Index")
    ax[1].set_yticks(token_ticks)
    ax[1].set_yticklabels(token_str)

    # Add a single colorbar to the right of the entire figure
    fig.colorbar(im2, ax=ax, label="Cosine Similarity", location="right")
    fig.suptitle(cfg.sae_file_str)

    fig_path = os.path.join(PLOTS_DIR, f"cossim_heatmap_llm_vs_latents_{cfg.output_file_str}.png")

    plt.savefig(fig_path)
    print(f"Saved {fig_path}")
    plt.close()
