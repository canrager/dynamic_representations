import os
import torch as th
from typing import Optional, List, Literal, Union
import matplotlib.pyplot as plt
import einops
from src.project_config import PLOTS_DIR, DEVICE
from src.exp_utils import compute_or_load_sae
from tqdm import trange


class Config:
    def __init__(self):
        self.debug: bool = False

        ### Model
        # self.model_name: str = "openai-community/gpt2"
        # self.layer_idx: int = 6
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 12
        self.llm_batch_size: str = 100

        self.sae_name: str = "EleutherAI/sae-llama-3-8b-32x"
        self.sae_batch_size: int = 100

        ### Dataset
        self.dataset_name: str = "SimpleStories/SimpleStories"
        # self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        self.hf_text_identifier: str = "story"
        self.num_total_stories: int = 100

        self.selected_story_idxs: Optional[List[int]] = [0]
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 75
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

        self.input_file_str = (
            f"{llm_str}" + f"_{dataset_str}" + f"_{self.num_total_stories}"
        )

        self.sae_file_str = self.input_file_str + f"_{sae_str}"

        self.output_file_str = (
            self.input_file_str
            + f"_lay_{self.layer_idx}"
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{story_idxs_str}"
        )


if __name__ == "__main__":

    cfg = Config()

    llm_act_BPD, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP = (
        compute_or_load_sae(cfg)
    )

    llm_act_PD = th.cat([llm_act_BPD[0, 0:20, :], llm_act_BPD[1, 0:20, :]], dim=1)
    # llm_act_centered_PD = llm_act_PD - llm_act_PD.mean(dim=-1, keepdim=True)
    llm_act_norm_PD = llm_act_PD / llm_act_PD.norm(dim=-1, keepdim=True)
    llm_act_PP = llm_act_norm_PD @ llm_act_norm_PD.T
    print(f"max: {llm_act_PP.max()}, min: {llm_act_PP.min()}")
    llm_act_PP = llm_act_PP.float().cpu().numpy()

    latent_acts_PS = th.cat([latent_acts_BPS[0, 0:20], latent_acts_BPS[1, 0:20]], dim=1)
    # latent_acts_centered_PS = latent_acts_PS - latent_acts_PS.mean(dim=-1, keepdim=True)
    latent_acts_norm_PS = latent_acts_PS / latent_acts_PS.norm(dim=-1, keepdim=True)
    latent_acts_PP = latent_acts_norm_PS @ latent_acts_norm_PS.T
    print(f"max: {latent_acts_PP.max()}, min: {latent_acts_PP.min()}")
    latent_acts_PP = latent_acts_PP.float().cpu().numpy()

    # Both are cosine similarity matrices, so they have the same scale [-1, 1]
    # vmin, vmax = -1, 1
    vmin, vmax = None, None

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot LLM activations
    im1 = ax[0].imshow(llm_act_PP, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax[0].set_title("LLM Activations")
    ax[0].set_xlabel("Token Index")
    ax[0].set_ylabel("Token Index")

    # Plot SAE outputs
    im2 = ax[1].imshow(latent_acts_PP, cmap="RdBu", vmin=vmin, vmax=vmax)
    ax[1].set_title("SAE Outputs")
    ax[1].set_xlabel("Token Index")
    ax[1].set_ylabel("Token Index")

    # Add a single colorbar to the right of the rightmost plot
    fig.colorbar(im2, ax=ax[1], label="Cosine Similarity")

    fig_path = os.path.join(
        PLOTS_DIR, f"cossim_heatmap_llm_vs_latents_{cfg.output_file_str}.png"
    )
    plt.savefig(fig_path)
    print(f"Saved {fig_path}")
    plt.close()
