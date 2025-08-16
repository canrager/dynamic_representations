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
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 12

        self.sae_name: str = "EleutherAI/sae-llama-3-8b-32x"
        self.sae_batch_size: int = 100

        ### Dataset
        self.dataset_name: str = "SimpleStories/SimpleStories"
        # self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        self.num_total_stories: int = 100

        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 75
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        self.latent_active_threshs: bool = [0.1, 0.2, 0.3, 0.4, 0.5, 1]

        ### String summarizing the parameters for loading and saving artifacts
        llm_str = self.llm_name.split("/")[-1]
        sae_str = self.sae_name.split("/")[-1]
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        story_idxs_str = (
            "_".join([str(i) for i in self.story_idxs])
            if self.selected_story_idxs is not None
            else "all"
        )

        self.input_file_str = f"{llm_str}" + f"_{dataset_str}" + f"_{self.num_total_stories}"

        self.sae_file_str = self.input_file_str + f"_{sae_str}"

        self.output_file_str = (
            self.sae_file_str
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{story_idxs_str}"
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
    ax.set_title(f"FVU over tokens\nsae {cfg.sae_name}, dataset {cfg.dataset_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(PLOTS_DIR, f"fvu_{cfg.output_file_str}.png")
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved FVU plot to {save_dir}")
    plt.close()


if __name__ == "__main__":

    cfg = Config()

    fvu_BP, latent_acts_BPS, latent_indices_BPK = compute_or_load_sae_artifacts(cfg)

    plot_num_active_latents(latent_acts_BPS, cfg)

    plot_fvu(fvu_BP, cfg)
