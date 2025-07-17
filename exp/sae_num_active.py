'''
Plot number of active sae features over tokens

'''

from typing import Optional, List
from src.exp_utils import compute_or_load_sae
from src.project_config import DEVICE

class Config():
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

        self.story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 75
        self.do_train_test_split: bool = False
        self.num_train_stories: int = 75
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )


        ### String summarizing the parameters for loading and saving artifacts
        llm_str = self.llm_name.split("/")[-1]
        sae_str = self.sae_name.split("/")[-1]
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        story_idxs_str = (
            "_".join([str(i) for i in self.story_idxs])
            if self.story_idxs is not None
            else "all"
        )

        self.input_file_str = (
            f"{llm_str}"
            + f"_{dataset_str}"
            + f"_{self.num_total_stories}"
        )

        self.sae_file_str = self.input_file_str + f"_{sae_str}"

        self.output_file_str = (
            self.sae_file_str
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{story_idxs_str}"
        )


if __name__ == "__main__":

    cfg = Config()

    fvu_BP, latent_acts_BPS, latent_indices_BPK = compute_or_load_sae(
        sae_name=cfg.sae_name,
        model_name=cfg.llm_name,
        num_stories=cfg.num_total_stories,
        layer_idx=cfg.layer_idx,
        batch_size=cfg.sae_batch_size,
        device=DEVICE,
        force_recompute=cfg.force_recompute,
        do_omit_BOS_token=cfg.omit_BOS_token,
        input_str=cfg.input_file_str,
        story_idxs=cfg.story_idxs,
        num_tokens_per_story=cfg.num_tokens_per_story,
        cfg = cfg,
    )

    print(f'fvu_BP {fvu_BP.shape}')
    print(f'latent_acts_BPS {latent_acts_BPS.shape}')
    print(f'latent_indices_BPK {latent_indices_BPK.shape}')