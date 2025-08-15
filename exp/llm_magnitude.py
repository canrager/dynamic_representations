'''
Plot magnitude of activations for syntactic complexity phrasal verb variations
'''

import torch as th
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import json
from typing import Optional, List
from collections import defaultdict

from src.project_config import INPUTS_DIR, PLOTS_DIR, MODELS_DIR
from src.exp_utils import compute_or_load_llm_artifacts, load_activation_split
from src.model_utils import load_tokenizer

class Config():
    def __init__(self):
        self.debug: bool = True

        # Model
        self.llm_name: str = "meta-llama/Llama-3.1-8B"
        self.layer_idx: int = 22
        self.llm_batch_size: int = 100

        # Dataset
        ### Dataset
        self.dataset_name: str = "SimpleStories/SimpleStories"
        # self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        # self.hf_text_identifier: str = "text"
        self.hf_text_identifier: str = "story"
        self.num_total_stories: int = 100
        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 100
        self.do_train_test_split: bool = False
        self.num_train_stories: int = 75
        self.force_recompute: bool = (
            True  # Always leave True, unless iterations with experiment iteration speed. force_recompute = False has the danger of using precomputed results with incorrect parameters.
        )

        # File names
        dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        model_str = self.llm_name.split("/")[-1]
        self.input_file_str = (
            f"{dataset_str}_{model_str}_{self.num_total_stories}_{self.num_tokens_per_story}"
        )
        self.output_file_str = (
            self.input_file_str
            + f"_l_{self.layer_idx}"
            + f"_didx_{self.selected_story_idxs}"
            + f"_nobos_{self.omit_BOS_token}"
        )

if __name__ == "__main__":
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

    act_BPD = act_train_LBPD[cfg.layer_idx]
    act_norm_BP = act_BPD.norm(dim=-1)
    act_norm_mean_P = act_norm_BP.mean(dim=0)
    act_norm_std_P = act_norm_BP.std(dim=0)
    B, P, D = act_BPD.shape
    act_norm_ci_P = 1.96 * act_norm_std_P / B**0.5

    fig, ax = plt.subplots(figsize=(8,6))
    range_P = range(P)
    ax.plot(range_P, act_norm_mean_P)
    ax.fill_between(
        range_P,
        act_norm_mean_P - act_norm_ci_P,
        act_norm_mean_P + act_norm_ci_P,
        alpha=0.2,
    )
    ax.grid(alpha=0.3)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Residual stream L2 Norm")
    ax.set_title(cfg.output_file_str)

    fig_path = os.path.join(PLOTS_DIR, f"llm_rep_magnitude_{cfg.output_file_str}.png")
    plt.savefig(fig_path, dpi=80)
    print(f"saved llm mag figure at {fig_path}.")
    plt.close()