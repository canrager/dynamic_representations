"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import math
import json
from typing import Optional, List, Any, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm, trange

from src.project_config import PLOTS_DIR, DEVICE, LLMConfig, DatasetConfig, BaseConfig
from src.exp_utils import compute_or_load_llm_artifacts, compute_or_load_surrogate_artifacts
from src.model_utils import load_tokenizer, load_nnsight_model, load_sae
from src.preprocessing_utils import load_subsampled_act_surr


def u_statistic(acts_BPD: th.Tensor, cfg):
    acts_BPD = acts_BPD.to(cfg.device)
    B, P, D = acts_BPD.shape
    id_P = th.zeros(P)

    # mean_D = acts_BPD.mean(dim=(0, 1))
    # acts_BPD = acts_BPD - mean_D
    acts_normalized_BPD = acts_BPD / acts_BPD.norm(dim=-1, keepdim=True)

    for p in range(P):
        X = acts_normalized_BPD[:, p, :]
        gram = X @ X.T
        fro2 = (gram**2).sum()

        id_P[p] = (B**2 - B) / (fro2 - B)

    return id_P


def main(cfg):
    act_BPD, surr_BPD, p_indices = load_subsampled_act_surr(cfg)
    ustat_act_P = u_statistic(act_BPD, cfg)
    ustat_surr_P = u_statistic(surr_BPD, cfg)

    results = {
        "p_indices": p_indices.cpu().tolist(),
        "ustat_act_P": ustat_act_P.cpu().tolist(),
        "ustat_surr_P": ustat_surr_P.cpu().tolist(),
    }
    artifact = {
        "config": asdict(cfg),
        "results": results
    }
    
    # Save
    save_path = os.path.join(cfg.interim_dir, "u_stat_llm.json")
    with open(save_path, 'w') as f:
        json.dump(artifact, f)


if __name__ == "__main__":

    cfg = BaseConfig(
        # Debugging
        verbose=False,
        # LLM
        llm_name="meta-llama/Llama-3.1-8B",
        revision=None,
        layer_idx=16,
        llm_batch_size=100,
        llm_hidden_dim=4096,
        dtype="bfloat16",
        # Dataset
        dataset_name="monology/pile-uncopyrighted",
        hf_text_identifier="text",
        # 80GB, 15min on A6000 for 10_000 sequences, 500 tokens, bf16
        num_sequences=1_000,
        context_length=500,
        # Preprocessing
        p_start=10,
        p_end=500,
        num_p=10,
        do_log_p=True,
        omit_bos=True,
        normalize_activations=False,
    )

    print(main(cfg))
