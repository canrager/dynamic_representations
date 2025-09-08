"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
from typing import Optional, List, Any, Tuple
from dataclasses import asdict

from src.project_config import BaseConfig
from src.preprocessing_utils import load_subsampled_act_surr, run_parameter_sweep


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


def run_single_experiment(cfg):
    import json
    import os
    
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
    
    # Save using the filename from config
    save_path = os.path.join(cfg.results_dir, f"{cfg.filename}.json")
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(artifact, f)
    
    print(f"Saved: {save_path}")




def main():
    base_cfg = BaseConfig(
        experiment_name="u_stat",
        verbose=False,
        llm_name="meta-llama/Llama-3.1-8B",
        revision=None,
        layer_idx=16,
        llm_batch_size=100,
        llm_hidden_dim=4096,
        dtype="bfloat16",
        dataset_name="monology/pile-uncopyrighted",
        hf_text_identifier="text",
        num_sequences=1_000,
        context_length=500,
        p_start=10,
        p_end=500,
        num_p=10,
        do_log_p=True,
        omit_bos=True,
        normalize_activations=False,
    )
    
    sweep_params = {
        'llm_name': ['meta-llama/Llama-3.1-8B'],
        'layer_idx': [0, 8, 16, 24, 31],
    }
    
    run_parameter_sweep(base_cfg, sweep_params, run_single_experiment)


if __name__ == "__main__":
    main()