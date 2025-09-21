"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
from dataclasses import dataclass, asdict
from datetime import datetime
from copy import deepcopy
import time
import gc

from src.configs import *


@dataclass
class IDConfig:
    reconstruction_threshold: int
    min_p: int
    max_p: int  # selected_context_length, inclusive
    num_p: int
    num_sequences: int
    do_log_spacing: bool
    act_path: str

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def compute_u_statistic(acts_BPD: th.Tensor, cfg):
    B, P, D = acts_BPD.shape
    id_P = th.zeros(P)

    acts_BPD = acts_BPD.to(cfg.env.device)
    acts_centered_BPD = acts_BPD - acts_BPD.mean(dim=0).float()
    acts_normalized_BPD = acts_centered_BPD / acts_centered_BPD.norm(dim=-1, keepdim=True)

    for p in range(P):
        X = acts_normalized_BPD[:, p, :]
        gram = X @ X.T
        fro2 = (gram**2).sum()

        id_P[p] = (B**2 - B) / (fro2 - B)

    return id_P.cpu()


def compute_rank(acts_BPD: th.Tensor, cfg):
    B, P, D = acts_BPD.shape
    id_P = th.zeros(P)

    acts_BPD = acts_BPD.to(cfg.env.device)
    acts_centered_BPD = acts_BPD - acts_BPD.mean(dim=0).float()
    acts_normalized_BPD = acts_centered_BPD / acts_centered_BPD.norm(dim=-1, keepdim=True)

    for p in range(P):
        X = acts_normalized_BPD[:, p, :]
        S = th.linalg.svdvals(X)
        S2 = S**2
        S2_normalized = S2 / S2.sum()
        components_mask = th.cumsum(S2_normalized, dim=0) < cfg.reconstruction_threshold

        id_P[p] = components_mask.sum()

    return id_P.cpu()


def single_pca_experiment(cfg: IDConfig, acts_BPD: th.Tensor):
    if cfg.do_log_spacing:
        # Log steps from smallest we to largest we, first and las indices of wes should be those
        log_start = th.log10(th.tensor(cfg.min_p, dtype=th.float))
        log_end = th.log10(th.tensor(cfg.max_p, dtype=th.float))
        log_steps = th.linspace(log_start, log_end, cfg.num_p)
        ps = th.round(10**log_steps).int()
    else:
        # Linear steps from smallest we to largest we, first and las indices of wes should be those
        ps = th.linspace(cfg.min_p, cfg.max_p, cfg.num_windows, dtype=th.int)

    acts_BpD = acts_BPD[: cfg.num_sequences, ps, :]

    ustat_p = compute_u_statistic(acts_BpD, cfg)
    rank_p = compute_rank(acts_BpD, cfg)
    results = dict(
        ps=ps.cpu().tolist(),
        ustat_p=ustat_p.cpu().tolist(),
        rank_p=rank_p.cpu().tolist(),
        config=asdict(cfg),
    )

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"id_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"saved results to: {save_path}")

    # Cleanup
    del acts_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_gemma_act_configs(
        cfg_class=IDConfig,
        act_paths=(
            (
                [None], 
                [
                    "activations", 
                    # "surrogate"
                ]
            ),
        ),
        reconstruction_threshold=0.9,
        min_p=20,
        max_p=499,  # 0-indexed
        num_p=7,
        do_log_spacing=True,
        num_sequences=[10, 100, 1000, 10000],
        # Artifacts
        env=ENV_CFG,
        data=DatasetConfig(
            name="Webtext",
            hf_name="monology/pile-uncopyrighted",
            num_sequences=10000,
            context_length=500,
        ),
        llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    for cfg in configs:
        artifacts, _ = load_matching_activations(
            cfg,
            [cfg.act_path],
            cfg.env.activations_dir,
            compared_attributes=["llm", "data"],
            verbose=False,
        )
        acts_BPD = artifacts[cfg.act_path]
        single_pca_experiment(cfg, acts_BPD)
        time.sleep(1)


if __name__ == "__main__":
    main()
