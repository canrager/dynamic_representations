"""
Compute intrinsic dimensionality of LLM activations across token positions
"""

import torch as th
from dataclasses import dataclass, asdict
from datetime import datetime
from copy import deepcopy
import time
import gc
import json

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


def diagonal_means_batch_vectorized(X):
    """
    More vectorized version using torch.diagonal's batch support.
    """
    B, P, _ = X.shape
    Y = th.zeros(B, P)

    for i in range(P):
        # Extract i-th diagonal for all batches at once
        diagonals = th.diagonal(X, offset=i, dim1=1, dim2=2)  # Shape: (B, P-i)
        Y[:, i] = diagonals.mean(dim=1)

    return Y


def compute_u_statistic(acts_BPD: th.Tensor, cfg):
    B, P, D = acts_BPD.shape
    id_P = th.zeros(P)
    surrogate_id_P = th.zeros(P)

    acts_BPD = acts_BPD.to(cfg.env.device)
    acts_centered_BPD = acts_BPD - acts_BPD.mean(dim=0).float()
    acts_normalized_BPD = acts_centered_BPD / acts_centered_BPD.norm(dim=-1, keepdim=True)

    # Create different random permutations across P dimension for each batch
    shuffled_acts_normalized_BPD = th.zeros_like(acts_normalized_BPD)
    for b in range(B):
        # Generate a random permutation for this batch
        perm = th.randperm(P, device=cfg.env.device)
        shuffled_acts_normalized_BPD[b] = acts_normalized_BPD[b, perm]

    for p in range(P):
        X = acts_normalized_BPD[:, p, :]
        gram = X @ X.T
        fro2 = (gram**2).sum()

        id_P[p] = (B**2 - B) / (fro2 - B)

        # Compute surrogate ID using shuffled data
        X_shuffled = shuffled_acts_normalized_BPD[:, p, :]
        surrogate_gram = X_shuffled @ X_shuffled.T
        surrogate_fro2 = (surrogate_gram**2).sum()
        surrogate_id_P[p] = (B**2 - B) / (surrogate_fro2 - B)

    return id_P.cpu(), surrogate_id_P.cpu()


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

    ustat_p, surrogate_ustat_p = compute_u_statistic(acts_BpD, cfg)
    rank_p = compute_rank(acts_BpD, cfg)
    results = dict(
        ps=ps.cpu().tolist(),
        ustat_p=ustat_p.cpu().tolist(),
        surrogate_ustat_p=surrogate_ustat_p.cpu().tolist(),
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
            # (
            #     [None],
            #     [
            #         # "activations",
            #         # "surrogate"
            #     ],
            # ),
            # (
            #     GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
            #     [
            #         "codes",
            #         # "recons"
            #     ]
            # ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         "codes",
            #         "recons"
            #     ]
            # ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    "novel_codes",
                    "novel_recons",
                    "pred_codes",
                    "pred_recons",
                    "total_recons",
                ]
            ),
        ),
        reconstruction_threshold=0.9,
        min_p=20,
        max_p=499,  # 0-indexed
        num_p=200,
        do_log_spacing=True,
        num_sequences=10000,
        # Artifacts
        env=ENV_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        # data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    for cfg in configs:
        act_path = cfg.act_path
        if cfg.sae is not None:
            act_path = os.path.join(cfg.sae.name, cfg.act_path)
        artifacts, _ = load_matching_activations(
            cfg,
            [act_path],
            cfg.env.activations_dir,
            compared_attributes=["llm", "data"],
            verbose=False,
        )
        acts_BPD = artifacts[act_path]
        single_pca_experiment(cfg, acts_BPD)
        time.sleep(1)


if __name__ == "__main__":
    main()
