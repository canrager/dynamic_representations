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
    do_log_spacing: bool

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

    acts_BpD = acts_BPD[:, ps, :]

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
    configs = get_configs(
        cfg_class=IDConfig,
        reconstruction_threshold=0.9,
        min_p=20,
        max_p=499,  # 0-indexed
        num_p=7,
        do_log_spacing=True,
        # Artifacts
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=[None] + GEMMA2_SAE_CFGS,
    )

    for cfg in configs:

        if cfg.sae is None:
            # Run on LLM activations
            keys = ["activations", "surrogate"]
        elif "temporal" in cfg.sae.name.lower():
            # Run on codes and reconstruction
            keys = [
                f"{cfg.sae.name}/novel_codes",
                f"{cfg.sae.name}/novel_recons",
                f"{cfg.sae.name}/pred_codes",
                f"{cfg.sae.name}/pred_recons",
                f"{cfg.sae.name}/total_recons",
            ]
        else:
            keys = [f"{cfg.sae.name}/latents", f"{cfg.sae.name}/reconstructions"]

        artifacts, _ = load_matching_artifacts(
            cfg, keys, cfg.env.activations_dir, compared_attributes=["llm", "data"]
        )
        for key in artifacts:
            acts_BPD = artifacts[key]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_pca_experiment(cfg, acts_BPD)
            time.sleep(1)


if __name__ == "__main__":
    main()
