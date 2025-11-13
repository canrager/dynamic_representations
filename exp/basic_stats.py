"""
Compute Basic stats of an SAE

- NMSE
- Cosine sim
- L0
- L1
"""

import torch as th
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import os
import json

from src.configs import *


@dataclass
class BasicStatsConfig:
    min_p: int
    max_p: int  # selected_context_length, inclusive
    num_p: int
    do_log_scale: bool

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def compute_l0_norm(f_BLS):
    l0_BL = (f_BLS != 0).float().sum(dim=-1)
    return l0_BL


def compute_l1_norm(f_BLS):
    l1_BL = f_BLS.abs().sum(dim=-1)
    return l1_BL


def compute_mse(x_hat_BLD, x_BLD):
    diff_squared_BLD = (x_hat_BLD - x_BLD) ** 2
    mse_L = diff_squared_BLD.sum(dim=-1).mean(dim=0)
    return mse_L


def compute_normalized_mse(x_hat_BLD, x_BLD):
    """Identical to fraction of variance explained if mean_dims=[0]"""
    mse_L = compute_mse(x_hat_BLD, x_BLD)
    x_mean_norm_L = x_BLD.norm(dim=-1).mean(dim=0)

    return mse_L / x_mean_norm_L


def compute_fraction_variance_explained(x_hat_BLD, x_BLD):
    total_variance = th.var(x_BLD, dim=0).sum(dim=-1)
    residual_variance = th.var(x_hat_BLD - x_BLD, dim=0).sum(dim=-1)
    return 1 - residual_variance / total_variance


def compute_cosine_similarity(x_hat_BLD, x_BLD):
    f_normed = x_hat_BLD / th.linalg.norm(x_hat_BLD, dim=-1, keepdim=True)
    x_normed = x_BLD / th.linalg.norm(x_BLD, dim=-1, keepdim=True)
    cos_sim_BL = (f_normed * x_normed).sum(dim=-1)
    return cos_sim_BL


def compute_fraction_alive_features(f_BLS):
    num_activations_S = (f_BLS != 0).float().sum(dim=(0, 1))
    is_alive_S = (num_activations_S > 0.5).float()
    frac_alive = is_alive_S.sum() / is_alive_S.shape[0]
    return frac_alive


def compute_mean_and_error(scores):
    if len(scores.shape) == 1:
        mean = scores.mean(dim=0)
        return {"score": scores.tolist(), "ci": None}
    elif len(scores.shape) == 2:
        mean = scores.mean(dim=0)
        std = scores.std(dim=0)
        ci = 1.96 * std / (scores.shape[0] ** 0.5)
        return {"score": mean.tolist(), "ci": ci.tolist()}
    else:
        ValueError("Unrecognized shape, expecting L or BL.")


@th.inference_mode()
def batch_compute_statistics(sae_cache_dir: str, cfg: BasicStatsConfig):
    results = {}
    dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]

    # Find relevant sequence position indices
    if cfg.do_log_scale:
        log_start = th.log10(th.tensor(cfg.min_p, dtype=th.float))
        log_end = th.log10(th.tensor(cfg.max_p, dtype=th.float))
        log_steps = th.linspace(log_start, log_end, cfg.num_p)
        ps = th.round(10**log_steps).int()
    else:
        ps = th.linspace(cfg.min_p, cfg.max_p, cfg.num_p, dtype=th.int)

    results["sequence_pos_indices"] = ps.tolist()
    ps = ps.to(cfg.env.device)

    # Identify all filenames in dir that contain "codes" or "recons".
    pt_files = [f for f in os.listdir(sae_cache_dir) if f.endswith(".pt")]
    code_filenames = [f for f in pt_files if "codes" in f]
    recons_filenames = [f for f in pt_files if "recons" in f]

    # Compute Metrics for code-like
    for fn in code_filenames:
        path = os.path.join(sae_cache_dir, fn)
        f_BLS = th.load(path, weights_only=False).to(device=cfg.env.device, dtype=dtype)
        f_BLS = f_BLS[:, ps, :]

        l0 = compute_l0_norm(f_BLS)
        l1 = compute_l1_norm(f_BLS)
        frac_alive = compute_fraction_alive_features(f_BLS)

        results[fn] = dict(
            l0=compute_mean_and_error(l0),
            l1=compute_mean_and_error(l1),
            fraction_alive=frac_alive.item(),
        )

        del f_BLS
        th.cuda.empty_cache()

    llm_activation_path = os.path.join(os.path.dirname(sae_cache_dir), "activations.pt")
    x_BLD = th.load(llm_activation_path, weights_only=False).to(device=cfg.env.device, dtype=dtype)
    x_BLD = x_BLD[:, ps, :]

    for fn in recons_filenames:
        path = os.path.join(sae_cache_dir, fn)
        x_hat_BLD = th.load(path, weights_only=False).to(device=cfg.env.device, dtype=dtype)
        x_hat_BLD = x_hat_BLD[:, ps, :]

        mse = compute_mse(x_hat_BLD, x_BLD)
        nmse = compute_normalized_mse(x_hat_BLD, x_BLD)
        fve = compute_fraction_variance_explained(x_hat_BLD, x_BLD)
        cos_sim = compute_cosine_similarity(x_hat_BLD, x_BLD)

        results[fn] = dict(
            mse=compute_mean_and_error(mse),
            normalized_mse=compute_mean_and_error(nmse),
            fraction_variance_explained=compute_mean_and_error(fve),
            cosine_similarity=compute_mean_and_error(cos_sim),
        )

        del x_hat_BLD
        th.cuda.empty_cache()

    del x_BLD
    th.cuda.empty_cache()

    return results


def single_pca_experiment(cfg: BasicStatsConfig):
    matching_artifact_path = find_matching_config_folder(
        cfg, cfg.env.activations_dir, recency_rank=0, compared_attributes=["llm", "data"]
    )
    matching_sae_path = os.path.join(matching_artifact_path, cfg.sae.name)
    results = batch_compute_statistics(matching_sae_path, cfg)

    results["config"] = asdict(cfg)

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"basic_stats_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"saved results to: {save_path}")


def main():
    configs = get_configs(
        cfg_class=BasicStatsConfig,
        min_p=1,
        max_p=499,
        num_p=10,
        do_log_scale=False,
        # Artifacts
        env=ENV_CFG,
        # data=WEBTEXT_DS_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        llm=GEMMA2_LLM_CFG,
        # sae=RELU_2X_DENSER_SELFTRAIN_SAE_CFG,
        sae=[
            # GEMMA2_BATCHTOPK_SAE_CFG,
            # GEMMA2_TOPK_SAE_CFG,
            # GEMMA2_RELU_SAE_CFG,
            # GEMMA2_TEMPORAL_SAE_CFG,
            GEMMA2_TEMPORAL_PRED_ONLY_SAE_CFG
        ],
    )

    for cfg in configs:
        single_pca_experiment(cfg)
        time.sleep(1)


if __name__ == "__main__":
    main()
