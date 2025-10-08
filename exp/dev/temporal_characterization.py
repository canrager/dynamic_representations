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
import einops

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
    total_variance = th.var(x_hat_BLD, dim=0).sum(dim=-1)
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
def batch_compute_statistics(llm_cache_dir: str, cfg: BasicStatsConfig):
    print("\n\n\n")
    print(cfg.data.name)
    results = {}
    dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
    sae_dir = os.path.join(llm_cache_dir, cfg.sae.name)

    # llm_act_path = os.path.join(llm_cache_dir, "activations.pt")
    # llm_act_BLD = th.load(llm_act_path, weights_only=False).to(device=cfg.env.device, dtype=dtype)
    # B, L, D = llm_act_BLD.shape

    # pred_recons_plus_b_dir = os.path.join(sae_dir, "pred_recons_plus_b.pt")
    # pred_recons_plus_b_BLD = th.load(pred_recons_plus_b_dir, weights_only=False).to(
    #     device=cfg.env.device, dtype=dtype
    # )
    # pred_recons_plus_b_nmse = compute_normalized_mse(pred_recons_plus_b_BLD, llm_act_BLD).mean()
    # print(f"==>> pred_recons_plus_b_nmse: {pred_recons_plus_b_nmse}")

    # novel_recons_plus_b_dir = os.path.join(sae_dir, "novel_recons_plus_b.pt")
    # novel_recons_plus_b_BLD = th.load(novel_recons_plus_b_dir, weights_only=False).to(
    #     device=cfg.env.device, dtype=dtype
    # )
    # novel_recons_plus_b_nmse = compute_normalized_mse(novel_recons_plus_b_BLD, llm_act_BLD).mean()
    # print(f"==>> novel_recons_plus_b_nmse: {novel_recons_plus_b_nmse}")

    # novel_residual_BLD = llm_act_BLD - novel_recons_plus_b_BLD
    # pred_residual_BLD = llm_act_BLD - pred_recons_plus_b_BLD

    # novel_residual_normalized_BLD = novel_residual_BLD / novel_residual_BLD.norm(
    #     dim=-1, keepdim=True
    # )
    # pred_residual_normalized_BLD = pred_residual_BLD / pred_residual_BLD.norm(dim=-1, keepdim=True)

    # error_similarity = (
    #     einops.einsum(
    #         novel_residual_normalized_BLD, pred_residual_normalized_BLD, "B L D, B L D ->"
    #     )
    #     / B
    #     / L
    # )
    # print(f"==>> error_similarity: {error_similarity}")

    novel_codes_dir = os.path.join(sae_dir, "novel_codes.pt")
    novel_codes_BLD = th.load(novel_codes_dir, weights_only=False).to(
        device=cfg.env.device, dtype=dtype
    )
    B, L, _ = novel_codes_BLD.shape
    pred_codes_dir = os.path.join(sae_dir, "pred_codes.pt")
    pred_codes_BLD = th.load(pred_codes_dir, weights_only=False).to(
        device=cfg.env.device, dtype=dtype
    )
    energy_novel_codes = (novel_codes_BLD**2).sum()
    energy_pred_codes = (pred_codes_BLD**2).sum()
    frac_energy_novel_codes = energy_novel_codes / (energy_novel_codes + energy_pred_codes)
    print(f"==>> frac_energy_novel_codes: {frac_energy_novel_codes}")
    frac_energy_pred_codes = energy_pred_codes / (energy_novel_codes + energy_pred_codes)
    print(f"==>> frac_energy_pred_codes: {frac_energy_pred_codes}")

    # Compute similarity batch by batch to reduce memory usage
    code_similarity = 0.0
    for b in range(B):
        novel_b = novel_codes_BLD[b]  # L D
        pred_b = pred_codes_BLD[b]    # L D
        novel_norm_b = novel_b.norm(dim=-1, keepdim=True)  # L 1
        pred_norm_b = pred_b.norm(dim=-1, keepdim=True)    # L 1

        similarity_b = (
            einops.einsum(novel_b.float(), pred_b.float(), "L D, L D -> L")
            / (novel_norm_b.squeeze(-1) * pred_norm_b.squeeze(-1))
        ).sum() / L

        code_similarity += similarity_b

    code_similarity /= B
    print(f"==>> code_similarity: {code_similarity}")

    return results


def single_pca_experiment(cfg: BasicStatsConfig):
    matching_artifact_path = find_matching_config_folder(
        cfg, cfg.env.activations_dir, recency_rank=0, compared_attributes=["llm", "data"]
    )
    # matching_sae_path = os.path.join(matching_artifact_path, cfg.sae.name)
    results = batch_compute_statistics(matching_artifact_path, cfg)

    results["config"] = asdict(cfg)

    # # Save results
    # datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_path = os.path.join(cfg.env.results_dir, f"basic_stats_{datetetime_str}.json")
    # with open(save_path, "w") as f:
    #     json.dump(results, f)
    # print(f"saved results to: {save_path}")


def main():
    configs = get_configs(
        cfg_class=BasicStatsConfig,
        min_p=1,
        max_p=499,
        num_p=20,
        do_log_scale=False,
        # Artifacts
        env=ENV_CFG,
        # data=WEBTEXT_DS_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        llm=GEMMA2_LLM_CFG,
        # sae=RELU_2X_DENSER_SELFTRAIN_SAE_CFG,
        sae=[TEMPORAL_SELFTRAIN_SAE_CFG],
    )

    for cfg in configs:
        single_pca_experiment(cfg)
        time.sleep(1)


if __name__ == "__main__":
    main()
