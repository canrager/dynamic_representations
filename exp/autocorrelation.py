from dataclasses import dataclass, asdict
from datetime import datetime
import json
import gc
from copy import deepcopy
import time

import torch as th
import einops

from src.configs import *


@dataclass
class AutocorrelationConfig:
    min_anchor: int
    max_anchor: int  # selected_context_length, inclusive
    num_anchors: int
    min_offset: int  # absolute value
    max_offset: int  # absolute value
    act_path: str
    # steps are linearly spaced

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def compute_autocorrelation_heatmap(cfg: AutocorrelationConfig, act_BPD: th.Tensor):
    act_BPD = act_BPD / act_BPD.norm(dim=-1, keepdim=True)
    anchors_W = th.linspace(
        cfg.min_anchor, cfg.max_anchor, cfg.num_anchors, dtype=th.int
    )  # 1-indexed
    anchor_act_WBD = act_BPD[:, anchors_W - 1, :].transpose(0, 1)

    B, P, D = act_BPD.shape
    window_size = cfg.max_offset - cfg.min_offset + 1

    relative_offsets_p = th.linspace(-cfg.max_offset, -cfg.min_offset, window_size, dtype=th.int)
    window_relative_pos_indices_WBp = (
        th.ones(cfg.num_anchors, B, window_size, dtype=th.int) * relative_offsets_p[None, None, :]
    )
    window_pos_indices_WBp = window_relative_pos_indices_WBp + anchors_W[:, None, None]
    window_batch_indices_WBp = (
        th.ones_like(window_pos_indices_WBp) * th.arange(B, dtype=th.int)[None, :, None]
    )
    window_act_WBpD = act_BPD[window_batch_indices_WBp, window_pos_indices_WBp, :]

    autocorr_WBp = einops.einsum(anchor_act_WBD, window_act_WBpD, "W B D, W B p D -> W B p")
    autocorr_Wp = autocorr_WBp.mean(dim=1)

    return autocorr_Wp, anchors_W, relative_offsets_p


def single_autocorrelation_experiment(cfg: AutocorrelationConfig, acts_BPD: th.Tensor):
    autocorr_Wp, anchors_W, relative_offsets_p = compute_autocorrelation_heatmap(cfg, acts_BPD)
    results = dict(
        autocorr_Wp=autocorr_Wp.cpu().tolist(),
        anchors_W=anchors_W.cpu().tolist(),
        relative_offsets_p=relative_offsets_p.cpu().tolist(),
        config=asdict(cfg),
    )

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"autocorr_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"saved results to: {save_path}")

    # Cleanup
    del acts_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_gemma_act_configs(
        cfg_class=AutocorrelationConfig,
        act_paths=(
            (
                [None], 
                [
                    # "activations", 
                    # "surrogate"
                ]
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    # "codes",
                    # "recons"
                    "residuals"
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    # "novel_codes",
                    # "novel_recons",
                    # "pred_codes",
                    # "pred_recons",
                    # "total_recons",
                    "residuals"
                ]
            ),
        ),
        min_anchor=50,
        max_anchor=500,  # selected_context_length, inclusive
        num_anchors=10,
        min_offset=10,  # absolute value
        max_offset=30,  # absolute value
        # Artifacts
        env=ENV_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
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
        single_autocorrelation_experiment(cfg, acts_BPD)
        time.sleep(1)


if __name__ == "__main__":
    main()
