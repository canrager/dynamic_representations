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
    # act_BPD = act_BPD / act_BPD.norm(dim=-1, keepdim=True)
    anchors_W = th.linspace(
        cfg.min_anchor, cfg.max_anchor, cfg.num_anchors, dtype=th.int
    )  # 1-indexed
    anchor_act_WBD = act_BPD[:, anchors_W, :].transpose(0, 1)

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
    # if "surrogate" in cfg.act_path:
    #     autocorr_Wp = autocorr_WBp.mean(dim=1)
    # else:
    #     autocorr_Wp = autocorr_WBp[:, 0, :] # single sample

    # Surrogate

    return autocorr_Wp, anchors_W, relative_offsets_p


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


def compute_autocorrelation_heatmap_vectorized(cfg: AutocorrelationConfig, act_BPD: th.Tensor):
    B, P, D = act_BPD.shape
    act_BPD = act_BPD.to(cfg.env.device)
    act_BPD = act_BPD / act_BPD.norm(dim=-1, keepdim=True)

    anchors_W = th.linspace(cfg.min_anchor, cfg.max_anchor, cfg.num_anchors, dtype=th.int)

    relative_offsets_p = th.linspace(
        -cfg.max_offset, -cfg.min_offset, cfg.max_offset - cfg.min_offset + 1, dtype=th.int
    )
    p = len(relative_offsets_p)
    W = cfg.num_anchors

    # Compute similarity matrix for each batch
    self_similarity_BPP = einops.einsum(act_BPD, act_BPD, "B P1 D, B P2 D -> B P1 P2")
    surrogate_similarity_BP = diagonal_means_batch_vectorized(self_similarity_BPP).to(
        cfg.env.device
    )

    # Create meshgrid for indexing
    B_indices_WBp = einops.repeat(th.arange(B), "B -> W B p", W=W, p=p)
    P1_indices_WBp = einops.repeat(anchors_W, "W -> W B p", B=B, p=p)
    absolute_window_indices_Wp = anchors_W[:, None] + relative_offsets_p[None, :]
    P2_indices_WBp = einops.repeat(absolute_window_indices_Wp, "W p -> W B p", B=B)

    # Collect the corresponding values
    B_indices_WBp = B_indices_WBp.to(cfg.env.device)
    P1_indices_WBp = P1_indices_WBp.to(cfg.env.device)
    P2_indices_WBp = P2_indices_WBp.to(cfg.env.device)
    autocorr_WBp = self_similarity_BPP[B_indices_WBp, P1_indices_WBp, P2_indices_WBp]

    positive_offset_indices_WBp = -einops.repeat(relative_offsets_p, "p -> W B p", W=W, B=B)
    positive_offset_indices_WBp = positive_offset_indices_WBp.to(cfg.env.device)
    surrogate_WBp = surrogate_similarity_BP[B_indices_WBp, positive_offset_indices_WBp]

    # Expected value across batch
    autocorr_Wp = autocorr_WBp.mean(dim=1)
    surrogate_Wp = surrogate_WBp.mean(dim=1)

    return autocorr_Wp, anchors_W, relative_offsets_p, surrogate_Wp


def test_autocorrelation_functions_match(cfg: AutocorrelationConfig, act_BPD: th.Tensor):
    """Test whether compute_autocorrelation_heatmap and compute_autocorrelation_heatmap_vectorized
    produce the same autocorr_Wp output."""

    autocorr_Wp_1, anchors_W_1, relative_offsets_p_1 = compute_autocorrelation_heatmap(cfg, act_BPD)
    autocorr_Wp_2, anchors_W_2, relative_offsets_p_2, _ = (
        compute_autocorrelation_heatmap_vectorized(cfg, act_BPD)
    )

    # Check if outputs match
    autocorr_match = th.allclose(autocorr_Wp_1, autocorr_Wp_2, atol=1e-6)
    anchors_match = th.allclose(anchors_W_1.float(), anchors_W_2.float())
    offsets_match = th.allclose(relative_offsets_p_1.float(), relative_offsets_p_2.float())

    print(f"autocorr_Wp match: {autocorr_match}")
    print(f"anchors_W match: {anchors_match}")
    print(f"relative_offsets_p match: {offsets_match}")

    if autocorr_match and anchors_match and offsets_match:
        print("✓ Both functions produce identical results!")
        return True
    else:
        print("✗ Functions produce different results")
        print(f"Max autocorr_Wp diff: {th.max(th.abs(autocorr_Wp_1 - autocorr_Wp_2))}")
        return False


def single_autocorrelation_experiment(cfg: AutocorrelationConfig, acts_BPD: th.Tensor):
    autocorr_Wp, anchors_W, relative_offsets_p, surrogate_Wp = (
        compute_autocorrelation_heatmap_vectorized(cfg, acts_BPD)
    )
    results = dict(
        autocorr_Wp=autocorr_Wp.cpu().tolist(),
        surrogate_Wp=surrogate_Wp.cpu().tolist(),
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
            # ([None], ["activations"]),
            (
                GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
                [
                    # "codes",
                    # "recons"
                    # "residuals"
                ]
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         "codes",
            #         "recons",
            #         "residuals"
            #     ]
            # ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    # "novel_codes",
                    # "novel_recons",
                    # "pred_codes",
                    # "pred_recons",
                    # "total_recons",
                    # "residuals"
                ]
            ),
        ),
        min_anchor=49,  # should be 0-indexed
        max_anchor=499,  # selected_context_length, inclusive
        num_anchors=10,
        min_offset=10,  # absolute value
        max_offset=30,  # absolute value
        # Artifacts
        env=ENV_CFG,
        # data=SIMPLESTORIES_DS_CFG,
        # data=WEBTEXT_DS_CFG,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
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
        single_autocorrelation_experiment(cfg, acts_BPD)
        # test_autocorrelation_functions_match(cfg, acts_BPD)
        time.sleep(1)


if __name__ == "__main__":
    main()
