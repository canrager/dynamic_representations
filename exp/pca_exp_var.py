from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import json
import gc
from tqdm import tqdm, trange
from copy import deepcopy
import time

import torch as th
import einops

from src.configs import *


@dataclass
class PCAExpVarConfig:
    window_sizes: List[str]
    selected_context_length: int
    num_windows: int
    do_log_spacing: bool
    reconstruction_thresh: float
    min_total_tokens_per_window: int
    act_path: str
    smallest_window_start: int

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def get_window_indices(cfg: PCAExpVarConfig, window_size=int):
    # An index window [ws, we] is spanned by window start (ws) and window end (we, inclusive) indices.
    smallest_we = window_size - 1 + cfg.smallest_window_start

    largest_we = (
        cfg.selected_context_length - 2
    )  # -1 inclusive index, -1 to leave a next token for reconstruction

    if cfg.do_log_spacing:
        # Log steps from smallest we to largest we, first and las indices of wes should be those
        log_start = th.log10(th.tensor(smallest_we, dtype=th.float))
        log_end = th.log10(th.tensor(largest_we, dtype=th.float))
        log_steps = th.linspace(log_start, log_end, cfg.num_windows)
        wes = th.round(10**log_steps).int()
    else:
        # Linear steps from smallest we to largest we, first and las indices of wes should be those
        wes = th.linspace(smallest_we, largest_we, cfg.num_windows, dtype=th.int)

    B = cfg.min_total_tokens_per_window // window_size + 1  # num sequences per token

    pos_indices_WBP = th.zeros(cfg.num_windows, B, window_size, dtype=th.int, device="cpu")
    for i, we in enumerate(wes):
        pos_indices_WBP[i, :] = th.arange(we + 1 - window_size, we + 1, dtype=th.int, device="cpu")

    batch_indices_WBP = th.ones_like(pos_indices_WBP) * th.arange(B)[None, :, None]

    return batch_indices_WBP, pos_indices_WBP, wes, B


def pca_var_explained(train_act_BPD, next_token_act_BD, cfg: PCAExpVarConfig):

    # Compute PCA Basis
    act_ND = train_act_BPD.flatten(0, 1)
    act_ND = act_ND.to(cfg.env.device)
    mean_act_D =  th.mean(act_ND, dim=0)
    centered_act_ND = act_ND - mean_act_D
    next_token_act_BD = next_token_act_BD.to(cfg.env.device)
    U, S, Vt = th.linalg.svd(centered_act_ND.float(), full_matrices=False)

    # Only keep components explaining <= cfg.reconstruction_thresh energy
    S2 = S**2
    S_normalized = S2 / S2.sum()
    components_mask = th.cumsum(S_normalized, dim=0) < cfg.reconstruction_thresh
    num_pca_components = components_mask.sum()
    Vt_CD = Vt[components_mask].to(dtype=act_ND.dtype)

    # Report the frac_variance_explained
    total_eval_var_D = th.var(next_token_act_BD, dim=0)
    pca_eval_act_BC = th.matmul(Vt_CD, next_token_act_BD.T).T
    pca_eval_var_D = th.var(pca_eval_act_BC, dim=0)
    frac_var_exp = pca_eval_var_D.sum() / total_eval_var_D.sum()

    # Baseline: Report frac variance explained by projecting next_token_act on to the mean of the prior window.
    # baseline_act_BD = ((next_token_act_BD / next_token_act_BD.norm(dim=-1, keepdim=True)) @ (mean_act_D / mean_act_D.norm(dim=-1, keepdim=True)))[:, None] * next_token_act_BD
    # baseline_var_D = th.var(baseline_act_BD, dim=0)
    # baseline_frac_var_exp = baseline_var_D.sum() / total_eval_var_D.sum()

    return frac_var_exp.cpu(), num_pca_components.cpu() #, baseline_frac_var_exp.cpu()


def context_var_explained(context_act_BpD, y_act_BD, cfg: PCAExpVarConfig):

    # Compute Contex Basis
    context_act_BpD = context_act_BpD.to(cfg.env.device)
    y_act_BD = y_act_BD.to(cfg.env.device)

    context_y_Bp = th.zeros()

    total_y_var_D = th.var(y_act_BD, dim=0)

    for b in range(B):
        y_D = y_act_BD[b]
        context_pD = context_act_BpD[b]
        U, S, Vt_pD = th.linalg.svd(context_pD.float(), full_matrices=False)

        # Report the frac_variance_explained
        context_y_p = Vt_pD @ y_D


    return frac_var_exp.cpu()

def prev_token_baseline(cur_token_act_BD, next_token_act_BD):
    # Baseline of projecting the next token onto the window of the previous token.
    normalized_cur_BD = cur_token_act_BD / cur_token_act_BD.norm(dim=-1, keepdim=True)
    normalized_next_BD = next_token_act_BD / next_token_act_BD.norm(dim=-1, keepdim=True)
    cosine_sim_B = einops.einsum(normalized_next_BD, normalized_cur_BD, "B D, B D -> B")
    baseline_act_BD = cosine_sim_B[:, None] * next_token_act_BD
    baseline_var_D = th.var(baseline_act_BD, dim=0)
    total_next_token_var_D = th.var(next_token_act_BD, dim=0)
    baseline_frac_var_exp = baseline_var_D.sum() / total_next_token_var_D.sum()
    return baseline_frac_var_exp


def single_pca_experiment(cfg: PCAExpVarConfig, acts_BPD: th.Tensor):
    assert cfg.selected_context_length > max(cfg.window_sizes)
    assert (
        cfg.min_total_tokens_per_window >= cfg.llm.hidden_dim
    ), "PCA requires at least hidden_dim datapoints."

    results = {}
    for window_size in tqdm(cfg.window_sizes, desc="Window size"):
        batch_indices_WBP, pos_indices_WBP, wes, B = get_window_indices(cfg, window_size)
        train_act_WbpD = acts_BPD[batch_indices_WBP, pos_indices_WBP, :]
        cur_act_BWD = acts_BPD[:, wes, :]
        eval_act_BWD = acts_BPD[:, wes + 1, :]  # Eval next token after window interval

        dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
        fve = th.zeros(cfg.num_windows, dtype=dtype)
        ncomp = th.zeros(cfg.num_windows, dtype=dtype)
        baseline = th.zeros(cfg.num_windows, dtype=dtype)

        for window_idx in trange(cfg.num_windows):
            train_act_bpD = train_act_WbpD[window_idx]
            eval_act_BD = eval_act_BWD[:, window_idx]
            cur_act_BD = cur_act_BWD[:, window_idx]
            frac_var_exp, num_pca_components = pca_var_explained(train_act_bpD, eval_act_BD, cfg)
            fve[window_idx] = frac_var_exp
            ncomp[window_idx] = num_pca_components
            baseline[window_idx] = prev_token_baseline(cur_act_BD, eval_act_BD)

        results[window_size] = dict(
            window_end_pos=wes.cpu().tolist(),
            frac_var_exp=fve.cpu().tolist(),
            num_pca_components=ncomp.cpu().tolist(),
            baseline_frac_var_exp=baseline.cpu().tolist(),
            num_seq_per_token=B,
        )

    # Save results
    results["config"] = asdict(cfg)
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"pca_exp_var_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"saved results to: {save_path}")

    # Cleanup
    del acts_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_gemma_act_configs(
        cfg_class=PCAExpVarConfig,
        window_sizes=[
            [
                10,
            ]
        ],
        min_total_tokens_per_window=9999,
        selected_context_length=500,
        num_windows=3,
        do_log_spacing=False,
        smallest_window_start=2,
        reconstruction_thresh=0.9,
        # Artifacts
        env=ENV_CFG,
        data=DatasetConfig(
            name="Webtext",
            hf_name="monology/pile-uncopyrighted",
            num_sequences=1000,
            context_length=500,
        ),
        llm=GEMMA2_LLM_CFG,
        sae=None,
        act_paths=(
            # (
            #     [None],
            #     [
            #         "activations",
            #         # "surrogate"
            #     ],
            # ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    "codes",
                    "recons",
                    "residuals"
                ]
            ),
            # (
            #     [TEMPORAL_SELFTRAIN_SAE_CFG],
            #     [
            #         "novel_codes",
            #         "novel_recons",
            #         "pred_codes",
            #         "pred_recons",
            #         "total_recons",
            #         "residuals"
            #     ]
            # ),
        ),
    )

    for cfg in configs:
        act_path = cfg.act_path
        if cfg.sae is not None:
            act_path = os.path.join(cfg.sae.name, cfg.act_path)

        artifacts, _ = load_matching_activations(
            cfg, [act_path], cfg.env.activations_dir, compared_attributes=["llm", "data"]
        )
        for key in artifacts:
            acts_BPD = artifacts[key]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_pca_experiment(key_config, acts_BPD)
            time.sleep(1)


if __name__ == "__main__":
    main()
