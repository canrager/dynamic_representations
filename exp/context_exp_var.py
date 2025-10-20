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

    B = min(cfg.data.num_sequences, 500) #// window_size
    batch_indices_WBP = einops.repeat(th.arange(B), "B -> W B P", W=cfg.num_windows, P=window_size)

    pos_indices_WBP = th.zeros(cfg.num_windows, B, window_size, dtype=th.int, device="cpu")
    for i, we in enumerate(wes):
        pos_indices_WBP[i, :] = th.arange(we + 1 - window_size, we + 1, dtype=th.int, device="cpu")

    return batch_indices_WBP, pos_indices_WBP, wes, B


def context_var_explained(context_act_BpD, y_act_BD, cfg: PCAExpVarConfig):
    """
    Question: How much variance can be explained by just projecting a token onto it's context
    Uncertainty: Should we center at all? By y_act_BD, since we're computing the the variance over that?
    """

    # Compute Contex Basis
    context_act_BpD = context_act_BpD.to(cfg.env.device)
    B, p, D = context_act_BpD.shape
    y_act_BD = y_act_BD.to(cfg.env.device)

    reconstructed_y_BD = th.zeros_like(y_act_BD)
    mean_rank_B = th.zeros(B)

    # Subtract mean
    y_mean_D = y_act_BD.mean(dim=0)
    y_act_BD = y_act_BD - y_mean_D
    context_act_BpD = context_act_BpD - y_mean_D

    for b in range(B):
        y_D = y_act_BD[b]
        context_pD = context_act_BpD[b]
        U, S, Vt_pD = th.linalg.svd(context_pD.float(), full_matrices=False)
        Vt_pD = Vt_pD.to(dtype=th.bfloat16)

        rank = int((S.abs() > 1e-8).float().sum().item())
        mean_rank_B[b] = rank

        V_Dp = Vt_pD[:rank].T
        reconstructed_y_BD[b] = V_Dp @ (V_Dp.T @ y_D)

    frac_var_exp = th.var(reconstructed_y_BD, dim=0).sum() / th.var(y_act_BD, dim=0).sum()

    mean_rank = mean_rank_B.mean()

    return frac_var_exp.cpu(), mean_rank.cpu()


def single_pca_experiment(cfg: PCAExpVarConfig, acts_BPD: th.Tensor):
    assert cfg.selected_context_length > max(cfg.window_sizes)

    results = {}
    for window_size in tqdm(cfg.window_sizes, desc="Window size"):
        dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
        fve = th.zeros(cfg.num_windows, dtype=dtype)
        ncomp = th.zeros(cfg.num_windows, dtype=dtype)
        batch_indices_WBP, pos_indices_WBP, wes, B = get_window_indices(cfg, window_size)

        for window_idx in trange(cfg.num_windows):
            batch_indices_BP = batch_indices_WBP[window_idx]
            pos_indices_BP = pos_indices_WBP[window_idx]
            train_act_BpD = acts_BPD[batch_indices_BP, pos_indices_BP]
            eval_act_BD = acts_BPD[
                :, wes[window_idx] + 1
            ]  # +1 since we're evaluating the next token

            frac_var_exp, num_pca_components = context_var_explained(
                train_act_BpD, eval_act_BD, cfg
            )
            fve[window_idx] = frac_var_exp
            ncomp[window_idx] = num_pca_components

        results[window_size] = dict(
            window_end_pos=wes.cpu().tolist(),
            frac_var_exp=fve.cpu().tolist(),
            num_pca_components=ncomp.cpu().tolist(),
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
        window_sizes=[[100]],
        # window_sizes=[[1, 3, 10, 32, 100, 316]],
        selected_context_length=500,
        # num_windows=50,
        num_windows=5,
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
        # llm=LLAMA3_LLM_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=None,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    # "surrogate"
                ],
            ),
            # (
            #     [BATCHTOPK_SELFTRAIN_SAE_CFG],
            #     [
            #         "codes",
            #         "recons",
            #         "residuals"
            #     ]
            # ),
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
            cfg,
            [act_path],
            cfg.env.activations_dir,
            compared_attributes=["llm", "data"],
            verbose=True,
        )
        for key in artifacts:
            acts_BPD = artifacts[key]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_pca_experiment(key_config, acts_BPD)
            time.sleep(1)


if __name__ == "__main__":
    main()
