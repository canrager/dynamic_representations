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
class ContextExpVarConfig:
    window_size: int
    selected_context_length: int
    num_windows_across_context: int
    do_log_spacing: bool
    act_path: str
    smallest_window_start: int
    num_sequences: int

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def get_window_indices(cfg: ContextExpVarConfig):
    if cfg.window_size == "growing":
        smallest_we = cfg.smallest_window_start
    else:
        # An index window [ws, we] is spanned by window start (ws) and window end (we, inclusive) indices.
        smallest_we = cfg.window_size - 1 + cfg.smallest_window_start

    largest_we = (
        cfg.selected_context_length - 2
    )  # -1 inclusive index, -1 to leave a next token for reconstruction

    if cfg.do_log_spacing:
        # Log steps from smallest we to largest we, first and las indices of wes should be those
        log_start = th.log10(th.tensor(smallest_we, dtype=th.float))
        log_end = th.log10(th.tensor(largest_we, dtype=th.float))
        # Scale num_windows by the fraction of log-space covered
        num_windows = max(
            (cfg.num_windows_across_context * (log_end - log_start) / log_end)
            .int()
            .item(),
            2,
        )
        log_steps = th.linspace(log_start, log_end, num_windows)
        wes = th.round(10**log_steps).int()
    else:
        # Linear steps from smallest we to largest we, first and las indices of wes should be those
        num_windows = (
            cfg.num_windows_across_context / cfg.selected_context_length * smallest_we
        )
        wes = th.linspace(smallest_we, largest_we, num_windows, dtype=th.int)

    if cfg.window_size == "growing":
        batch_indices_WBP, context_indices_WBP = [], []

        for we in wes:
            context_indices_BP = th.arange(smallest_we, we + 1, dtype=th.int)
            context_indices_WBP.append(context_indices_BP)
            batch_indices_WBP.append(
                einops.repeat(
                    th.arange(cfg.num_sequences),
                    "B -> B P",
                    P=context_indices_BP.shape[0],
                )
            )

    else:
        batch_indices_WBP = einops.repeat(
            th.arange(cfg.num_sequences), "B -> W B P", W=num_windows, P=cfg.window_size
        )

        context_indices_WBP = th.zeros(
            num_windows, cfg.num_sequences, cfg.window_size, dtype=th.int, device="cpu"
        )
        for i, we in enumerate(wes):
            context_indices_WBP[i, :] = th.arange(
                we + 1 - cfg.window_size, we + 1, dtype=th.int, device="cpu"
            )

    return batch_indices_WBP, context_indices_WBP, wes, num_windows


def context_var_explained(context_act_BpD, y_act_BD, cfg: ContextExpVarConfig):
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

    frac_var_exp = (
        th.var(reconstructed_y_BD, dim=0).sum() / th.var(y_act_BD, dim=0).sum()
    )

    mean_rank = mean_rank_B.mean()

    return frac_var_exp.cpu(), mean_rank.cpu()


def random_directions_baseline(
    context_act_BpD, y_act_BD, cfg: ContextExpVarConfig, seed=42
):
    """
    Baseline: Project target token onto random orthogonal directions matching the rank of the context.
    This provides a comparison to see how much variance can be explained by random directions
    vs. the actual context directions.
    """
    # Set random seed for reproducibility
    th.manual_seed(seed)

    # Compute Context Basis
    context_act_BpD = context_act_BpD.to(cfg.env.device)
    B, p, D = context_act_BpD.shape
    y_act_BD = y_act_BD.to(cfg.env.device)

    reconstructed_y_BD = th.zeros_like(y_act_BD)

    # Subtract mean (same as main function)
    y_mean_D = y_act_BD.mean(dim=0)
    y_act_BD = y_act_BD - y_mean_D
    context_act_BpD = context_act_BpD - y_mean_D

    for b in range(B):
        y_D = y_act_BD[b]
        context_pD = context_act_BpD[b]

        # Compute rank of context (same as in context_var_explained)
        U, S, Vt_pD = th.linalg.svd(context_pD.float(), full_matrices=False)
        rank = int((S.abs() > 1e-8).float().sum().item())

        # Generate random orthogonal directions matching the rank
        # Create random matrix and use QR decomposition to get orthogonal basis
        random_matrix_Dr = th.randn(D, rank, dtype=th.float32, device=cfg.env.device)
        Q_Dr, R_rr = th.linalg.qr(random_matrix_Dr)
        # Q_Dr is now an orthogonal basis of rank dimensions

        # Project target token onto random directions
        reconstructed_y_BD[b] = Q_Dr @ (Q_Dr.T @ y_D.to(th.float32))

    frac_var_exp = (
        th.var(reconstructed_y_BD, dim=0).sum() / th.var(y_act_BD, dim=0).sum()
    )

    return frac_var_exp.cpu()


def first_token_baseline(context_act_BpD, y_act_BD, cfg: ContextExpVarConfig):
    """
    Baseline: Project target token onto the first token's activation direction.
    This tests how much variance can be explained by just the first token in the context.
    """
    # Compute Context Basis
    context_act_BpD = context_act_BpD.to(cfg.env.device)
    B, p, D = context_act_BpD.shape
    y_act_BD = y_act_BD.to(cfg.env.device)

    reconstructed_y_BD = th.zeros_like(y_act_BD)

    # Subtract mean (same as main function)
    y_mean_D = y_act_BD.mean(dim=0)
    y_act_BD = y_act_BD - y_mean_D
    context_act_BpD = context_act_BpD - y_mean_D

    for b in range(B):
        y_D = y_act_BD[b]
        context_pD = context_act_BpD[b]

        # Extract first token activation (index 0)
        first_token_D = context_pD[0]

        # Normalize the first token direction
        first_token_norm = first_token_D.norm()
        if first_token_norm > 1e-8:
            first_token_normalized_D = first_token_D / first_token_norm
        else:
            first_token_normalized_D = first_token_D

        # Project target token onto first token direction
        # Projection: (y · first_token) * first_token
        projection_coeff = th.dot(y_D, first_token_normalized_D)
        reconstructed_y_BD[b] = projection_coeff * first_token_normalized_D

    frac_var_exp = (
        th.var(reconstructed_y_BD, dim=0).sum() / th.var(y_act_BD, dim=0).sum()
    )

    return frac_var_exp.cpu()


def single_pca_experiment(cfg: ContextExpVarConfig, acts_BPD: th.Tensor):

    dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
    batch_indices_WBP, context_indices_WBP, wes, num_windows = get_window_indices(cfg)
    fve = th.zeros(num_windows, dtype=dtype)
    ncomp = th.zeros(num_windows, dtype=dtype)

    # Initialize baseline arrays only for growing window size
    compute_baselines = cfg.window_size == "growing"
    if compute_baselines:
        fve_random = th.zeros(num_windows, dtype=dtype)
        fve_first_token = th.zeros(num_windows, dtype=dtype)

    for window_idx in trange(num_windows):
        batch_indices_BP = batch_indices_WBP[window_idx]
        context_indices_BP = context_indices_WBP[window_idx]
        train_act_BpD = acts_BPD[batch_indices_BP, context_indices_BP]

        arange_B = th.arange(cfg.num_sequences, dtype=th.int)
        eval_index_B = th.ones(cfg.num_sequences, dtype=th.int) * (
            wes[window_idx] + 1
        )  # +1 since were evaluating the next
        eval_act_BD = acts_BPD[arange_B, eval_index_B]

        frac_var_exp, num_pca_components = context_var_explained(
            train_act_BpD, eval_act_BD, cfg
        )
        fve[window_idx] = frac_var_exp
        ncomp[window_idx] = num_pca_components

        # Compute baselines only for growing window size
        if compute_baselines:
            frac_var_exp_random = random_directions_baseline(
                train_act_BpD, eval_act_BD, cfg
            )
            frac_var_exp_first_token = first_token_baseline(
                train_act_BpD, eval_act_BD, cfg
            )
            fve_random[window_idx] = frac_var_exp_random
            fve_first_token[window_idx] = frac_var_exp_first_token

    results = dict(
        window_end_pos=wes.cpu().tolist(),
        frac_var_exp=fve.cpu().tolist(),
        num_pca_components=ncomp.cpu().tolist(),
        num_seq_per_token=cfg.num_sequences,
    )

    # Add baseline results if computed
    if compute_baselines:
        results["frac_var_exp_random"] = fve_random.cpu().tolist()
        results["frac_var_exp_first_token"] = fve_first_token.cpu().tolist()

    # Save results
    results["config"] = asdict(cfg)
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        cfg.env.results_dir, f"context_exp_var_{datetetime_str}.json"
    )
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"saved results to: {save_path}")

    # Cleanup
    del acts_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_gemma_act_configs(
        cfg_class=ContextExpVarConfig,
        # window_size=[1, 3, 8, 10, 20, 32, 50, 100, 120, 316, "growing"],
        # window_size=[10, 23, 58, 161, "growing"],
        # window_size=[1, 9, 40, 161, "growing"],
        # window_size=[9],
        # window_size=[9, 40],
        # window_size=[1, 3, 10, 32, "growing"],
        window_size=["growing"],
        # window_size=[490],
        selected_context_length=500,
        num_windows_across_context=100,
        do_log_spacing=True,
        smallest_window_start=10,
        num_sequences=1000,
        # Artifacts
        env=ENV_CFG,
        # data=DatasetConfig(
        #     name="Webtext",
        #     hf_name="monology/pile-uncopyrighted",
        #     num_sequences=50000,
        #     context_length=100,
        # ),
        data=DatasetConfig(
            name="Webtext",
            hf_name="monology/pile-uncopyrighted",
            num_sequences=1000,
            context_length=500,
        ),
        # data=[CODE_DS_CFG, SIMPLESTORIES_DS_CFG],
        # num_sequences=[2, 10, 100, 1000, 1000, 50000],
        # num_sequences=[2, 10, 100, 1000],
        llm=LLAMA3_L15_LLM_CFG,
        # llm=GEMMA2_LLM_CFG,
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
            verbose=False,
        )
        for key in artifacts:
            acts_BPD = artifacts[key]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_pca_experiment(key_config, acts_BPD)
            time.sleep(1)


if __name__ == "__main__":
    main()
