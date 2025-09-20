from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import json
import gc

import torch as th

from src.configs import *


@dataclass
class PCAExpVarConfig:
    window_sizes: List[str]
    selected_context_length: int
    num_windows: int
    do_log_spacing: bool
    omit_bos_token: bool
    reconstruction_thresh: float
    min_total_tokens_per_window: int

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig


def get_window_indices(cfg: PCAExpVarConfig, window_size=int):
    # An index window [ws, we] is spanned by window start (ws) and window end (we, inclusive) indices.
    if cfg.omit_bos_token:
        smallest_we = window_size
    else:
        smallest_we = window_size - 1

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

    B = cfg.min_total_tokens_per_window // window_size + 1 # num sequences per token

    pos_indices_WBP = th.zeros(cfg.num_windows, B, window_size, dtype=th.int, device="cpu")
    for i, we in enumerate(wes):
        pos_indices_WBP[i, :] = th.arange(we+1 - window_size, we+1, dtype=th.int, device="cpu")

    batch_indices_WBP = th.ones_like(pos_indices_WBP) * th.arange(B)[None, : , None]

    return batch_indices_WBP, pos_indices_WBP, wes

def pca_var_explained(train_act_BPD, eval_act_BD, cfg: PCAExpVarConfig):
    
    # Compute PCA Basis
    act_ND = train_act_BPD.flatten(0, 1)
    centered_act_ND = act_ND - th.mean(act_ND, dim=0)
    U, S, Vt = th.linalg.svd(centered_act_ND, full_matrices=False)
    print(f"==>> U.shape: {U.shape}")
    print(f"==>> S.shape: {S.shape}")
    print(f"==>> Vt.shape: {Vt.shape}")

    # Only keep components explaining <= cfg.reconstruction_thresh energy
    S_normalized = S / S.sum()
    components_mask = th.cumsum(S_normalized) < cfg.reconstruction_thresh
    num_pca_components = components_mask.sum()
    Vt_CD = Vt[components_mask]

    # Report the frac_variance_explained
    total_eval_var_D = th.var(eval_act_BD, dim=0)
    pca_eval_act_BC = th.matmul(Vt_CD @ eval_act_BD.T).T
    pca_eval_var_D = th.var(pca_eval_act_BC, dim=0)
    frac_var_exp = pca_eval_var_D.sum() / total_eval_var_D.sum()

    return frac_var_exp, num_pca_components



def single_pca_experiment(cfg: PCAExpVarConfig):
    # Load activations from artifacts
    if cfg.sae is None:
        acts_BPD = load_llm_activations(cfg)
        artifacts = None
    else:
        key = f"{cfg.sae.name}/activations"
        artifacts, _ = load_matching_artifacts(cfg, key)
        acts_BPD = artifacts[key]

    results = {}
    for window_size in cfg.window_sizes:
        batch_indices_WBP, pos_indices_WBP, wes = get_window_indices(cfg, window_size)
        train_act_WbpD = acts_BPD[batch_indices_WBP, pos_indices_WBP, :]
        eval_act_WB = acts_BPD[:, wes + 1, :] # Eval next token after window interval

        fve = th.zeros(cfg.num_windows, dtype=cfg.env.dtype)
        ncomp = th.zeros(cfg.num_windows, dtype=cfg.env.dtype)

        for window_idx in range(cfg.num_windows):
            train_act_bpD = train_act_WbpD[window_idx]
            eval_act_B =  eval_act_WB[window_idx]
            frac_var_exp, num_pca_components = pca_var_explained(train_act_bpD, eval_act_B, cfg)
            fve[window_idx] = frac_var_exp
            ncomp[window_idx] = num_pca_components

        results[window_size] = dict(
            window_end_pos=wes.cpu().tolist(),
            frac_var_exp=fve.cpu().tolist(),
            num_pca_components=ncomp.cpu().tolist()
        )

    # Save results
    results["config"] = asdict(cfg)
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"pca_exp_var_{datetetime_str}")
    with open(save_path, "w") as f:
        json.dump(results, f)

    
    # Cleanup
    del acts_BPD, artifacts
    th.cuda.empty_cache()
    gc.collect()


def main():
    configs = get_configs(
        cfg_class=PCAExpVarConfig,
        window_sizes=(490),  # Choosing tuple so it will not be resulting in separate configs
        selected_context_length=50,
        num_windows=3,
        do_log_spacing=False,
        omit_bos_token=True,
        reconstruction_thresh=0.9,
        min_total_tokens_per_window=5000,
        # Artifacts
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=[None],
    )
    for cfg in configs:
        reuslts = single_pca_experiment(cfg)


if __name__ == "__main__":
    main()
