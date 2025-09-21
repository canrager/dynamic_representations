from dataclasses import dataclass, asdict
from typing import List
from datetime import datetime
import json
import gc
from tqdm import tqdm
from copy import deepcopy
import time

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
    act_path: str

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

    B = cfg.min_total_tokens_per_window // window_size + 1  # num sequences per token

    pos_indices_WBP = th.zeros(cfg.num_windows, B, window_size, dtype=th.int, device="cpu")
    for i, we in enumerate(wes):
        pos_indices_WBP[i, :] = th.arange(we + 1 - window_size, we + 1, dtype=th.int, device="cpu")

    batch_indices_WBP = th.ones_like(pos_indices_WBP) * th.arange(B)[None, :, None]

    return batch_indices_WBP, pos_indices_WBP, wes, B


def pca_var_explained(train_act_BPD, eval_act_BD, cfg: PCAExpVarConfig):

    # Compute PCA Basis
    act_ND = train_act_BPD.flatten(0, 1)
    act_ND = act_ND.to(cfg.env.device)
    eval_act_BD = eval_act_BD.to(cfg.env.device)
    centered_act_ND = act_ND - th.mean(act_ND, dim=0)
    U, S, Vt = th.linalg.svd(centered_act_ND.float(), full_matrices=False)

    # Only keep components explaining <= cfg.reconstruction_thresh energy
    S2 = S**2
    S_normalized = S2 / S2.sum()
    components_mask = th.cumsum(S_normalized, dim=0) < cfg.reconstruction_thresh
    num_pca_components = components_mask.sum()
    Vt_CD = Vt[components_mask].to(dtype=act_ND.dtype)

    # Report the frac_variance_explained
    total_eval_var_D = th.var(eval_act_BD, dim=0)
    pca_eval_act_BC = th.matmul(Vt_CD, eval_act_BD.T).T
    pca_eval_var_D = th.var(pca_eval_act_BC, dim=0)
    frac_var_exp = pca_eval_var_D.sum() / total_eval_var_D.sum()

    return frac_var_exp.cpu(), num_pca_components.cpu()


def single_pca_experiment(cfg: PCAExpVarConfig, acts_BPD: th.Tensor):
    assert cfg.selected_context_length > max(cfg.window_sizes)
    assert (
        cfg.min_total_tokens_per_window >= cfg.llm.hidden_dim
    ), "PCA requires at least hidden_dim datapoints."

    results = {}
    for window_size in tqdm(cfg.window_sizes, desc="Window size"):
        batch_indices_WBP, pos_indices_WBP, wes, B = get_window_indices(cfg, window_size)
        train_act_WbpD = acts_BPD[batch_indices_WBP, pos_indices_WBP, :]
        eval_act_WB = acts_BPD[:, wes + 1, :]  # Eval next token after window interval

        dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
        fve = th.zeros(cfg.num_windows, dtype=dtype)
        ncomp = th.zeros(cfg.num_windows, dtype=dtype)

        for window_idx in range(cfg.num_windows):
            train_act_bpD = train_act_WbpD[window_idx]
            eval_act_B = eval_act_WB[window_idx]
            frac_var_exp, num_pca_components = pca_var_explained(train_act_bpD, eval_act_B, cfg)
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
    configs = get_configs(
        cfg_class=PCAExpVarConfig,
        window_sizes=(
            20,
            34,
            58,
            99,
            169,
            288,
            490,
        ),  # Choosing tuple so it will not be resulting in separate configs
        selected_context_length=500,
        num_windows=5,
        do_log_spacing=False,
        omit_bos_token=True,
        reconstruction_thresh=0.9,
        min_total_tokens_per_window=19999,
        act_path=None,
        # Artifacts
        env=ENV_CFG,
        data=WEBTEXT_DS_CFG,
        llm=GEMMA2_LLM_CFG,
        sae=GEMMA2_SAE_CFGS,
    )

    for cfg in configs:

        if cfg.sae is None:
            # Run on LLM activations
            keys = ["activations"]
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

        print(cfg.env.activations_dir)

        artifacts, _ = load_matching_activations(
            cfg, keys, cfg.env.activations_dir, compared_attributes=["llm", "data"]
        )
        for key in artifacts:
            acts_BPD = artifacts[key]
            key_config = deepcopy(cfg)
            key_config.act_path = key
            single_pca_experiment(key_config, acts_BPD)
            time.sleep(1)


if __name__ == "__main__":
    main()
