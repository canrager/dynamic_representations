"""
Script for caching: 

1. LLM activations
2. Stationary surrogate activations
3. SAE activations and reconstruction

and saving as artifacts for further analysis.
"""

from dataclasses import asdict
from src.exp_utils import compute_llm_artifacts
import os
import torch as th
import json
import numpy as np
from datetime import datetime
from src.model_utils import load_nnsight_model

from src.configs import *


def cache_llm_activations(cfg):
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(cfg.env.activations_dir, folder_name)
    os.makedirs(folder_dir)

    model, submodule, _ = load_nnsight_model(cfg)

    acts_LBPD, masks_LBPD, tokens_BP, _ = compute_llm_artifacts(cfg, model, submodule)

    # Move tensors to CPU before saving to free GPU memory
    acts_LBPD = acts_LBPD.cpu()
    masks_LBPD = masks_LBPD.cpu()
    tokens_BP = tokens_BP.cpu()

    # Save
    with open(os.path.join(folder_dir, f"activations.pt"), "wb") as f:
        th.save(acts_LBPD, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, f"masks.pt"), "wb") as f:
        th.save(masks_LBPD, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, f"tokens.pt"), "wb") as f:
        th.save(tokens_BP, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, f"config.json"), "w") as f:
        json.dump(asdict(cfg), f)

    print(f"Cached activations to: {folder_dir}")

    # Explicit cleanup to prevent OOM between sweep iterations
    del acts_LBPD, masks_LBPD, tokens_BP, model, submodule
    th.cuda.empty_cache()
    import gc

    gc.collect()


def phase_randomized_surrogate(X_BPD: th.Tensor) -> th.Tensor:
    """
    Phase-randomized surrogate per (B, D) series along time P.
    Preserves power spectrum per dim, randomizes phases -> stationary.

    Args:
        X_BPD: Tensor of shape (B, P, D)

    Returns:
        Phase-randomized surrogate with same shape
    """
    B, P, D = X_BPD.shape
    X_sur = th.empty_like(X_BPD)
    X_np = X_BPD.detach().cpu().numpy()

    for b in range(B):
        for d in range(D):
            x = X_np[b, :, d]
            fft_x = np.fft.rfft(x)
            mag = np.abs(fft_x)
            # random phases in [0, 2π), keep DC/Nyquist magnitudes
            rand_phase = np.exp(1j * np.random.uniform(0.0, 2 * np.pi, size=fft_x.shape))
            # ensure DC (0-freq) has zero phase
            rand_phase[0] = 1.0 + 0.0j
            fft_new = mag * rand_phase
            x_new = np.fft.irfft(fft_new, n=P)
            X_sur[b, :, d] = th.from_numpy(x_new).to(X_BPD)
    return X_sur


@dataclass
class CacheConfig:
    data: DatasetConfig
    llm: LLMConfig
    sae: (
        SAEConfig | None
    )  # If None is passed, Cache the LLM and Surrogate. SAE cache requires existing LLM cache.
    env: EnvironmentConfig


def main():
    cache_configs = get_configs(
        CacheConfig,
        data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        llm=LLAMA3_LLM_CFG,
        sae=[None] + LLAMA3_SAE_CFGS,
        env=ENV_CFG,
    )

    cache_llm_activations(cache_configs[0])


if __name__ == "__main__":
    main()
