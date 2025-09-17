"""
Script for caching:

1. LLM activations
2. Stationary surrogate activations
3. SAE activations and reconstruction

and saving as artifacts for further analysis.
"""

from dataclasses import asdict, dataclass
import os
import torch as th
import gc
import json
import numpy as np
from datetime import datetime

from src.model_utils import load_nnsight_model, load_sae
from src.exp_utils import compute_llm_artifacts
from src.cache_utils import batch_sae_cache

from src.configs import (
    DatasetConfig,
    LLMConfig,
    SAEConfig,
    EnvironmentConfig,
    LLAMA3_LLM_CFG,
    LLAMA3_SAE_CFGS,
    GEMMA2_LLM_CFG,
    GEMMA2_SAE_CFGS,
    ENV_CFG,
    WEBTEXT_DS_CFG,
    SIMPLESTORIES_DS_CFG,
    CODE_DS_CFG,
    get_configs,
    load_llm_activations,
)


@dataclass
class CacheConfig:
    data: DatasetConfig
    llm: LLMConfig
    sae: (
        SAEConfig | None
    )  # If None is passed, Cache the LLM and Surrogate. SAE cache requires existing LLM cache.
    env: EnvironmentConfig


def cache_llm_activations(cfg):
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(cfg.env.activations_dir, folder_name)
    os.makedirs(folder_dir)

    model, submodule, _ = load_nnsight_model(cfg)

    acts_LBPD, masks_LBPD, tokens_BP = compute_llm_artifacts(cfg, model, submodule)

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

    del acts_LBPD, masks_LBPD, tokens_BP, model, submodule
    th.cuda.empty_cache()
    gc.collect()


def compute_phase_randomized_surrogate(X_BPD: th.Tensor) -> th.Tensor:
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
    X_np = X_BPD.detach().float().cpu().numpy()

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

def cache_surrogate_activations(cfg: CacheConfig):
    act_BPD, save_dir = load_llm_activations(cfg, return_target_dir=True)
    surrogate_BPD = compute_phase_randomized_surrogate(act_BPD)

    with open(os.path.join(save_dir, "surrogate.pt"), "wb") as f:
        th.save(surrogate_BPD, f, pickle_protocol=5)

    print(f"Cached surrogate to: {save_dir}")

def cache_sae_activations(cfg: CacheConfig):
    llm_act_BPD, save_dir = load_llm_activations(cfg, return_target_dir=True)
    save_dir = os.path.join(save_dir, cfg.sae.dict_class)

    # Check whether precomputed acts for this sae already exists
    if os.path.isdir(save_dir):
        print(f'Cached activations already exist in {save_dir}. Skipping activation cache.')
        return
    else:
        os.makedirs(save_dir)

    sae = load_sae(cfg)
    sae_recon_BPD, sae_act_BPS, _ = batch_sae_cache(sae, llm_act_BPD, cfg)

    sae_act_BPS = sae_act_BPS.cpu()
    sae_recon_BPD = sae_recon_BPD.cpu()

    with open(os.path.join(save_dir, "latents.py"), "wb") as f:
        th.save(sae_act_BPS, f)
    with open(os.path.join(save_dir, "reconstructions.py"), "wb") as f:
        th.save(sae_recon_BPD, f)

    del sae, sae_act_BPS, sae_recon_BPD
    th.cuda.empty_cache()
    gc.collect()


def main():
    cache_configs = get_configs(
        CacheConfig,
        data=DatasetConfig(
            name="Webtext",
            hf_name="monology/pile-uncopyrighted",
            num_sequences=10,
            context_length=500,
        ),
        llm=LLMConfig(
            name="Gemma-2-2B",
            hf_name="google/gemma-2-2b",
            revision=None,
            layer_idx=12,
            hidden_dim=2304,
            batch_size=100,
        ),
        sae=GEMMA2_SAE_CFGS,
        env=ENV_CFG,
    )

    for cfg in cache_configs:
        if cfg.sae is None:
            pass
            # Cache LLM activations and compute surrogate
            cache_llm_activations(cfg)
            cache_surrogate_activations(cfg)
        else:
            cache_sae_activations(cfg)


if __name__ == "__main__":
    main()
