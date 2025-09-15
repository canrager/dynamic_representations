"""
Script for computing a stationary surrogate from cached LLM activations, and saving as artifacts for further analysis.
"""

from src.project_config import BaseConfig
import os
import torch as th
import json
import numpy as np

from src.preprocessing_utils import load_llm_artifacts, run_parameter_sweep


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

def load_cfg_from_dir(dir):
    cfg_path = os.path.join(dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = BaseConfig(**cfg_dict)
    return cfg


def cache_surrogate_single_experiment(cfg):
    # Load cached activations and get the artifact directory
    artifacts_dict, artifact_dir = load_llm_artifacts(cfg, ['activations'])
    acts_BPD = artifacts_dict['activations']
    
    # Generate surrogate data
    surr_BPD = phase_randomized_surrogate(acts_BPD)

    # Save surrogate to the same folder where activations are stored
    with open(os.path.join(artifact_dir, "surrogate.pt"), "wb") as f:
        th.save(surr_BPD, f, pickle_protocol=5)
    
    print(f"Cached surrogate to: {artifact_dir}")


def main():
    base_cfg = BaseConfig(
        experiment_name="cache_llm_activations",  # Match the activation cache naming
        llm_name="meta-llama/Llama-3.1-8B",
        revision=None,
        layer_idx=16,
        llm_batch_size=100,
        llm_hidden_dim=4096,
        dtype="bfloat16",
        dataset_name="monology/pile-uncopyrighted",
        hf_text_identifier="text",
        num_sequences=1_000,
        context_length=500,
        save_artifacts=False,
        overwrite_existing=True,
        verbose=False,
    )
    
    sweep_params = {
        'llm_name': ['meta-llama/Llama-3.1-8B'],
        'layer_idx': [31],
        'dataset_name': ['monology/pile-uncopyrighted']
    }
    
    run_parameter_sweep(base_cfg, sweep_params, cache_surrogate_single_experiment)


if __name__ == "__main__":
    # takes ~10mins for 1k sequences, 500tokens on CPU
    main()
