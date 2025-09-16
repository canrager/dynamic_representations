"""
Script for computing a stationary surrogate from cached LLM activations, and saving as artifacts for further analysis.
"""

from src.project_config import BaseConfig
import os
import torch as th
import json
import numpy as np

from src.preprocessing_utils import load_llm_artifacts, run_parameter_sweep




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
