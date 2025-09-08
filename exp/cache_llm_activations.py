"""
Script for caching LLM activations, and saving as artifacts for further analysis.
"""

from src.project_config import BaseConfig
from dataclasses import asdict
from src.exp_utils import compute_llm_artifacts
import os
import torch as th
import json
from src.model_utils import load_nnsight_model

from src.preprocessing_utils import run_parameter_sweep


def cache_single_experiment(cfg):
    folder_dir = os.path.join(cfg.activations_dir, cfg.filename)
    os.makedirs(folder_dir, exist_ok=cfg.overwrite_existing)

    model, submodules, _ = load_nnsight_model(cfg)

    acts_LBPD, masks_LBPD, tokens_BP, _ = compute_llm_artifacts(cfg, model, submodules)

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
    del acts_LBPD, masks_LBPD, tokens_BP, model, submodules
    th.cuda.empty_cache()
    import gc
    gc.collect()


def main():
    base_cfg = BaseConfig(
        experiment_name="cache_llm_activations",
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
        save_artifacts=False,  # Suppress automatic saving of compute_llm_artifacts
        overwrite_existing=True,
        verbose=False,
    )
    
    sweep_params = {
        'llm_name': ['meta-llama/Llama-3.1-8B'],
        'layer_idx': [31],
        'dataset_name': ['monology/pile-uncopyrighted']
    }
    
    run_parameter_sweep(base_cfg, sweep_params, cache_single_experiment)


if __name__ == "__main__":
    main()
