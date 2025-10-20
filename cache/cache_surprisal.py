"""
Script for caching surprisal sentence activations:

1. Load surprisal sentences from artifacts/text_inputs/surprisal.json
2. Cache LLM activations for the sentences
3. Save as artifacts for further analysis

The script creates a timestamped folder in artifacts/activations/ and saves:
- sentences.pt (activations)
- masks.pt (attention masks)
- tokens.pt (token ids)
- config.json (configuration metadata)
"""

from dataclasses import asdict, dataclass
import os
import torch as th
import gc
import json
from datetime import datetime

from src.model_utils import load_nnsight_model, load_sae
from src.exp_utils import compute_llm_artifacts
from src.cache_utils import batch_sae_cache

from src.configs import *


@dataclass
class SurprisalCacheConfig:
    scaling_factor: float
    data: DatasetConfig  # Will be used for metadata only
    llm: LLMConfig
    sae: (
        SAEConfig | None
    )  # If None is passed, Cache the LLM only. SAE cache requires existing LLM cache.
    env: EnvironmentConfig


def load_surprisal_sentences(filename):
    """Load surprisal sentences from JSON file."""
    json_path = os.path.join("artifacts", "text_inputs", filename)
    with open(json_path, 'r') as f:
        sentences = json.load(f)
    return sentences


def cache_surprisal_llm_activations(cfg):
    """Cache activations for surprisal sentences."""
    # Create timestamped folder
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(cfg.env.activations_dir, folder_name)
    os.makedirs(folder_dir)

    # Load model once
    model, submodule, _ = load_nnsight_model(cfg)

    # Load surprisal sentences
    sentences = load_surprisal_sentences(cfg.data.hf_name)

    print(f"Processing {len(sentences)} surprisal sentences...")

    # Compute activations using loaded_sequences
    acts_BPD, masks_BPD, tokens_BP, attn_patterns_BHPP = compute_llm_artifacts(
        cfg, model, submodule, loaded_dataset_sequences=sentences, cache_attn=True
    )

    # Apply scaling factor and move to CPU
    acts_BPD = cfg.scaling_factor * acts_BPD.cpu()
    masks_BPD = masks_BPD.cpu()
    tokens_BP = tokens_BP.cpu()

    # Save activations, masks, and tokens
    with open(os.path.join(folder_dir, "activations.pt"), "wb") as f:
        th.save(acts_BPD, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, "masks.pt"), "wb") as f:
        th.save(masks_BPD, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, "tokens.pt"), "wb") as f:
        th.save(tokens_BP, f, pickle_protocol=5)
    with open(os.path.join(folder_dir, "attn_patterns.pt"), "wb") as f:
        th.save(attn_patterns_BHPP, f, pickle_protocol=5)

    print(attn_patterns_BHPP.shape)

    print(f"Saved surprisal activations: {acts_BPD.shape}")

    # Save configuration
    with open(os.path.join(folder_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Cached surprisal activations to: {folder_dir}")

    # Clean up
    del acts_BPD, masks_BPD, tokens_BP, model, submodule
    th.cuda.empty_cache()
    gc.collect()


def load_surprisal_llm_activations(cfg, return_target_dir=False):
    """Load LLM activations for surprisal sentences."""
    artifacts, target_dir = load_matching_activations(
        source_object=cfg,
        target_filenames=["activations"],
        target_folder=cfg.env.activations_dir,
        recency_rank=0,
        compared_attributes=["llm", "data"],
    )
    if not return_target_dir:
        return artifacts["activations"]
    else:
        return artifacts["activations"], target_dir


def cache_surprisal_sae_activations(cfg):
    """Cache SAE activations for surprisal sentences."""
    print("Processing SAE activations for surprisal sentences...")

    # Load LLM activations
    llm_act_BPD, save_dir = load_surprisal_llm_activations(cfg, return_target_dir=True)

    # Create SAE directory structure
    sae_save_dir = os.path.join(save_dir, cfg.sae.name)

    # Check whether precomputed acts for this sae already exists
    if os.path.isdir(sae_save_dir):
        print(f'Cached SAE activations already exist in {sae_save_dir}. Skipping SAE cache.')
        return
    else:
        os.makedirs(sae_save_dir, exist_ok=True)

    # Load SAE and compute activations
    sae = load_sae(cfg)

    if "selftrain" in cfg.sae.local_weights_path:
        # Use temporal SAE caching for selftrain SAEs
        results_dict = batch_sae_cache(sae, llm_act_BPD, cfg)

        for key in results_dict:
            with open(os.path.join(sae_save_dir, f"{key}.pt"), "wb") as f:
                th.save(results_dict[key], f)

        del results_dict
    else:
        # Use dictionary learning SAE caching
        sae_recon_BPD, sae_act_BPS, _ = batch_sae_cache(sae, llm_act_BPD, cfg)

        sae_act_BPS = sae_act_BPS.cpu()
        sae_recon_BPD = sae_recon_BPD.cpu()

        with open(os.path.join(sae_save_dir, "codes.pt"), "wb") as f:
            th.save(sae_act_BPS, f)
        with open(os.path.join(sae_save_dir, "recons.pt"), "wb") as f:
            th.save(sae_recon_BPD, f)

        del sae_act_BPS, sae_recon_BPD

    del sae
    th.cuda.empty_cache()
    gc.collect()

    print(f"Saved SAE activations to: {sae_save_dir}")


def main():
    """Main function to configure and run surprisal caching."""
    cache_configs = get_configs(
        SurprisalCacheConfig,
        scaling_factor=0.00666666667,  # For gemma2-2b on pile
        # data=DatasetConfig(
        #     name="Surprisal",
        #     hf_name="surprisal.json",
        #     num_sequences=59,  # Total number of sentences in JSON
        #     context_length=None,  # Use variable length with padding
        # ),
        data=DatasetConfig(
            name="Twist",
            hf_name="twist.json",
            num_sequences=1,  # Total number of sentences in JSON
            context_length=500,  # Use variable length with padding
        ),
        # llm=LLMConfig(
        #     name="Gemma-2-2B",
        #     hf_name="google/gemma-2-2b",
        #     revision=None,
        #     layer_idx=12,
        #     hidden_dim=2304,
        #     batch_size=50,
        # ),
        llm=GEMMA2_LLM_CFG,
        # sae=None,  # Uncomment to cache only LLM activations
        sae=[None, TEMPORAL_SELFTRAIN_SAE_CFG, BATCHTOPK_SELFTRAIN_SAE_CFG],  # Cache temporal selftrain SAEs
        env=ENV_CFG,
    )

    for cfg in cache_configs:
        if cfg.sae is None:
            # Cache LLM activations only
            cache_surprisal_llm_activations(cfg)
        else:
            # Cache SAE activations (requires existing LLM cache)
            cache_surprisal_sae_activations(cfg)


if __name__ == "__main__":
    main()