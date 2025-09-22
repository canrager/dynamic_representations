"""
Script for caching garden path sentence activations:

1. Load garden path sentences from artifacts/text_inputs/garden_path.csv
2. Cache LLM activations for sentence_ambiguous, sentence_gp, and sentence_post
3. Save as separate artifacts for further analysis

The script creates a timestamped folder in artifacts/activations/ and saves:
- sentence_ambiguous.pt, sentence_gp.pt, sentence_post.pt (activations)
- masks_ambiguous.pt, masks_gp.pt, masks_post.pt (attention masks)
- tokens_ambiguous.pt, tokens_gp.pt, tokens_post.pt (token ids)
- config.json (configuration metadata)
"""

from dataclasses import asdict, dataclass
import os
import torch as th
import gc
import json
import pandas as pd
from datetime import datetime

from src.model_utils import load_nnsight_model, load_sae
from src.exp_utils import compute_llm_artifacts
from src.cache_utils import batch_sae_cache

from src.configs import *


@dataclass
class GardenPathCacheConfig:
    scaling_factor: float
    data: DatasetConfig  # Will be used for metadata only
    llm: LLMConfig
    sae: (
        SAEConfig | None
    )  # If None is passed, Cache the LLM only. SAE cache requires existing LLM cache.
    env: EnvironmentConfig


def load_garden_path_sentences():
    """Load garden path sentences from CSV file."""
    csv_path = os.path.join("artifacts", "text_inputs", "garden_path.csv")
    df = pd.read_csv(csv_path)

    # Extract the three sentence types
    sentence_ambiguous = df['sentence_ambiguous'].tolist()
    sentence_gp = df['sentence_gp'].tolist()
    sentence_post = df['sentence_post'].tolist()

    return sentence_ambiguous, sentence_gp, sentence_post


def cache_garden_path_llm_activations(cfg):
    """Cache activations for all three garden path sentence types."""
    # Create timestamped folder
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(cfg.env.activations_dir, folder_name)
    os.makedirs(folder_dir)

    # Load model once
    model, submodule, _ = load_nnsight_model(cfg)

    # Load garden path sentences
    sentence_ambiguous, sentence_gp, sentence_post = load_garden_path_sentences()

    # Dictionary to store sentence types and their data
    sentence_types = {
        'ambiguous': sentence_ambiguous,
        'gp': sentence_gp,
        'post': sentence_post
    }

    print(f"Processing {len(sentence_ambiguous)} garden path sentences...")

    # Process each sentence type
    for sentence_type, sentences in sentence_types.items():
        print(f"Processing {sentence_type} sentences...")

        # Compute activations using loaded_sequences
        acts_LBPD, masks_LBPD, tokens_BP = compute_llm_artifacts(
            cfg, model, submodule, loaded_dataset_sequences=sentences
        )

        # Apply scaling factor and move to CPU
        acts_LBPD = cfg.scaling_factor * acts_LBPD.cpu()
        masks_LBPD = masks_LBPD.cpu()
        tokens_BP = tokens_BP.cpu()

        # Save activations, masks, and tokens for this sentence type
        with open(os.path.join(folder_dir, f"sentence_{sentence_type}.pt"), "wb") as f:
            th.save(acts_LBPD, f, pickle_protocol=5)
        with open(os.path.join(folder_dir, f"masks_{sentence_type}.pt"), "wb") as f:
            th.save(masks_LBPD, f, pickle_protocol=5)
        with open(os.path.join(folder_dir, f"tokens_{sentence_type}.pt"), "wb") as f:
            th.save(tokens_BP, f, pickle_protocol=5)

        print(f"Saved {sentence_type} activations: {acts_LBPD.shape}")

        # Clean up memory
        del acts_LBPD, masks_LBPD, tokens_BP
        th.cuda.empty_cache()
        gc.collect()

    # Save configuration
    with open(os.path.join(folder_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"Cached garden path activations to: {folder_dir}")

    # Clean up model
    del model, submodule
    th.cuda.empty_cache()
    gc.collect()


def load_garden_path_llm_activations(cfg, sentence_type, return_target_dir=False):
    """Load LLM activations for a specific garden path sentence type."""
    artifacts, target_dir = load_matching_activations(
        source_object=cfg,
        target_filenames=[f"sentence_{sentence_type}"],
        target_folder=cfg.env.activations_dir,
        recency_rank=0,
        compared_attributes=["llm", "data"],
    )
    if not return_target_dir:
        return artifacts[f"sentence_{sentence_type}"]
    else:
        return artifacts[f"sentence_{sentence_type}"], target_dir


def cache_garden_path_sae_activations(cfg):
    """Cache SAE activations for all three garden path sentence types."""
    sentence_types = ['ambiguous', 'gp', 'post']

    for sentence_type in sentence_types:
        print(f"Processing SAE activations for {sentence_type} sentences...")

        # Load LLM activations for this sentence type
        llm_act_BPD, save_dir = load_garden_path_llm_activations(cfg, sentence_type, return_target_dir=True)

        # Create SAE directory structure: sae_name/sentence_type/
        sae_base_dir = os.path.join(save_dir, cfg.sae.name)
        sae_save_dir = os.path.join(sae_base_dir, sentence_type)

        # Check whether precomputed acts for this sae and sentence type already exists
        if os.path.isdir(sae_save_dir):
            print(f'Cached SAE activations already exist in {sae_save_dir}. Skipping SAE cache for {sentence_type}.')
            continue
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

            with open(os.path.join(sae_save_dir, "latents.pt"), "wb") as f:
                th.save(sae_act_BPS, f)
            with open(os.path.join(sae_save_dir, "reconstructions.pt"), "wb") as f:
                th.save(sae_recon_BPD, f)

            del sae_act_BPS, sae_recon_BPD

        del sae
        th.cuda.empty_cache()
        gc.collect()

        print(f"Saved SAE activations for {sentence_type} to: {sae_save_dir}")


def main():
    """Main function to configure and run garden path caching."""
    cache_configs = get_configs(
        GardenPathCacheConfig,
        scaling_factor=0.00666666667,  # For gemma2-2b on pile
        data=DatasetConfig(
            name="GardenPath",
            hf_name="garden_path.csv",
            num_sequences=97,  # Total number of sentences in CSV
            context_length=None,  # Use variable length with padding
        ),
        llm=LLMConfig(
            name="Gemma-2-2B",
            hf_name="google/gemma-2-2b",
            revision=None,
            layer_idx=12,
            hidden_dim=2304,
            batch_size=50,
        ),
        # sae=None,  # Uncomment to cache only LLM activations
        sae=[TEMPORAL_SELFTRAIN_SAE_CFG, BATCHTOPK_SELFTRAIN_SAE_CFG],  # Cache temporal selftrain SAEs
        env=ENV_CFG,
    )

    for cfg in cache_configs:
        if cfg.sae is None:
            # Cache LLM activations only
            cache_garden_path_llm_activations(cfg)
        else:
            # Cache SAE activations (requires existing LLM cache)
            cache_garden_path_sae_activations(cfg)


if __name__ == "__main__":
    main()