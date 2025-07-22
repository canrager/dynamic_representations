#!/usr/bin/env python3
"""
Standalone script for debugging plot_activation_distribution function
Uses the exact same dataset loading and preprocessing pipeline as the original
"""

import os
import torch
import json
from nnsight import LanguageModel
from tqdm import trange
from typing import List, Tuple, Optional
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding, AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

from sparsify import Sae

# Project configuration (inlined)
DEVICE = "cuda"
MODELS_DIR = "/home/can/models"
PLOTS_DIR = "debug_plots"

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)


class Config:
    """Exact copy of the Config class from the original file"""

    def __init__(self):
        self.llm_name: str = "meta-llama/Meta-Llama-3-8B"
        self.layer_idx: int = 13
        self.llm_batch_size: int = 100

        self.sae_name: str = "EleutherAI/sae-llama-3-8b-32x-v2"
        self.sae_batch_size: int = 100

        ### Dataset
        self.dataset_name: str = "SimpleStories/SimpleStories"
        # self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        self.hf_text_identifier: str = "story"
        self.num_total_stories: int = 100

        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 10
        self.force_recompute: bool = (
            True  # Set to True for debugging to always recompute
        )

        self.latent_active_threshs: bool = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
        self.sort_variance: bool = True
        self.reconstruction_thresholds: List[float] = [0.8, 0.9, 0.95, 0.99]

        # For debugging, add this attribute
        self.debug: bool = False

        ### String summarizing the parameters for loading and saving artifacts
        self.llm_str = self.llm_name.split("/")[-1]
        self.sae_str = self.sae_name.split("/")[-1]
        self.dataset_str = self.dataset_name.split("/")[-1].split(".")[0]
        self.story_idxs_str = (
            "_".join([str(i) for i in self.selected_story_idxs])
            if self.selected_story_idxs is not None
            else "all"
        )

        self.input_file_str = (
            f"{self.llm_str}" + f"_{self.dataset_str}" + f"_{self.num_total_stories}"
        )

        self.sae_file_str = self.input_file_str + f"_{self.sae_str}"

        self.output_file_str = (
            self.sae_file_str
            + f"_ntok_{self.num_tokens_per_story}"
            + f"_nobos_{self.omit_BOS_token}"
            + f"_didx_{self.story_idxs_str}"
        )


# Model utilities (inlined from src/model_utils.py)
def load_tokenizer(llm_name: str, cache_dir: str):
    if "gpt2" in llm_name:
        tokenizer = GPT2Tokenizer.from_pretrained(llm_name, cache_dir=cache_dir)
        tokenizer.add_bos_token = True
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir)
    return tokenizer


def load_nnsight_model(cfg):
    model = LanguageModel(
        cfg.llm_name,
        cache_dir=MODELS_DIR,
        device_map=DEVICE,
        dispatch=True,
    )

    if "gpt2" in cfg.llm_name:
        print(model)
        print(model.config)
        hidden_dim = model.config.n_embd
        submodules = [model.transformer.h[l] for l in range(model.config.n_layer)]
        # Language Model loads the AutoTokenizer, which does not use the add_bos_token method.
        model.tokenizer = load_tokenizer(cfg.llm_name, cache_dir=MODELS_DIR)
    elif "Llama" in cfg.llm_name:
        print(model)
        print(model.config)
        hidden_dim = model.config.hidden_size
        submodules = [
            model.model.layers[l] for l in range(model.config.num_hidden_layers)
        ]
    return model, submodules, hidden_dim


def load_sae(sae_name, layer_idx):
    sae_hookpoint_str = f"layers.{layer_idx}"
    sae = Sae.load_from_hub(sae_name, sae_hookpoint_str, cache_dir=MODELS_DIR)
    sae = sae.to(DEVICE)
    print(f"Loaded SAE with k={sae.cfg.k}")
    return sae


# Cache utilities (inlined from src/cache_utils.py)
def collect_from_hf(
    tokenizer, dataset_name, num_stories, num_tokens, hf_text_identifier="story"
):
    all_stories = load_dataset(path=dataset_name, cache_dir=MODELS_DIR, streaming=True)[
        "train"
    ]

    inputs_BP = []
    selected_story_idxs = []

    for story_idx, story_item in enumerate(all_stories):
        if len(inputs_BP) >= num_stories:
            break

        story_text = story_item[hf_text_identifier]
        input_ids_P = tokenizer(story_text, return_tensors="pt").input_ids

        if input_ids_P.shape[1] >= num_tokens:
            inputs_BP.append(input_ids_P[0, :num_tokens])
            selected_story_idxs.append(story_idx)

    inputs_BP = torch.stack(inputs_BP)
    masks_BP = torch.ones_like(inputs_BP)  # All ones since we filtered by length
    return inputs_BP, masks_BP, selected_story_idxs

def batch_llm_cache(
    model: LanguageModel,
    submodules: List[Module],
    inputs_BP: Tensor,
    masks_BP: Tensor,
    hidden_dim: int,
    batch_size: int,
    device: str,
    debug: bool = False,
) -> Tensor:
    all_acts_LBPD = torch.zeros(
        (
            len(submodules),
            inputs_BP.shape[0],
            inputs_BP.shape[1],
            hidden_dim,
        )
    )
    all_masks_BP = torch.zeros(
        (
            inputs_BP.shape[0],
            inputs_BP.shape[1],
        )
    )

    for batch_start in trange(
        0, inputs_BP.shape[0], batch_size, desc="Batched Forward"
    ):
        batch_end = batch_start + batch_size
        batch_input_ids = inputs_BP[batch_start:batch_end].to(device)
        batch_mask = masks_BP[batch_start:batch_end].to(device)
        batch_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
        }

        if debug:
            print(
                f"computing activations for batch {batch_start} to {batch_end}. Tokens:"
            )
            for ids_P in batch_input_ids:
                decoded_tokens = [
                    model.tokenizer.decode(id, skip_special_tokens=False)
                    for id in ids_P
                ]
                print(decoded_tokens)
                print()
            print(f"batch_mask:\n {batch_mask}")

        all_masks_BP[batch_start:batch_end] = batch_mask
        with (
            torch.inference_mode(),
            model.trace(batch_inputs, scan=False, validate=False),
        ):
            for l, sm in enumerate(submodules):
                all_acts_LBPD[l, batch_start:batch_end] = sm.output[0].save()

    all_acts_LBPD = all_acts_LBPD.to("cpu")
    all_masks_BP = all_masks_BP.to("cpu")
    return all_acts_LBPD, all_masks_BP


def batch_sae_cache(
    sae, act_BPD: Tensor, batch_size: int = 100, device: str = "cuda"
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Process activations through SAE in batches.
    """
    B, P, D = act_BPD.shape
    K = sae.cfg.k

    act_flattened = act_BPD.reshape(B * P, D)
    out_flattened = []
    fvu_flattened = []
    latent_acts_flattened = []
    latent_indices_flattened = []

    for i in trange(0, len(act_flattened), batch_size, desc="SAE forward"):
        batch = act_flattened[i : i + batch_size]
        batch = batch.to(device)
        batch_sae = sae.forward(batch)
        out_flattened.append(batch_sae.sae_out.detach().cpu())
        fvu_flattened.append(batch_sae.fvu.detach().cpu())
        latent_acts_flattened.append(batch_sae.latent_acts.detach().cpu())
        latent_indices_flattened.append(batch_sae.latent_indices.detach().cpu())
        print(f"batch_sae {batch_sae.latent_acts.shape}")

    out_flattened = torch.cat(out_flattened, dim=0)
    fvu_flattened = torch.cat(fvu_flattened, dim=0)
    latent_acts_flattened = torch.cat(latent_acts_flattened, dim=0)
    latent_indices_flattened = torch.cat(latent_indices_flattened, dim=0)

    out = out_flattened.reshape(B, P, D)
    fvu = fvu_flattened.reshape(B, P)
    latent_acts = latent_acts_flattened.reshape(B, P, K)
    latent_indices = latent_indices_flattened.reshape(B, P, K)

    return out, fvu, latent_acts, latent_indices


def compute_llm_artifacts(cfg):
    # Load model
    model, submodules, hidden_dim = load_nnsight_model(cfg)

    inputs_BP, masks_BP, dataset_story_idxs = collect_from_hf(
        tokenizer=model.tokenizer,
        dataset_name=cfg.dataset_name,
        num_stories=cfg.num_total_stories,
        num_tokens=cfg.num_tokens_per_story,
        hf_text_identifier=cfg.hf_text_identifier,
    )

    # Call batch_act_cache
    all_acts_LbPD, all_masks_BP = batch_llm_cache(
        model=model,
        submodules=submodules,
        inputs_BP=inputs_BP,
        masks_BP=masks_BP,
        hidden_dim=hidden_dim,
        batch_size=cfg.llm_batch_size,
        device=DEVICE,
        debug=cfg.debug,
    )

    if cfg.selected_story_idxs is not None:
        dataset_story_idxs = torch.tensor(dataset_story_idxs)
        dataset_story_idxs = dataset_story_idxs[cfg.selected_story_idxs]
        all_acts_LbPD = all_acts_LbPD[:, dataset_story_idxs, :, :]
        all_masks_BP = all_masks_BP[dataset_story_idxs, :]

    if cfg.omit_BOS_token:
        all_acts_LbPD = all_acts_LbPD[:, :, 1:, :]
        inputs_BP = inputs_BP[:, 1:]
        masks_BP = masks_BP[:, 1:]

    if cfg.num_tokens_per_story is not None:
        all_acts_LbPD = all_acts_LbPD[:, :, : cfg.num_tokens_per_story, :]
        inputs_BP = inputs_BP[:, : cfg.num_tokens_per_story]
        masks_BP = masks_BP[:, : cfg.num_tokens_per_story]

    return all_acts_LbPD, all_masks_BP, inputs_BP, dataset_story_idxs



def compute_or_load_sae(cfg) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute or load SAE results for a given model, layer, and number of stories."""


    llm_act_LBPD, all_masks_BP, inputs_BP, dataset_story_idxs = compute_llm_artifacts(cfg)
    llm_act_BPD = llm_act_LBPD[cfg.layer_idx]

    # Load SAE
    sae = load_sae(cfg.sae_name, cfg.layer_idx)

    # Compute data
    sae_out_BPD, fvu_BP, latent_acts_BPS, latent_indices_BPK = batch_sae_cache(
        sae, llm_act_BPD, cfg.sae_batch_size, DEVICE
    )

    # Save data
    sae_data = {
        "llm_acts": llm_act_BPD,
        "sae_out": sae_out_BPD,
        "fvu": fvu_BP,
        "latent_acts": latent_acts_BPS,
        "latent_indices": latent_indices_BPK,
        "model_name": cfg.llm_name,
        "num_total_stories": cfg.num_total_stories,
        "sae_name": cfg.sae_name,
        "layer_idx": cfg.layer_idx,
    }
    

    return llm_act_BPD, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP


# Plotting function (exact copy from original)
def plot_activation_distribution(latent_act_BPS, cfg):
    latent_act_BPS = latent_act_BPS.float().detach().cpu()

    if cfg.sort_variance:
        latent_act_BPS, _ = torch.sort(latent_act_BPS, dim=-1, descending=True)

    latent_act_mean_S = latent_act_BPS.mean(dim=(0, 1))
    latent_act_std_S = latent_act_BPS.std(dim=(0, 1))
    B, P, S = latent_act_BPS.shape
    latent_act_ci_S = 1.96 * latent_act_std_S / (B**0.5)  # only stories are independent

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(latent_act_mean_S, color="blue")
    ax.scatter(
        range(len(latent_act_mean_S)),
        latent_act_mean_S,
        label="activation value",
        color="blue",
    )
    ax.fill_between(
        range(len(latent_act_mean_S)),
        latent_act_mean_S - latent_act_ci_S,
        latent_act_mean_S + latent_act_ci_S,
        alpha=0.2,
        label=f"95% CI",
        color="blue",
    )

    ax.set_xlabel("rank")
    ax.set_ylabel("mean activation magnitude")
    ax.set_title(
        f"Activation, mean over batch and token pos, sorted by magnitude\nsae {cfg.sae_str}, dataset {cfg.dataset_str}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    save_dir = os.path.join(
        PLOTS_DIR, f"activation_distribution_{cfg.output_file_str}.png"
    )
    plt.savefig(save_dir, dpi=80)
    print(f"\nSaved SAE latent var distribution over tokens plot to {save_dir}")
    plt.close()


def main():
    """Main function to debug plot_activation_distribution with exact same pipeline"""
    print("Starting debug script for plot_activation_distribution...")
    print(f"Using device: {DEVICE}")

    cfg = Config()
    print(f"Configuration:")
    print(f"  Model: {cfg.llm_name}")
    print(f"  SAE: {cfg.sae_name}")
    print(f"  Dataset: {cfg.dataset_name}")
    print(f"  Layer: {cfg.layer_idx}")
    print(f"  Stories: {cfg.num_total_stories}")
    print(f"  Tokens per story: {cfg.num_tokens_per_story}")

    print("\nLoading data using exact same pipeline as original...")

    llm_act_BPD, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP = (
        compute_or_load_sae(cfg)
    )

    print(f"\nData loaded successfully!")
    print(f"  LLM activations shape: {llm_act_BPD.shape}")
    print(f"  SAE latent activations shape: {latent_acts_BPS.shape}")
    print(f"  SAE output shape: {sae_out_BPD.shape}")
    print(f"  FVU shape: {fvu_BP.shape}")

    print(f"\nData statistics:")
    print(f"  Latent acts mean: {latent_acts_BPS.mean():.4f}")
    print(f"  Latent acts std: {latent_acts_BPS.std():.4f}")
    print(f"  Latent acts max: {latent_acts_BPS.max():.4f}")
    print(f"  Non-zero fraction: {(latent_acts_BPS > 0).float().mean():.4f}")

    print("\nCalling plot_activation_distribution...")
    plot_activation_distribution(latent_acts_BPS, cfg)

    print("Debug script completed successfully!")
    print(f"Check {PLOTS_DIR}/ for the output plot.")


if __name__ == "__main__":
    main()
