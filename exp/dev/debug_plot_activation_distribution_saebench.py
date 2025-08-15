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

from src.custom_saes.topk_sae import load_dictionary_learning_topk_sae
from src.custom_saes.relu_sae import load_dictionary_learning_relu_sae

# Project configuration (inlined)
DEVICE = "cuda"
MODELS_DIR = "/home/can/models"
PLOTS_DIR = "debug_plots"

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)


class Config:
    """Exact copy of the Config class from the original file"""

    def __init__(self):
        self.llm_name: str = "google/gemma-2-2b"
        self.layer_idx: int = 12
        self.llm_batch_size: int = 100

        self.dtype = torch.float32

        self.sae_architecture = "topk"
        self.sae_repo_id = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
        self.sae_filename = "gemma-2-2b_top_k_width-2pow14_date-0107/resid_post_layer_12/trainer_2/ae.pt"
        self.sae_name: str = "saebench_gemma-2-2b_topk-80_width-2pow14_layer_12_trainer_2"
        self.d_sae: int = 192 # 16384

        # self.sae_architecture = "relu"
        # self.sae_repo_id = "canrager/saebench_gemma-2-2b_width-2pow14_date-0107"
        # self.sae_filename = "gemma-2-2b_standard_new_width-2pow14_date-0107/resid_post_layer_12/trainer_4/ae.pt"
        # self.sae_name: str = "saebench_gemma-2-2b_relu_width-2pow14_layer_12_trainer_4"
        # self.d_sae: int = 16384

        # self.sae_name: str = "gemma-2-2b_top_k_width-2pow14_date-0107_resid_post_layer_12_trainer_2"

        self.sae_batch_size: int = 100

        ### Dataset
        # self.dataset_name: str = "SimpleStories/SimpleStories"
        self.dataset_name: str = "monology/pile-uncopyrighted"
        # self.dataset_name: str = "NeelNanda/code-10k"
        # self.hf_text_identifier: str = "story"
        self.hf_text_identifier: str = "text"
        self.num_total_stories: int = 100

        self.selected_story_idxs: Optional[List[int]] = None
        self.omit_BOS_token: bool = True
        self.num_tokens_per_story: int = 10
        self.force_recompute: bool = (
            True  # Set to True for debugging to always recompute
        )

        self.latent_active_threshs: bool = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
        self.sort_variance: bool = False
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
    elif any([s in cfg.llm_name for s in ["Llama", "gemma"]]):
        print(model)
        print(model.config)
        hidden_dim = model.config.hidden_size
        submodules = [
            model.model.layers[l] for l in range(model.config.num_hidden_layers)
        ]
    return model, submodules, hidden_dim



def load_sae_saebench(cfg):
    if cfg.sae_architecture == "topk":
        sae = load_dictionary_learning_topk_sae(
            repo_id=cfg.sae_repo_id,
            filename=cfg.sae_filename,
            model_name=cfg.llm_name,
            device=DEVICE,
            layer=cfg.layer_idx,
            dtype=cfg.dtype
        )
    elif cfg.sae_architecture == "relu":
        sae = load_dictionary_learning_relu_sae(
            repo_id=cfg.sae_repo_id,
            filename=cfg.sae_filename,
            model_name=cfg.llm_name,
            device=DEVICE,
            layer=cfg.layer_idx,
            dtype=cfg.dtype
        )
    else:
        raise NotImplementedError
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


def batch_sae_cache_saebench(
    sae, act_BPD: Tensor, batch_size: int = 100, device: str = "cuda"
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Process activations through SAE in batches.
    """
    B, P, D = act_BPD.shape

    sae_outs = []
    fvus = []
    latent_actss = []
    latent_indicess = []

    for i in trange(0, B, batch_size, desc="SAE forward"):
        batch = act_BPD[i : i + batch_size]
        batch = batch.to(device)
        batch_sae = sae.forward(batch)
        sae_outs.append(batch_sae.sae_out.detach().cpu())
        fvus.append(batch_sae.fvu.detach().cpu())
        latent_actss.append(batch_sae.latent_acts.detach().cpu())
        latent_indicess.append(batch_sae.latent_indices.detach().cpu())
        print(f"batch_sae {batch_sae.latent_acts.shape}")

    print(f'len sae_outs {len(sae_outs)}')

    sae_outs = torch.cat(sae_outs, dim=0)
    fvus = torch.cat(fvus, dim=0)
    latent_actss = torch.cat(latent_actss, dim=0)
    latent_indicess = torch.cat(latent_indicess, dim=0)

    return sae_outs, fvus, latent_actss, latent_indicess


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
    sae = load_sae_saebench(cfg)

    # Compute data
    sae_out_BPD, fvu_BP, latent_acts_BPS, latent_indices_BPS = batch_sae_cache_saebench(
        sae, llm_act_BPD, cfg.sae_batch_size, DEVICE
    )
    print(f'sae_out_BPD shape = {sae_out_BPD.shape}')
    print(f'fvu_BP shape = {fvu_BP.shape}')
    print(f'latent_acts_BPS shape = {latent_acts_BPS.shape}')
    print(f'latent_indices_BPS shape = {latent_indices_BPS.shape}')

    # Save data
    sae_data = {
        "llm_acts": llm_act_BPD,
        "sae_out": sae_out_BPD,
        "fvu": fvu_BP,
        "latent_acts": latent_acts_BPS,
        "latent_indices": latent_indices_BPS,
        "model_name": cfg.llm_name,
        "num_total_stories": cfg.num_total_stories,
        "sae_name": cfg.sae_name,
        "layer_idx": cfg.layer_idx,
    }
    

    return llm_act_BPD, latent_acts_BPS, latent_indices_BPS, sae_out_BPD, fvu_BP


# Plotting function (exact copy from original)
def plot_activation_distribution(latent_act_BPS, cfg):
    latent_act_BPS = latent_act_BPS.float().detach().cpu()

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Sorted variance (cfg.sort_variance=True)
    latent_act_sorted_BPS, _ = torch.sort(latent_act_BPS, dim=-1, descending=True)
    latent_act_mean_sorted_S = latent_act_sorted_BPS.mean(dim=(0, 1))
    latent_act_std_sorted_S = latent_act_sorted_BPS.std(dim=(0, 1))
    latent_sorted_max_act_var = latent_act_sorted_BPS.max(-1).values.var()
    B, P, S = latent_act_BPS.shape
    latent_act_ci_sorted_S = 1.96 * latent_act_std_sorted_S / (B**0.5)

    ax1.plot(latent_act_mean_sorted_S, color="blue")
    ax1.scatter(
        range(len(latent_act_mean_sorted_S)),
        latent_act_mean_sorted_S,
        label="activation value",
        color="blue",
    )
    ax1.fill_between(
        range(len(latent_act_mean_sorted_S)),
        latent_act_mean_sorted_S - latent_act_ci_sorted_S,
        latent_act_mean_sorted_S + latent_act_ci_sorted_S,
        alpha=0.2,
        label=f"95% CI",
        color="blue",
    )
    ax1.set_xlabel("rank")
    ax1.set_ylabel("mean activation magnitude")
    ax1.set_title(f"Sorted by magnitude \n(variance across tokens for max latent act:{latent_sorted_max_act_var})")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right subplot: Not sorted (cfg.sort_variance=False)
    latent_act_mean_S = latent_act_BPS.mean(dim=(0, 1))
    latent_act_std_S = latent_act_BPS.std(dim=(0, 1))
    latent_max_act_var = latent_act_BPS.max(-1).values.var()
    latent_act_ci_S = 1.96 * latent_act_std_S / (B**0.5)

    ax2.plot(latent_act_mean_S, color="red")
    ax2.scatter(
        range(len(latent_act_mean_S)),
        latent_act_mean_S,
        label="activation value",
        color="red",
    )
    ax2.fill_between(
        range(len(latent_act_mean_S)),
        latent_act_mean_S - latent_act_ci_S,
        latent_act_mean_S + latent_act_ci_S,
        alpha=0.2,
        label=f"95% CI",
        color="red",
    )
    ax2.set_xlabel("latent index")
    ax2.set_ylabel("mean activation magnitude")
    ax2.set_title(f"Original order \n(variance across tokens for max latent act:{latent_max_act_var})")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Main title for the entire figure
    fig.suptitle(
        f"SAE Latent Activations\nsae {cfg.sae_str}, dataset {cfg.dataset_str}",
        fontsize=14
    )

    plt.tight_layout()

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

    llm_act_BPD, latent_acts_BPS, latent_indices_BPS, sae_out_BPD, fvu_BP = (
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
