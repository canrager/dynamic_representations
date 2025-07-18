import os
import torch
import json
from nnsight import LanguageModel
from tqdm import trange
from typing import List, Tuple
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding
from datasets import load_dataset

from src.model_utils import load_nnsight_model
from src.project_config import DEVICE, MODELS_DIR, INTERIM_DIR, INPUTS_DIR

def collect_from_hf(tokenizer, dataset_name, num_stories, num_tokens, hf_text_identifier="story"):
    # Use stories with the same amounts of tokens
    # For equal weighting across position and avoiding padding errors
    # NOTE: exact tokenization varies by model, therefore, it can be that different models see different stories

    all_stories = load_dataset(path=dataset_name, cache_dir=MODELS_DIR, streaming=True)["train"]

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

    return inputs_BP, selected_story_idxs


def collect_from_local(tokenizer, dataset_name, num_sentences, num_tokens):
    with open(os.path.join(INPUTS_DIR, dataset_name), "r") as f:
        all_sentences = json.load(f)["sentences"]
    print(f"Loaded {len(all_sentences)} sentences.")

    inputs_BP = []
    selected_sentence_idxs = []

    min_length = 1e5
    max_length = 0
    for sentence_idx, sentence_item in enumerate(all_sentences):
        if len(inputs_BP) >= num_sentences:
            break

        input_ids_BP = tokenizer(sentence_item, return_tensors="pt").input_ids
        input_ids_P = input_ids_BP[0, :]
        P = input_ids_P.shape[0]

        if P < min_length:
            min_length = P

        if P > max_length:
            max_length = P

        if P >= num_tokens:
            inputs_BP.append(input_ids_P[:num_tokens])
            selected_sentence_idxs.append(sentence_idx)

    assert len(inputs_BP) == num_sentences, (
        f"Expected {num_sentences} sentences. Collected {len(inputs_BP)} sentences. "
        f"Minimum length: {min_length}, Maximum length: {max_length}"
    )

    print(f"Tokenized {len(inputs_BP)} sentences")
    inputs_BP = torch.stack(inputs_BP)
    return inputs_BP, selected_sentence_idxs

def batch_llm_cache(
    model: LanguageModel,
    submodules: List[Module],
    inputs_BP: BatchEncoding,
    hidden_dim: int,
    batch_size: int,
    device: str,
) -> Tensor:
    all_acts_LBPD = torch.zeros(
        (
            len(submodules),
            inputs_BP.shape[0],
            inputs_BP.shape[1],
            hidden_dim,
        )
    )

    for batch_start in trange(
        0, inputs_BP.shape[0], batch_size, desc="Batched Forward"
    ):
        batch_end = batch_start + batch_size
        batch_input_ids = inputs_BP[batch_start:batch_end].to(device)
        batch_mask = torch.ones_like(batch_input_ids)
        batch_inputs = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_mask,
        }

        with (
            torch.inference_mode(),
            model.trace(batch_inputs, scan=False, validate=False),
        ):
            for l, sm in enumerate(submodules):
                all_acts_LBPD[l, batch_start:batch_end] = sm.output[0].save()

    all_acts_LBPD = all_acts_LBPD.to("cpu")
    return all_acts_LBPD


def batch_sae_cache(
    sae, act_BPD: Tensor, batch_size: int = 100, device: str = "cuda"
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Process activations through SAE in batches.

    Args:
        sae: The SAE model
        act_BPD: Activations tensor [B, P, D]
        batch_size: Batch size for processing, initialize to 1 for getting fva for every token
        device: Device to use for computation

    Returns:
        tuple: (fvu_BP, latent_acts_BPS, latent_indices_BPK)
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
        if i == 0:
            print(f"batch_sae {batch_sae}")

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

    # Load dataset
    if cfg.dataset_name.endswith(".json"):
        inputs_BP, selected_story_idxs = collect_from_local(
            tokenizer=model.tokenizer,
            dataset_name=cfg.dataset_name,
            num_sentences=cfg.num_total_stories,
            num_tokens=cfg.num_tokens_per_story
        )
    else:
        inputs_BP, selected_story_idxs = collect_from_hf(
            tokenizer=model.tokenizer, 
            dataset_name=cfg.dataset_name, 
            num_stories=cfg.num_total_stories, 
            num_tokens=cfg.num_tokens_per_story,
            hf_text_identifier=cfg.hf_text_identifier
        )

    # Call batch_act_cache
    all_acts_LbPD = batch_llm_cache(
        model=model,
        submodules=submodules,
        inputs_BP=inputs_BP,
        hidden_dim=hidden_dim,
        batch_size=cfg.llm_batch_size,
        device=DEVICE,
    )

    # Save artifacts
    with open(os.path.join(INTERIM_DIR, f"activations_{cfg.input_file_str}.pt"), "wb") as f:
        torch.save(all_acts_LbPD, f, pickle_protocol=5)
    with open(os.path.join(INTERIM_DIR, f"story_idxs_{cfg.input_file_str}.pt"), "wb") as f:
        torch.save(selected_story_idxs, f, pickle_protocol=5)
    with open(os.path.join(INTERIM_DIR, f"tokens_{cfg.input_file_str}.pt"), "wb") as f:
        torch.save(inputs_BP, f, pickle_protocol=5)

    return all_acts_LbPD, selected_story_idxs, inputs_BP