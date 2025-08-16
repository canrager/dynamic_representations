import os
import torch as th
import json
from nnsight import LanguageModel
from tqdm import trange
from typing import List, Tuple
from torch import Tensor
from torch.nn import Module
from transformers import BatchEncoding
from datasets import load_dataset
import gc

from src.model_utils import load_nnsight_model
from src.project_config import DEVICE, MODELS_DIR, INTERIM_DIR, INPUTS_DIR

def tokenize_from_sequences(tokenizer, sequences):
    # Tokenize a list of sequences, using all tokens, no max lenght
    tokenized = tokenizer.batch_encode_plus(
        sequences,
        padding=True,
        padding_side="right",
        truncation=True,
        return_tensors="pt",
    )
    inputs_BP = tokenized["input_ids"]
    masks_BP = tokenized["attention_mask"]
    selected_story_idxs = list(range(len(sequences)))
    return inputs_BP, masks_BP, selected_story_idxs

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

    inputs_BP = th.stack(inputs_BP)
    masks_BP = th.ones_like(inputs_BP)

    return inputs_BP, masks_BP, selected_story_idxs


def collect_from_local_with_length_filter(tokenizer, dataset_name, num_sentences, num_tokens):
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
    inputs_BP = th.stack(inputs_BP)
    masks_BP = th.ones_like(inputs_BP)
    return inputs_BP, masks_BP, selected_sentence_idxs

def tokenize_with_pad_from_local(tokenizer, dataset_name, num_sentences=None, num_tokens=64):
    # Num tokens == all
    with open(os.path.join(INPUTS_DIR, dataset_name), "r") as f:
        all_sentences = json.load(f)["sentences"]
    print(f"Loaded {len(all_sentences)} sentences.")

    if num_sentences is not None:
        all_sentences = all_sentences[:num_sentences]

    tokenized = tokenizer.batch_encode_plus(
        all_sentences,
        padding=True,
        padding_side="right",
        truncation=True,
        max_length=num_tokens,
        return_tensors="pt",
    )

    inputs_BP = tokenized["input_ids"]
    masks_BP = tokenized["attention_mask"]
    selected_sentence_idxs = list(range(len(all_sentences)))
    return inputs_BP, masks_BP, selected_sentence_idxs

def collect_from_local(tokenizer, dataset_name, num_sentences, num_tokens):
    if num_tokens is not None:
        return collect_from_local_with_length_filter(tokenizer, dataset_name, num_sentences, num_tokens)
    else:
        return tokenize_with_pad_from_local(tokenizer, dataset_name, num_sentences)


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
    all_acts_LBPD = th.zeros(
        (
            len(submodules),
            inputs_BP.shape[0],
            inputs_BP.shape[1],
            hidden_dim,
        )
    )
    all_masks_BP = th.zeros(
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
            print(f"computing activations for batch {batch_start} to {batch_end}. Tokens:")
            for ids_P in batch_input_ids:
                decoded_tokens = [model.tokenizer.decode(id, skip_special_tokens=False) for id in ids_P]
                print(decoded_tokens)
                print()

        all_masks_BP[batch_start:batch_end] = batch_mask
        with (
            th.inference_mode(),
            model.trace(batch_inputs, scan=False, validate=False),
        ):
            for l, sm in enumerate(submodules):
                all_acts_LBPD[l, batch_start:batch_end] = sm.output[0].save()

    all_acts_LBPD = all_acts_LBPD.to("cpu")
    all_masks_BP = all_masks_BP.to("cpu")
    return all_acts_LBPD, all_masks_BP


def batch_sae_cache_eleuther(
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
        batch_sae = sae(batch)
        out_flattened.append(batch_sae.sae_out.detach().cpu())
        fvu_flattened.append(batch_sae.fvu.detach().cpu())
        latent_acts_flattened.append(batch_sae.latent_acts.detach().cpu())
        latent_indices_flattened.append(batch_sae.latent_indices.detach().cpu())

    out_flattened = th.stack(out_flattened)
    fvu_flattened = th.stack(fvu_flattened)
    latent_acts_flattened = th.stack(latent_acts_flattened)
    latent_indices_flattened = th.stack(latent_indices_flattened)

    out = out_flattened.reshape(B, P, D)
    fvu = fvu_flattened.reshape(B, P)
    latent_acts = latent_acts_flattened.reshape(B, P, K)
    latent_indices = latent_indices_flattened.reshape(B, P, K)

    return out, latent_acts, latent_indices


def batch_sae_cache_saebench(
    sae, act_BPD: Tensor, batch_size: int = 100, device: str = "cuda"
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Process activations through SAE in batches.
    """
    B, P, D = act_BPD.shape

    recons = []
    latent_actss = []

    for i in trange(0, B, batch_size, desc="SAE forward"):
        batch = act_BPD[i : i + batch_size]
        batch = batch.to(device)
        recon_BPD, sae_act_BPS = sae.forward(batch, output_features=True)
        recons.append(recon_BPD.detach().cpu())
        latent_actss.append(sae_act_BPS.detach().cpu())

    recons = th.cat(recons, dim=0)
    latent_actss = th.cat(latent_actss, dim=0)
    latent_indicess = None # TODO

    return recons, latent_actss, latent_indicess


def batch_sae_cache(sae, act_BPD, sae_cfg):
    if "eleuther" in sae_cfg.name.lower():
        forward_output = batch_sae_cache_eleuther(sae, act_BPD, sae_cfg.batch_size, DEVICE)
    elif any([name in sae_cfg.name.lower() for name in ["saebench", "neurons"]]):
        forward_output = batch_sae_cache_saebench(sae, act_BPD, sae_cfg.batch_size, DEVICE)
    else:
        raise ValueError("SAE distribution unknown")
    
    return forward_output


def compute_llm_artifacts(cfg, loaded_dataset_sequences=None):
    # Load model
    model, submodules, hidden_dim = load_nnsight_model(cfg.llm)

    # Load dataset
    if loaded_dataset_sequences is not None:
        inputs_BP, masks_BP, selected_story_idxs = tokenize_from_sequences(
            tokenizer=model.tokenizer,
            sequences=loaded_dataset_sequences
        )
    elif cfg.dataset.name.endswith(".json"):
        inputs_BP, masks_BP, selected_story_idxs = collect_from_local(
            tokenizer=model.tokenizer,
            dataset_name=cfg.dataset.name,
            num_sentences=cfg.num_total_stories,
            num_tokens=cfg.num_tokens_per_story
        )
    else:
        inputs_BP, masks_BP, selected_story_idxs = collect_from_hf(
            tokenizer=model.tokenizer, 
            dataset_name=cfg.dataset.name, 
            num_stories=cfg.num_total_stories, 
            num_tokens=cfg.num_tokens_per_story,
            hf_text_identifier=cfg.dataset.hf_text_identifier
        )

    # Call batch_act_cache
    all_acts_LbPD, all_masks_BP = batch_llm_cache(
        model=model,
        submodules=submodules,
        inputs_BP=inputs_BP,
        masks_BP=masks_BP,
        hidden_dim=hidden_dim,
        batch_size=cfg.llm.batch_size,
        device=DEVICE,
        debug=cfg.debug,
    )

    # Save artifacts
    if cfg.save_artifacts:
        with open(os.path.join(INTERIM_DIR, f"activations_{hash(cfg)}.pt"), "wb") as f:
            th.save(all_acts_LbPD, f, pickle_protocol=5)
        with open(os.path.join(INTERIM_DIR, f"story_idxs_{hash(cfg)}.pt"), "wb") as f:
            th.save(selected_story_idxs, f, pickle_protocol=5)
        with open(os.path.join(INTERIM_DIR, f"tokens_{hash(cfg)}.pt"), "wb") as f:
            th.save(inputs_BP, f, pickle_protocol=5)
        with open(os.path.join(INTERIM_DIR, f"masks_{hash(cfg)}.pt"), "wb") as f:
            th.save(all_masks_BP, f, pickle_protocol=5)

    # Memory cleanup
    del model
    th.cuda.empty_cache()
    gc.collect()

    return all_acts_LbPD, all_masks_BP, inputs_BP, selected_story_idxs