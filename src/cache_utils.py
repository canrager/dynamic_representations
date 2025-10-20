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
from collections import defaultdict

from src.model_utils import load_nnsight_model
from src.project_config import DEVICE, MODELS_DIR, INTERIM_DIR, INPUTS_DIR

DTYPE_STR_TO_TORCH = {
    "float32": th.float32,
    "bfloat16": th.bfloat16
}

def tokenize_from_sequences(tokenizer, sequences, max_length: int = None):
    # Tokenize a list of sequences, using all tokens, no max lenght
    tokenized = tokenizer.batch_encode_plus(
        sequences,
        padding=True,
        padding_side="right",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs_BP = tokenized["input_ids"]
    masks_BP = tokenized["attention_mask"]
    selected_story_idxs = list(range(len(sequences)))
    return inputs_BP, masks_BP, selected_story_idxs

def collect_from_hf(tokenizer, dataset_name, num_stories, num_tokens):
    # Use stories with the same amounts of tokens
    # For equal weighting across position and avoiding padding errors
    # NOTE: exact tokenization varies by model, therefore, it can be that different models see different stories

    if dataset_name == "SimpleStories/SimpleStories":
        hf_text_identifier = "story"
        hf_split_identifier = "train"
        do_chat_template = False

    elif dataset_name == "HuggingFaceH4/ultrachat_200k":
        hf_text_identifier = "messages"
        hf_split_identifier = "train_sft"
        do_chat_template = True

    else:
        hf_text_identifier = "text"
        hf_split_identifier = "train"
        do_chat_template = False


    all_stories = load_dataset(path=dataset_name, cache_dir=MODELS_DIR, streaming=True)[hf_split_identifier]

    inputs_BP = []
    selected_story_idxs = []

    for story_idx, story_item in enumerate(all_stories):
        if len(inputs_BP) >= num_stories:
            break

        story_text = story_item[hf_text_identifier]

        if do_chat_template:
            input_ids_P = tokenizer.apply_chat_template(story_text, return_tensors="pt")
        else:
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


# def batch_llm_cache(
#     model: LanguageModel,
#     submodule: Module,
#     inputs_BP: Tensor,
#     masks_BP: Tensor,
#     hidden_dim: int,
#     batch_size: int,
#     device: str,
#     dtype: str,
#     debug: bool = False,
# ) -> Tensor:
    
#     dtype = DTYPE_STR_TO_TORCH[dtype]
#     all_acts_BPD = th.zeros(
#         (
#             inputs_BP.shape[0],
#             inputs_BP.shape[1],
#             hidden_dim,
#         ), dtype=dtype, device="cpu"
#     )
#     all_masks_BP = th.zeros(
#         (
#             inputs_BP.shape[0],
#             inputs_BP.shape[1],
#         ), dtype=th.int, device="cpu"
#     )

#     for batch_start in trange(
#         0, inputs_BP.shape[0], batch_size, desc="Batched Forward"
#     ):
#         batch_end = batch_start + batch_size
#         batch_input_ids = inputs_BP[batch_start:batch_end].to(device)
#         print(f"==>> batch_input_ids.shape: {batch_input_ids.shape}")
#         batch_mask = masks_BP[batch_start:batch_end].to(device)
#         batch_inputs = {
#             "input_ids": batch_input_ids,
#             "attention_mask": batch_mask,
#         }
#         all_masks_BP[batch_start:batch_end] = batch_mask

#         if debug:
#             print(f"computing activations for batch {batch_start} to {batch_end}. Tokens:")
#             for ids_P in batch_input_ids:
#                 decoded_tokens = [model.tokenizer.decode(id, skip_special_tokens=False) for id in ids_P]
#                 print(decoded_tokens)
#                 print()
            

#         with th.inference_mode():
#             with model.trace(batch_inputs, scan=False, validate=False):
#                 out = submodule.output
#                 if isinstance(out, tuple):
#                     out = out[0]
#                 out.save()
#         all_acts_BPD[batch_start:batch_end] = out.to("cpu")

#         del batch_input_ids, batch_mask, batch_inputs, out
#         th.cuda.empty_cache()
#         gc.collect()

#     return all_acts_BPD, all_masks_BP

def batch_llm_cache(
    model: LanguageModel,
    submodule: Module,
    inputs_BP: Tensor,
    masks_BP: Tensor,
    hidden_dim: int,
    batch_size: int,
    device: str,
    dtype: str,
    debug: bool = False,
    cache_attn: bool = None,
    num_heads: bool = None
) -> Tensor:
    
    dtype = DTYPE_STR_TO_TORCH[dtype]
    all_acts_BPD = th.zeros(
        (
            inputs_BP.shape[0],
            inputs_BP.shape[1],
            hidden_dim,
        ), dtype=dtype, device="cpu"
    )
    all_masks_BP = th.zeros(
        (
            inputs_BP.shape[0],
            inputs_BP.shape[1],
        ), dtype=th.int, device="cpu"
    )

    all_attn_patterns_BHPP = None
    if cache_attn:
        all_attn_patterns_BHPP = th.zeros(
            (
                inputs_BP.shape[0],
                num_heads,
                inputs_BP.shape[1],
                inputs_BP.shape[1],
            ), dtype=th.int, device="cpu"
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
        all_masks_BP[batch_start:batch_end] = batch_mask

        if debug:
            print(f"computing activations for batch {batch_start} to {batch_end}. Tokens:")
            for ids_P in batch_input_ids:
                decoded_tokens = [model.tokenizer.decode(id, skip_special_tokens=False) for id in ids_P]
                print(decoded_tokens)
                print()
            

        with th.inference_mode():
            with model.trace(batch_inputs, scan=False, validate=False, output_attentions=cache_attn):
                if cache_attn:
                    attn_pattern_bHPP = submodule.self_attn.output[1].save()
                out = submodule.output
                if isinstance(out, tuple):
                    out = out[0]
                out.save()
        all_acts_BPD[batch_start:batch_end] = out.to("cpu")
        if cache_attn:
            all_attn_patterns_BHPP[batch_start:batch_end] = attn_pattern_bHPP.to("cpu")

        del batch_input_ids, batch_mask, batch_inputs, out
        if cache_attn:
            del attn_pattern_bHPP
        th.cuda.empty_cache()
        gc.collect()

    return all_acts_BPD, all_masks_BP, all_attn_patterns_BHPP


def compute_llm_artifacts(cfg, model, submodule, loaded_dataset_sequences=None, cache_attn=False):
    # Load dataset
    if loaded_dataset_sequences is not None:
        inputs_BP, masks_BP, _ = tokenize_from_sequences(
            tokenizer=model.tokenizer,
            sequences=loaded_dataset_sequences
        )
    elif cfg.data.hf_name.endswith(".json"):
        inputs_BP, masks_BP, _ = collect_from_local(
            tokenizer=model.tokenizer,
            dataset_name=cfg.data.hf_name,
            num_sentences=cfg.data.num_sequences,
            num_tokens=cfg.data.context_length
        )
    else:
        inputs_BP, masks_BP, _ = collect_from_hf(
            tokenizer=model.tokenizer, 
            dataset_name=cfg.data.hf_name, 
            num_stories=cfg.data.num_sequences, 
            num_tokens=cfg.data.context_length,
        )

    # Call batch_act_cache
    all_acts_bPD, all_masks_BP, all_attn_patterns_BHPP = batch_llm_cache(
        model=model,
        submodule=submodule,
        inputs_BP=inputs_BP,
        masks_BP=masks_BP,
        hidden_dim=cfg.llm.hidden_dim,
        batch_size=cfg.llm.batch_size,
        device=cfg.env.device,
        dtype=cfg.env.dtype,
        debug=False,
        cache_attn=cache_attn,
        num_heads=model.config.num_attention_heads
    ) # all_attn_patterns_BHPP is None if cache_attn==False

    if cache_attn:
        return all_acts_bPD, all_masks_BP, inputs_BP, all_attn_patterns_BHPP
    else:
        return all_acts_bPD, all_masks_BP, inputs_BP


def batch_eleuther_sae_cache(
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


def batch_dictionarylearning_sae_cache(
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

@th.inference_mode
def batch_temporal_sae_cache(
    sae, act_BPD: Tensor, cfg
) -> Tuple[Tensor, Tensor, Tensor]:
    B, P, D = act_BPD.shape
    dtype = DTYPE_STR_TO_TORCH[cfg.env.dtype]

    results = defaultdict(list)
    for i in trange(0, B, cfg.sae.batch_size, desc="SAE forward"):
        batch = act_BPD[i : i + cfg.sae.batch_size]
        batch = batch.to(device=cfg.env.device, dtype=dtype)
        recon_BPD, batch_dict = sae.forward(batch, return_graph=True)
        residuals_BPD = batch - recon_BPD
        results["total_recons"].append(recon_BPD.detach().cpu())
        results["residuals"].append(residuals_BPD.detach().cpu())
        results["novel_codes"].append(batch_dict["novel_codes"].detach().cpu())
        results["novel_recons"].append(batch_dict["novel_recons"].detach().cpu())
        results["pred_codes"].append(batch_dict["pred_codes"].detach().cpu())
        results["pred_recons"].append(batch_dict["pred_recons"].detach().cpu())
        results["attn_graphs"].append(batch_dict["attn_graphs"].detach().cpu())

        # Add the reconstruction bias
        results["novel_recons_plus_b"].append(
            (batch_dict["novel_recons"] + sae.b).detach().cpu()
        )
        results["pred_recons_plus_b"].append(
            (batch_dict["pred_recons"] + sae.b).detach().cpu()
        )


    for key in results:
        results[key] = th.cat(results[key], dim=0)

    return results

@th.inference_mode
def batch_standard_sae_cache(
    sae, act_BPD: Tensor, cfg
) -> Tuple[Tensor, Tensor, Tensor]:
    B, P, D = act_BPD.shape
    dtype = DTYPE_STR_TO_TORCH[cfg.env.dtype]

    results = defaultdict(list)
    for i in trange(0, B, cfg.sae.batch_size, desc="SAE forward"):
        batch = act_BPD[i : i + cfg.sae.batch_size]
        batch = batch.to(device=cfg.env.device, dtype=dtype)
        b = batch.shape[0]
        batch_ND = batch.view(b * P, D)
        with th.inference_mode():
            recons_ND, codes_ND, _ = sae.forward(batch_ND, return_hidden=True)
        recons_BPD = recons_ND.view(b, P, D)
        codes_BPD = codes_ND.view(b, P, cfg.sae.dict_size)
        residuals_BPD = batch - recons_BPD

        results["codes"].append(codes_BPD.detach().cpu())
        results["recons"].append(recons_BPD.detach().cpu())
        results["residuals"].append(residuals_BPD.detach().cpu())

    for key in results:
        results[key] = th.cat(results[key], dim=0)

    return results


def batch_sae_cache(sae, act_BPD, cfg):
    if "eleuther" in cfg.sae.name.lower():
        forward_output = batch_eleuther_sae_cache(sae, act_BPD, cfg.sae.batch_size, cfg.env.device)
    elif "temporal" in cfg.sae.name.lower():
        forward_output = batch_temporal_sae_cache(sae, act_BPD, cfg)
    elif (not "temporal" in cfg.sae.name.lower()) and ("selftrain" in cfg.sae.local_weights_path.lower()):
        forward_output = batch_standard_sae_cache(sae, act_BPD, cfg)
    else:
        forward_output = batch_dictionarylearning_sae_cache(sae, act_BPD, cfg.sae.batch_size, cfg.env.device)
    
    return forward_output