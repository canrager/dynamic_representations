import os
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from tqdm import trange
from sparsify import Sae

from src.project_config import ARTIFACTS_DIR, MODELS_DIR
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoTokenizer


def load_activations(
    model_name: str,
    num_stories: int,
    story_idxs: Optional[List[int]] = None,
    omit_BOS_token: bool = False,
    truncate_to_min_seq_length: bool = False,
    truncate_seq_length: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[int]]:
    """
    Load activations and attention mask tensors for a given model and number of stories.

    Args:
        model_name: Name of the model (e.g., "openai-community/gpt2")
        num_stories: Number of stories used to generate activations

    Returns:
        tuple: (activations tensor, attention mask tensor)
    """
    model_str = model_name.replace("/", "--")
    acts_save_fname = f"activations_{model_str}_simple-stories_first-{num_stories}.pt"
    mask_save_fname = f"mask_{model_str}_simple-stories_first-{num_stories}.pt"

    acts_path = os.path.join(ARTIFACTS_DIR, acts_save_fname)
    mask_path = os.path.join(ARTIFACTS_DIR, mask_save_fname)

    attention_mask_BP = torch.load(mask_path, weights_only=False).to("cpu")
    activations_LBPD = torch.load(acts_path, weights_only=False).to("cpu")

    if story_idxs is not None:
        activations_LBPD = activations_LBPD[:, story_idxs, :, :]
        attention_mask_BP = attention_mask_BP[story_idxs, :]

    if omit_BOS_token:
        activations_LBPD = activations_LBPD[:, :, 1:, :]
        attention_mask_BP = attention_mask_BP[:, 1:]

    if truncate_seq_length is not None:
        activations_LBPD = activations_LBPD[:, :, :truncate_seq_length, :]
        attention_mask_BP = attention_mask_BP[:, :truncate_seq_length]

    if truncate_to_min_seq_length:
        assert truncate_seq_length is None, "Either truncate_seq_length or truncate_to_min_seq_length can be passed, not both."
        min_seq_length = None
        min_seq_length = min(attention_mask_BP.sum(dim=-1))
        print(f"truncating to minimum sequence length of {min_seq_length}")
        activations_LBPD = activations_LBPD[:, :, :min_seq_length, :]
        attention_mask_BP = attention_mask_BP[:, :min_seq_length]
        assert torch.all(
            attention_mask_BP == 1
        ), "Attention mask should be all ones after truncation"

    # Reshape activations to [L, B*P, D] and mask to [B*P]
    L, B, P, D = activations_LBPD.shape
    activations_LbD = activations_LBPD.reshape(L, B * P, D)
    attention_mask_b = attention_mask_BP.reshape(B * P)

    # Create a mask for valid positions (where attention_mask == 1)
    valid_positions = attention_mask_b == 1
    activations_LbD = activations_LbD[:, valid_positions, :]

    return activations_LbD, activations_LBPD, attention_mask_BP


def load_mask_only(
    model_name: str,
    num_stories: int,
    story_idxs: Optional[List[int]] = None,
    omit_BOS_token: bool = True,
    max_tokens_per_story: Optional[int] = None,
    truncate_seq_length: bool = False,
) -> Tuple[Tensor, Optional[int]]:
    """
    Load and preprocess only the attention mask for the specified stories.

    Args:
        model_name: Name of the model
        num_stories: Number of stories to load
        story_idxs: Optional list of story indices to load
        omit_BOS_token: Whether to omit the BOS token
        max_tokens_per_story: Optional maximum number of tokens per story
        truncate_seq_length: Whether to truncate to minimum sequence length

    Returns:
        Tuple containing:
        - attention_mask_BP: Attention mask tensor [B, P]
        - min_seq_length: Minimum sequence length if truncate_seq_length is True, None otherwise
    """
    model_str = model_name.replace("/", "--")
    mask_save_fname = f"mask_{model_str}_simple-stories_first-{num_stories}.pt"
    mask_path = os.path.join(ARTIFACTS_DIR, mask_save_fname)

    attention_mask_BP = torch.load(mask_path, weights_only=False).to("cpu")

    if story_idxs is not None:
        attention_mask_BP = attention_mask_BP[story_idxs, :]

    if omit_BOS_token:
        attention_mask_BP = attention_mask_BP[:, 1:]

    if max_tokens_per_story is not None:
        attention_mask_BP = attention_mask_BP[:, :max_tokens_per_story]

    min_seq_length = None
    if truncate_seq_length:
        min_seq_length = min(attention_mask_BP.sum(dim=-1))
        print(f"truncating to minimum sequence length of {min_seq_length}")
        attention_mask_BP = attention_mask_BP[:, :min_seq_length]
        assert torch.all(
            attention_mask_BP == 1
        ), "Attention mask should be all ones after truncation"

    return attention_mask_BP, min_seq_length


def save_svd_results(
    U_LbC: Tensor,
    S_LC: Tensor,
    Vt_LCD: Tensor,
    means_LD: Tensor,
    model_name: str,
    num_stories: int,
    layer_idx: Optional[int] = None,
) -> None:
    """
    Save SVD results to disk.

    Args:
        U_LbC: Left singular vectors tensor [L, b, C]
        S_LC: Singular values tensor [L, C]
        Vt_LCD: Right singular vectors tensor [L, C, D]
        means_LD: Mean values used for centering [L, D]
        model_name: Name of the model
        num_stories: Number of stories used
        layer_idx: Optional layer index for single-layer SVD
    """
    model_str = model_name.replace("/", "--")
    if layer_idx is not None:
        svd_save_fname = (
            f"svd_{model_str}_simple-stories_first-{num_stories}_layer-{layer_idx}.pt"
        )
    else:
        svd_save_fname = f"svd_{model_str}_simple-stories_first-{num_stories}.pt"
    svd_path = os.path.join(ARTIFACTS_DIR, svd_save_fname)

    svd_data = {
        "U": U_LbC,
        "S": S_LC,
        "Vt": Vt_LCD,
        "means": means_LD,
        "model_name": model_name,
        "num_stories": num_stories,
        "layer_idx": layer_idx,
    }

    torch.save(svd_data, svd_path)
    print(f"SVD results saved to: {svd_path}")


def load_svd_results(
    model_name: str, num_stories: int, layer_idx: Optional[int] = None
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Load SVD results from disk if they exist.

    Args:
        model_name: Name of the model
        num_stories: Number of stories used
        layer_idx: Optional layer index for single-layer SVD

    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) if file exists, None otherwise
    """
    model_str = model_name.replace("/", "--")
    if layer_idx is not None:
        svd_save_fname = (
            f"svd_{model_str}_simple-stories_first-{num_stories}_layer-{layer_idx}.pt"
        )
    else:
        svd_save_fname = f"svd_{model_str}_simple-stories_first-{num_stories}.pt"
    svd_path = os.path.join(ARTIFACTS_DIR, svd_save_fname)

    if os.path.exists(svd_path):
        try:
            svd_data = torch.load(svd_path, weights_only=False)
            print(f"SVD results loaded from: {svd_path}")
            # Handle backwards compatibility - old files might not have means
            if "means" in svd_data:
                return svd_data["U"], svd_data["S"], svd_data["Vt"], svd_data["means"]
            else:
                print(
                    "Warning: Loaded SVD file does not contain means. PCA may be incorrect."
                )
                # Return None for means to indicate they need to be recomputed
                return svd_data["U"], svd_data["S"], svd_data["Vt"], None
        except Exception as e:
            print(f"Error loading SVD results: {e}")
            return None
    else:
        print(f"No saved SVD results found at: {svd_path}")
        return None


def compute_or_load_svd(
    act_LbD: Tensor,
    model_name: str,
    num_stories: int,
    force_recompute: bool = False,
    layer_idx: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SVD or load from disk if available.

    Args:
        act_LbD: Activations tensor [L, b, D]
        model_name: Name of the model
        num_stories: Number of stories used
        force_recompute: If True, recompute SVD even if saved results exist
        layer_idx: Optional layer index to compute SVD for single layer only
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) where C = min(b, D)
               If layer_idx is specified, L dimension becomes singleton
    """
    if not force_recompute:
        # Try to load existing SVD results
        svd_results = load_svd_results(model_name, num_stories, layer_idx)
        if svd_results is not None and svd_results[3] is not None:
            return svd_results

    # Compute SVD layer by layer on GPU for efficiency and memory management
    if layer_idx is not None:
        print(f"Computing SVD for single layer {layer_idx} with proper centering...")
    else:
        print("Computing SVD layer by layer with proper centering...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    L, b, D = act_LbD.shape
    C = min(b, D)  # Number of components

    # Determine which layers to process
    if layer_idx is not None:
        # Validate layer_idx
        if layer_idx < 0 or layer_idx >= L:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {L-1}]")
        layers_to_process = [layer_idx]
        # Initialize tensors with singleton L dimension for single layer
        U_LbC = torch.zeros(1, b, C)
        S_LC = torch.zeros(1, C)
        Vt_LCD = torch.zeros(1, C, D)
        means_LD = torch.zeros(1, D)
    else:
        layers_to_process = list(range(L))
        # Initialize tensors with full L dimension for all layers
        U_LbC = torch.zeros(L, b, C)
        S_LC = torch.zeros(L, C)
        Vt_LCD = torch.zeros(L, C, D)
        means_LD = torch.zeros(L, D)

    # Process each layer
    for i in trange(len(layers_to_process), desc="Computing SVD per layer"):
        actual_layer = layers_to_process[i]
        print(f'{i}. computing SVD for layer {actual_layer}')
        act_bD = act_LbD[actual_layer].to(device) 

        # Center the data before SVD (proper PCA)
        mean_D = torch.mean(act_bD, dim=0, keepdim=True)
        act_centered_bD = act_bD - mean_D

        # Compute SVD
        U_layer, S_layer, Vt_layer = torch.linalg.svd(
            act_centered_bD, full_matrices=False
        )

        # Store results
        U_LbC[i] = U_layer.cpu()
        S_LC[i] = S_layer.cpu()
        Vt_LCD[i] = Vt_layer.cpu()
        means_LD[i] = mean_D.cpu()

        # Clear GPU memory after each layer
        del act_bD, mean_D, act_centered_bD, U_layer, S_layer, Vt_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the results
    save_svd_results(U_LbC, S_LC, Vt_LCD, means_LD, model_name, num_stories, layer_idx)

    return U_LbC, S_LC, Vt_LCD, means_LD


def load_tokens_of_story(
    story_idx: int,
    model_name: str,
    omit_BOS_token: bool = False,
    seq_length: Optional[int] = None,
    max_tokens_per_story: Optional[int] = None,
) -> List[str]:
    dname = "SimpleStories/SimpleStories"
    all_stories = load_dataset(path=dname, cache_dir=MODELS_DIR)["train"]
    story_str = all_stories[story_idx]["story"]

    if "gpt2" in model_name:
        # Language Model loads the AutoTokenizer, which does not use the add_bos_token method.
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
        tokenizer.add_bos_token = True
        tokenizer.pad_token = tokenizer.eos_token
    elif "Llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown model: {model_name}")

    token_ids_L = tokenizer(
        story_str,
        padding=True,
        padding_side="right",
        truncation=False,
    ).input_ids

    if omit_BOS_token:
        token_ids_L = token_ids_L[1:]

    if seq_length is not None:
        token_ids_L = token_ids_L[:seq_length]

    if max_tokens_per_story is not None:
        token_ids_L = token_ids_L[:max_tokens_per_story]

    token_str_L = [tokenizer.decode(t, skip_special_tokens=False) for t in token_ids_L]
    return token_str_L


def load_tokens_of_stories(
    story_idxs: List[int],
    model_name: str,
    omit_BOS_token: bool = False,
    seq_length: Optional[int] = None,
    max_tokens_per_story: Optional[int] = None,
) -> List[List[str]]:
    tokens = []
    for story_idx in story_idxs:
        tokens.append(
            load_tokens_of_story(
                story_idx, model_name, omit_BOS_token, seq_length, max_tokens_per_story
            )
        )
    return tokens


def save_sae_results(
    fvu_BP: Tensor,
    latent_acts_BPS: Tensor,
    latent_indices_BPK: Tensor,
    model_name: str,
    num_stories: int,
    sae_name: str,
    layer_idx: int,
) -> None:
    """
    Save SAE results to disk.

    Args:
        fvu_BP: Fraction of variance unexplained tensor [B, P]
        latent_acts_BPS: Latent activations tensor [B, P, S]
        latent_indices_BPK: Latent indices tensor [B, P, K]
        model_name: Name of the model
        num_stories: Number of stories used
        sae_name: Name of the SAE
        layer_idx: Layer index
    """
    model_str = model_name.replace("/", "--")
    sae_str = sae_name.replace("/", "--")
    sae_save_fname = f"sae_{model_str}_{sae_str}_layer-{layer_idx}_simple-stories_first-{num_stories}.pt"
    sae_path = os.path.join(ARTIFACTS_DIR, sae_save_fname)

    sae_data = {
        "fvu": fvu_BP,
        "latent_acts": latent_acts_BPS,
        "latent_indices": latent_indices_BPK,
        "model_name": model_name,
        "num_stories": num_stories,
        "sae_name": sae_name,
        "layer_idx": layer_idx,
    }

    torch.save(sae_data, sae_path)
    print(f"SAE results saved to: {sae_path}")


def load_sae_results(
    model_name: str, num_stories: int, sae_name: str, layer_idx: int
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """
    Load SAE results from disk if they exist.

    Args:
        model_name: Name of the model
        num_stories: Number of stories used
        sae_name: Name of the SAE
        layer_idx: Layer index

    Returns:
        tuple: (fvu_LBP, latent_acts_LBPS, latent_indices_LBPK) if file exists, None otherwise
    """
    model_str = model_name.replace("/", "--")
    sae_str = sae_name.replace("/", "--")
    sae_save_fname = f"sae_{model_str}_{sae_str}_layer-{layer_idx}_simple-stories_first-{num_stories}.pt"
    sae_path = os.path.join(ARTIFACTS_DIR, sae_save_fname)

    if os.path.exists(sae_path):
        try:
            sae_data = torch.load(sae_path, weights_only=False)
            print(f"SAE results loaded from: {sae_path}")
            return sae_data["fvu"], sae_data["latent_acts"], sae_data["latent_indices"]
        except Exception as e:
            print(f"Error loading SAE results: {e}")
            return None
    else:
        print(f"No saved SAE results found at: {sae_path}")
        return None


def batch_sae_forward(
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
    fvu_flattened = []
    latent_acts_flattened = []
    latent_indices_flattened = []

    for i in trange(0, len(act_flattened), batch_size, desc="SAE forward"):
        batch = act_flattened[i : i + batch_size]
        batch = batch.to(device)
        batch_sae = sae.forward(batch)
        fvu_flattened.append(batch_sae.fvu.detach().cpu())
        latent_acts_flattened.append(batch_sae.latent_acts.detach().cpu())
        latent_indices_flattened.append(batch_sae.latent_indices.detach().cpu())
        if i == 0:
            print(f"batch_sae {batch_sae}")

    fvu_flattened = torch.cat(fvu_flattened, dim=0)
    latent_acts_flattened = torch.cat(latent_acts_flattened, dim=0)
    latent_indices_flattened = torch.cat(latent_indices_flattened, dim=0)

    fvu = fvu_flattened.reshape(B, P)
    latent_acts = latent_acts_flattened.reshape(B, P, K)
    latent_indices = latent_indices_flattened.reshape(B, P, K)

    return fvu, latent_acts, latent_indices


def compute_or_load_sae(
    sae_name: str,
    model_name: str,
    num_stories: int,
    layer_idx: int,
    batch_size: int = 100,
    device: str = "cuda",
    force_recompute: bool = False,
    do_omit_BOS_token: bool = True,
    do_truncate_seq_length: bool = False,
    max_tokens_per_story: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute SAE activations or load from disk if available.

    Args:
        sae: The SAE model
        act_BPD: Activations tensor [B, P, D]
        model_name: Name of the model
        num_stories: Number of stories used
        sae_name: Name of the SAE
        layer_idx: Layer index
        batch_size: Batch size for processing
        device: Device to use for computation
        force_recompute: If True, recompute SAE even if saved results exist

    Returns:
        tuple: (fvu_BP, latent_acts_BPS, latent_indices_BPK)
    """
    if not force_recompute:
        # Try to load existing SAE results
        sae_results = load_sae_results(model_name, num_stories, sae_name, layer_idx)
        if sae_results is not None:
            fvu_BP, latent_acts_BPS, latent_indices_BPK = sae_results
            mask_BP, _ = load_mask_only(
                model_name,
                num_stories,
                story_idxs=None,
                omit_BOS_token=do_omit_BOS_token,
                truncate_seq_length=do_truncate_seq_length,
                max_tokens_per_story=max_tokens_per_story,
            )
            return fvu_BP, latent_acts_BPS, latent_indices_BPK, mask_BP

    _, act_LBPD, mask_BP, _ = load_activations(
        model_name,
        num_stories,
        story_idxs=None,
        omit_BOS_token=do_omit_BOS_token,
        truncate_to_min_seq_length=do_truncate_seq_length,
        truncate_seq_length=max_tokens_per_story,
    )
    act_BPD = act_LBPD[layer_idx]

    sae = Sae.load_from_hub(sae_name, hookpoint=f"layers.{layer_idx}")
    sae = sae.to(device)
    print(sae.cfg)

    # Compute SAE activations
    print("Computing SAE activations...")
    fvu, latent_acts, latent_indices = batch_sae_forward(
        sae, act_BPD, batch_size, device
    )

    # Save the results
    save_sae_results(
        fvu, latent_acts, latent_indices, model_name, num_stories, sae_name, layer_idx
    )

    return fvu, latent_acts, latent_indices, mask_BP
