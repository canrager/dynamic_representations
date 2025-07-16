import os
import json
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from tqdm import trange
from sparsify import Sae

from src.project_config import INTERIM_DIR, MODELS_DIR, INPUTS_DIR
from src.model_utils import load_tokenizer


def load_activations(
    model_name: str,
    num_stories: int,
    story_idxs: Optional[List[int]] = None,
    omit_BOS_token: bool = False,
    dataset_name: str = "SimpleStories/SimpleStories",
    num_tokens_per_story: Optional[int] = None,
    input_str: str = "",
) -> Tuple[Tensor, Tensor, List[int]]:
    """
    Load activations and attention mask tensors for a given model and number of stories.

    Args:
        model_name: Name of the model (e.g., "openai-community/gpt2")
        num_stories: Number of stories used to generate activations

    Returns:
        tuple: (activations tensor, attention mask tensor)
    """

    # Load indicies of stories in the full dataset
    actual_story_idxs_fname = f"story_idxs_{input_str}.pt"
    actual_story_idxs_path = os.path.join(INTERIM_DIR, actual_story_idxs_fname)
    actual_story_idxs = torch.load(actual_story_idxs_path, weights_only=False)

    # Load tokens of stories
    tokens_save_fname = f"tokens_{input_str}.pt"
    tokens_path = os.path.join(INTERIM_DIR, tokens_save_fname)
    tokens_BP = torch.load(tokens_path, weights_only=False)

    # Load activations
    acts_save_fname = f"activations_{input_str}.pt"
    acts_path = os.path.join(INTERIM_DIR, acts_save_fname)
    activations_LBPD = torch.load(acts_path, weights_only=False).to("cpu")

    if story_idxs is not None:
        actual_story_idxs = actual_story_idxs[story_idxs]
        activations_LBPD = activations_LBPD[:, story_idxs, :, :]

    if omit_BOS_token:
        activations_LBPD = activations_LBPD[:, :, 1:, :]

    if num_tokens_per_story is not None:
        activations_LBPD = activations_LBPD[:, :, :num_tokens_per_story, :]
        tokens_BP = tokens_BP[:, :num_tokens_per_story]

    return activations_LBPD, actual_story_idxs, tokens_BP

def load_activation_split(
        cfg,
    ):

        act_LBPD, dataset_story_idxs, tokens_BP = load_activations(
            cfg.model_name,
            cfg.num_total_stories,
            story_idxs=cfg.story_idxs,
            omit_BOS_token=cfg.omit_BOS_token,
            dataset_name=cfg.dataset_name,
            num_tokens_per_story=cfg.num_tokens_per_story,
            input_str=cfg.input_str,
        )

        # Do train-test split
        if cfg.do_train_test_split:
            rand_idxs = torch.randperm(cfg.num_total_stories)
            train_idxs = rand_idxs[:cfg.num_train_stories]
            test_idxs = rand_idxs[cfg.num_train_stories:]

            act_train_LBPD = act_LBPD[:, train_idxs, :, :]
            act_test_LBPD = act_LBPD[:, test_idxs, :, :]

            tokens_test_BP = [tokens_BP[i] for i in test_idxs]
            dataset_idxs_test = [dataset_story_idxs[i] for i in test_idxs]
            num_test_stories = len(test_idxs)
        else:
            act_train_LBPD = act_LBPD
            act_test_LBPD = act_LBPD
            tokens_test_BP = tokens_BP
            dataset_idxs_test = dataset_story_idxs
            num_test_stories = cfg.num_total_stories

        return act_train_LBPD, act_test_LBPD, tokens_test_BP, num_test_stories, dataset_idxs_test


def save_svd_results(
    U_LbC: Tensor,
    S_LC: Tensor,
    Vt_LCD: Tensor,
    means_LD: Tensor,
    model_name: str,
    dataset_name: str,
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
        dataset_name: Name of the dataset
        num_stories: Number of stories used
        layer_idx: Optional layer index for single-layer SVD
    """
    model_str = model_name.replace("/", "--")
    dataset_str = dataset_name.split("/")[-1].split(".")[0]
    if layer_idx is not None:
        svd_save_fname = (
            f"svd_{model_str}_{dataset_str}_first-{num_stories}_layer-{layer_idx}.pt"
        )
    else:
        svd_save_fname = f"svd_{model_str}_{dataset_str}_first-{num_stories}.pt"
    svd_path = os.path.join(INTERIM_DIR, svd_save_fname)

    svd_data = {
        "U": U_LbC,
        "S": S_LC,
        "Vt": Vt_LCD,
        "means": means_LD,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_stories": num_stories,
        "layer_idx": layer_idx,
    }

    torch.save(svd_data, svd_path)
    print(f"SVD results saved to: {svd_path}")


def load_svd_results(
    model_name: str,
    dataset_name: str,
    num_stories: int,
    layer_idx: Optional[int] = None,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Load SVD results from disk if they exist.

    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        num_stories: Number of stories used
        layer_idx: Optional layer index for single-layer SVD

    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) if file exists, None otherwise
    """
    model_str = model_name.replace("/", "--")
    dataset_str = dataset_name.split("/")[-1].split(".")[0]
    if layer_idx is not None:
        svd_save_fname = (
            f"svd_{model_str}_{dataset_str}_first-{num_stories}_layer-{layer_idx}.pt"
        )
    else:
        svd_save_fname = f"svd_{model_str}_{dataset_str}_first-{num_stories}.pt"
    svd_path = os.path.join(INTERIM_DIR, svd_save_fname)

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


def compute_centered_svd(
    act_LbD: Tensor,
    layer_idx: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SVD for a given layer of activations.
    """

    # Compute SVD layer by layer on GPU for efficiency and memory management
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        if layer_idx is not None:
            print(
                f"Computing SVD for single layer {layer_idx} with proper centering..."
            )
        else:
            print("Computing SVD layer by layer with proper centering...")

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
    else:  # All layers, starting at 0
        layers_to_process = list(range(L))
        # Initialize tensors with full L dimension for all layers
        U_LbC = torch.zeros(L, b, C)
        S_LC = torch.zeros(L, C)
        Vt_LCD = torch.zeros(L, C, D)
        means_LD = torch.zeros(L, D)

    # Process each layer
    for i in range(len(layers_to_process)):
        if verbose:
            print(f"{i}. computing SVD for layer {actual_layer}")
        actual_layer = layers_to_process[i]
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

    return U_LbC, S_LC, Vt_LCD, means_LD


def compute_or_load_svd(
    act_LBPD: Tensor,
    model_name: str,
    dataset_name: str,
    num_stories: int,
    force_recompute: bool = False,
    layer_idx: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SVD or load from disk if available.

    Args:
        act_LbD: Activations tensor [L, b, D]
        model_name: Name of the model
        dataset_name: Name of the dataset
        num_stories: Number of stories used
        force_recompute: If True, recompute SVD even if saved results exist
        layer_idx: Optional layer index to compute SVD for single layer only
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) where C = min(b, D)
               If layer_idx is specified, L dimension becomes singleton
    """
    if not force_recompute:
        # Try to load existing SVD results
        svd_results = load_svd_results(model_name, dataset_name, num_stories, layer_idx)
        if svd_results is not None and svd_results[3] is not None:
            return svd_results

    # Reshape activations to [L, B*P, D]
    L, B, P, D = act_LBPD.shape
    act_LbD = act_LBPD.reshape(L, B * P, D)

    U_LbC, S_LC, Vt_LCD, means_LD = compute_centered_svd(act_LbD, layer_idx)

    # Save the results
    save_svd_results(
        U_LbC, S_LC, Vt_LCD, means_LD, model_name, dataset_name, num_stories, layer_idx
    )

    return U_LbC, S_LC, Vt_LCD, means_LD


def load_tokens_of_story(
    dataset_name: str,
    dataset_num_stories: int,
    story_idx: int,
    model_name: str,
    omit_BOS_token: bool = False,
    seq_length: Optional[int] = None,
) -> List[str]:
    """
    Load tokens for a single story.

    Args:
        story_idx: Index of the story to load
        model_name: Name of the model to determine the tokenizer
        omit_BOS_token: Whether to omit the beginning-of-sequence (BOS) token
        seq_length: Optional sequence length to truncate or pad tokens

    Returns:
        A list of tokens for the specified story.
    """
    tokenizer = load_tokenizer(model_name, MODELS_DIR)
    model_str = model_name.replace("/", "--")
    dataset_str = dataset_name.split("/")[-1].split(".")[0]
    tokens_fname = f"tokens_{model_str}_{dataset_str}_samples{dataset_num_stories}.pt"
    inputs_BP = torch.load(os.path.join(INTERIM_DIR, tokens_fname), weights_only=False)
    tokens = [
        tokenizer.decode(t, skip_special_tokens=False) for t in inputs_BP[story_idx]
    ]

    if omit_BOS_token:
        # GPT2 doesn't add a BOS token, but other models like Llama do.
        # It's safer to check if the first token is a BOS token before removing it.
        if tokens and tokenizer.bos_token and tokens[0] == tokenizer.bos_token:
            tokens = tokens[1:]
        elif tokens and tokens[0] == "<s>":
            tokens = tokens[1:]

    if seq_length is not None:
        tokens = tokens[:seq_length]

    return tokens


def load_tokens_of_stories(
    story_idxs: List[int],
    model_name: str,
    omit_BOS_token: bool = False,
    seq_length: Optional[int] = None,
) -> List[List[str]]:
    tokens_of_stories = []
    for story_idx in story_idxs:
        tokens_of_stories.append(
            load_tokens_of_story(story_idx, model_name, omit_BOS_token, seq_length)
        )
    return tokens_of_stories


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
    sae_path = os.path.join(INTERIM_DIR, sae_save_fname)

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
    sae_path = os.path.join(INTERIM_DIR, sae_save_fname)

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
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute or load SAE results for a given model, layer, and number of stories.

    Args:
        sae_name: Name of the SAE model to load.
        model_name: Name of the model (e.g., "openai-community/gpt2").
        num_stories: Number of stories used for activations.
        layer_idx: Index of the layer to get activations from.
        batch_size: Batch size for SAE forward pass.
        device: Device to run computation on.
        force_recompute: If True, recompute SAE results even if they exist.
        do_omit_BOS_token: Whether to omit the BOS token from activations.

    Returns:
        Tuple containing:
        - fvu_BP: Frame variance unexplained tensor.
        - latent_acts_BPS: Latent activations tensor.
        - latent_indices_BPK: Latent indices tensor.
    """
    sae_results_exist = (
        load_sae_results(model_name, num_stories, sae_name, layer_idx) is not None
    )

    if force_recompute or not sae_results_exist:
        _, act_LBPD = load_activations(
            model_name=model_name,
            num_stories=num_stories,
            story_idxs=None,
            omit_BOS_token=do_omit_BOS_token,
        )
        act_BPD = act_LBPD[layer_idx]

        sae_path = os.path.join(MODELS_DIR, f"{sae_name}.pt")
        sae = Sae.load_from_pretrained(sae_path)
        sae = sae.to(device)

        fvu_BP, latent_acts_BPS, latent_indices_BPK = batch_sae_forward(
            sae, act_BPD, batch_size, device
        )

        save_sae_results(
            fvu_BP,
            latent_acts_BPS,
            latent_indices_BPK,
            model_name,
            num_stories,
            sae_name,
            layer_idx,
        )
    else:
        fvu_BP, latent_acts_BPS, latent_indices_BPK = load_sae_results(
            model_name, num_stories, sae_name, layer_idx
        )

    return fvu_BP, latent_acts_BPS, latent_indices_BPK
