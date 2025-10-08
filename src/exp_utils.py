import os
import json
import csv
from typing import Tuple, Optional, List, Dict, Any

from transformers import AutoTokenizer

import torch as th
from tqdm import trange
from dataclasses import dataclass
import gc
import numpy as np

from src.project_config import INTERIM_DIR, MODELS_DIR, INPUTS_DIR, DEVICE
from src.model_utils import load_tokenizer, load_sae
from src.cache_utils import compute_llm_artifacts, batch_sae_cache


def _match_word_labels_to_tokens(tokens_BP, word_lists, word_labels, cfg):
    """
    Match word-level labels to tokens using character-position based matching.

    This implementation:
    1. Joins words with spaces to reconstruct text
    2. Tokenizes the full text to get token positions
    3. Maps each token to its character span
    4. Maps character positions back to original words
    5. Assigns labels based on word ownership

    Args:
        tokens_BP: Token tensor [B, P]
        word_lists: List of word lists, one per sequence
        word_labels: List of label lists, one per word list
        cfg: Configuration object with tokenizer info

    Returns:
        token_labels_BP: Token label tensor [B, P] where each element is a string label
    """
    from src.model_utils import load_tokenizer

    tokenizer = load_tokenizer(cfg.llm.name, MODELS_DIR)
    B, P = tokens_BP.shape

    # Initialize with empty strings
    token_labels_BP = [["" for _ in range(P)] for _ in range(B)]

    for b in range(B):
        if b >= len(word_lists) or b >= len(word_labels):
            continue

        words = word_lists[b]
        labels = word_labels[b]

        if len(words) != len(labels):
            print(
                f"Warning: Mismatched word/label counts for sequence {b}: {len(words)} words, {len(labels)} labels"
            )
            continue

        # Reconstruct text by joining words with spaces
        text = " ".join(words)

        # Create character-to-word mapping
        char_to_word = {}
        char_pos = 0
        for word_idx, word in enumerate(words):
            # Skip space before first word
            if word_idx > 0:
                char_pos += 1  # space character

            # Map each character in this word to the word index
            word_start = char_pos
            word_end = char_pos + len(word)
            for char_idx in range(word_start, word_end):
                char_to_word[char_idx] = word_idx
            char_pos = word_end

        # Tokenize the full text to get character offsets
        # Use return_offsets_mapping to get character positions
        try:
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
            token_ids = encoded["input_ids"]
            offsets = encoded.get("offset_mapping", [])

            # Match our pre-computed token IDs with the fresh tokenization
            for p in range(min(P, len(token_ids))):
                if p < len(offsets) and offsets[p] is not None:
                    start_char, end_char = offsets[p]

                    # Find which word(s) this token spans
                    if start_char < len(text):
                        # Use the start character to determine word ownership
                        if start_char in char_to_word:
                            word_idx = char_to_word[start_char]
                            if word_idx < len(labels):
                                token_labels_BP[b][p] = labels[word_idx]
                        else:
                            # Token might start with a space, try next character
                            if start_char + 1 in char_to_word:
                                word_idx = char_to_word[start_char + 1]
                                if word_idx < len(labels):
                                    token_labels_BP[b][p] = labels[word_idx]

        except Exception as e:
            print(
                f"Warning: Could not get offsets for sequence {b}, falling back to simple matching: {e}"
            )
            # Fallback: simple word-by-word tokenization approach
            _match_tokens_fallback(token_labels_BP[b], tokens_BP[b], words, labels, tokenizer, P)

    return token_labels_BP


def _match_tokens_fallback(token_labels_P, tokens_P, words, labels, tokenizer, P):
    """
    Fallback method for token-label matching when offset mapping is not available.
    Uses incremental word-by-word tokenization.
    """
    token_idx = 0

    for word_idx, (word, label) in enumerate(zip(words, labels)):
        # Tokenize this word individually (without special tokens)
        word_tokens = tokenizer(word, add_special_tokens=False)["input_ids"]

        # Assign this label to the next N tokens
        for _ in range(len(word_tokens)):
            if token_idx < P:
                # Skip special tokens at the beginning
                while token_idx < P and tokens_P[token_idx].item() in [
                    tokenizer.bos_token_id,
                    tokenizer.cls_token_id,
                ]:
                    token_idx += 1

                if token_idx < P:
                    token_labels_P[token_idx] = label
                    token_idx += 1


def compute_or_load_llm_artifacts(
    cfg, loaded_dataset_sequences=None, loaded_word_lists=None, loaded_word_labels=None
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Optional[List[List[str]]], List[int]]:
    """
    Load activations and attention mask tensors for a given model and number of stories.
    Optionally handles word-level labels that need to be matched to tokens.

    Args:
        cfg: Configuration object
        loaded_dataset_sequences: Pre-loaded dataset sequences (optional)
        loaded_word_lists: List of word lists, where each word list corresponds to a sequence (optional)
        loaded_word_labels: List of label lists, where each label list corresponds to a word list (optional)

    Returns:
        tuple: (activations tensor, attention mask tensor, tokens tensor, token labels list or None, story indices)
    """

    artifact_fnames = {
        "act": f"activations_{cfg.exp_name}.pt",
        "mask": f"masks_{cfg.exp_name}.pt",
        "tokens": f"tokens_{cfg.exp_name}.pt",
        "all_idxs": f"story_idxs_{cfg.exp_name}.pt",
        "labels": f"token_labels_{cfg.exp_name}.pt",
    }
    artifact_dirs = {k: os.path.join(INTERIM_DIR, v) for k, v in artifact_fnames.items()}
    # Check if we need labels (only check label artifacts if word labels are provided)
    has_word_labels = loaded_word_lists is not None and loaded_word_labels is not None
    core_artifacts = ["act", "mask", "tokens", "all_idxs"]
    artifacts_to_check = core_artifacts + (["labels"] if has_word_labels else [])

    is_existing_each = [os.path.exists(artifact_dirs[k]) for k in artifacts_to_check]
    is_existing = all(is_existing_each)
    if cfg.verbose:
        print(f"looking for files {"\n".join(artifact_dirs.values())}")
        print(f"Are all existing? {is_existing}")


    if not is_existing or cfg.llm.force_recompute:
        acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_llm_artifacts(
            cfg, loaded_dataset_sequences=loaded_dataset_sequences
        )

        # Generate token labels if word labels are provided
        if has_word_labels:
            token_labels_BP = _match_word_labels_to_tokens(
                tokens_BP, loaded_word_lists, loaded_word_labels, cfg
            )
            # Save token labels
            if cfg.save_artifacts:
                th.save(token_labels_BP, artifact_dirs["labels"])
        else:
            token_labels_BP = None
    else:
        acts_LBPD = th.load(artifact_dirs["act"], weights_only=False).to("cpu")
        masks_BP = th.load(artifact_dirs["mask"], weights_only=False)
        tokens_BP = th.load(artifact_dirs["tokens"], weights_only=False)
        dataset_story_idxs = th.load(artifact_dirs["all_idxs"], weights_only=False)

        # Load token labels if they exist
        if has_word_labels and os.path.exists(artifact_dirs["labels"]):
            token_labels_BP = th.load(artifact_dirs["labels"], weights_only=False)
        else:
            token_labels_BP = None

    if cfg.selected_story_idxs is not None:
        dataset_story_idxs = th.tensor(dataset_story_idxs)
        dataset_story_idxs = dataset_story_idxs[cfg.selected_story_idxs]
        acts_LBPD = acts_LBPD[:, dataset_story_idxs, :, :]
        masks_BP = masks_BP[dataset_story_idxs, :]
        if token_labels_BP is not None:
            token_labels_BP = [token_labels_BP[i] for i in dataset_story_idxs]

    if cfg.omit_BOS_token:
        acts_LBPD = acts_LBPD[:, :, 1:, :]
        tokens_BP = tokens_BP[:, 1:]
        masks_BP = masks_BP[:, 1:]
        if token_labels_BP is not None:
            token_labels_BP = [seq[1:] for seq in token_labels_BP]

    if cfg.num_tokens_per_story is not None:
        acts_LBPD = acts_LBPD[:, :, : cfg.num_tokens_per_story, :]
        tokens_BP = tokens_BP[:, : cfg.num_tokens_per_story]
        masks_BP = masks_BP[:, : cfg.num_tokens_per_story]
        if token_labels_BP is not None:
            token_labels_BP = [seq[: cfg.num_tokens_per_story] for seq in token_labels_BP]

    return acts_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs


def phase_randomized_surrogate(X_BPD: th.Tensor) -> th.Tensor:
    """
    Phase-randomized surrogate per (B, D) series along time P.
    Preserves power spectrum per dim, randomizes phases -> stationary.

    Args:
        X_BPD: Tensor of shape (B, P, D)

    Returns:
        Phase-randomized surrogate with same shape
    """
    B, P, D = X_BPD.shape
    X_sur = th.empty_like(X_BPD)
    X_np = X_BPD.detach().cpu().numpy()

    for b in range(B):
        for d in range(D):
            x = X_np[b, :, d]
            fft_x = np.fft.rfft(x)
            mag = np.abs(fft_x)
            # random phases in [0, 2π), keep DC/Nyquist magnitudes
            rand_phase = np.exp(1j * np.random.uniform(0.0, 2 * np.pi, size=fft_x.shape))
            # ensure DC (0-freq) has zero phase
            rand_phase[0] = 1.0 + 0.0j
            fft_new = mag * rand_phase
            x_new = np.fft.irfft(fft_new, n=P)
            X_sur[b, :, d] = th.from_numpy(x_new).to(X_BPD)
    return X_sur


def compute_or_load_surrogate_artifacts(cfg, original_acts_BPD: th.Tensor) -> th.Tensor:
    """
    Compute or load phase-randomized surrogate data with caching.
    
    Args:
        cfg: Configuration object
        original_acts_BPD: Original activation tensor of shape (B, P, D)
        
    Returns:
        Phase-randomized surrogate tensor with same shape
    """
    from src.project_config import INTERIM_DIR
    
    # Create artifact filename
    surrogate_fname = f"surrogate_{cfg.exp_name}.pt"
    surrogate_path = os.path.join(INTERIM_DIR, surrogate_fname)
    
    # Check if surrogate data exists and should be loaded
    if os.path.exists(surrogate_path) and not cfg.force_recompute:
        if cfg.verbose:
            print(f"Loading surrogate data from {surrogate_path}")
        surrogate_acts_BPD = th.load(surrogate_path)
        return surrogate_acts_BPD
    else:
        if cfg.verbose:
            print("Computing phase-randomized surrogate data...")
        surrogate_acts_BPD = phase_randomized_surrogate(original_acts_BPD)
        
        # Save if configured
        if cfg.save_artifacts:
            os.makedirs(INTERIM_DIR, exist_ok=True)
            th.save(surrogate_acts_BPD, surrogate_path)
            if cfg.verbose:
                print(f"Saved surrogate data to {surrogate_path}")
        
        return surrogate_acts_BPD


def load_activation_split(cfg, loaded_dataset_sequences=None):
    act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = (
        compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=loaded_dataset_sequences)
    )

    # Do train-test split
    if cfg.do_train_test_split:
        rand_idxs = th.randperm(cfg.num_total_stories)
        train_idxs = rand_idxs[: cfg.num_train_stories]
        test_idxs = rand_idxs[cfg.num_train_stories :]

        act_train_LBPD = act_LBPD[:, train_idxs, :, :]
        act_test_LBPD = act_LBPD[:, test_idxs, :, :]

        masks_train_BP = masks_BP[train_idxs, :]
        masks_test_BP = masks_BP[test_idxs, :]

        tokens_test_BP = [tokens_BP[i] for i in test_idxs]
        dataset_idxs_test = [dataset_story_idxs[i] for i in test_idxs]
        num_test_stories = len(test_idxs)
    else:
        act_train_LBPD = act_LBPD
        act_test_LBPD = act_LBPD
        masks_train_BP = masks_BP
        masks_test_BP = masks_BP
        tokens_test_BP = tokens_BP
        dataset_idxs_test = dataset_story_idxs
        num_test_stories = cfg.num_total_stories

    return (
        act_train_LBPD,
        act_test_LBPD,
        masks_train_BP,
        masks_test_BP,
        tokens_test_BP,
        num_test_stories,
        dataset_idxs_test,
    )


@dataclass
class SAEArtifact:
    llm_act_BPD: th.Tensor
    sae_act_BPS: th.Tensor
    sae_indices_BPS: th.Tensor
    recon_BPD: th.Tensor
    sae_cfg: any


def compute_or_load_sae_artifacts(llm_act_BPD, cfg) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute or load SAE results for a given model, layer, and number of stories.

    Returns:
        Tuple containing:
        - fvu_BP: Frame variance unexplained tensor.
        - latent_acts_BPS: Latent activations tensor.
        - latent_indices_BPK: Latent indices tensor.
    """
    sae_artifact_path = os.path.join(INTERIM_DIR, f"sae_activations_{hash(cfg)}.pt")

    if cfg.sae.force_recompute or not os.path.exists(sae_artifact_path):
        sae = load_sae(cfg)
        recon_BPD, sae_act_BPS, sae_indices_BPS = batch_sae_cache(sae, llm_act_BPD, cfg.sae)

        sae_artifact = SAEArtifact(llm_act_BPD, sae_act_BPS, sae_indices_BPS, recon_BPD, cfg.sae)
        if cfg.save_artifacts:
            with open(sae_artifact_path, "wb") as f:
                th.save(sae_artifact, f)

        del sae
        th.cuda.empty_cache()
        gc.collect()
    else:
        # Load precomputed artifacts
        sae_artifact = th.load(sae_artifact_path, weights_only=False)
        print(f"SAE results loaded from: {sae_artifact_path}")

    return sae_artifact


def save_svd_results(
    U_LbC: th.Tensor,
    S_LC: th.Tensor,
    Vt_LCD: th.Tensor,
    means_LD: th.Tensor,
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
        svd_save_fname = f"svd_{model_str}_{dataset_str}_first-{num_stories}_layer-{layer_idx}.pt"
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

    th.save(svd_data, svd_path)
    print(f"SVD results saved to: {svd_path}")


def load_svd_results(
    model_name: str,
    dataset_name: str,
    num_stories: int,
    layer_idx: Optional[int] = None,
) -> Optional[Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]]:
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
        svd_save_fname = f"svd_{model_str}_{dataset_str}_first-{num_stories}_layer-{layer_idx}.pt"
    else:
        svd_save_fname = f"svd_{model_str}_{dataset_str}_first-{num_stories}.pt"
    svd_path = os.path.join(INTERIM_DIR, svd_save_fname)

    if os.path.exists(svd_path):
        try:
            svd_data = th.load(svd_path, weights_only=False)
            print(f"SVD results loaded from: {svd_path}")
            # Handle backwards compatibility - old files might not have means
            if "means" in svd_data:
                return svd_data["U"], svd_data["S"], svd_data["Vt"], svd_data["means"]
            else:
                print("Warning: Loaded SVD file does not contain means. PCA may be incorrect.")
                # Return None for means to indicate they need to be recomputed
                return svd_data["U"], svd_data["S"], svd_data["Vt"], None
        except Exception as e:
            print(f"Error loading SVD results: {e}")
            return None
    else:
        print(f"No saved SVD results found at: {svd_path}")
        return None


def compute_centered_svd(
    act_LbD: th.Tensor,
    layer_idx: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute SVD for a given layer of activations.
    """

    # Compute SVD layer by layer on GPU for efficiency and memory management
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
        if layer_idx is not None:
            print(f"Computing SVD for single layer {layer_idx} with proper centering...")
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
        U_LbC = th.zeros(1, b, C)
        S_LC = th.zeros(1, C)
        Vt_LCD = th.zeros(1, C, D)
        means_LD = th.zeros(1, D)
    else:  # All layers, starting at 0
        layers_to_process = list(range(L))
        # Initialize tensors with full L dimension for all layers
        U_LbC = th.zeros(L, b, C)
        S_LC = th.zeros(L, C)
        Vt_LCD = th.zeros(L, C, D)
        means_LD = th.zeros(L, D)

    # Process each layer
    for i in range(len(layers_to_process)):
        if verbose:
            print(f"{i}. computing SVD for layer {actual_layer}")
        actual_layer = layers_to_process[i]
        act_bD = act_LbD[actual_layer].to(device)

        # Center the data before SVD (proper PCA)
        mean_D = th.mean(act_bD, dim=0, keepdim=True)
        act_centered_bD = act_bD - mean_D

        # Compute SVD
        U_layer, S_layer, Vt_layer = th.linalg.svd(act_centered_bD, full_matrices=False)

        # Store results
        U_LbC[i] = U_layer.cpu()
        S_LC[i] = S_layer.cpu()
        Vt_LCD[i] = Vt_layer.cpu()
        means_LD[i] = mean_D.cpu()

        # Clear GPU memory after each layer
        del act_bD, mean_D, act_centered_bD, U_layer, S_layer, Vt_layer
        if th.cuda.is_available():
            th.cuda.empty_cache()

    return U_LbC, S_LC, Vt_LCD, means_LD


def compute_centered_svd_single_layer(
    act_pD: th.Tensor,
    verbose: bool = False,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """
    Compute SVD for a single layer of activations.
    """

    act_LpD = act_pD.unsqueeze(0)
    U_LpC, S_LC, Vt_LCD, means_LD = compute_centered_svd(act_LpD, layer_idx=0, verbose=verbose)
    return U_LpC[0], S_LC[0], Vt_LCD[0], means_LD[0]


def compute_or_load_svd(
    act_LBPD: th.Tensor,
    model_name: str,
    dataset_name: str,
    num_stories: int,
    force_recompute: bool = False,
    layer_idx: Optional[int] = None,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
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
    tokens_BP,
    story_idx: int,
    model_name: str,
    omit_BOS_token: bool = False,
    seq_length: Optional[int] = None,
    tokenizer: AutoTokenizer = None,
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
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name, MODELS_DIR)
        
    tokens = [tokenizer.decode(t, skip_special_tokens=False) for t in tokens_BP[story_idx]]
    tokens = [t.replace("\n\n", "\\n") for t in tokens]

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


def compute_mean_and_ci_to_csv(results_dict: Dict[str, Any], output_path: str) -> None:
    """
    Compute mean and confidence intervals across values and format into CSV.

    Args:
        results_dict: Dictionary with structure like the JSON example:
            {
                "sae_name.pt": {
                    "metric_name": {
                        "score": [list of values],
                        "ci": [list of ci values] or None
                    }
                }
            }
        output_path: Path where to save the CSV file
    """
    rows = []

    # Extract SAE names (everything except config and sequence_pos_indices)
    sae_names = [key for key in results_dict.keys()
                 if key not in ['config', 'sequence_pos_indices']]

    for sae_name in sae_names:
        sae_data = results_dict[sae_name]

        # Get all metrics for this SAE
        metrics = [key for key in sae_data.keys() if key != 'fraction_alive']

        row = {'sae_name': sae_name}

        for metric in metrics:
            metric_data = sae_data[metric]
            scores = metric_data.get('score', [])
            cis = metric_data.get('ci', [])

            if scores:
                # Compute mean
                mean_score = th.tensor(scores).float().mean().item()
                row[f'{metric}_mean'] = mean_score

                # Compute CI mean if available
                if cis and any(ci is not None for ci in cis):
                    # Filter out None values
                    valid_cis = [ci for ci in cis if ci is not None]
                    if valid_cis:
                        mean_ci = th.tensor(valid_cis).float().mean().item()
                        row[f'{metric}_ci'] = mean_ci
                    else:
                        row[f'{metric}_ci'] = None
                else:
                    row[f'{metric}_ci'] = None

        # Add fraction_alive if it exists
        if 'fraction_alive' in sae_data:
            row['fraction_alive'] = sae_data['fraction_alive']

        rows.append(row)

    # Write to CSV
    if rows:
        fieldnames = list(rows[0].keys())

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
