import os
import json
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from tqdm import trange

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
    
    tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
    B, P = tokens_BP.shape
    
    # Initialize with empty strings
    token_labels_BP = [["" for _ in range(P)] for _ in range(B)]
    
    for b in range(B):
        if b >= len(word_lists) or b >= len(word_labels):
            continue
            
        words = word_lists[b]
        labels = word_labels[b]
        
        if len(words) != len(labels):
            print(f"Warning: Mismatched word/label counts for sequence {b}: {len(words)} words, {len(labels)} labels")
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
            token_ids = encoded['input_ids']
            offsets = encoded.get('offset_mapping', [])
            
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
            print(f"Warning: Could not get offsets for sequence {b}, falling back to simple matching: {e}")
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
        word_tokens = tokenizer(word, add_special_tokens=False)['input_ids']
        
        # Assign this label to the next N tokens
        for _ in range(len(word_tokens)):
            if token_idx < P:
                # Skip special tokens at the beginning
                while (token_idx < P and 
                       tokens_P[token_idx].item() in [tokenizer.bos_token_id, tokenizer.cls_token_id]):
                    token_idx += 1
                
                if token_idx < P:
                    token_labels_P[token_idx] = label
                    token_idx += 1


def compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=None, loaded_word_lists=None, loaded_word_labels=None) -> Tuple[Tensor, Tensor, Tensor, Optional[List[List[str]]], List[int]]:
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
        "act": f"activations_{cfg.input_file_str}.pt",
        "mask": f"masks_{cfg.input_file_str}.pt",
        "tokens": f"tokens_{cfg.input_file_str}.pt",
        "all_idxs": f"story_idxs_{cfg.input_file_str}.pt",
        "labels": f"token_labels_{cfg.input_file_str}.pt",
    }
    artifact_dirs = {
        k: os.path.join(INTERIM_DIR, v) for k, v in artifact_fnames.items()
    }
    # Check if we need labels (only check label artifacts if word labels are provided)
    has_word_labels = loaded_word_lists is not None and loaded_word_labels is not None
    core_artifacts = ["act", "mask", "tokens", "all_idxs"]
    artifacts_to_check = core_artifacts + (["labels"] if has_word_labels else [])
    
    is_existing_each = [os.path.exists(artifact_dirs[k]) for k in artifacts_to_check]
    is_existing = all(is_existing_each)

    if not is_existing or cfg.force_recompute:
        acts_LBPD, masks_BP, tokens_BP, dataset_story_idxs = compute_llm_artifacts(cfg, loaded_dataset_sequences=loaded_dataset_sequences)
        
        # Generate token labels if word labels are provided
        if has_word_labels:
            token_labels_BP = _match_word_labels_to_tokens(tokens_BP, loaded_word_lists, loaded_word_labels, cfg)
            # Save token labels
            torch.save(token_labels_BP, artifact_dirs["labels"])
        else:
            token_labels_BP = None
    else:
        acts_LBPD = torch.load(artifact_dirs["act"], weights_only=False).to("cpu")
        masks_BP = torch.load(artifact_dirs["mask"], weights_only=False)
        tokens_BP = torch.load(artifact_dirs["tokens"], weights_only=False)
        dataset_story_idxs = torch.load(artifact_dirs["all_idxs"], weights_only=False)
        
        # Load token labels if they exist
        if has_word_labels and os.path.exists(artifact_dirs["labels"]):
            token_labels_BP = torch.load(artifact_dirs["labels"], weights_only=False)
        else:
            token_labels_BP = None

    if cfg.selected_story_idxs is not None:
        dataset_story_idxs = torch.tensor(dataset_story_idxs)
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


def load_activation_split(cfg, loaded_dataset_sequences=None):
    act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=loaded_dataset_sequences)

    # Do train-test split
    if cfg.do_train_test_split:
        rand_idxs = torch.randperm(cfg.num_total_stories)
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


def compute_or_load_sae(cfg, loaded_dataset_sequences=None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute or load SAE results for a given model, layer, and number of stories.

    Returns:
        Tuple containing:
        - fvu_BP: Frame variance unexplained tensor.
        - latent_acts_BPS: Latent activations tensor.
        - latent_indices_BPK: Latent indices tensor.
    """
    sae_path = os.path.join(INTERIM_DIR, f"sae_activations_{cfg.sae_file_str}.pt")

    if cfg.force_recompute or not os.path.exists(sae_path):
        llm_act_LBPD, masks_BP, tokens_BP, token_labels_BP, dataset_story_idxs = compute_or_load_llm_artifacts(cfg, loaded_dataset_sequences=loaded_dataset_sequences)
        llm_act_BPD = llm_act_LBPD[cfg.layer_idx]

        # Load SAE
        sae = load_sae(cfg)

        # Compute data
        sae_out_BPD, fvu_BP, latent_acts_BPS, latent_indices_BPK = batch_sae_cache(sae, llm_act_BPD, cfg)

        # Save data
        sae_data = {
            "llm_acts": llm_act_BPD,
            "masks": masks_BP,
            "sae_out": sae_out_BPD,
            "fvu": fvu_BP,
            "latent_acts": latent_acts_BPS,
            "latent_indices": latent_indices_BPK,
            "model_name": cfg.llm_name,
            "num_total_stories": cfg.num_total_stories,
            "sae_name": cfg.sae_name,
            "layer_idx": cfg.layer_idx,
        }
        with open(sae_path, "wb") as f:
            torch.save(sae_data, f)
    else:
        # Load precomputed artifacts
        sae_data = torch.load(sae_path, weights_only=False)
        llm_act_BPD = sae_data["llm_acts"]
        masks_BP = sae_data["masks"]
        latent_acts_BPS = sae_data["latent_acts"]
        latent_indices_BPK = sae_data["latent_indices"]
        sae_out_BPD = sae_data["sae_out"]
        fvu_BP = sae_data["fvu"]
        print(f"SAE results loaded from: {sae_path}")

    return llm_act_BPD, masks_BP, latent_acts_BPS, latent_indices_BPK, sae_out_BPD, fvu_BP


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

def compute_centered_svd_single_layer(
    act_pD: Tensor,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SVD for a single layer of activations.
    """

    act_LpD = act_pD.unsqueeze(0)
    U_LpC, S_LC, Vt_LCD, means_LD = compute_centered_svd(act_LpD, layer_idx=0, verbose=verbose)
    return U_LpC[0], S_LC[0], Vt_LCD[0], means_LD[0]


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
    input_file_str: str,
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
    inputs_BP = torch.load(os.path.join(INTERIM_DIR, f"tokens_{input_file_str}.pt"), weights_only=False)
    tokens = [
        tokenizer.decode(t, skip_special_tokens=False) for t in inputs_BP[story_idx]
    ]
    tokens = [t.replace("\n", "<newline>") for t in tokens]

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
