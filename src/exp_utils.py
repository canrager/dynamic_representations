import os
from typing import Tuple, Optional, List

import torch
from torch import Tensor
from tqdm import trange

from src.project_config import ARTIFACTS_DIR, MODELS_DIR
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoTokenizer

def load_activations(model_name: str, num_stories: int, story_idxs: Optional[List[int]] = None, omit_BOS_token: bool = False, truncate_seq_length: bool = False, subtract_mean: bool = False, max_tokens_per_story: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Optional[int]]:
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

    activations_LBPD = torch.load(acts_path, weights_only=False).to("cpu")
    attention_mask_BP = torch.load(mask_path, weights_only=False).to("cpu")

    if story_idxs is not None:
        activations_LBPD = activations_LBPD[:, story_idxs, :, :]
        attention_mask_BP = attention_mask_BP[story_idxs, :]

    if subtract_mean:
        activations_LBPD = activations_LBPD - activations_LBPD.mean(dim=-1, keepdim=True)

    if omit_BOS_token:
        activations_LBPD = activations_LBPD[:, :, 1:, :]
        attention_mask_BP = attention_mask_BP[:, 1:]

    if max_tokens_per_story is not None:
        activations_LBPD = activations_LBPD[:, :, :max_tokens_per_story, :]
        attention_mask_BP = attention_mask_BP[:, :max_tokens_per_story]

    min_seq_length = None
    if truncate_seq_length:
        min_seq_length = min(attention_mask_BP.sum(dim=-1))
        print(f'truncating to minimum sequence length of {min_seq_length}')
        activations_LBPD = activations_LBPD[:, :, :min_seq_length, :]
        attention_mask_BP = attention_mask_BP[:, :min_seq_length]
        assert torch.all(attention_mask_BP == 1), "Attention mask should be all ones after truncation"
        
    # Reshape activations to [L, B*P, D] and mask to [B*P]
    print(f"activations_LBPD.shape: {activations_LBPD.shape}")
    print(f"attention_mask_BP.shape: {attention_mask_BP.shape}")
    L, B, P, D = activations_LBPD.shape
    activations_LbD = activations_LBPD.reshape(L, B*P, D)
    attention_mask_b = attention_mask_BP.reshape(B*P)
    
    # Create a mask for valid positions (where attention_mask == 1)
    valid_positions = attention_mask_b == 1
    activations_LbD = activations_LbD[:, valid_positions, :]

    return activations_LbD, activations_LBPD, attention_mask_BP, min_seq_length

def save_svd_results(U_LbC: Tensor, S_LC: Tensor, Vt_LCD: Tensor, means_LD: Tensor, model_name: str, num_stories: int) -> None:
    """
    Save SVD results to disk.
    
    Args:
        U_LbC: Left singular vectors tensor [L, b, C]
        S_LC: Singular values tensor [L, C]
        Vt_LCD: Right singular vectors tensor [L, C, D]
        means_LD: Mean values used for centering [L, D]
        model_name: Name of the model
        num_stories: Number of stories used
    """
    model_str = model_name.replace("/", "--")
    svd_save_fname = f"svd_{model_str}_simple-stories_first-{num_stories}.pt"
    svd_path = os.path.join(ARTIFACTS_DIR, svd_save_fname)
    
    svd_data = {
        'U': U_LbC,
        'S': S_LC,
        'Vt': Vt_LCD,
        'means': means_LD,
        'model_name': model_name,
        'num_stories': num_stories
    }
    
    torch.save(svd_data, svd_path)
    print(f"SVD results saved to: {svd_path}")

def load_svd_results(model_name: str, num_stories: int) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Load SVD results from disk if they exist.
    
    Args:
        model_name: Name of the model
        num_stories: Number of stories used
        
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) if file exists, None otherwise
    """
    model_str = model_name.replace("/", "--")
    svd_save_fname = f"svd_{model_str}_simple-stories_first-{num_stories}.pt"
    svd_path = os.path.join(ARTIFACTS_DIR, svd_save_fname)
    
    if os.path.exists(svd_path):
        try:
            svd_data = torch.load(svd_path, weights_only=False)
            print(f"SVD results loaded from: {svd_path}")
            # Handle backwards compatibility - old files might not have means
            if 'means' in svd_data:
                return svd_data['U'], svd_data['S'], svd_data['Vt'], svd_data['means']
            else:
                print("Warning: Loaded SVD file does not contain means. PCA may be incorrect.")
                # Return None for means to indicate they need to be recomputed
                return svd_data['U'], svd_data['S'], svd_data['Vt'], None
        except Exception as e:
            print(f"Error loading SVD results: {e}")
            return None
    else:
        print(f"No saved SVD results found at: {svd_path}")
        return None

def compute_or_load_svd(act_LbD: Tensor, model_name: str, num_stories: int, force_recompute: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute SVD or load from disk if available.
    
    Args:
        act_LbD: Activations tensor [L, b, D]
        model_name: Name of the model
        num_stories: Number of stories used
        force_recompute: If True, recompute SVD even if saved results exist
        
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD, means_LD) where C = min(b, D)
    """
    if not force_recompute:
        # Try to load existing SVD results
        svd_results = load_svd_results(model_name, num_stories)
        if svd_results is not None and svd_results[3] is not None:
            return svd_results
    
    # Compute SVD layer by layer on GPU for efficiency and memory management
    print("Computing SVD layer by layer with proper centering...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    L, b, D = act_LbD.shape
    C = min(b, D)  # Number of components
    
    # Initialize tensors with correct shapes
    # U has shape [L, b, C] (left singular vectors)
    # S has shape [L, C] (singular values)
    # Vt has shape [L, C, D] (right singular vectors - these are the principal components)
    # means has shape [L, D] (mean values used for centering)
    U_LbC = torch.zeros(L, b, C)
    S_LC = torch.zeros(L, C)
    Vt_LCD = torch.zeros(L, C, D)
    means_LD = torch.zeros(L, D)
    
    # Process each layer separately
    for layer in trange(L, desc="Computing SVD per layer"):
        act_layer_gpu = act_LbD[layer].to(device)  # Shape: [b, D]
        
        # IMPORTANT: Center the data before SVD (proper PCA)
        mean_D = torch.mean(act_layer_gpu, dim=0)  # Shape: [D]
        act_centered_gpu = act_layer_gpu - mean_D[None, :]  # Center the data
        
        U_layer, S_layer, Vt_layer = torch.linalg.svd(act_centered_gpu, full_matrices=False)
        
        # U_layer: [b, C], S_layer: [C], Vt_layer: [C, D]
        U_LbC[layer] = U_layer.cpu()
        S_LC[layer] = S_layer.cpu()
        Vt_LCD[layer] = Vt_layer.cpu()
        means_LD[layer] = mean_D.cpu()
        
        # Clear GPU memory after each layer
        del act_layer_gpu, mean_D, act_centered_gpu, U_layer, S_layer, Vt_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save the results
    save_svd_results(U_LbC, S_LC, Vt_LCD, means_LD, model_name, num_stories)
    
    return U_LbC, S_LC, Vt_LCD, means_LD

def load_tokens_of_story(story_idx: int, model_name: str, omit_BOS_token: bool = False, seq_length: Optional[int] = None, max_tokens_per_story: Optional[int] = None) -> List[str]:
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

def load_tokens_of_stories(story_idxs: List[int], model_name: str, omit_BOS_token: bool = False, seq_length: Optional[int] = None, max_tokens_per_story: Optional[int] = None) -> List[List[str]]:
    tokens = []
    for story_idx in story_idxs:
        tokens.append(load_tokens_of_story(story_idx, model_name, omit_BOS_token, seq_length, max_tokens_per_story))
    return tokens
