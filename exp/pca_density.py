import os
import torch
from src.project_config import MODELS_DIR, ARTIFACTS_DIR
from torch import Tensor
from typing import Tuple, Optional
from tqdm import trange
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset

from typing import List, Optional

def load_activations(model_name: str, num_stories: int, omit_BOS_token: bool = False, truncate_seq_length: bool = False) -> Tuple[Tensor, Tensor, Tensor, Optional[int]]:
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

    if omit_BOS_token:
        activations_LBPD = activations_LBPD[:, :, 1:, :]
        attention_mask_BP = attention_mask_BP[:, 1:]

    min_seq_length = None
    if truncate_seq_length:
        min_seq_length = min(attention_mask_BP.sum(dim=-1))
        print(f'truncating to minimum sequence length of {min_seq_length}')
        activations_LBPD = activations_LBPD[:, :, :min_seq_length, :]
        attention_mask_BP = attention_mask_BP[:, :min_seq_length]
        assert torch.all(attention_mask_BP == 1), "Attention mask should be all ones after truncation"
        
    # Reshape activations to [L, B*P, D] and mask to [B*P]
    L, B, P, D = activations_LBPD.shape
    activations_LbD = activations_LBPD.reshape(L, B*P, D)
    attention_mask_b = attention_mask_BP.reshape(B*P)
    
    # Create a mask for valid positions (where attention_mask == 1)
    valid_positions = attention_mask_b == 1
    activations_LbD = activations_LbD[:, valid_positions, :]

    return activations_LbD, activations_LBPD, attention_mask_BP, min_seq_length

def save_svd_results(U_LbC: Tensor, S_LC: Tensor, Vt_LCD: Tensor, model_name: str, num_stories: int) -> None:
    """
    Save SVD results to disk.
    
    Args:
        U_LbC: Left singular vectors tensor [L, b, C]
        S_LC: Singular values tensor [L, C]
        Vt_LCD: Right singular vectors tensor [L, C, D]
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
        'model_name': model_name,
        'num_stories': num_stories
    }
    
    torch.save(svd_data, svd_path)
    print(f"SVD results saved to: {svd_path}")

def load_svd_results(model_name: str, num_stories: int) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """
    Load SVD results from disk if they exist.
    
    Args:
        model_name: Name of the model
        num_stories: Number of stories used
        
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD) if file exists, None otherwise
    """
    model_str = model_name.replace("/", "--")
    svd_save_fname = f"svd_{model_str}_simple-stories_first-{num_stories}.pt"
    svd_path = os.path.join(ARTIFACTS_DIR, svd_save_fname)
    
    if os.path.exists(svd_path):
        try:
            svd_data = torch.load(svd_path, weights_only=False)
            print(f"SVD results loaded from: {svd_path}")
            return svd_data['U'], svd_data['S'], svd_data['Vt']
        except Exception as e:
            print(f"Error loading SVD results: {e}")
            return None
    else:
        print(f"No saved SVD results found at: {svd_path}")
        return None

def compute_or_load_svd(act_LbD: Tensor, model_name: str, num_stories: int, force_recompute: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute SVD or load from disk if available.
    
    Args:
        act_LbD: Activations tensor [L, b, D]
        model_name: Name of the model
        num_stories: Number of stories used
        force_recompute: If True, recompute SVD even if saved results exist
        
    Returns:
        tuple: (U_LbC, S_LC, Vt_LCD) where C = min(b, D)
    """
    if not force_recompute:
        # Try to load existing SVD results
        svd_results = load_svd_results(model_name, num_stories)
        if svd_results is not None:
            return svd_results
    
    # Compute SVD layer by layer on GPU for efficiency and memory management
    print("Computing SVD layer by layer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    L, b, D = act_LbD.shape
    C = min(b, D)  # Number of components
    
    # Initialize tensors with correct shapes
    # U has shape [L, b, C] (left singular vectors)
    # S has shape [L, C] (singular values)
    # Vt has shape [L, C, D] (right singular vectors - these are the principal components)
    U_LbC = torch.zeros(L, b, C)
    S_LC = torch.zeros(L, C)
    Vt_LCD = torch.zeros(L, C, D)
    
    # Process each layer separately
    for layer in trange(L, desc="Computing SVD per layer"):
        act_layer_gpu = act_LbD[layer].to(device)  # Shape: [b, D]
        U_layer, S_layer, Vt_layer = torch.linalg.svd(act_layer_gpu, full_matrices=False)
        
        # U_layer: [b, C], S_layer: [C], Vt_layer: [C, D]
        U_LbC[layer] = U_layer.cpu()
        S_LC[layer] = S_layer.cpu()
        Vt_LCD[layer] = Vt_layer.cpu()
        
        # Clear GPU memory after each layer
        del act_layer_gpu, U_layer, S_layer, Vt_layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save the results
    save_svd_results(U_LbC, S_LC, Vt_LCD, model_name, num_stories)
    
    return U_LbC, S_LC, Vt_LCD

def plot_single_evolution(evolution_P: torch.Tensor, plot_label: str, plot_title: str, save_fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(evolution_P, label=plot_label)
    ax.legend()
    ax.set_title(plot_title)
    fig.savefig(os.path.join(ARTIFACTS_DIR, save_fname))
    print(f"Figure saved to: {os.path.join(ARTIFACTS_DIR, save_fname)}")
    plt.close(fig)

def plot_evolutions_across_layers_pca_components(evolutions_LCP: torch.Tensor, plot_title: str, save_fname: str, sequence_tokens: Optional[List[str]] = None) -> None:
    num_layers, num_pca_components, seq_length = evolutions_LCP.shape
    if sequence_tokens is not None:
        assert len(sequence_tokens) == seq_length
    
    # Scale figsize based on number of subplots and sequence length
    # Increase width more aggressively when we have tokens to display
    base_width = 4 * num_pca_components
    if sequence_tokens is not None:
        # Add extra width for rotated labels - roughly 0.1 inch per token
        extra_width = max(2, seq_length * 0.3)
        fig_width = max(base_width + extra_width, 12)
    else:
        fig_width = max(base_width, 8)
    
    fig_height = max([3 * num_layers, fig_width*1.2])  # At least 6 inches tall
    
    fig, axs = plt.subplots(num_layers, num_pca_components, figsize=(fig_width, fig_height))
    
    # Handle case where we might have only 1 row or 1 column
    if num_layers == 1 and num_pca_components == 1:
        axs = [[axs]]
    elif num_layers == 1:
        axs = [axs]
    elif num_pca_components == 1:
        axs = [[ax] for ax in axs]
    
    for l in range(num_layers):
        for c in range(num_pca_components):
            # Plot the evolution for this layer and PCA component
            axs[l][c].plot(evolutions_LCP[l, c, :].numpy())
            axs[l][c].set_title(f'Layer {l}, PCA {c}')
            axs[l][c].grid(True, alpha=0.3)
            if sequence_tokens is not None:
                axs[l][c].set_xticks(range(len(sequence_tokens)))
                axs[l][c].set_xticklabels(sequence_tokens, rotation=90, ha='center', fontsize=8)
            
    # Adjust layout with extra bottom padding for rotated labels
    if sequence_tokens is not None:
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave more space at bottom for rotated labels
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Add main title with some space
    fig.suptitle(plot_title, y=0.98, fontsize=14)
    
    # Save and close
    fig.savefig(os.path.join(ARTIFACTS_DIR, save_fname), dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(ARTIFACTS_DIR, save_fname)}")
    plt.close(fig)


def plot_evolutions_across_stories_layers_pca_components(evolutions_LBCP: torch.Tensor, plot_title: str, save_fname: str, sequence_tokens: Optional[List[str]] = None) -> None:
    # TODO remove sequence tokens
    num_layers, num_stories, num_pca_components, seq_length = evolutions_LBCP.shape
    if sequence_tokens is not None:
        assert len(sequence_tokens) == seq_length
    
    # Scale figsize based on number of subplots and sequence length
    # Increase width more aggressively when we have tokens to display
    base_width = 4 * num_pca_components
    if sequence_tokens is not None:
        # Add extra width for rotated labels - roughly 0.1 inch per token
        extra_width = max(2, seq_length * 0.3)
        fig_width = max(base_width + extra_width, 12)
    else:
        fig_width = max(base_width, 8)
    
    fig_height = max([3 * num_layers, fig_width*1.2])  # At least 6 inches tall
    
    fig, axs = plt.subplots(num_layers, num_pca_components, figsize=(fig_width, fig_height))
    
    # Handle case where we might have only 1 row or 1 column
    if num_layers == 1 and num_pca_components == 1:
        axs = [[axs]]
    elif num_layers == 1:
        axs = [axs]
    elif num_pca_components == 1:
        axs = [[ax] for ax in axs]
    
    for l in range(num_layers):
        for c in range(num_pca_components):
            for s in range(num_stories):
                # Plot the evolution for this layer and PCA component
                axs[l][c].plot(evolutions_LBCP[l, s, c, :].numpy(), label=f"story {s}")
                axs[l][c].set_title(f'Layer {l}, PCA {c}')
                axs[l][c].grid(True, alpha=0.3)
                axs[l][c].legend()
                if sequence_tokens is not None:
                    axs[l][c].set_xticks(range(len(sequence_tokens)))
                    axs[l][c].set_xticklabels(sequence_tokens, rotation=90, ha='center', fontsize=8)
            
    # Adjust layout with extra bottom padding for rotated labels
    if sequence_tokens is not None:
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave more space at bottom for rotated labels
    else:
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    
    # Add main title with some space
    fig.suptitle(plot_title, y=0.98, fontsize=14)
    
    # Save and close
    fig.savefig(os.path.join(ARTIFACTS_DIR, save_fname), dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(ARTIFACTS_DIR, save_fname)}")
    plt.close(fig)

def load_tokens_of_story(story_idx: int, model_name: str, omit_BOS_token: bool = False, seq_length: Optional[int] = None) -> List[str]:
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

    token_str_L = [tokenizer.decode(t, skip_special_tokens=False) for t in token_ids_L]
    return token_str_L


if __name__ == "__main__":
    num_stories = 100

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"
    
    # Set to True to force recomputation of SVD even if saved results exist
    force_recompute = False
    do_omit_BOS_token = True
    do_truncate_seq_length = True

    model_str = model_name.replace("/", "--")

    act_LbD, act_LBPD, mask_BP, trunc_seq_length = load_activations(model_name, num_stories, do_omit_BOS_token, do_truncate_seq_length)
    print(act_LbD.shape)
    

    # Compute or load PCA - Tensorized version
    # act_LbD has shape [L, b, D] where L is layers, b is batch*positions, D is hidden_dim
    # the columns of U are sing vectors as well as rows of Vt
    U_LbC, S_LC, Vt_LCD = compute_or_load_svd(act_LbD, model_name, num_stories, force_recompute)
    

    # # Select layer, select PCA component, select Story
    # layer_idx = 6
    # pca_component_idx = 0
    # story_idx = 0

    # seq_tokens = load_tokens_of_story(story_idx, model_name, do_omit_BOS_token, trunc_seq_length)
    # print("Sequence_tokens:")
    # print("".join(seq_tokens))

    # story_PD = act_LBPD[layer_idx, story_idx, mask_BP[story_idx].bool(), :]
    # pca_D = Vt_LCD[layer_idx, pca_component_idx, :]

    # story_pca = story_PD @ pca_D

    # plot_title=f"story-{story_idx}_pca-{pca_component_idx}_layer{layer_idx}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    # plot_save_fname = f"single-evolution_{model_str}_{plot_title}.png"
    # plot_single_evolution(story_pca, plot_label="story0", plot_title=plot_title, save_fname=plot_save_fname)



    # # All layers, range of PCA components, select story

    # top_pca_components = 10
    # story_idx = 0

    # seq_tokens = load_tokens_of_story(story_idx, model_name, do_omit_BOS_token, trunc_seq_length)
    # print("Sequence_tokens:")
    # print("".join(seq_tokens))
    

    # # TODO: Check the actual tokens and masks align, BOS token handling?
    # story_LPD = act_LBPD[:, story_idx, mask_BP[story_idx].bool(), :]
    # pca_LcD = Vt_LCD[:, :top_pca_components, :]  # Use all layers and top_pca_components
    # story_pca_LcP = torch.einsum("LPD,LcD->LcP", story_LPD, pca_LcD)

    # plot_title=f"story-{story_idx}_top-pca-components-{top_pca_components}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    # plot_save_fname = f"evolution-single-story_{model_str}_{plot_title}.png"
    # plot_evolutions_across_layers_pca_components(story_pca_LcP, plot_title=plot_title, save_fname=plot_save_fname, sequence_tokens=seq_tokens)


    # All layers, range of PCA components, range of stories

    top_pca_components = 10
    top_story_idxs = 3

    for i in range(top_story_idxs):
        seq_tokens = load_tokens_of_story(i, model_name, do_omit_BOS_token, trunc_seq_length)
        print(f"{i}. story sequence_tokens:")
        print("".join(seq_tokens), "\n--------------------\n")

    # Note: with constant truncation, should already be truncated 
    story_LBPD = act_LBPD[:, :top_story_idxs, :trunc_seq_length, :]
    pca_LcD = Vt_LCD[:, :top_pca_components, :]  # Use all layers and top_pca_components
    story_pca_LBcP = torch.einsum("LBPD,LcD->LBcP", story_LBPD, pca_LcD)

    plot_title=f"stories-first-{top_story_idxs}_top-pca-components-{top_pca_components}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    plot_save_fname = f"plot_{model_str}_{plot_title}.png"
    plot_evolutions_across_stories_layers_pca_components(story_pca_LBcP, plot_title=plot_title, save_fname=plot_save_fname)