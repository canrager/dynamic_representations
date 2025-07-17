import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
from src.project_config import ARTIFACTS_DIR
from src.exp_utils import compute_or_load_svd, load_tokens_of_story

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
    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(act_LbD, model_name, num_stories, force_recompute)
    

    # # Select layer, select PCA component, select Story
    layer_idx = 6
    pca_component_idx = 0
    story_idx = 0

    seq_tokens = load_tokens_of_story(story_idx, model_name, do_omit_BOS_token, trunc_seq_length)
    print("Sequence_tokens:")
    print("".join(seq_tokens))

    story_PD = act_LBPD[layer_idx, story_idx, mask_BP[story_idx].bool(), :]
    story_centered_PD = story_PD - means_LD[:, None, None, :]
    pca_D = Vt_LCD[layer_idx, pca_component_idx, :]

    story_pca = story_centered_PD @ pca_D

    plot_title=f"story-{story_idx}_pca-{pca_component_idx}_layer{layer_idx}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    plot_save_fname = f"single-evolution_{model_str}_{plot_title}.png"
    plot_single_evolution(story_pca, plot_label="story0", plot_title=plot_title, save_fname=plot_save_fname)



    # # All layers, range of PCA components, select story

    top_pca_components = 10
    story_idx = 0

    seq_tokens = load_tokens_of_story(story_idx, model_name, do_omit_BOS_token, trunc_seq_length)
    print("Sequence_tokens:")
    print("".join(seq_tokens))
    

    # TODO: Check the actual tokens and masks align, BOS token handling?
    story_LPD = act_LBPD[:, story_idx, mask_BP[story_idx].bool(), :]
    story_centered_LPD = story_LPD - means_LD[:, None, None, :]
    pca_LcD = Vt_LCD[:, :top_pca_components, :]  # Use all layers and top_pca_components
    story_pca_LcP = torch.einsum("LPD,LcD->LcP", story_centered_LPD, pca_LcD)

    plot_title=f"story-{story_idx}_top-pca-components-{top_pca_components}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    plot_save_fname = f"evolution-single-story_{model_str}_{plot_title}.png"
    plot_evolutions_across_layers_pca_components(story_pca_LcP, plot_title=plot_title, save_fname=plot_save_fname, sequence_tokens=seq_tokens)


    ## All layers, range of PCA components, range of stories

    top_pca_components = 10
    top_story_idxs = 3

    for i in range(top_story_idxs):
        seq_tokens = load_tokens_of_story(i, model_name, do_omit_BOS_token, trunc_seq_length)
        print(f"{i}. story sequence_tokens:")
        print("".join(seq_tokens), "\n--------------------\n")

    # Note: with constant truncation, should already be truncated 
    story_LBPD = act_LBPD[:, :top_story_idxs, :trunc_seq_length, :]
    story_centered_LBPD = story_LBPD - means_LD[:, None, None, :]
    pca_LcD = Vt_LCD[:, :top_pca_components, :]  # Use all layers and top_pca_components
    story_pca_LBcP = torch.einsum("LBPD,LcD->LBcP", story_centered_LBPD, pca_LcD)

    plot_title=f"stories-first-{top_story_idxs}_top-pca-components-{top_pca_components}_omit-bos-{do_omit_BOS_token}_trunc-story-{do_truncate_seq_length}"
    plot_save_fname = f"plot_{model_str}_{plot_title}.png"
    plot_evolutions_across_stories_layers_pca_components(story_pca_LBcP, plot_title=plot_title, save_fname=plot_save_fname)