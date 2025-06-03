import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
from src.project_config import ARTIFACTS_DIR
from src.exp_utils import compute_or_load_svd, load_tokens_of_story, load_activations




if __name__ == "__main__":
    num_stories = 100
    reconstruction_threshold = 0.8

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    layer_idx = 16
    story_idx = 0
    
    # Set to True to force recomputation of SVD even if saved results exist
    force_recompute = False
    do_omit_BOS_token = True
    do_truncate_seq_length = False

    model_str = model_name.split("/")[-1]

    act_LbD, act_LBPD, mask_BP, trunc_seq_length = load_activations(model_name, num_stories, do_omit_BOS_token, do_truncate_seq_length)
    
    # Compute or load PCA - Tensorized version
    # act_LbD has shape [L, b, D] where L is layers, b is batch*positions, D is hidden_dim
    # the columns of U are sing vectors as well as rows of Vt
    U_LbC, S_LC, Vt_LCD, means_LD = compute_or_load_svd(act_LbD, model_name, num_stories, force_recompute)
    
    num_layers, num_components, hidden_dim = Vt_LCD.shape[0], Vt_LCD.shape[1], Vt_LCD.shape[2]

    # Exp2 PCA density. How many PCA components are needed to reconstruct the activations?

    # Exp 2.1: For a given token, plot the reconstruction error as a function of the number of PCA components.
    story_PD = act_LBPD[layer_idx, story_idx, mask_BP[story_idx].bool(), :]
    story_centered_PD = story_PD - means_LD[layer_idx][None, :] # Center the data by subtracting the mean that was used during PCA computation
    
    # Simplified computation of fraction of variance reconstructed
    # Project centered data onto PCA components
    story_pca_PC = torch.einsum("pd,cd->pc", story_centered_PD, Vt_LCD[layer_idx, :, :])
    
    # Compute cumulative squared projections (variance explained)
    squared_projections_PC = story_pca_PC ** 2
    cumulative_variance_PC = torch.cumsum(squared_projections_PC, dim=1)
    
    # Total variance for each position (squared norm of centered data)
    total_variance_P = torch.sum(story_centered_PD ** 2, dim=1)
    
    # Fraction of variance reconstructed
    frac_variance_reconstructed_PC = cumulative_variance_PC / total_variance_P[:, None]
    
    step_size = 10
    fig, ax = plt.subplots()
    for p in range(0, story_PD.shape[0], step_size):
        ax.plot(range(num_components), frac_variance_reconstructed_PC[p], label=f"Token {p}")
    ax.legend()
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Fraction of variance reconstructed")
    ax.set_title(f"Fraction of variance reconstructed for story {story_idx} in layer {layer_idx}\n{model_name}")
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"frac_variance_reconstructed_{model_str}_story_{story_idx}_layer_{layer_idx}.png"))
    print(f"Saved figure to {os.path.join(ARTIFACTS_DIR, f'frac_variance_reconstructed_{model_str}_story_{story_idx}_layer_{layer_idx}.png')}")
    plt.close()

    print(f"num_components: {num_components}")


    ## Exp 2.2: Plot explained variance for a given token index, across all stories in the dataset.

    tokens_BPD = act_LBPD[layer_idx, :, :, :]
    
    # Center the data properly using the stored mean
    data_mean_D = means_LD[layer_idx]  # Use the stored mean from PCA computation
    tokens_centered_BPD = tokens_BPD - data_mean_D[None, None, :]  # Broadcast across batch and position
    
    Vt_CD = Vt_LCD[layer_idx, :, :]

    total_var_P = torch.var(tokens_centered_BPD, dim=0).sum(dim=-1)

    # Compute PCA coefficients on centered data
    pca_BPC = torch.einsum("bpd,cd->bpc", tokens_centered_BPD, Vt_CD)
    pca_var_PC = torch.var(pca_BPC, dim=0)
    pca_cumsum_var_PC = torch.cumsum(pca_var_PC, dim=-1)

    pca_var_exp_PC = pca_cumsum_var_PC / total_var_P[:, None]

    step_size = 10
    fig, ax = plt.subplots()
    for p in range(0, pca_var_exp_PC.shape[0], step_size):
        ax.plot(range(num_components), pca_var_exp_PC[p], label=f"Token {p}")
    ax.legend()
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Variance explained")
    ax.set_title(f"Variance explained for fixed token positions across stories in layer {layer_idx}")
    fig_name = f"explained_variance_layer_{layer_idx}_{model_str}"
    fig_path = os.path.join(ARTIFACTS_DIR, f"{fig_name}.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
    plt.close()


    ## Exp 2.3: Plot explained variance for a story, x: tokens, y: number of PCA components required to reconstruct thresh% of variance.

    story_idxs = [0,1,2,3,4]
    
    tokens_of_story = None
    if len(story_idxs) == 1:
        tokens_of_story = load_tokens_of_story(story_idxs[0], model_name, do_omit_BOS_token, trunc_seq_length)
        print(f"tokens_of_story: {"".join(tokens_of_story)}")

    story_BPD = act_LBPD[layer_idx, story_idxs, :, :]
    story_BPD = story_BPD.reshape(-1, story_BPD.shape[-2], story_BPD.shape[-1])
    story_centered_BPD = story_BPD - means_LD[layer_idx][None, None, :] # Center the data by subtracting the mean that was used during PCA computation
    total_variance_BP = torch.sum(story_centered_BPD ** 2, dim=-1)

    pca_BPC = torch.einsum("bpd,cd->bpc", story_centered_BPD, Vt_LCD[layer_idx, :, :])
    variance_pca_BPC = pca_BPC ** 2
    cumulative_variance_BPC = torch.cumsum(variance_pca_BPC, dim=-1)

    explained_variance_cumulative_BPC = cumulative_variance_BPC / total_variance_BP[:, :, None]
    
    ## Find first component that meets threshold
    meets_threshold_BPC = explained_variance_cumulative_BPC >= reconstruction_threshold
    min_components_required_BP = torch.argmax(meets_threshold_BPC.int(), dim=-1)
    
    # If no solution is found, set to max number of components
    has_solution_BP = torch.any(meets_threshold_BPC, dim=-1)
    min_components_required_BP.masked_fill_(~has_solution_BP, num_components)

    fig, ax = plt.subplots(figsize=(15, 6))  # Increase figure width for better readability
    num_stories, num_tokens, hidden_dim = story_BPD.shape
    for b in range(num_stories):
        ax.plot(range(num_tokens), min_components_required_BP[b], label=f"Story {story_idxs[b]}")
    ax.legend()
    ax.set_xlabel("Token Position")
    ylabel = f"PCA Components needed for {reconstruction_threshold*100}% variance"
    ax.set_ylabel(ylabel)
    if tokens_of_story is not None:
        ax.set_xticks(range(num_tokens))
        ax.set_xticklabels(tokens_of_story, rotation=90, ha='right')  # Rotate labels and align right
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    ax.set_title(f"Number of PCA components required to reconstruct {reconstruction_threshold*100}% of variance for stories {story_idxs} in layer {layer_idx}")
    stories_str = "_".join([str(s) for s in story_idxs])
    fig_name = f"num_components_required_to_reconstruct_{reconstruction_threshold}_stories_{stories_str}_layer_{layer_idx}"
    fig_path = os.path.join(ARTIFACTS_DIR, f"{fig_name}.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
    plt.close()


