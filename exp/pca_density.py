import os
import torch
from typing import Optional, List
import matplotlib.pyplot as plt
from src.project_config import ARTIFACTS_DIR
from src.exp_utils import compute_or_load_svd, load_tokens_of_story, load_activations




if __name__ == "__main__":
    num_stories = 100
    reconstruction_threshold = 0.9

    # model_name = "openai-community/gpt2"
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "google/gemma-3-12b-pt"
    # model_name = "allenai/Llama-3.1-Tulu-3-8B"
    # model_name = "google/gemma-2-2b"

    layer_idx = 6
    story_idx = 0
    
    # Set to True to force recomputation of SVD even if saved results exist
    force_recompute = False
    do_omit_BOS_token = True
    do_truncate_seq_length = True

    model_str = model_name.replace("/", "--")

    act_LbD, act_LBPD, mask_BP, trunc_seq_length = load_activations(model_name, num_stories, do_omit_BOS_token, do_truncate_seq_length)
    
    # Compute or load PCA - Tensorized version
    # act_LbD has shape [L, b, D] where L is layers, b is batch*positions, D is hidden_dim
    # the columns of U are sing vectors as well as rows of Vt
    U_LbC, S_LC, Vt_LCD = compute_or_load_svd(act_LbD, model_name, num_stories, force_recompute)
    
    num_layers, num_components, hidden_dim = Vt_LCD.shape[0], Vt_LCD.shape[1], Vt_LCD.shape[2]

    # Exp2 PCA density. How many PCA components are needed to reconstruct the activations?

    # Exp 2.1: For a given token, plot the reconstruction error as a function of the number of PCA components.
    story_PD = act_LBPD[layer_idx, story_idx, mask_BP[story_idx].bool(), :]
    story_pca_PC = torch.einsum("pd,cd->pc", story_PD, Vt_LCD[layer_idx, :, :])
    story_pca_PCD = story_pca_PC[:, :, None] * Vt_LCD[layer_idx, :, :]
    story_pca_cumsum_PCD = story_pca_PCD.cumsum(dim=1)
    reconstruction_error_PC = torch.norm(story_PD[:, None, :] - story_pca_cumsum_PCD, dim=2)
    
    step_size = 10
    fig, ax = plt.subplots()
    for p in range(0, story_PD.shape[0], step_size):
        ax.plot(range(num_components), reconstruction_error_PC[p], label=f"Token {p}")
    ax.legend()
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Reconstruction error")
    ax.set_title(f"Reconstruction error for story {story_idx} in layer {layer_idx}")
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"reconstruction_error_story_{story_idx}_layer_{layer_idx}.png"))
    print(f"Saved figure to {os.path.join(ARTIFACTS_DIR, f'reconstruction_error_story_{story_idx}_layer_{layer_idx}.png')}")
    plt.close()

    # Exp 2.1b: Plot variance explained by PCA components
    # Compute variance explained by each component
    eigenvalues = S_LC[layer_idx] ** 2  # Convert singular values to eigenvalues
    total_variance = torch.sum(eigenvalues)
    cumulative_variance = torch.cumsum(eigenvalues, dim=0)
    variance_explained_percentage = (cumulative_variance / total_variance) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Variance explained percentage vs number of components
    ax1.plot(range(1, len(variance_explained_percentage) + 1), variance_explained_percentage)
    ax1.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% variance')
    ax1.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
    ax1.axhline(y=99, color='green', linestyle='--', alpha=0.7, label='99% variance')
    ax1.set_xlabel("Number of PCA components")
    ax1.set_ylabel("Cumulative variance explained (%)")
    ax1.set_title(f"Variance explained by PCA components (Layer {layer_idx})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual component variance contribution (first 50 components)
    n_show = min(50, len(eigenvalues))
    individual_variance_pct = (eigenvalues[:n_show] / total_variance) * 100
    ax2.bar(range(1, n_show + 1), individual_variance_pct)
    ax2.set_xlabel("PCA component")
    ax2.set_ylabel("Individual variance explained (%)")
    ax2.set_title(f"Individual component variance (Layer {layer_idx}, first {n_show} components)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"variance_explained_layer_{layer_idx}.png"))
    print(f"Saved variance plot to {os.path.join(ARTIFACTS_DIR, f'variance_explained_layer_{layer_idx}.png')}")
    plt.close()
    
    # Print some key statistics
    for threshold in [90, 95, 99]:
        components_needed = torch.sum(variance_explained_percentage < threshold).item() + 1
        actual_variance = variance_explained_percentage[components_needed - 1].item()
        print(f"Layer {layer_idx}: {components_needed} components explain {actual_variance:.2f}% of variance (target: {threshold}%)")
    
    # Exp 2.1c: Compare reconstruction error vs variance explained
    # This shows the relationship between the two metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Use mean reconstruction error across all tokens for cleaner visualization
    mean_reconstruction_error = torch.mean(reconstruction_error_PC, dim=0)
    
    ax1.plot(range(num_components), mean_reconstruction_error, label='Mean reconstruction error')
    ax1.set_xlabel("Number of PCA components")
    ax1.set_ylabel("Mean L2 reconstruction error")
    ax1.set_title(f"Mean reconstruction error (Layer {layer_idx})")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(range(1, len(variance_explained_percentage) + 1), 100 - variance_explained_percentage, 
             label='Unexplained variance %', color='orange')
    ax2.set_xlabel("Number of PCA components")
    ax2.set_ylabel("Unexplained variance (%)")
    ax2.set_title(f"Unexplained variance (Layer {layer_idx})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, f"reconstruction_vs_variance_layer_{layer_idx}.png"))
    print(f"Saved comparison plot to {os.path.join(ARTIFACTS_DIR, f'reconstruction_vs_variance_layer_{layer_idx}.png')}")
    plt.close()

    # Exp 2.2: Plot explained variance of tokens at pos=0

    tokens_BPD = act_LBPD[layer_idx, :, :, :]
    Vt_CD = Vt_LCD[layer_idx, :, :]

    total_var_P = torch.var(tokens_BPD, dim=0).sum(dim=-1)

    # Compute PCA coefficients
    pca_BPC = torch.einsum("bpd,cd->bpc", tokens_BPD, Vt_CD)
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


