import os
from tqdm import trange
from matplotlib import pyplot as plt
import torch
from src.exp_utils import compute_or_load_llm_artifacts, compute_or_load_sae
from src.project_config import PLOTS_DIR


def plot_fvu_stories(fvu_BP, mask_BP, num_stories, model_str, sae_str, layer_idx):
    """
    Plot FVU (Fraction of Variance Unexplained) for multiple stories.

    Args:
        fvu_BP: FVU tensor with shape (batch, position)
        mask_BP: Mask tensor with shape (batch, position)
        num_stories: Number of stories to plot
        model_str: Model name string for filename
        sae_str: SAE name string for filename
        layer_idx: Layer index for filename
    """
    save_fname = f"fvu_BP_model-{model_str}_{sae_str}_layer-{layer_idx}.png"
    save_path = os.path.join(PLOTS_DIR, save_fname)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for story_idx in range(num_stories):
        masked_fvu = fvu_BP[story_idx, mask_BP[story_idx].bool()]
        ax.plot(masked_fvu.cpu().numpy(), label=f"Story {story_idx}")
        plt.grid(True)
    ax.legend()
    plt.savefig(save_path)
    print(f"Saved fvu plot to {save_path}")
    plt.close()


def plot_num_active_stories(
    num_active_BP, mask_BP, num_stories, model_str, sae_str, layer_idx, threshold
):
    """
    Plot number of active latents for multiple stories.

    Args:
        num_active_BP: Number of active latents tensor with shape (batch, position)
        mask_BP: Mask tensor with shape (batch, position)
        num_stories: Number of stories to plot
        model_str: Model name string for filename
        sae_str: SAE name string for filename
        layer_idx: Layer index for filename
        threshold: Threshold used to determine active latents
    """
    save_fname = f"num_active_BP_model-{model_str}_{sae_str}_layer-{layer_idx}_thresh-{threshold:.0e}.png"
    save_path = os.path.join(PLOTS_DIR, save_fname)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for story_idx in range(num_stories):
        masked_num_active = num_active_BP[story_idx, mask_BP[story_idx].bool()]
        ax.plot(masked_num_active.cpu().numpy(), label=f"Story {story_idx}")
        plt.grid(True)
    ax.legend()
    ax.set_ylabel("Number of Active Latents")
    plt.savefig(save_path)
    print(f"Saved num_active plot to {save_path}")
    plt.close()


def plot_latent_acts_distribution(
    latent_acts_BPS, model_str, sae_str, layer_idx, bins=100
):
    """
    Plot the distribution of all flattened latent activations.

    Args:
        latent_acts_BPS: Latent activations tensor with shape (batch, position, sparse_dim)
        model_str: Model name string for filename
        sae_str: SAE name string for filename
        layer_idx: Layer index for filename
        bins: Number of bins for histogram
    """
    save_fname = f"latent_acts_dist_model-{model_str}_{sae_str}_layer-{layer_idx}.png"
    save_path = os.path.join(PLOTS_DIR, save_fname)

    # Flatten all latent activations
    flattened_acts = latent_acts_BPS.flatten().cpu().to(torch.float32).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Full distribution
    ax1.hist(flattened_acts, bins=bins, alpha=0.7, density=True)
    ax1.set_xlabel("Latent Activation Value")
    ax1.set_ylabel("Density")
    ax1.set_title("Full Distribution of Latent Activations")
    ax1.grid(True)

    # Log scale for better visualization of small values
    ax2.hist(flattened_acts, bins=bins, alpha=0.7, density=True)
    ax2.set_xlabel("Latent Activation Value")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of Latent Activations (Log Scale)")
    ax2.set_yscale("log")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved latent acts distribution plot to {save_path}")
    plt.close()


def plot_num_active_thresholds_single_story(
    latent_acts_BPS,
    mask_BP,
    story_idx,
    model_str,
    sae_str,
    layer_idx,
    thresholds,
    fvu_BP,
):
    """
    Plot number of active latents for a single story across multiple threshold values,
    and FVU over position for the same story.

    Args:
        latent_acts_BPS: Latent activations tensor with shape (batch, position, sparse_dim)
        mask_BP: Mask tensor with shape (batch, position)
        story_idx: Index of the story to plot
        model_str: Model name string for filename
        sae_str: SAE name string for filename
        layer_idx: Layer index for filename
        thresholds: List of threshold values to plot
        fvu_BP: FVU tensor with shape (batch, position)
    """
    save_fname = f"num_active_thresholds_story-{story_idx}_model-{model_str}_{sae_str}_layer-{layer_idx}.png"
    save_path = os.path.join(PLOTS_DIR, save_fname)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Get the mask for this story
    story_mask = mask_BP[story_idx].bool()

    # Top subplot: Number of active latents for different thresholds
    for threshold in thresholds:
        # Compute number of active latents for this threshold
        num_active_P = (latent_acts_BPS[story_idx].abs() > threshold).sum(dim=-1)
        # Apply mask to get only valid positions
        masked_num_active = num_active_P[story_mask]

        ax1.plot(
            masked_num_active.cpu().numpy(),
            label=f"Threshold {threshold:.1f}",
            linewidth=2,
        )

    ax1.set_xlabel("Position")
    ax1.set_ylabel("Number of Active Latents")
    ax1.set_title(f"Number of Active Latents vs Threshold (Story {story_idx})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: FVU over position
    masked_fvu = fvu_BP[story_idx, story_mask]
    ax2.plot(masked_fvu.cpu().numpy(), color="red", linewidth=2)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Fraction of Variance Unexplained (FVU)")
    ax2.set_title(f"FVU over Position (Story {story_idx})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved num_active thresholds + FVU plot to {save_path}")
    plt.close()


def plot_mean_stats_across_stories(
    latent_acts_BPS, mask_BP, fvu_BP, model_str, sae_str, layer_idx, thresholds
):
    """
    Plot mean number of active latents and mean FVU across all stories with 95% confidence intervals.

    Args:
        latent_acts_BPS: Latent activations tensor with shape (batch, position, sparse_dim)
        mask_BP: Mask tensor with shape (batch, position)
        fvu_BP: FVU tensor with shape (batch, position)
        model_str: Model name string for filename
        sae_str: SAE name string for filename
        layer_idx: Layer index for filename
        thresholds: List of threshold values to plot
    """
    save_fname = (
        f"mean_stats_across_stories_model-{model_str}_{sae_str}_layer-{layer_idx}.png"
    )
    save_path = os.path.join(PLOTS_DIR, save_fname)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9))

    # Find maximum position across all stories
    max_pos = mask_BP.sum(dim=-1).max().item()

    # Top subplot: Mean number of active latents with 95% CI bands
    for threshold in thresholds:
        # Compute number of active latents for this threshold
        num_active_BP = (latent_acts_BPS.abs() > threshold).sum(
            dim=-1
        )  # (batch, position)

        mean_num_active = []
        ci_num_active = []
        positions = []

        for pos in range(max_pos):
            # Get valid stories for this position
            valid_stories = mask_BP[:, pos].bool()
            num_valid = valid_stories.sum().item()
            if num_valid > 0:  # At least one story has valid token at this position
                valid_num_active = num_active_BP[valid_stories, pos]
                mean_val = valid_num_active.float().mean().cpu().item()

                if num_valid > 1:
                    std_val = valid_num_active.float().std().cpu().item()
                    # 95% confidence interval: 1.96 * std / sqrt(n)
                    ci_val = 1.96 * std_val / (num_valid**0.5)
                else:
                    ci_val = 0.0

                mean_num_active.append(mean_val)
                ci_num_active.append(ci_val)
                positions.append(pos)

        positions = torch.tensor(positions)
        mean_num_active = torch.tensor(mean_num_active)
        ci_num_active = torch.tensor(ci_num_active)

        # Plot mean line
        ax1.plot(
            positions, mean_num_active, label=f"Threshold {threshold:.2f}", linewidth=2
        )

        # Plot 95% CI band
        ax1.fill_between(
            positions,
            mean_num_active - ci_num_active,
            mean_num_active + ci_num_active,
            alpha=0.2,
        )

    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Mean Number of Active Latents")
    ax1.set_title("Mean Number of Active Latents Across All Stories (95% CI)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Mean FVU with 95% CI bands
    mean_fvu = []
    ci_fvu = []
    positions = []

    for pos in range(max_pos):
        # Get valid stories for this position
        valid_stories = mask_BP[:, pos].bool()
        num_valid = valid_stories.sum().item()
        if num_valid > 0:  # At least one story has valid token at this position
            valid_fvu = fvu_BP[valid_stories, pos]
            mean_val = valid_fvu.float().mean().cpu().item()

            if num_valid > 1:
                std_val = valid_fvu.float().std().cpu().item()
                # 95% confidence interval: 1.96 * std / sqrt(n)
                ci_val = 1.96 * std_val / (num_valid**0.5)
            else:
                ci_val = 0.0

            mean_fvu.append(mean_val)
            ci_fvu.append(ci_val)
            positions.append(pos)

    positions = torch.tensor(positions)
    mean_fvu = torch.tensor(mean_fvu)
    ci_fvu = torch.tensor(ci_fvu)

    # Plot mean line
    ax2.plot(positions, mean_fvu, color="red", linewidth=2, label="Mean FVU")

    # Plot 95% CI band
    ax2.fill_between(
        positions,
        mean_fvu - ci_fvu,
        mean_fvu + ci_fvu,
        color="red",
        alpha=0.2,
        label="95% CI",
    )

    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Mean Fraction of Variance Unexplained (FVU)")
    ax2.set_title("Mean FVU Across All Stories (95% CI)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved mean stats across stories plot to {save_path}")
    plt.close()


if __name__ == "__main__":

    layer_idx = 12
    device = "cuda"
    model_name = "meta-llama/Llama-3.1-8B"
    sae_name = "EleutherAI/sae-llama-3-8b-32x"
    num_stories = 100

    force_recompute = False
    # Changing the following parameters will require to force recompute
    do_omit_BOS_token = True
    do_truncate_seq_length = False
    max_tokens_per_story = 300

    model_str = model_name.split("/")[-1]
    sae_str = sae_name.split("/")[-1]

    fvu_BP, latent_acts_BPS, latent_indices_BPK, mask_BP = compute_or_load_sae(
        sae_name=sae_name,
        model_name=model_name,
        num_stories=num_stories,
        layer_idx=layer_idx,
        batch_size=1000,  # Need to adapt sparsify.SAE to not sum fvu loss over batch dimension
        device=device,
        force_recompute=force_recompute,
        do_omit_BOS_token=do_omit_BOS_token,
        do_truncate_seq_length=do_truncate_seq_length,
        max_tokens_per_story=max_tokens_per_story,
    )

    # Call the externalized plotting function
    plot_fvu_stories(fvu_BP, mask_BP, num_stories, model_str, sae_str, layer_idx)

    threshold = 0.1
    num_active_BP = (latent_acts_BPS.abs() > threshold).sum(dim=-1)

    # Call the plotting function for num_active
    plot_num_active_stories(
        num_active_BP, mask_BP, num_stories, model_str, sae_str, layer_idx, threshold
    )

    # Plot multiple thresholds for a single story
    thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
    plot_num_active_thresholds_single_story(
        latent_acts_BPS,
        mask_BP,
        story_idx=1,
        model_str=model_str,
        sae_str=sae_str,
        layer_idx=layer_idx,
        thresholds=thresholds,
        fvu_BP=fvu_BP,
    )

    # Plot the activation density histogram
    plot_latent_acts_distribution(latent_acts_BPS, model_str, sae_str, layer_idx)

    # Plot mean stats across stories
    mean_thresholds = [0.25, 0.4]
    plot_mean_stats_across_stories(
        latent_acts_BPS, mask_BP, fvu_BP, model_str, sae_str, layer_idx, mean_thresholds
    )
