"""
Garden Path Sentence Entropy Analysis:

1. Find pairs of garden path sentence variations.
2. Tokenize. Hopefully, they're token-aligned.
3. Cache LLM and SAE activations
4. Compute entropy of LLM and SAE activations.

Entropy is computed as softmax across the final hidden_dim/sae_latent_dim
followed by Shannon entropy. We only compute entropy for the final num_final_tokens
as determined by the attention masks.

Results are saved as [batch, num_final_tokens] tensors for both magnitude and entropy,
structured as gp_type: tensor.tolist() (e.g., "ambiguous": list).
"""

import torch as th
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
import gc

from src.configs import *


@dataclass
class GardenPathEntropyConfig:
    num_final_tokens: int  # Number of final tokens to analyze
    act_path: str  # Activation file to analyze (e.g., "activations", "codes", "novel_codes")
    scaling_factor: 0.00666666667

    env: EnvironmentConfig
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig | None  # None for LLM activations


def compute_entropy_and_magnitude(acts_BPD: th.Tensor, masks_BP: th.Tensor, num_final_tokens: int, debug=True):
    """
    Compute Shannon entropy and L2 magnitude for the final num_final_tokens.

    Args:
        acts_BPD: Activations tensor [B, P, D]
        masks_BP: Attention masks [B, P]
        num_final_tokens: Number of final tokens to analyze

    Returns:
        entropy_BF: Shannon entropy for final tokens [B, num_final_tokens]
        magnitude_BF: L2 magnitude for final tokens [B, num_final_tokens]
        final_positions_BF: Token position indices for final tokens [B, num_final_tokens]
    """
    B, P, D = acts_BPD.shape

    if debug:
        print(f"  Debug: acts_BPD shape: {acts_BPD.shape}")
        print(f"  Debug: acts_BPD range: [{acts_BPD.min():.4f}, {acts_BPD.max():.4f}]")
        print(f"  Debug: acts_BPD mean: {acts_BPD.mean():.4f}, std: {acts_BPD.std():.4f}")

    # Find the final tokens for each sequence based on masks
    entropy_BF = []
    magnitude_BF = []
    final_positions_BF = []

    for b in range(B):
        # Find the last num_final_tokens valid positions
        valid_positions = th.where(masks_BP[b] == 1)[0]
        if len(valid_positions) < num_final_tokens:
            # If sequence is shorter than num_final_tokens, use all valid positions
            final_positions = valid_positions
        else:
            # Take the last num_final_tokens positions
            final_positions = valid_positions[-num_final_tokens:]

        # if debug and b == 0:  # Debug first sequence
            # print(f"  Debug: valid_positions length: {len(valid_positions)}")
            # print(f"  Debug: final_positions: {final_positions.tolist()}")

        # Extract activations for final positions
        final_acts_FD = acts_BPD[b, final_positions, :]  # [F, D]

        # if debug and b == 0:
            # print(f"==>> final_acts_FD: {final_acts_FD}")
            # print(f"  Debug: final_acts_FD shape: {final_acts_FD.shape}")
            # print(f"  Debug: final_acts_FD range: [{final_acts_FD.min():.4f}, {final_acts_FD.max():.4f}]")

        # Compute L2 magnitude
        magnitude_F = th.norm(final_acts_FD, dim=-1)  # [F]

        # Compute Shannon entropy with temperature scaling
        temperature = 1  # You can experiment with different values
        scaled_acts_FD = final_acts_FD / temperature
        scaled_acts_FD = scaled_acts_FD / ((scaled_acts_FD > 0).float().mean(dim=-1, keepdim=True) *0.008)
        print((scaled_acts_FD > 0).float().mean(dim=-1, keepdim=True) *0.008)

        # Apply softmax across the feature dimension
        probs_FD = F.softmax(scaled_acts_FD, dim=-1)  # [F, D]

        # if debug and b == 0:
            # print(f"==>> probs_FD: {probs_FD}")
            # print(f"  Debug: probs_FD max per token: {probs_FD.max(dim=-1)[0]}")
            # print(f"  Debug: probs_FD min per token: {probs_FD.min(dim=-1)[0]}")
            # print(f"  Debug: probs_FD entropy should be < {th.log(th.tensor(float(D))):.4f}")

        # Compute Shannon entropy: -sum(p * log(p))
        # Use more stable computation
        log_probs_FD = F.log_softmax(
            scaled_acts_FD, dim=-1
        )
        entropy_F = -th.sum(probs_FD * log_probs_FD, dim=-1)  # [F]

        if debug and b == 0:
            print(f"  Debug: entropy values: {entropy_F}")

        # Pad positions and values to num_final_tokens if needed
        if len(final_positions) < num_final_tokens:
            pad_size = num_final_tokens - len(final_positions)
            entropy_F = F.pad(entropy_F, (0, pad_size), value=0.0)
            magnitude_F = F.pad(magnitude_F, (0, pad_size), value=0.0)
            # Pad positions with -1 to indicate invalid positions
            final_positions = F.pad(final_positions, (0, pad_size), value=-1)

        entropy_BF.append(entropy_F)
        magnitude_BF.append(magnitude_F)
        final_positions_BF.append(final_positions)

    entropy_BF = th.stack(entropy_BF)  # [B, num_final_tokens]
    magnitude_BF = th.stack(magnitude_BF)  # [B, num_final_tokens]
    final_positions_BF = th.stack(final_positions_BF)  # [B, num_final_tokens]

    if debug:
        print(f"  Debug: Final entropy range: [{entropy_BF.min():.4f}, {entropy_BF.max():.4f}]")
        print(f"  Debug: Final magnitude range: [{magnitude_BF.min():.4f}, {magnitude_BF.max():.4f}]")
        print(f"  Debug: Entropy variance: {entropy_BF.var():.4f}")

    return entropy_BF, magnitude_BF, final_positions_BF


def load_garden_path_activations(cfg):
    """
    Load garden path activations for all three sentence types.

    Returns:
        Dict with keys 'ambiguous', 'gp', 'post', each containing:
        - acts_BPD: Activations tensor
        - masks_BP: Attention masks tensor
        - tokens_BP: Token IDs tensor
    """
    sentence_types = ['ambiguous', 'gp', 'post']
    results = {}

    for sentence_type in sentence_types:
        if cfg.sae is None:
            # Load LLM activations
            acts_BPD, target_dir = load_matching_activations(
                source_object=cfg,
                target_filenames=[f"sentence_{sentence_type}"],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
            )
            acts_BPD = acts_BPD[f"sentence_{sentence_type}"]
            # acts_BPD /= cfg.scaling_factor

            # Load corresponding masks
            masks_artifacts, _ = load_matching_activations(
                source_object=cfg,
                target_filenames=[f"masks_{sentence_type}"],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
            )
            masks_BP = masks_artifacts[f"masks_{sentence_type}"]

            # Load corresponding tokens
            tokens_artifacts, _ = load_matching_activations(
                source_object=cfg,
                target_filenames=[f"tokens_{sentence_type}"],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
            )
            tokens_BP = tokens_artifacts[f"tokens_{sentence_type}"]

        else:
            # Load SAE activations
            target_fname = os.path.join(cfg.sae.name, sentence_type, cfg.act_path)
            acts_BPD, target_dir = load_matching_activations(
                source_object=cfg,
                target_filenames=[target_fname],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
                # subdirectory_path=os.path.join(cfg.sae.name, sentence_type),
                verbose=False,
            )
            acts_BPD = acts_BPD[target_fname]
            # acts_BPD /= cfg.scaling_factor

            # Load corresponding masks (from LLM cache)
            masks_artifacts, _ = load_matching_activations(
                source_object=cfg,
                target_filenames=[f"masks_{sentence_type}"],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
            )
            masks_BP = masks_artifacts[f"masks_{sentence_type}"]

            # Load corresponding tokens (from LLM cache)
            tokens_artifacts, _ = load_matching_activations(
                source_object=cfg,
                target_filenames=[f"tokens_{sentence_type}"],
                target_folder=cfg.env.activations_dir,
                recency_rank=0,
                compared_attributes=["llm", "data"],
            )
            tokens_BP = tokens_artifacts[f"tokens_{sentence_type}"]

        results[sentence_type] = {
            'acts_BPD': acts_BPD,
            'masks_BP': masks_BP,
            'tokens_BP': tokens_BP
        }

    return results


def run_garden_path_entropy_analysis(cfg):
    """
    Run entropy analysis on garden path sentences.

    Returns:
        results: Dict containing entropy and magnitude results for each sentence type
    """
    print(f"Loading garden path activations for {cfg.act_path}...")

    # Load activations for all sentence types
    activations_data = load_garden_path_activations(cfg)

    # Create position indices: negative relative indices leading up to 0
    # e.g., num_final=3 -> [-2, -1, 0]
    pos_indices = list(range(-cfg.num_final_tokens + 1, 1))

    results = {
        'entropy': {},
        'magnitude': {},
        'tokens': {},
        'pos_indices': pos_indices,
        'config': asdict(cfg)
    }

    for sentence_type, data in activations_data.items():
        print(f"Computing entropy and magnitude for {sentence_type} sentences...")

        acts_BPD = data['acts_BPD']
        masks_BP = data['masks_BP']
        tokens_BP = data['tokens_BP']

        # Move to GPU for computation
        acts_BPD = acts_BPD.to(cfg.env.device)
        masks_BP = masks_BP.to(cfg.env.device)

        # Compute entropy and magnitude
        entropy_BF, magnitude_BF, final_positions_BF = compute_entropy_and_magnitude(
            acts_BPD, masks_BP, cfg.num_final_tokens
        )

        # Move back to CPU and convert to lists
        entropy_BF = entropy_BF.cpu()
        magnitude_BF = magnitude_BF.cpu()
        final_positions_BF = final_positions_BF.cpu()

        # Extract the corresponding tokens for each sequence
        B, F = final_positions_BF.shape
        tokens_for_sequences = []
        for b in range(B):
            sequence_tokens = []
            for f in range(F):
                pos = final_positions_BF[b, f].item()
                if pos >= 0:  # Valid position (not padded)
                    sequence_tokens.append(tokens_BP[b, pos].item())
                else:
                    sequence_tokens.append(-1)  # Placeholder for invalid positions
            tokens_for_sequences.append(sequence_tokens)

        results['entropy'][sentence_type] = entropy_BF.tolist()
        results['magnitude'][sentence_type] = magnitude_BF.tolist()
        results['tokens'][sentence_type] = tokens_for_sequences

        print(f"  {sentence_type}: entropy shape {entropy_BF.shape}, magnitude shape {magnitude_BF.shape}")

        # Clean up GPU memory
        del acts_BPD, masks_BP, entropy_BF, magnitude_BF, final_positions_BF
        th.cuda.empty_cache()
        gc.collect()

    return results


def main():
    """Main function to run garden path entropy analysis."""
    configs = get_gemma_act_configs(
        cfg_class=GardenPathEntropyConfig,
        scaling_factor=0.00666666667,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    "surrogate"
                ]
            ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    "codes",
                    "recons"
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    "novel_codes",
                    "novel_recons",
                    "pred_codes",
                    "pred_recons",
                    "total_recons",
                ]
            ),
        ),
        num_final_tokens=5,  # Analyze the final 3 tokens
        data=DatasetConfig(
            name="GardenPath",
            hf_name="garden_path.csv",
            num_sequences=97,
            context_length=None,
        ),
        llm=LLMConfig(
            name="Gemma-2-2B",
            hf_name="google/gemma-2-2b",
            revision=None,
            layer_idx=12,
            hidden_dim=2304,
            batch_size=50,
        ),
        env=ENV_CFG,
    )

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Processing: {cfg.act_path}")
        if cfg.sae is not None:
            print(f"SAE: {cfg.sae.name}")
        print(f"{'='*60}")

        # Run entropy analysis
        results = run_garden_path_entropy_analysis(cfg)

        # Save results
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sae_name = cfg.sae.name if cfg.sae is not None else "llm"
        filename = f"garden_path_entropy_{sae_name}_{cfg.act_path}_{datetime_str}.json"
        save_path = os.path.join(cfg.env.results_dir, filename)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {save_path}")



if __name__ == "__main__":
    main()