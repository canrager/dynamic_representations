# Which parts of the decoder vector are used for the novel reconstruction?
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import torch as th
import matplotlib.pyplot as plt

from src.model_utils import load_sae, load_hf_model
from src.exp_utils import load_tokens_of_story
from src.configs import *
from src.plotting_utils import savefig
from einops import repeat
import time
import gc


@dataclass
class ExperimentConfig:
    llm: LLMConfig
    sae: SAEConfig
    env: EnvironmentConfig
    data: DatasetConfig

    act_path: str

    # Position subsampling
    min_p: int
    max_p: int  # inclusive
    num_p: int
    do_log_scale: bool

    # PCA parameters
    n_components: int = 4
    center: bool = True

    # Plotting parameters
    num_sequences: int | None = None  # Number of sequences to plot (None = all)
    connect_sequences: bool = False  # Connect consecutive points within sequences
    hover_window: int = 50  # Number of tokens to show before/after current position in hover text


def generate_pca(act_BLD, n_components=4, center=True):
    """
    Apply PCA dimensionality reduction to activations.

    Args:
        act_BLD: torch.Tensor of shape (batch, layers, dim)
        n_components: number of principal components to compute
        center: whether to center the data before PCA

    Returns:
        embedding: torch.Tensor of shape (batch, n_components)
        explained_variance_ratio: torch.Tensor of shape (n_components,)
    """
    # Reshape to (batch, features)
    B, L, D = act_BLD.shape
    X = act_BLD.reshape(B * L, D).float()

    # Center the data
    if center:
        X_mean = X.mean(dim=0, keepdim=True)
        X_centered = X - X_mean
    else:
        X_centered = X

    # Compute SVD
    U, S, Vt = th.linalg.svd(X_centered, full_matrices=False)

    # Project onto principal components
    embedding = U[:, :n_components] * S[:n_components].unsqueeze(0)

    # Compute explained variance ratio
    explained_variance = S**2 / (X_centered.shape[0] - 1)
    explained_variance_ratio = explained_variance[:n_components] / explained_variance.sum()

    embedding = embedding.reshape(B, L, -1)

    return embedding, explained_variance_ratio


def experiment(act_BLD, tokens_BL, tokenizer, cfg: ExperimentConfig):
    """
    Run PCA dimensionality reduction and save results.

    Args:
        act_BLD: activations tensor of shape (batch, layers, dim)
        tokens_BL: tokens tensor of shape (batch, seq_len)
        tokenizer: tokenizer for decoding tokens
        cfg: experiment configuration
    """
    # Compute position indices for subsampling
    if cfg.do_log_scale:
        log_start = th.log10(th.tensor(cfg.min_p, dtype=th.float))
        log_end = th.log10(th.tensor(cfg.max_p, dtype=th.float))
        log_steps = th.linspace(log_start, log_end, cfg.num_p)
        ps = th.round(10**log_steps).int()
    else:
        ps = th.linspace(cfg.min_p, cfg.max_p, cfg.num_p, dtype=th.int)

    print(f"Position indices: {ps.tolist()}")

    # Subsample activations at selected positions
    # act_BLD shape: (batch, layers, dim)
    # We select specific layer positions
    act_BLD_subsampled = act_BLD[:, ps, :]  # (batch, num_p, dim)

    # Create position labels for each point
    # Each batch sample gets all positions
    B = act_BLD_subsampled.shape[0]
    pos_labels = repeat(ps, 'L -> B L', B=B)  # (batch, num_p)

    # Generate PCA embedding
    print(f"Generating PCA embedding for {act_BLD_subsampled.shape[0]} samples...")
    embedding, explained_variance_ratio = generate_pca(
        act_BLD_subsampled,
        n_components=cfg.n_components,
        center=cfg.center,
    )

    print(f"Explained variance ratio: {explained_variance_ratio.tolist()}")
    print(f"Cumulative explained variance: {explained_variance_ratio.cumsum(0).tolist()}")

    # Generate hover texts with token context
    print(f"Generating hover texts with token context...")
    hover_texts = []
    for b_idx in range(B):
        # Load tokens for this story once
        token_strs_L = load_tokens_of_story(
            tokens_BP=tokens_BL,
            story_idx=b_idx,
            model_name=cfg.llm.hf_name,
            omit_BOS_token=False,
            seq_length=None,
            tokenizer=tokenizer,
        )

        for l_idx_in_seq, pos_idx in enumerate(ps.tolist()):
            # pos_idx is the actual position in the full sequence
            # Show previous hover_window tokens + current token (bolded last)
            start_idx = max(0, pos_idx - cfg.hover_window)
            end_idx = min(len(token_strs_L), pos_idx + 1, cfg.max_p)

            story_str = ""
            token_count = 0
            for i in range(start_idx, end_idx):
                t = token_strs_L[i]

                # Add linebreak every 15 tokens
                if token_count > 0 and token_count % 15 == 0:
                    story_str += "<br>"

                if i == pos_idx:
                    story_str += f"<b>{t}</b>"  # Bold the current token (last one)
                else:
                    story_str += t

                token_count += 1

            hover_texts.append(f"Position: {pos_idx}<br>Story: {story_str}")

    # Prepare results
    results = {
        "embedding": embedding.cpu().tolist(),
        "pos_labels": pos_labels.tolist(),
        "pos_indices": ps.tolist(),
        "explained_variance_ratio": explained_variance_ratio.cpu().tolist(),
        "hover_texts": hover_texts,
        "config": asdict(cfg),
    }

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"pred_structure_pca_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to: {save_path}")

    return embedding, pos_labels, explained_variance_ratio


def main():
    configs = get_gemma_act_configs(
        cfg_class=ExperimentConfig,
        act_paths=(
            (
                [None],
                [
                    "activations",
                    # "surrogate"
                ],
            ),
            # (
            #     GEMMA2_STANDARD_SELFTRAIN_SAE_CFGS,
            #     [
            #         "codes",
            #         # "recons"
            #     ]
            # ),
            (
                [BATCHTOPK_SELFTRAIN_SAE_CFG],
                [
                    "codes",
                    # "recons"
                ]
            ),
            (
                [TEMPORAL_SELFTRAIN_SAE_CFG],
                [
                    # "novel_codes",
                    # "novel_recons",
                    "pred_codes",
                    # "pred_recons",
                    # "total_recons",
                ],
            ),
        ),
        # Position subsampling
        min_p=10,
        max_p=499,
        num_p=20,
        do_log_scale=False,
        # PCA parameters
        n_components=4,
        center=True,
        # Artifacts
        env=ENV_CFG,
        # data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        # data=WEBTEXT_DS_CFG,
        # data=SIMPLESTORIES_DS_CFG,
        # data=CODE_DS_CFG,
        data=CHAT_DS_CFG,
        llm=IT_GEMMA2_LLM_CFG,
        sae=None,  # set by act_paths
        act_path=None,  # set by act_paths
    )

    for cfg in configs:
        # Load activations
        if cfg.sae is not None:
            act_type = os.path.join(cfg.sae.name, cfg.act_path)
        else:
            act_type = cfg.act_path

        artifacts, _ = load_matching_activations(
            cfg,
            [act_type, "tokens"],
            cfg.env.activations_dir,
            compared_attributes=["llm", "data"],
            verbose=False,
        )
        act_BLD = artifacts[act_type]
        act_BLD = act_BLD.to(cfg.env.device)
        tokens_BL = artifacts["tokens"]

        print(f"==>> act_BLD.shape: {act_BLD.shape}")

        # Load tokenizer
        tokenizer = load_hf_model(cfg, tokenizer_only=True)

        # Run experiment
        experiment(act_BLD, tokens_BL, tokenizer, cfg)

        # Cleanup
        del act_BLD, tokens_BL
        th.cuda.empty_cache()
        gc.collect()
        time.sleep(1)


if __name__ == "__main__":
    main()
