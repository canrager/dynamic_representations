# Which parts of the decoder vector are used for the novel reconstruction?
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import torch as th
import matplotlib.pyplot as plt
from umap import UMAP
from einops import repeat

from src.model_utils import load_sae, load_hf_model
from src.exp_utils import load_tokens_of_story
from src.configs import *
from src.plotting_utils import savefig
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

    # UMAP parameters
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42

    # Plotting parameters
    num_sequences: int | None = None  # Number of sequences to plot (None = all)
    connect_sequences: bool = False  # Connect consecutive points within sequences
    hover_window: int = 500  # Number of tokens to show before/after current position in hover text


def generate_umap(
    act_BLD, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
):
    """
    Apply UMAP dimensionality reduction to activations.

    Args:
        act_BLD: torch.Tensor of shape (batch, layers, dim)
        n_components: number of dimensions to reduce to (typically 2 or 3)
        n_neighbors: balance local vs global structure (15-50 typical)
        min_dist: minimum distance between points in low-D space (0.0-0.99)
        metric: distance metric ('euclidean', 'cosine', 'manhattan', etc.)
        random_state: random seed for reproducibility

    Returns:
        embedding: torch.Tensor of shape (batch, n_components)
    """
    # Reshape to (batch, features)
    B, L, D = act_BLD.shape
    X = act_BLD.reshape(B * L, D).cpu().float().numpy()

    # Initialize and fit UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)
    embedding = th.from_numpy(embedding).reshape(B, L, -1)
    print(embedding.shape)
    return embedding


def experiment(act_BLD, tokens_BL, tokenizer, cfg: ExperimentConfig):
    """
    Run UMAP dimensionality reduction and save results.

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
    pos_labels = repeat(ps, "L -> B L", B=B)  # (batch, num_p)

    # Generate UMAP embedding
    print(f"Generating UMAP embedding for {act_BLD_subsampled.shape[0]} samples...")
    embedding = generate_umap(
        act_BLD_subsampled,
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.random_state,
    )

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
        "hover_texts": hover_texts,
        "config": asdict(cfg),
    }

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.act_path == "activations":
        title_prefix = "LLM_activations"
    elif cfg.act_path == "codes":
        title_prefix = f"{cfg.sae.name}_codes"
    elif cfg.act_path == "pred_codes":
        title_prefix = f"{cfg.sae.name}_pred_codes"
    elif cfg.act_path == "novel_codes":
        title_prefix = f"{cfg.sae.name}_novel_codes"
    else:
        raise ValueError()

    save_path = os.path.join(
        cfg.env.results_dir,
        f"pred_structure_{datetetime_str}_{title_prefix}_{cfg.llm.name}_{cfg.data.name}.json",
    )
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to: {save_path}")

    return embedding, pos_labels


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
                ],
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
        min_p=0,
        max_p=499,
        num_p=499,
        do_log_scale=False,
        # UMAP parameters
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        # Artifacts
        env=ENV_CFG,
        # data=[WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG, CODE_DS_CFG],
        # data=WEBTEXT_DS_CFG,
        # data=DatasetConfig(
        #     name="Twist",
        #     hf_name="twist.json",
        #     num_sequences=1,  # Total number of sentences in JSON
        #     context_length=500,  # Use variable length with padding
        # ),
        # data=CHAT_DS_CFG,
        data=SIMPLESTORIES_DS_CFG,
        # llm=IT_GEMMA2_LLM_CFG,
        llm=GEMMA2_LLM_CFG,
        # llm=LLAMA3_LLM_CFG,
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
