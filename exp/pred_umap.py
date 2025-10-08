# Which parts of the decoder vector are used for the novel reconstruction?
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import torch as th
import matplotlib.pyplot as plt
from umap import UMAP

from src.model_utils import load_sae, load_hf_model
from src.exp_utils import load_tokens_of_story
from src.configs import *
from src.plotting_utils import savefig


@dataclass
class ExperimentConfig:
    llm: LLMConfig
    sae: SAEConfig
    env: EnvironmentConfig
    data: DatasetConfig

    act_type: str

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
    X = act_BLD.reshape(B, L * D).cpu().float().numpy()

    # Initialize and fit UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)
    return th.from_numpy(embedding)


def plot_umap(
    embedding,
    labels=None,
    title="UMAP Projection",
    save_path=None,
    figsize=(10, 8),
    s=5,
    cmap="tab10",
    alpha=0.6,
):
    """
    Visualize UMAP embedding.

    Args:
        embedding: torch.Tensor or numpy array of shape (n_samples, 2 or 3)
        labels: optional labels for coloring points
        title: plot title
        save_path: optional path to save figure
        figsize: figure size tuple
        s: point size
        cmap: colormap for labels
        alpha: point transparency
    """
    if isinstance(embedding, th.Tensor):
        embedding = embedding.cpu().numpy()

    fig = plt.figure(figsize=figsize)

    if embedding.shape[1] == 2:
        if labels is not None:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=cmap, s=s, alpha=alpha)
            plt.colorbar()
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], s=s, alpha=alpha)
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
    elif embedding.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                embedding[:, 2],
                c=labels,
                cmap=cmap,
                s=s,
                alpha=alpha,
            )
            plt.colorbar(scatter)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=s, alpha=alpha)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")

    plt.title(title)

    if save_path:
        savefig(save_path)
    else:
        plt.show()

    return fig


def experiment(act_BLD, cfg: ExperimentConfig):
    """
    Run UMAP dimensionality reduction and save results.

    Args:
        act_BLD: activations tensor of shape (batch, layers, dim)
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
    # Each batch sample gets all positions, so we repeat position indices
    B = act_BLD_subsampled.shape[0]
    pos_labels = ps.repeat(B)  # (batch * num_p,)

    # Reshape for UMAP: treat each (batch, position) as a separate sample
    B, L, D = act_BLD_subsampled.shape
    act_reshaped = act_BLD_subsampled.permute(0, 1, 2).reshape(B * L, D)  # (batch*num_p, dim)

    # Generate UMAP embedding
    print(f"Generating UMAP embedding for {act_reshaped.shape[0]} samples...")
    embedding = generate_umap(
        act_reshaped.unsqueeze(1),  # Add dummy layer dimension for compatibility
        n_components=cfg.n_components,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=cfg.random_state,
    )

    # Prepare results
    results = {
        "embedding": embedding.cpu().tolist(),
        "pos_labels": pos_labels.tolist(),
        "pos_indices": ps.tolist(),
        "config": asdict(cfg),
    }

    # Save results
    datetetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f"pred_structure_{datetetime_str}.json")
    with open(save_path, "w") as f:
        json.dump(results, f)
    print(f"Saved results to: {save_path}")

    return embedding, pos_labels


def main():
    cfg = ExperimentConfig(
        sae=TEMPORAL_SELFTRAIN_SAE_CFG,
        act_type="pred_codes",
        # sae=BATCHTOPK_SELFTRAIN_SAE_CFG,
        # act_type="codes",
        # sae=None,
        # act_type="activations",
        llm=GEMMA2_LLM_CFG,
        env=ENV_CFG,
        # data=WEBTEXT_DS_CFG,
        data=CODE_DS_CFG,
        # data=SIMPLESTORIES_DS_CFG,
        # Position subsampling
        min_p=10,
        max_p=499,
        num_p=20,
        do_log_scale=False,
        # UMAP parameters
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    # Load activations
    if cfg.sae is not None:
        act_type = os.path.join(cfg.sae.name, cfg.act_type)
    else:
        act_type = cfg.act_type

    artifacts, _ = load_matching_activations(
        cfg,
        [act_type],
        cfg.env.activations_dir,
        compared_attributes=["llm", "data"],
        verbose=False,
    )
    act_BLD = artifacts[act_type]
    act_BLD = act_BLD.to(cfg.env.device)

    print(f"==>> act_BLD.shape: {act_BLD.shape}")

    # Run experiment
    experiment(act_BLD, cfg)


if __name__ == "__main__":
    main()
