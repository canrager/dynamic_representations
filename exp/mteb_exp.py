# import mteb
# from sentence_transformers import SentenceTransformer

# # Define the sentence-transformers model name
# model_name = "sentence-transformers/all-MiniLM-L6-v2"


# # load the model using MTEB
# model = mteb.get_model(
#     model_name
# )  # will default to SentenceTransformers(model_name) if not implemented in MTEB
# # or using SentenceTransformers
# # model = SentenceTransformers(model_name)

# # select the desired tasks and evaluate
# tasks = mteb.get_tasks(tasks=["Banking77Classification"])
# evaluation = mteb.MTEB(tasks=tasks)
# results = evaluation.run(model)


import mteb
from mteb.encoder_interface import PromptType
import torch as th
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import hashlib
import json
from sentence_transformers.model_card import SentenceTransformerModelCardData

from src.configs import *
from src.model_utils import load_nnsight_model, load_sae, load_tokenizer
from src.cache_utils import tokenize_from_sequences, batch_llm_cache, batch_sae_cache


@dataclass
class MTEBConfig:
    llm: LLMConfig
    sae: SAEConfig
    env: EnvironmentConfig

    sae_act_type: str | None = None  # the index into SAE results
    max_length: int | None = 500


def generate_cache_key(sentences, llm_config, batch_size, device, dtype):
    """Generate a unique hash key for caching based on inputs."""
    # Create a deterministic representation of the inputs
    cache_dict = {
        "sentences": sentences,
        "llm_name": llm_config.name,
        "llm_hf_name": llm_config.hf_name,
        "layer_idx": llm_config.layer_idx,
        "hidden_dim": llm_config.hidden_dim,
        "batch_size": batch_size,
        "device": device,
        "dtype": dtype,
    }
    # Create hash from JSON string
    cache_str = json.dumps(cache_dict, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()


def load_llm_acts_cache(cache_key):
    """Try to load cached llm_acts_BPD from disk.

    Args:
        cache_key: The unique cache key

    Returns:
        llm_acts_BPD tensor if cache exists, None otherwise
    """
    cache_dir = os.path.join("artifacts", "llm_cache", cache_key)
    cache_path = os.path.join(cache_dir, "llm_acts.pt")

    if os.path.exists(cache_path):
        print(f"Loading cached llm_acts_BPD from {cache_path}")
        return th.load(cache_path, weights_only=False)
    return None


def save_llm_acts_cache(
    cache_key, llm_acts_BPD, llm_config, batch_size, device, dtype, num_sentences
):
    """Save llm_acts_BPD to cache with metadata.

    Args:
        cache_key: The unique cache key
        llm_acts_BPD: The activations tensor to save
        llm_config: LLM configuration
        batch_size: Batch size used
        device: Device used
        dtype: Data type used
        num_sentences: Number of sentences processed
    """
    cache_dir = os.path.join("artifacts", "llm_cache", cache_key)
    cache_path = os.path.join(cache_dir, "llm_acts.pt")
    metadata_path = os.path.join(cache_dir, "metadata.json")

    os.makedirs(cache_dir, exist_ok=True)
    th.save(llm_acts_BPD, cache_path)

    metadata = {
        "llm_name": llm_config.name,
        "llm_hf_name": llm_config.hf_name,
        "layer_idx": llm_config.layer_idx,
        "hidden_dim": llm_config.hidden_dim,
        "batch_size": batch_size,
        "device": device,
        "dtype": dtype,
        "num_sentences": num_sentences,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved llm_acts_BPD to cache: {cache_path}")


def pairwise_cosine_similarity(x_LD):
    eps = 1e-8
    x_norm_LD = x_LD / (th.norm(x_LD, dim=-1, keepdim=True) + eps)
    return th.matmul(x_norm_LD, x_norm_LD.T)


def save_similarity_heatmaps(sae_acts_BPD, inputs_BP, masks_BP, tokenizer, cfg, num_sequences=3):
    """Save similarity heatmaps for the first num_sequences."""
    os.makedirs(cfg.env.plots_dir, exist_ok=True)

    num_sequences = min(num_sequences, sae_acts_BPD.shape[0])

    for seq_idx in range(num_sequences):
        # Get activations for this sequence
        acts_PD = sae_acts_BPD[seq_idx]  # P x D

        # Compute pairwise cosine similarity
        sim_matrix = pairwise_cosine_similarity(acts_PD).float().cpu()

        # Tokenize to get tokens for labels
        input_ids = inputs_BP[seq_idx][masks_BP[seq_idx].bool()]
        tokens = [tokenizer.decode(t) for t in input_ids]
        num_tokens = len(tokens)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(sim_matrix[:num_tokens, :num_tokens], cmap="magma")

        ax.set_title(f"Seq {seq_idx} - {cfg.sae_act_type}", fontsize=12)
        ax.set_xticks(range(num_tokens))
        ax.set_yticks(range(num_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Tokens")

        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        # Save
        model_name = cfg.llm.name
        if cfg.sae is not None:
            model_name += f"_{cfg.sae.name}"

        save_name = f"similarity_seq{seq_idx}_{model_name}_{cfg.sae_act_type}.png"
        plot_path = os.path.join(cfg.env.plots_dir, save_name)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"saved figure to: {plot_path}")
        plt.close()


class CustomModel:
    def __init__(self, cfg):
        self.cfg = cfg
        embedder_name = cfg.llm.hf_name

        # Load Gemma 2 only if needed
        self.llm = None
        self.tokenizer = load_tokenizer(cfg)
        self.submodule = None

        # Load SAE
        if cfg.sae is not None:
            self.sae = load_sae(cfg)
            embedder_name += f"_{cfg.sae.name}"

        # Initialize model card data
        self.model_card_data = SentenceTransformerModelCardData(
            model_name=embedder_name,
            language="eng-Latn",
            license="apache-2.0",
            task_name="semantic textual similarity, text classification, semantic search",
            tags=[
                "sentence-transformers",
                "sparse-autoencoders",
                "interpretability",
            ],
        )
        self.similarity_fn_name = "cosine"

    def encode(
        self,
        sentences: list[str],
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs,
    ) -> th.tensor:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences
        """

        # Generate cache key
        cache_key = generate_cache_key(
            sentences,
            self.cfg.llm,
            self.cfg.llm.batch_size,
            self.cfg.env.device,
            self.cfg.env.dtype,
        )

        inputs_BP, masks_BP, _ = tokenize_from_sequences(
            self.tokenizer, sentences, max_length=self.cfg.max_length
        )
        
        # Try to load from cache
        llm_acts_BPD = load_llm_acts_cache(cache_key)

        if llm_acts_BPD is None:

            if self.llm is None:
                self.llm, self.submodule, _ = load_nnsight_model(cfg)

            llm_acts_BPD, _ = batch_llm_cache(
                self.llm,
                self.submodule,
                inputs_BP,
                masks_BP,
                hidden_dim=self.cfg.llm.hidden_dim,
                batch_size=self.cfg.llm.batch_size,
                device=self.cfg.env.device,
                dtype=self.cfg.env.dtype,
                debug=False,
            )

            # Save to cache
            save_llm_acts_cache(
                cache_key,
                llm_acts_BPD,
                self.cfg.llm,
                self.cfg.llm.batch_size,
                self.cfg.env.device,
                self.cfg.env.dtype,
                len(sentences),
            )

        if self.cfg.sae is not None:
            sae_results = batch_sae_cache(self.sae, llm_acts_BPD, cfg)
            sae_acts_BPD = sae_results[self.cfg.sae_act_type]

            acts_BPD = sae_acts_BPD
        else:
            acts_BPD = llm_acts_BPD

        # Save similarity heatmaps for first 3 sequences
        save_similarity_heatmaps(
            acts_BPD,
            inputs_BP,
            masks_BP,
            self.tokenizer,
            self.cfg,
            num_sequences=3,
        )

        # Select the final token embedding
        final_token_indices_B = masks_BP.sum(dim=1) - 1  # B
        batch_arange_B = th.arange(final_token_indices_B.shape[0])
        final_acts_BD = acts_BPD[batch_arange_B, final_token_indices_B, :]

        return final_acts_BD.cpu().float()


# evaluating the model:
# cfg = MTEBConfig(llm=GEMMA2_LLM_CFG, sae=None, env=ENV_CFG)
# cfg = MTEBConfig(llm=GEMMA2_LLM_CFG, sae=BATCHTOPK_SELFTRAIN_SAE_CFG, env=ENV_CFG, sae_act_type="codes")
cfg = MTEBConfig(
    llm=GEMMA2_LLM_CFG,
    sae=TEMPORAL_2X_WIDER_SELFTRAIN_SAE_CFG,
    env=ENV_CFG,
    sae_act_type="pred_codes",
)
# cfg = MTEBConfig(
#     llm=GEMMA2_LLM_CFG, sae=TEMPORAL_SELFTRAIN_SAE_CFG, env=ENV_CFG, sae_act_type="pred_codes"
# )
model = CustomModel(cfg)
# tasks = mteb.get_tasks(tasks=["Banking77Classification"])
# tasks = mteb.get_tasks(tasks=["ToxicConversationsClassification"])
# tasks = mteb.get_tasks(tasks=["AmazonCounterfactualClassification"])
task_names = [
    "ArxivClassification.v2",
    "PoemSentimentClassification.v2",
    "EmotionClassification.v2",
    "AmazonCounterfactualClassification",
    "MAUDLegalBenchClassification.v2",
]
tasks = mteb.get_tasks(
    task_types=["Classification"],
    tasks=task_names,
    languages=["eng"],
    script=["Latn"],
    modalities=["text"],
    exclusive_modality_filter=True,
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, verbosity=3)
