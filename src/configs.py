from dataclasses import dataclass, is_dataclass, fields
from typing import Any, List, Dict
from itertools import product
import os
import json
from pathlib import Path
import torch as th

from src.dictionary import AutoEncoder, JumpReluAutoEncoder, AutoEncoderTopK, BatchTopKSAE
from sae.saeTemporal import TemporalSAE
from sae.saeStandard import SAEStandard


######## Environment #########


@dataclass
class EnvironmentConfig:
    device: str
    dtype: str

    hf_cache_dir: str
    plots_dir: str
    results_dir: str
    text_inputs_dir: str
    activations_dir: str


ENV_CFG = EnvironmentConfig(
    device="cuda",
    dtype="bfloat16",
    hf_cache_dir="/home/can/models",
    plots_dir="artifacts/plots",
    results_dir="artifacts/results",
    text_inputs_dir="artifacts/text_inputs",
    activations_dir="artifacts/activations",
)

DTYPE_STR_TO_CLASS = {"bfloat16": th.bfloat16, "float32": th.float32}


######## LLM #########


@dataclass
class LLMConfig:
    name: str
    hf_name: str
    revision: str
    layer_idx: int
    hidden_dim: int
    batch_size: int


LLAMA3_L15_LLM_CFG = LLMConfig(
    name="Llama-3.1-8B",
    hf_name="meta-llama/Llama-3.1-8B",
    revision=None,
    layer_idx=15,
    hidden_dim=4096,
    batch_size=10,
)

LLAMA3_L26_LLM_CFG = LLMConfig(
    name="Llama-3.1-8B",
    hf_name="meta-llama/Llama-3.1-8B",
    revision=None,
    layer_idx=26,
    hidden_dim=4096,
    batch_size=10,
)

GEMMA2_LLM_CFG = LLMConfig(
    name="Gemma-2-2B",
    hf_name="google/gemma-2-2b",
    revision=None,
    layer_idx=12,
    hidden_dim=2304,
    batch_size=50,
)

IT_GEMMA2_LLM_CFG = LLMConfig(
    name="Gemma-2-2B-IT",
    hf_name="google/gemma-2-2b-it",
    revision=None,
    layer_idx=12,
    hidden_dim=2304,
    batch_size=50,
)


######## Dataset ########


@dataclass
class DatasetConfig:
    name: str
    hf_name: str
    num_sequences: int
    context_length: int


WEBTEXT_DS_CFG = DatasetConfig(
    name="Webtext",
    hf_name="monology/pile-uncopyrighted",
    num_sequences=1000, 
    context_length=500,
)

WEBTEXT_SMALL_DS_CFG = DatasetConfig(
    name="Webtext",
    hf_name="monology/pile-uncopyrighted",
    num_sequences=10,
    context_length=500,
)

CHAT_DS_CFG = DatasetConfig(
    name="Chat",
    hf_name="HuggingFaceH4/ultrachat_200k",
    num_sequences=1000,
    context_length=500,
)
ALPACA_DS_CFG = DatasetConfig(
    name="Alpaca",
    hf_name="tatsu-lab/alpaca",
    num_sequences=1000,
    context_length=50,
)

SIMPLESTORIES_DS_CFG = DatasetConfig(
    name="SimpleStories",
    hf_name="SimpleStories/SimpleStories",
    num_sequences=1000,
    context_length=500,
)

SIMPLESTORIES_SMALL_DS_CFG = DatasetConfig(
    name="SimpleStories",
    hf_name="SimpleStories/SimpleStories",
    num_sequences=10,
    context_length=500,
)

CODE_DS_CFG = DatasetConfig(
    name="Code",
    hf_name="neelnanda/code-10k",
    num_sequences=1000,
    context_length=500,
)

TWIST_DS_CFG = DatasetConfig(
    name="Twist",
    hf_name="twist.json",
    num_sequences=1,  # Total number of sentences in JSON
    context_length=None,  # Use variable length with padding
)

######### SAE ##########


@dataclass
class SAEConfig:
    release: str
    name: str
    local_weights_path: str
    dict_class: Any
    dict_size: int
    batch_size: int


SAE_STR_TO_CLASS = {
    "batch_top_k": BatchTopKSAE,
    "top_k": AutoEncoderTopK,
    "relu": AutoEncoder,
    "jump_relu": JumpReluAutoEncoder,
    "temporal": TemporalSAE,
    "standard": SAEStandard,
}

# Gemma-2-2B SAE configs (layer 12)
GEMMA2_RELU_SAE_CFG = SAEConfig(
    release="singh",
    name="relu",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/relu",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_RELU_2X_DENSER_SAE_CFG = SAEConfig(
    release="singh",
    name="relu_2x_denser",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/relu_2x_denser",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_RELU_2X_WIDER_SAE_CFG = SAEConfig(
    release="singh",
    name="relu_2x_wider",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/relu_2x_wider",
    dict_class="standard",
    dict_size=18432,
    batch_size=10,
)

GEMMA2_TOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="topk",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/topk",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TOPK_2X_DENSER_SAE_CFG = SAEConfig(
    release="singh",
    name="topk_2x_denser",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/topk_2x_denser",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TOPK_2X_WIDER_SAE_CFG = SAEConfig(
    release="singh",
    name="topk_2x_wider",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/topk_2x_wider",
    dict_class="standard",
    dict_size=18432,
    batch_size=10,
)

GEMMA2_BATCHTOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="batchtopk",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/batchtopk",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_BATCHTOPK_2X_DENSER_SAE_CFG = SAEConfig(
    release="singh",
    name="batchtopk_2x_denser",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/batchtopk_2x_denser",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_BATCHTOPK_2X_WIDER_SAE_CFG = SAEConfig(
    release="singh",
    name="batchtopk_2x_wider",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/batchtopk_2x_wider",
    dict_class="standard",
    dict_size=18432,
    batch_size=10,
)

GEMMA2_MP_SAE_CFG = SAEConfig(
    release="singh",
    name="mp",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/mp",
    dict_class="standard",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_2X_DENSER_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_2x_denser",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_2x_denser",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_2X_WIDER_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_2x_wider",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_2x_wider",
    dict_class="temporal",
    dict_size=18432,
    batch_size=10,
)

GEMMA2_TEMPORAL_2X_LAYERS_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_2x_layers",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_2x_layers",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_4X_HEADS_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_4x_heads",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_4x_heads",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_RANK_1_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_rank_1",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_rank_1",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_RANK_2_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_rank_2",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_rank_2",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_RANK_1_LAYERS_2_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_rank_1_layers_2",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_rank_1_layers_2",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_TEMPORAL_PRED_ONLY_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal_pred_only",
    local_weights_path="artifacts/trained_saes/gemma-2-2B/layer_12/temporal_pred_only",
    dict_class="temporal",
    dict_size=9216,
    batch_size=10,
)

GEMMA2_SAE_CFGS = [
    GEMMA2_RELU_SAE_CFG,
    GEMMA2_RELU_2X_DENSER_SAE_CFG,
    GEMMA2_RELU_2X_WIDER_SAE_CFG,
    GEMMA2_TOPK_SAE_CFG,
    GEMMA2_TOPK_2X_DENSER_SAE_CFG,
    GEMMA2_TOPK_2X_WIDER_SAE_CFG,
    GEMMA2_BATCHTOPK_SAE_CFG,
    GEMMA2_BATCHTOPK_2X_DENSER_SAE_CFG,
    GEMMA2_BATCHTOPK_2X_WIDER_SAE_CFG,
    GEMMA2_MP_SAE_CFG,
    GEMMA2_TEMPORAL_SAE_CFG,
    GEMMA2_TEMPORAL_2X_DENSER_SAE_CFG,
    GEMMA2_TEMPORAL_2X_WIDER_SAE_CFG,
    GEMMA2_TEMPORAL_2X_LAYERS_SAE_CFG,
    GEMMA2_TEMPORAL_4X_HEADS_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_1_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_2_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_1_LAYERS_2_SAE_CFG,
    GEMMA2_TEMPORAL_PRED_ONLY_SAE_CFG,
]

GEMMA2_STANDARD_SAE_CFGS = [
    GEMMA2_RELU_SAE_CFG,
    GEMMA2_RELU_2X_DENSER_SAE_CFG,
    GEMMA2_RELU_2X_WIDER_SAE_CFG,
    GEMMA2_TOPK_SAE_CFG,
    GEMMA2_TOPK_2X_DENSER_SAE_CFG,
    GEMMA2_TOPK_2X_WIDER_SAE_CFG,
    GEMMA2_BATCHTOPK_SAE_CFG,
    GEMMA2_BATCHTOPK_2X_DENSER_SAE_CFG,
    GEMMA2_BATCHTOPK_2X_WIDER_SAE_CFG,
    GEMMA2_MP_SAE_CFG,
]

GEMMA2_TEMPORAL_SAE_CFGS = [
    GEMMA2_TEMPORAL_SAE_CFG,
    GEMMA2_TEMPORAL_2X_DENSER_SAE_CFG,
    GEMMA2_TEMPORAL_2X_WIDER_SAE_CFG,
    GEMMA2_TEMPORAL_2X_LAYERS_SAE_CFG,
    GEMMA2_TEMPORAL_4X_HEADS_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_1_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_2_SAE_CFG,
    GEMMA2_TEMPORAL_RANK_1_LAYERS_2_SAE_CFG,
    GEMMA2_TEMPORAL_PRED_ONLY_SAE_CFG,
]


# Llama-3.1-8B SAE configs (layer 16)
LLAMA3_L16_RELU_SAE_CFG = SAEConfig(
    release="singh",
    name="relu",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_16/relu",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L16_TOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="topk",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_16/topk",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L16_BATCHTOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="batchtopk",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_16/batchtopk",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L16_TEMPORAL_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_16/temporal",
    dict_class="temporal",
    dict_size=16384,
    batch_size=10,
)

# Llama-3.1-8B SAE configs (layer 26)
LLAMA3_L26_RELU_SAE_CFG = SAEConfig(
    release="singh",
    name="relu",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_26/relu",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L26_TOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="topk",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_26/topk",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L26_BATCHTOPK_SAE_CFG = SAEConfig(
    release="singh",
    name="batchtopk",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_26/batchtopk",
    dict_class="standard",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L26_TEMPORAL_SAE_CFG = SAEConfig(
    release="singh",
    name="temporal",
    local_weights_path="artifacts/trained_saes/llama-3.1-8B/layer_26/temporal",
    dict_class="temporal",
    dict_size=16384,
    batch_size=10,
)

LLAMA3_L16_SAE_CFGS = [
    LLAMA3_L16_RELU_SAE_CFG,
    LLAMA3_L16_TOPK_SAE_CFG,
    LLAMA3_L16_BATCHTOPK_SAE_CFG,
    LLAMA3_L16_TEMPORAL_SAE_CFG,
]

LLAMA3_L26_SAE_CFGS = [
    LLAMA3_L26_RELU_SAE_CFG,
    LLAMA3_L26_TOPK_SAE_CFG,
    LLAMA3_L26_BATCHTOPK_SAE_CFG,
    LLAMA3_L26_TEMPORAL_SAE_CFG,
]

LLAMA3_SAE_CFGS = LLAMA3_L16_SAE_CFGS + LLAMA3_L26_SAE_CFGS

######## Activation Cache Config ########


######## Utils ########


def get_configs(cfg_class, **kwargs):
    # Separate list arguments from single arguments
    list_args = {}

    for key, value in kwargs.items():
        if isinstance(value, list):
            list_args[key] = value
        else:
            list_args[key] = [value]  # Convert single items to lists

    # Get all combinations
    keys = list(list_args.keys())
    values = list(list_args.values())

    configs = []
    for combination in product(*values):
        config_kwargs = dict(zip(keys, combination))
        configs.append(cfg_class(**config_kwargs))

    return configs


def get_gemma_act_configs(cfg_class, act_paths, **kwargs):
    all_cfgs = []

    for sae_cfg, act_path in act_paths:
        kwargs["sae"] = sae_cfg
        kwargs["act_path"] = act_path
        cfgs = get_configs(cfg_class, **kwargs)
        for cfg in cfgs: 
            if cfg.sae is not None:
                cfg.act_path = os.path.join(cfg.sae.name, cfg.act_path)
            all_cfgs.append(cfg)

    return all_cfgs


def check_dataclass_overlap(source, target, path="", verbose=False, compared_attributes=None):
    """
    Check full overlap between two dataclass objects.

    For each attribute in source:
    1. Check if target has the same attribute name
    2. Check if the attribute values are equal (recursively for nested dataclasses)

    Args:
        source: Source dataclass object
        target: Target dataclass object
        path: Current path for error reporting (used in recursion)
        verbose: Print error messages instead of raising exceptions
        compared_attributes: List of attribute names to compare. If None, compare all attributes.

    Raises:
        ValueError: If overlap check fails with detailed error message
    """
    if not is_dataclass(source):
        raise ValueError(f"Source object at path '{path}' is not a dataclass")
    if not is_dataclass(target):
        raise ValueError(f"Target object at path '{path}' is not a dataclass")

    source_fields = {field.name: getattr(source, field.name) for field in fields(source)}

    # Filter fields based on compared_attributes
    if compared_attributes is not None:
        source_fields = {k: v for k, v in source_fields.items() if k in compared_attributes}

    for attr_name, source_value in source_fields.items():
        current_path = f"{path}.{attr_name}" if path else attr_name

        # Check if target has the attribute
        if not hasattr(target, attr_name):
            if verbose:
                print(f"Target object missing attribute '{attr_name}' at path '{current_path}'")
            return False

        target_value = getattr(target, attr_name)

        # If both values are dataclasses, recurse
        if is_dataclass(source_value) and is_dataclass(target_value):
            # Handle nested compared_attributes
            nested_compared_attributes = None
            if compared_attributes is not None and hasattr(compared_attributes, "__getitem__"):
                # If compared_attributes is a dict/dataclass, get nested attributes for this field
                if hasattr(compared_attributes, attr_name):
                    nested_compared_attributes = getattr(compared_attributes, attr_name)
                elif isinstance(compared_attributes, dict) and attr_name in compared_attributes:
                    nested_compared_attributes = compared_attributes[attr_name]

            if not check_dataclass_overlap(
                source_value, target_value, current_path, verbose, nested_compared_attributes
            ):
                return False
        else:
            # Check if values are equal
            if source_value != target_value:
                if verbose:
                    print(
                        f"Attribute value mismatch at path '{current_path}': "
                        f"source has '{source_value}' but target has '{target_value}'"
                    )
                return False
    return True


def check_dataclass_dict_overlap(
    source, target_dict, path="", verbose=False, compared_attributes=None
):
    """
    Check full overlap between a dataclass object and a dictionary (e.g., from asdict).

    For each attribute in source:
    1. Check if target_dict has the same key name
    2. Check if the values are equal (recursively for nested dataclasses)

    Args:
        source: Source dataclass object
        target_dict: Target dictionary (e.g., from asdict(dataclass_obj))
        path: Current path for error reporting (used in recursion)
        verbose: Print error messages instead of raising exceptions
        compared_attributes: List of attribute names to compare. If None, compare all attributes.

    Raises:
        ValueError: If overlap check fails with detailed error message
    """
    if not is_dataclass(source):
        raise ValueError(f"Source object at path '{path}' is not a dataclass")
    if not isinstance(target_dict, dict):
        raise ValueError(f"Target object at path '{path}' is not a dictionary")

    source_fields = {field.name: getattr(source, field.name) for field in fields(source)}

    # Filter fields based on compared_attributes
    if compared_attributes is not None:
        source_fields = {k: v for k, v in source_fields.items() if k in compared_attributes}

    for attr_name, source_value in source_fields.items():
        current_path = f"{path}.{attr_name}" if path else attr_name

        # Check if target_dict has the key
        if attr_name not in target_dict:
            if verbose:
                print(f"Target dictionary missing key '{attr_name}' at path '{current_path}'")
            return False

        target_value = target_dict[attr_name]

        # If source value is a dataclass and target value is a dict, recurse
        if is_dataclass(source_value) and isinstance(target_value, dict):
            # Handle nested compared_attributes
            nested_compared_attributes = None
            if compared_attributes is not None and hasattr(compared_attributes, "__getitem__"):
                # If compared_attributes is a dict/dataclass, get nested attributes for this field
                if hasattr(compared_attributes, attr_name):
                    nested_compared_attributes = getattr(compared_attributes, attr_name)
                elif isinstance(compared_attributes, dict) and attr_name in compared_attributes:
                    nested_compared_attributes = compared_attributes[attr_name]

            if not check_dataclass_dict_overlap(
                source_value, target_value, current_path, verbose, nested_compared_attributes
            ):
                return False
        else:
            # Check if values are equal
            if source_value != target_value:
                if source_value == "artifacts/text_inputs":
                    continue
                if verbose:
                    print(
                        f"Attribute value mismatch at path '{current_path}': "
                        f"source has '{source_value}' but target has '{target_value}'"
                    )
                return False
    return True


def find_matching_config_folder(
    source_object,
    target_folder: str,
    recency_rank: int = 0,
    compared_attributes=None,
    verbose=False,
) -> str:
    """
    Find a config folder that matches the source dataclass object based on recency.

    Args:
        source_object: Source dataclass object to match against
        target_folder: Path to folder containing datetime-named subfolders
        recency_rank: Rank order from most recent (0=most recent, 1=second most recent, etc.)
        compared_attributes: List of attribute names to compare. If None, compare all attributes.

    Returns:
        target_path: Full path to the selected target folder

    Raises:
        ValueError: If no matching configs found or recency_rank is out of range
        FileNotFoundError: If target_folder doesn't exist
    """
    if not is_dataclass(source_object):
        raise ValueError("source_object must be a dataclass")

    target_path = Path(target_folder)
    if not target_path.exists():
        raise FileNotFoundError(f"Target folder does not exist: {target_folder}")

    # Get all one-level subfolders
    subfolders = [d for d in target_path.iterdir() if d.is_dir()]

    # Load configs from subfolders that have config.json
    all_configs_dict = {}
    for subfolder in subfolders:
        config_file = subfolder / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                all_configs_dict[subfolder.name] = config_dict
            except (json.JSONDecodeError, IOError):
                # Skip folders with invalid/unreadable config.json
                continue

    if not all_configs_dict:
        raise ValueError(f"No valid config.json files found in subfolders of {target_folder}")

    # Filter configs that have overlap with source_object
    target_configs = {}
    for folder_name, config_dict in all_configs_dict.items():
        if check_dataclass_dict_overlap(
            source_object, config_dict, compared_attributes=compared_attributes, verbose=verbose
        ):
            target_configs[folder_name] = config_dict

    if not target_configs:
        raise ValueError("No configs found that match the source object")

    # Sort folder names by datetime (most recent first)
    # Assuming format like "20250916_101815"
    sorted_folders = sorted(target_configs.keys(), reverse=True)

    if recency_rank >= len(sorted_folders):
        raise ValueError(
            f"recency_rank_order {recency_rank} is out of range. "
            f"Only {len(sorted_folders)} matching configs found."
        )

    selected_folder = sorted_folders[recency_rank]
    return str(target_path / selected_folder)


def load_matching_activations(
    source_object,
    target_filenames: List[str],
    target_folder: str,
    recency_rank: int = 0,
    compared_attributes: List[str] = None,
    verbose: bool = False,
) -> Dict[str, th.Tensor]:
    target_dir = find_matching_config_folder(
        source_object, target_folder, recency_rank, compared_attributes, verbose
    )
    artifacts = {}

    for filetype in target_filenames:
        path = os.path.join(target_dir, f"{filetype}.pt")

        with open(path, "rb") as f:
            artifacts[filetype] = th.load(f, weights_only=False)
    return artifacts, target_dir


def load_matching_results(
    exp_name: str,
    source_cfg: Any,
    target_folder: str,
    recency_rank: int,
    compared_attributes: List[str],
    verbose,
):
    # Get all .json filenames in target folder
    json_files = [f for f in os.listdir(target_folder) if f.endswith(".json")]

    # Filter by exp_name in filename
    filtered_files = [f for f in json_files if f.startswith(exp_name)]

    # Load filtered results
    matching_results = []
    for filename in filtered_files:
        filepath = os.path.join(target_folder, filename)
        with open(filepath, "r") as f:
            result = json.load(f)
            result["_filename"] = filename

            # Filter by matching config in compared attributes
            if check_dataclass_dict_overlap(
                source_cfg,
                result["config"],
                verbose=verbose,
                compared_attributes=compared_attributes,
                path=filepath,
            ):
                matching_results.append(result)

    # Sort by recency (assuming filename contains timestamp or creation time)
    matching_results.sort(
        key=lambda x: os.path.getctime(os.path.join(target_folder, x["_filename"])), reverse=True
    )

    # Return result with selected recency as dict
    if len(matching_results) > recency_rank:
        selected_result = matching_results[recency_rank]
        if verbose:
            print(f"Loaded result from {selected_result['_filename']} (rank {recency_rank})")
        return selected_result
    else:
        if verbose:
            print(f"No result found at recency rank {recency_rank}")
        return None


def load_results_multiple_configs(
    exp_name: str,
    source_cfgs: List[Any],
    target_folder: str,
    recency_rank: int,
    compared_attributes: List[str],
    verbose: bool,
):
    results = {}
    for cfg in source_cfgs:
        result = load_matching_results(
            exp_name, cfg, target_folder, recency_rank, compared_attributes, verbose
        )
        results[result["_filename"]] = result

    return results


def load_llm_activations(cfg, recency_rank=0, return_target_dir=False):
    """Wrapper for load_matching_artifacts specifically for loading llm activations."""
    artifacts, target_dir = load_matching_activations(
        source_object=cfg,
        target_filenames=["activations"],
        target_folder=cfg.env.activations_dir,
        recency_rank=recency_rank,
        compared_attributes=["llm", "data"],
    )
    if not return_target_dir:
        return artifacts["activations"]
    else:
        return artifacts["activations"], target_dir
