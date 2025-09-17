from dataclasses import dataclass, is_dataclass, fields
from typing import Any, List, Dict
from itertools import product
import os
import json
from pathlib import Path
import torch as th

from src.dictionary import AutoEncoder, JumpReluAutoEncoder, AutoEncoderTopK, BatchTopKSAE


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
    text_inputs_dir="artifacts/inputs",
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


LLAMA3_LLM_CFG = LLMConfig(
    name="Llama-3.1-8B",
    hf_name="meta-llama/Llama-3.1-8B",
    revision=None,
    layer_idx=16,
    hidden_dim=4096,
    batch_size=100,
)

GEMMA2_LLM_CFG = LLMConfig(
    name="Gemma-2-2B",
    hf_name="google/gemma-2-2b",
    revision=None,
    layer_idx=12,
    hidden_dim=2304,
    batch_size=100,
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

SIMPLESTORIES_DS_CFG = DatasetConfig(
    name="SimpleStories",
    hf_name="SimpleStories/SimpleStories",
    num_sequences=1000,
    context_length=500,
)

CODE_DS_CFG = DatasetConfig(
    name="Code",
    hf_name="neelnanda/code-10K",
    num_sequences=1000,
    context_length=500,
)


######### SAE ##########


@dataclass
class SAEConfig:
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
}

TOPK_GEMMA2_SAE_CFG = SAEConfig(
    name="Top K",
    local_weights_path="artifacts/trained_saes/gemma2/top_k",
    dict_class="top_k",
    dict_size=4096,
    batch_size=100,
)
BTOPK_GEMMA2_SAE_CFG = SAEConfig(
    name="Batch Top K",
    local_weights_path="artifacts/trained_saes/gemma2/batch_top_k",
    dict_class="batch_top_k",
    dict_size=4096,
    batch_size=100,
)
RELU_GEMMA2_SAE_CFG = SAEConfig(
    name="ReLU",
    local_weights_path="artifacts/trained_saes/gemma2/relu",
    dict_class="relu",
    dict_size=4096,
    batch_size=100,
)
JUMPRELU_GEMMA2_SAE_CFG = SAEConfig(
    name="Jump ReLU",
    local_weights_path="artifacts/trained_saes/gemma2/jump_relu",
    dict_class="jump_relu",
    dict_size=4096,
    batch_size=100,
)

GEMMA2_SAE_CFGS = [
    RELU_GEMMA2_SAE_CFG,
    JUMPRELU_GEMMA2_SAE_CFG,
    TOPK_GEMMA2_SAE_CFG,
    BTOPK_GEMMA2_SAE_CFG,
]

TOPK_LLAMA3_SAE_CFG = SAEConfig(
    name="Top K",
    local_weights_path="artifacts/trained_saes/llama3/top_k",
    dict_class="top_k",
    dict_size=16384,
    batch_size=100,
)
BTOPK_LLAMA3_SAE_CFG = SAEConfig(
    name="Batch Top K",
    local_weights_path="artifacts/trained_saes/llama3/batch_top_k",
    dict_class="batch_top_k",
    dict_size=16384,
    batch_size=100,
)
RELU_LLAMA3_SAE_CFG = SAEConfig(
    name="ReLU",
    local_weights_path="artifacts/trained_saes/llama3/relu",
    dict_class="relu",
    dict_size=16384,
    batch_size=100,
)
JUMPRELU_LLAMA3_SAE_CFG = SAEConfig(
    name="Jump ReLU",
    local_weights_path="artifacts/trained_saes/llama3/jump_relu",
    dict_class="jump_relu",
    dict_size=16384,
    batch_size=100,
)

LLAMA3_SAE_CFGS = [
    RELU_LLAMA3_SAE_CFG,
    JUMPRELU_LLAMA3_SAE_CFG,
    TOPK_LLAMA3_SAE_CFG,
    BTOPK_LLAMA3_SAE_CFG,
]

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
                if verbose:
                    print(
                        f"Attribute value mismatch at path '{current_path}': "
                        f"source has '{source_value}' but target has '{target_value}'"
                    )
                return False
    return True


def find_matching_config_folder(
    source_object, target_folder: str, recency_rank: int = 0, compared_attributes=None
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
            source_object, config_dict, compared_attributes=compared_attributes
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


def load_matching_artifacts(
    source_object,
    target_filenames: List[str],
    target_folder: str,
    recency_rank: int = 0,
    compared_attributes=None,
) -> Dict[str, th.Tensor]:
    target_dir = find_matching_config_folder(
        source_object, target_folder, recency_rank, compared_attributes
    )
    artifacts = {}

    for fn in target_filenames:
        path = os.path.join(target_dir, f"{fn}.pt")
        with open(path, "rb") as f:
            artifacts[fn] = th.load(f, weights_only=False)
    return artifacts, target_dir


def load_llm_activations(cfg, recency_rank=0, return_target_dir=False):
    """Wrapper for load_matching_artifacts specifically for loading llm activations."""
    artifacts, target_dir = load_matching_artifacts(
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
