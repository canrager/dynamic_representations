"""
Defining config structures.

Current Experiment Hierarchy:
TextDataset -> LLM activations (-> SAE activations) -> Analysis -> Plot

text dataset
LLM activations
    SAE activations
single experiment

We use a grouped config config management.
Stuff that's interchanged frequently should be grouped together.
for a given sweep, only one group will be changed at a time.
"""

from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
from typing import Any
import torch as th

DEVICE = "cuda"
MODELS_DIR = Path("/home/can/models")
PLOTS_DIR = Path("artifacts/plots")
INTERIM_DIR = Path("artifacts/interim")
INPUTS_DIR = Path("artifacts/inputs")
ACTIVATIONS_DIR = Path("artifacts/activations")


@dataclass
class BaseConfig:

    # Experiment
    experiment_name: str = "experiment"

    # Dataset
    dataset_name: str = ""
    hf_text_identifier: str = ""
    num_sequences: int = 0
    context_length: int = 0  # Num tokens per sequence

    # LLM
    llm_name: str = ""
    layer_idx: int = 0
    llm_hidden_dim: int = 0
    llm_batch_size: int = 0
    revision: str | None = None
    # force_recompute: bool
    save_artifacts: bool = False # TODO remove this one by adapting compute llm artifacts
    overwrite_existing: bool = False

    # Environment
    verbose: bool = False

    device: str = "cuda"
    dtype: str = "float32"

    hfcache_dir = Path("/home/can/models")
    plots_dir = Path("artifacts/plots")
    results_dir = Path("artifacts/results")
    inputs_dir = Path("artifacts/inputs")
    activations_dir = Path("artifacts/activations")
    interim_dir = Path("artifacts/interim")

    # Ustat
    normalize_activations: bool = None
    p_start: int = None
    p_end: int = None
    num_p: int = None
    do_log_p: bool = None
    omit_bos: bool = None
    
    # Rank measurement
    reconstruction_threshold: float = 0.9
    
    # Unique identifier for experiments
    unique_id: str = None
    
    # Optional filename for saving artifacts
    filename: str = None



@dataclass(frozen=True)
class LLMConfig:
    name: str
    layer_idx: int
    batch_size: int
    revision: str | None
    force_recompute: bool
    dtype: str


@dataclass(frozen=True)
class SAEConfig:
    dict_class: Any
    dict_size: int
    batch_size: int
    name: str
    local_filename: str | None
    force_recompute: bool


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    hf_text_identifier: str
    # num_sequences: int
    # ctx_length: int # Num tokens per sequence
