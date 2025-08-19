from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
from typing import Any

DEVICE = "cuda"
MODELS_DIR = Path("/home/can/models")
PLOTS_DIR = Path("artifacts/plots")
INTERIM_DIR = Path("artifacts/interim")
INPUTS_DIR = Path("artifacts/inputs")


@dataclass(frozen=True)
class LLMConfig:
    name: str
    layer_idx: int
    batch_size: int
    revision: str | None
    force_recompute: bool


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
