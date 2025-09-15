from dataclasses import dataclass
from typing import Any

from src.dictionary import AutoEncoder, JumpReluAutoEncoder, AutoEncoderTopK, BatchTopKSAE


######## Environment #########

@dataclass
class EnvironmentConfig:
    device: str
    dtype: str

    hfcache_dir: str
    plots_dir: str
    results_dir: str
    text_inputs_dir: str
    activations_dir: str


CAN_ENV_CFG = EnvironmentConfig(
    device = "cuda",
    dtype = "float32",
    hfcache_dir = "/home/can/models",
    plots_dir = "artifacts/plots",
    results_dir = "artifacts/results",
    text_inputs_dir = "artifacts/inputs",
    activations_dir = "artifacts/activations",
)


######## LLM #########

@dataclass
class LLMConfig:
    name: str
    hf_name: str
    revision: str
    hidden_dim: int
    batch_size: int

LLAMA3_LLM_CFG = LLMConfig(
    name="Llama-3.1-8B",
    hf_name="meta-llama/Llama-3.1-8B",
    revision=None,
    hidden_dim=4096,
    batch_size=500
)

GEMMA2_LLM_CFG = LLMConfig(
    name="Gemma-2-2B",
    hf_name="google/gemma-2-2b",
    revision=None,
    hidden_dim=2304,
    batch_size=500
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

DTYPE_STR_TO_CLASS = {
    "batch_top_k": BatchTopKSAE,
    "top_k": AutoEncoderTopK,
    "relu": AutoEncoder,
    "jump_relu": JumpReluAutoEncoder
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
    RELU_GEMMA2_SAE_CFG, JUMPRELU_GEMMA2_SAE_CFG, TOPK_GEMMA2_SAE_CFG, BTOPK_GEMMA2_SAE_CFG
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
    RELU_LLAMA3_SAE_CFG, JUMPRELU_LLAMA3_SAE_CFG, TOPK_LLAMA3_SAE_CFG, BTOPK_LLAMA3_SAE_CFG
]