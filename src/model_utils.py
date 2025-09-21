import os
from nnsight import LanguageModel
import torch as th
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from src.project_config import MODELS_DIR, DEVICE
from src.custom_saes.relu_sae import load_dictionary_learning_relu_sae
from src.custom_saes.topk_sae import load_dictionary_learning_topk_sae
from src.configs import SAE_STR_TO_CLASS, DTYPE_STR_TO_CLASS


def load_tokenizer(llm_name: str, cache_dir: str):
    if "gpt2" in llm_name:
        tokenizer = GPT2Tokenizer.from_pretrained(llm_name, cache_dir=MODELS_DIR)
        tokenizer.add_bos_token = True
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=cache_dir)

    return tokenizer


def load_nnsight_model(cfg):
    model = LanguageModel(
        cfg.llm.hf_name,
        revision=cfg.llm.revision,
        cache_dir=cfg.env.hf_cache_dir,
        device_map=cfg.env.device,  # Use the defined device
        torch_dtype=cfg.env.dtype,
        dispatch=True,
    )

    if "gpt2" in cfg.llm.name:
        print(model)
        print(model.config)
        hidden_dim = model.config.n_embd
        submodule = model.transformer.h[cfg.llm.layer_idx]

        # Language Model loads the AutoTokenizer, which does not use the add_bos_token method.
        model.tokenizer = load_tokenizer(cfg.llm.name, cache_dir=MODELS_DIR)

    elif any([s in cfg.llm.name.lower() for s in ["llama", "gemma", "allenai", "qwen", "mistral"]]):
        print(model)
        print(model.config)
        hidden_dim = model.config.hidden_size
        submodule = model.model.layers[cfg.llm.layer_idx]
    else:
        raise ValueError("Unknown model")

    return model, submodule, hidden_dim


def load_hf_model(
    llm_name: str,
    cache_dir: str,
    device: str,
    quantization_bits: int = None,
    tokenizer_only: bool = False,
    verbose: bool = True,
):
    """
    Load a huggingface model and tokenizer.

    Args:
        device (str): Device to load the model on ('auto', 'cuda', 'cpu', etc.)
        cache_dir (str, optional): Directory to cache the downloaded model
    """

    if verbose and cache_dir is not None:
        local_path = os.path.join(cache_dir, f"models--{llm_name.replace('/', '--')}")
        local_path_exists = os.path.exists(local_path)
        if local_path_exists:
            print(f"Model exists in {local_path}")
        else:
            print(f"Model does not exist in {local_path}")

    # Load tokenizer
    tokenizer = load_tokenizer(llm_name, cache_dir)
    if tokenizer_only:
        return tokenizer

    # Determine quantization
    torch_dtype = th.bfloat16

    if quantization_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quantization_config_str = "8-bit"
    elif quantization_bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        quantization_config_str = "4-bit"
    else:
        quantization_config = None
        quantization_config_str = "no"

    if verbose:
        print(f"Using {quantization_config_str} quantization and bfloat16")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch_dtype,
        device_map=device,
        quantization_config=quantization_config,
        cache_dir=cache_dir,
    )

    # Optimize for inference
    model.eval()
    model = th.compile(model)

    return model, tokenizer

def load_dictionarylearning_sae(cfg):
    dict_class = SAE_STR_TO_CLASS[cfg.sae.dict_class]
    dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
    sae = dict_class.from_pretrained(
        path=f"{cfg.sae.local_weights_path}/ae.pt",
        device=cfg.env.device,
        dtype=dtype,
    )
    sae.activation_dim = cfg.sae.dict_size
    return sae

def load_selftrain_sae(cfg):
    dict_class = SAE_STR_TO_CLASS[cfg.sae.dict_class]
    dtype = DTYPE_STR_TO_CLASS[cfg.env.dtype]
    sae = dict_class.from_pretrained(
        folder_path=cfg.sae.local_weights_path,
        device=cfg.env.device,
        dtype=dtype,
    )
    return sae

def load_sae(cfg):
    if "selftrain" in cfg.sae.local_weights_path:
        return load_selftrain_sae(cfg)
    else:
        return load_dictionarylearning_sae(cfg)
