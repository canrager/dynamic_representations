import os
import sys
from typing import Optional
from nnsight import LanguageModel
import torch
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from sparsify import Sae
from src.project_config import MODELS_DIR, DEVICE
from src.custom_saes.relu_sae import load_dictionary_learning_relu_sae
from src.custom_saes.topk_sae import load_dictionary_learning_topk_sae


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
        cfg.llm_name,
        revision=cfg.llm_revision,
        cache_dir=MODELS_DIR,
        device_map=DEVICE,  # Use the defined device
        dispatch=True,
    )

    if "gpt2" in cfg.llm_name:
        print(model)
        print(model.config)
        hidden_dim = model.config.n_embd
        submodules = [model.transformer.h[l] for l in range(model.config.n_layer)]

        # Language Model loads the AutoTokenizer, which does not use the add_bos_token method.
        model.tokenizer = load_tokenizer(cfg.llm_name, cache_dir=MODELS_DIR)

    elif any([s in cfg.llm_name.lower() for s in ["llama", "gemma", "allenai", "qwen", "mistral"]]):
        print(model)
        print(model.config)
        hidden_dim = model.config.hidden_size
        submodules = [
            model.model.layers[l] for l in range(model.config.num_hidden_layers)
        ]
    else:
        raise ValueError("Unknown model")

    return model, submodules, hidden_dim


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
    torch_dtype = torch.bfloat16

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
    model = torch.compile(model)

    return model, tokenizer


def load_sae(cfg):
    if "eleuther" in cfg.sae_name.lower():
        sae_hookpoint_str = f"layers.{cfg.layer_idx}"
        sae = Sae.load_from_hub(cfg.sae_name, sae_hookpoint_str, cache_dir=MODELS_DIR)
        sae = sae.to(DEVICE)
    elif "saebench" in cfg.sae_name.lower():
        if cfg.sae_architecture == "topk":
            sae = load_dictionary_learning_topk_sae(
                repo_id=cfg.sae_repo_id,
                filename=cfg.sae_filename,
                model_name=cfg.llm_name,
                device=DEVICE,
                layer=cfg.layer_idx,
                dtype=cfg.dtype
            )
        elif cfg.sae_architecture == "relu":
            sae = load_dictionary_learning_relu_sae(
                repo_id=cfg.sae_repo_id,
                filename=cfg.sae_filename,
                model_name=cfg.llm_name,
                device=DEVICE,
                layer=cfg.layer_idx,
                dtype=cfg.dtype
            )
    else:
        raise NotImplementedError
    return sae
