"""
Test script to load chat dataset, tokenize, and decode single tokens.
"""

import torch as th
from src.configs import CHAT_DS_CFG, IT_GEMMA2_LLM_CFG, ENV_CFG
from src.model_utils import load_nnsight_model
from src.cache_utils import collect_from_hf

def main():
    # Setup config for instruction-tuned Gemma
    llm_cfg = IT_GEMMA2_LLM_CFG
    env_cfg = ENV_CFG
    data_cfg = CHAT_DS_CFG

    # Load model and tokenizer
    print(f"Loading model: {llm_cfg.hf_name}")
    model, submodule, hidden_dim = load_nnsight_model(
        type('Config', (), {
            'llm': llm_cfg,
            'env': env_cfg
        })()
    )

    tokenizer = model.tokenizer
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")

    # Load a small sample of chat data
    print(f"\nLoading chat dataset: {data_cfg.hf_name}")
    print(f"Number of sequences: {data_cfg.num_sequences}")
    print(f"Context length: {data_cfg.context_length}")

    inputs_BP, masks_BP, selected_idxs = collect_from_hf(
        tokenizer=tokenizer,
        dataset_name=data_cfg.hf_name,
        num_stories=5,  # Just load 5 examples for testing
        num_tokens=data_cfg.context_length
    )

    print(f"\nTokenized shape: {inputs_BP.shape}")
    print(f"Masks shape: {masks_BP.shape}")

    # Decode single tokens for the first sequence
    print("\n" + "="*80)
    print("DECODING SINGLE TOKENS (First Sequence)")
    print("="*80)

    for idx, token_id in enumerate(inputs_BP[0]):  # First 50 tokens
        token_str = tokenizer.decode(token_id, skip_special_tokens=False)
        # Replace newlines for better display
        token_str_display = token_str.replace("\n", "\\n")
        print(f"Token {idx:3d} | ID: {token_id:6d} | '{token_str_display}'")

    # Decode the full sequence
    print("\n" + "="*80)
    print("FULL DECODED SEQUENCE")
    print("="*80)
    full_text = tokenizer.decode(inputs_BP[0], skip_special_tokens=False)
    print(full_text)  # First 500 chars
    print("...")

if __name__ == "__main__":
    main()
