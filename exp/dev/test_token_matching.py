#!/usr/bin/env python3
"""
Test script to verify token-label matching functionality.
"""
import json
import torch
from types import SimpleNamespace

# Create a minimal config object for testing
cfg = SimpleNamespace()
cfg.llm_name = "meta-llama/Llama-3.1-8B"
cfg.input_file_str = "test_relative_clause"
cfg.force_recompute = True
cfg.selected_story_idxs = None
cfg.omit_BOS_token = True
cfg.num_tokens_per_story = None
cfg.llm_batch_size = 4
cfg.debug = False
cfg.dataset_name = "test"
cfg.num_total_stories = 7
cfg.hf_text_identifier = "text"

# Load the relative clause data
with open("artifacts/inputs/relative_clause_loc.json", "r") as f:
    data = json.load(f)

# Extract a few examples for testing
test_examples = data["sentence_pairs"][:3]  # Use first 3 examples

# Prepare the word lists and labels
word_lists = []
word_labels = []
sequences = []

# Add a simple adversarial example where each word maps to one token
print("Test data:")
print("Adversarial example (simple case):")
simple_words = ["The", "detokenization", "works"]
simple_labels = ["article", "subject", "verb"]
word_lists.append(simple_words)
word_labels.append(simple_labels)
sequences.append(" ".join(simple_words))
print(f"  Words: {simple_words}")
print(f"  Labels: {simple_labels}")
print(f"  Text: {' '.join(simple_words)}")
print()

for i, example in enumerate(test_examples):
    # Test both mid and late relative clause versions
    for version in ["mid_rel_clause", "late_rel_clause"]:
        word_list = example[f"{version}_string"]
        label_list = example[f"{version}_label"]
        
        word_lists.append(word_list)
        word_labels.append(label_list)
        sequences.append(" ".join(word_list))
        
        print(f"Example {i+1} ({version}):")
        print(f"  Words: {word_list}")
        print(f"  Labels: {label_list}")
        print(f"  Text: {' '.join(word_list)}")
        print()

print(f"Total test sequences: {len(sequences)}")

# Test the function
try:
    from src.exp_utils import compute_or_load_llm_artifacts
    
    print("Testing compute_or_load_llm_artifacts with word labels...")
    
    acts_LBPD, masks_BP, tokens_BP, token_labels_BP, story_idxs = compute_or_load_llm_artifacts(
        cfg, 
        loaded_dataset_sequences=sequences,
        loaded_word_lists=word_lists,
        loaded_word_labels=word_labels
    )
    
    print("Success! Results:")
    print(f"Acts shape: {acts_LBPD.shape}")
    print(f"Masks shape: {masks_BP.shape}")
    print(f"Tokens shape: {tokens_BP.shape}")
    print(f"Token labels type: {type(token_labels_BP)}")
    
    if token_labels_BP:
        print(f"Number of sequences with labels: {len(token_labels_BP)}")
        
        # Load tokenizer to decode tokens for inspection
        from src.model_utils import load_tokenizer
        from src.project_config import MODELS_DIR
        
        tokenizer = load_tokenizer(cfg.llm_name, MODELS_DIR)
        
        # Show token-label mapping for first few sequences
        for seq_idx in range(min(3, len(sequences))):
            print(f"\nToken-label mapping for sequence {seq_idx}:")
            print(f"Original: {sequences[seq_idx]}")
            print(f"Words: {word_lists[seq_idx]}")
            print(f"Labels: {word_labels[seq_idx]}")
            print("Tokens and their labels:")
            
            for token_idx in range(min(15, tokens_BP.shape[1])):  # Show first 15 tokens
                token_id = tokens_BP[seq_idx, token_idx].item()
                if token_id == tokenizer.pad_token_id:
                    break
                    
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                label = token_labels_BP[seq_idx][token_idx] if token_idx < len(token_labels_BP[seq_idx]) else ""
                print(f"  {token_idx:2d}: '{token_text}' -> '{label}'")
            
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()