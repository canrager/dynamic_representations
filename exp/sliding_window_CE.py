"""
This experiment evaluates how sliding window size affects a language model's cross-entropy loss performance.

Experiment Overview:
- Goal: Measure the relationship between sliding window attention size and model performance (via cross-entropy loss) on a text dataset
- Dataset: Uses the "monology/pile-uncopyrighted" dataset with 1000 sentences, each containing 100 tokens
- Sliding Window Sizes Tested: [1, 5, 10, 25, 50, 100, None]
  - None represents full attention (no sliding window)
  - Range from very restrictive (1 token) to moderately large (100 tokens)

Expected Workflow:
1. Data Loading: Use collect_from_hf function from src/cache_utils.py to load the Pile dataset
2. Model Testing: For each sliding window size:
   - Load Qwen model with that specific sliding window configuration
   - Compute cross-entropy loss on the dataset
3. Visualization: Create a bar plot showing cross-entropy loss vs sliding window size

Expected Results:
The experiment likely aims to demonstrate the trade-off between:
- Computational efficiency (smaller windows are faster)
- Model performance (larger windows may capture more context, leading to lower loss)

The results should show whether there's a "sweet spot" for sliding window size where performance remains good while maintaining efficiency benefits.
"""

import os
import torch as th
import th.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from src.project_config import MODELS_DIR, PLOTS_DIR
from src.cache_utils import collect_from_hf
import numpy as np

def compute_cross_entropy_loss(model, tokenizer, inputs_BP, masks_BP):
    """
    Compute cross-entropy loss on a given dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        inputs_BP: Input token IDs [batch_size, seq_len]
        masks_BP: Attention masks [batch_size, seq_len]
    
    Returns:
        float: Average cross-entropy loss
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with th.no_grad():
        # Move inputs to device
        inputs_BP = inputs_BP.to(model.device)
        masks_BP = masks_BP.to(model.device)
        
        # Forward pass
        outputs = model(input_ids=inputs_BP, attention_mask=masks_BP)
        logits = outputs.logits
        
        # Shift for language modeling: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs_BP[..., 1:].contiguous()
        shift_masks = masks_BP[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_masks = shift_masks.view(-1)
        
        # Compute loss only on non-padded tokens
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        masked_loss = loss * shift_masks
        
        total_loss += masked_loss.sum().item()
        total_tokens += shift_masks.sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else 0.0

def make_line_plot(sliding_windows, losses, num_sentences, save_fname="sliding_window_ce_results"):
    """
    Make line plot with scatter points of cross entropy loss over sliding window size.
    
    Args:
        sliding_windows: List of sliding window sizes (None for full attention)
        losses: List of corresponding cross-entropy losses
        num_sentences: Number of sentences used for positioning None value
        save_fname: Base filename to save the plot (without extension)
    """
    # Create x-axis values: window sizes, with None placed at num_sentences
    x_values = []
    y_values = []
    labels = []
    
    for window, loss in zip(sliding_windows, losses):
        if window is None:
            x_values.append(num_sentences)
            labels.append("Full")
        else:
            x_values.append(window)
            labels.append(str(window))
        y_values.append(loss)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line connecting points
    ax.plot(x_values, y_values, linewidth=2, color='C0', marker='o', markersize=8)
    
    # Add scatter points with labels
    ax.scatter(x_values, y_values, color='C0', s=50, zorder=5)
    
    # Add value labels on points
    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold')
    
    ax.set_xlabel('Sliding Window Size', fontsize=12)
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax.set_title('Cross-Entropy Loss vs Sliding Window Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis labels to show both numeric and "Full" for None case
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    
    # Improve layout
    plt.tight_layout()
    
    # Save plot using PLOTS_DIR
    fig_path = os.path.join(PLOTS_DIR, f"{save_fname}.png")
    plt.savefig(fig_path, dpi=80, bbox_inches="tight")
    print(f"Saved plot to {fig_path}")
    plt.close()


if __name__ == "__main__":
    dataset_name = "monology/pile-uncopyrighted"
    num_sentences = 100
    num_tokens_per_sentence = 100
    model_name = "Qwen/Qwen2.5-7B"
    sliding_windows = th.arange(0, num_tokens_per_sentence, 2).tolist() + [None]

    print("Loading dataset...")
    # Load a small tokenizer first to get the dataset
    temp_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
    inputs_BP, masks_BP, selected_story_idxs = collect_from_hf(
        tokenizer=temp_tokenizer,
        dataset_name=dataset_name,
        num_stories=num_sentences,
        num_tokens=num_tokens_per_sentence,
        hf_text_identifier="text"  # For monology/pile-uncopyrighted
    )
    print(f"Loaded {inputs_BP.shape[0]} sequences with {inputs_BP.shape[1]} tokens each")
    
    losses = []
    
    # Compute cross entropy loss for different sliding window sizes
    for i, window_size in enumerate(sliding_windows):
        print(f"\n--- Testing sliding window size: {window_size} ({i+1}/{len(sliding_windows)}) ---")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)
        
        # Configure model with sliding window (if not None)
        if window_size is None:
            # Full attention
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=th.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                cache_dir=MODELS_DIR,
            )
            print("Using full attention")
            print(f"Model config sliding window: {getattr(model.config, 'sliding_window', 'None')}")
        else:
            # Sliding window attention
            config = Qwen2Config.from_pretrained(model_name)
            config.use_sliding_window = True
            config.sliding_window = window_size
            # Set max_window_layers to ensure all layers use sliding window
            config.max_window_layers = 0 # Apply to all layers
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=th.float16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                cache_dir=MODELS_DIR,
            )
            print(f"Using sliding window with size: {window_size}")
            print(f"Model config sliding window: {model.config.sliding_window}")
            print(f"Model config use_sliding_window: {model.config.use_sliding_window}")
            print(f"Model config max_window_layers: {model.config.max_window_layers}/{model.config.num_hidden_layers}")
            
            # Verify the configuration is actually applied
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window != window_size:
                print(f"WARNING: Expected sliding window {window_size}, but got {model.config.sliding_window}")
            if not getattr(model.config, 'use_sliding_window', False):
                print("WARNING: use_sliding_window is False!")
        
        # Compute cross-entropy loss
        loss = compute_cross_entropy_loss(model, tokenizer, inputs_BP, masks_BP)
        losses.append(loss)
        print(f"Cross-entropy loss: {loss:.4f}")
        
        # Clean up memory
        del model, tokenizer
        th.cuda.empty_cache()
    
    # Create and save line plot
    print("\nCreating visualization...")
    make_line_plot(sliding_windows, losses, num_sentences, "sliding_window_ce_results")
    
    # Print results summary
    print("\n--- RESULTS SUMMARY ---")
    for window, loss in zip(sliding_windows, losses):
        window_str = "Full" if window is None else str(window)
        print(f"Window size {window_str:>4}: {loss:.4f}")
    
    print("\nExperiment completed!")
