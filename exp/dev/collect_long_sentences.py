from datasets import load_dataset
import re
from src.project_config import MODELS_DIR
from src.exp_utils import load_tokenizer


def get_english_only_tokens(tokenizer):
    vocab = tokenizer.get_vocab()
    english_only_tokens = set()
    for token, token_id in vocab.items():
        if re.match(r"^[A-Za-z ]*$", token):
            english_only_tokens.add(token_id)
    return english_only_tokens


def collect_long_sentences(tokenizer, all_stories, num_sentences, num_tokens):

    story_idxs = []
    inputs_BP = []

    # Find all tokens in vocab that exclusively contain English letters and spaces
    english_only_tokens = get_english_only_tokens(tokenizer)
    print(english_only_tokens)
    print(
        f"english only tokens decoded: {[tokenizer.decode(token_id) for token_id in english_only_tokens]}"
    )

    print(f"Found {len(english_only_tokens)} English-only tokens in vocabulary")

    for story_idx, story_item in enumerate(all_stories):
        if len(inputs_BP) >= num_sentences:
            break

        story_text = story_item["text"]
        # Tokenize the full story
        story_tokens = tokenizer(story_text, return_tensors="pt").input_ids[0]

        # Find the first consecutive sequence of English-only tokens >= num_tokens
        consecutive_start = None
        consecutive_length = 0

        for i, token_id in enumerate(story_tokens):
            if token_id.item() in english_only_tokens:
                if consecutive_start is None:
                    consecutive_start = i
                consecutive_length += 1

                # Check if we have enough consecutive English tokens
                if consecutive_length >= num_tokens:
                    story_idxs.append(story_idx)
                    english_sequence = story_tokens[
                        consecutive_start : consecutive_start + num_tokens
                    ]
                    inputs_BP.append(english_sequence)
                    print(f"decoded story: {tokenizer.decode(english_sequence)}")
                    break
            else:
                # Reset if we hit a non-English token
                consecutive_start = None
                consecutive_length = 0

        if consecutive_length > 0:
            print(
                f"Story {story_idx} max consecutive English tokens: {consecutive_length}"
            )
        else:
            print(f"Story {story_idx} has no English-only tokens.")

    return story_idxs, inputs_BP


if __name__ == "__main__":

    num_sentences = 100
    num_tokens = 20

    model_name = "gpt2"
    model_name = "meta-llama/Llama-3.1-8B"

    hf_dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        cache_dir=MODELS_DIR,
        streaming=True,
    )["train"]

    tokenizer = load_tokenizer("gpt2", cfg.llm.hf_cache_dir=MODELS_DIR)
    story_idxs, inputs_BP = collect_long_sentences(
        tokenizer, hf_dataset, num_sentences, num_tokens
    )
    print(story_idxs)
