from src.project_config import MODELS_DIR
from transformers import AutoTokenizer, GPT2Tokenizer

model_name = "openai-community/gpt2"
# model_name = "meta-llama/Llama-3.1-8B"

input_str = "My life goal:"

tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)

tokenizer.add_special_tokens({
    'pad_token': '[PAD]'
}) # both gpt2 and llama3.1 tokenizers need this
tokenizer.add_bos_token = True # works with both both gpt2 and llama3.1, essentially

token_ids_L = tokenizer(
    input_str,
    padding=True,
    padding_side="right",
    truncation=False
).input_ids

print(token_ids_L)

token_str_L = [tokenizer.decode(t) for t in token_ids_L]

print(token_str_L)