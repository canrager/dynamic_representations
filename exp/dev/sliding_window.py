
#%%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from src.project_config import MODELS_DIR

# Load Qwen2.5 with sliding window attention
model_name = "Qwen/Qwen2.5-7B"
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODELS_DIR)

# # Configure sliding window
# config = Qwen2Config.from_pretrained(model_name)
# config.use_sliding_window = True
# config.sliding_window = 10

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # config=config,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    cache_dir=MODELS_DIR, 
)

#%%
# print(f"Model loaded: {model_name}")
# print(f"Sliding window enabled: {model.config.use_sliding_window}")
# print(f"Sliding window size: {model.config.sliding_window}")

print(model)
print(model.config)
#%%
e = tokenizer.encode("hello world")
d = [tokenizer.decode(t) for t in e]
print(d)
# %%
