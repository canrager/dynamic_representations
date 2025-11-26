import torch

# Load the checkpoint
ckpt_path = "artifacts/trained_saes/gemma-2-2B/layer_12/split_tfa/latest_ckpt.pt"
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Print keys and shapes
print("Checkpoint contents:")
print("-" * 60)
for key, value in state_dict.items():
    if isinstance(value, torch.Tensor):
        print(f"{key:40s} {str(value.shape):20s}")
    else:
        print(f"{key:40s} {type(value).__name__}")

print(state_dict["sae"].keys())
