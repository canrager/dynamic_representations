from dictionary_learning.dictionary import AutoEncoder

path = "artifacts/trained_saes/Standard_gemma-2-2b__0108/resid_post_layer_12/trainer_2/ae.pt"
sae = AutoEncoder.from_pretrained(path)