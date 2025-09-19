from sae.saeTemporal import TemporalSAE
import torch as th

path = "/home/can/dynamic_representations/artifacts/trained_saes/temporal_gemma2/llxloq3x/"
sae = TemporalSAE.from_pretrained(path, dtype=th.bfloat16, device="cuda")
