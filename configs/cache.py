from dataclasses import dataclass
from configs.defaults import *

@dataclass
class CacheConfig:
    data: DatasetConfig
    llm: LLMConfig
    sae: SAEConfig | None # If None is passed, Cache the LLM and Surrogate. SAE cache requires existing LLM cache.

dataset_configs = [WEBTEXT_DS_CFG, SIMPLESTORIES_DS_CFG]
llm_configs = LLAMA3_LLM_CFG
sae_configs = [None] + LLAMA3_SAE_CFGS




# --> Cache configs