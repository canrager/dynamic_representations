import os
import torch as th
import itertools
import json
from copy import deepcopy
from datetime import datetime
from src.project_config import BaseConfig

def generate_llm_dataset_folder_dir(cfg):
    model_str = cfg.llm_name.split("/")[-1]
    dataset_name = cfg.dataset_name.split("/")[-1].split(".")[0]
    
    # Add unique identifier if available
    unique_id = getattr(cfg, 'unique_id', None)
    if unique_id:
        folder_name = f"{dataset_name}_{model_str}_l{cfg.layer_idx}_seq{cfg.num_sequences}_ctx{cfg.context_length}_{unique_id}"
    else:
        folder_name = f"{dataset_name}_{model_str}_l{cfg.layer_idx}_seq{cfg.num_sequences}_ctx{cfg.context_length}"
    
    folder_dir = os.path.join(cfg.activations_dir, folder_name)
    return folder_dir
    
def load_acts(cfg: BaseConfig):
    folder_dir = generate_llm_dataset_folder_dir(cfg)
    activation_dir = os.path.join(folder_dir, "activations.pt")
    masks_dir = os.path.join(folder_dir, "masks.pt")

    # Check whether all masks are 1
    with open(masks_dir, "rb") as f:
        masks_BP = th.load(f, weights_only=False)
    assert th.all(masks_BP == 1)

    with open(activation_dir, "rb") as f:
        acts_LBPD = th.load(f, weights_only=False)

    return acts_LBPD[0]

def load_llm_artifacts(cfg: BaseConfig, artifacts=['activations', 'surrogate']):
    # Find all cached activation directories
    activations_base_dir = cfg.activations_dir
    if not os.path.exists(activations_base_dir):
        raise FileNotFoundError(f"Activations directory not found: {activations_base_dir}")
    
    candidates = []
    
    # Scan all subdirectories for config.json files
    for item in os.listdir(activations_base_dir):
        item_path = os.path.join(activations_base_dir, item)
        if os.path.isdir(item_path):
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        cached_config = json.load(f)
                    
                    # Check if this candidate matches the required parameters
                    matches = True
                    match_params = ['llm_name', 'layer_idx', 'dataset_name', 'num_sequences', 'context_length']
                    
                    for param in match_params:
                        if getattr(cfg, param) != cached_config.get(param):
                            matches = False
                            break
                    
                    if matches:
                        # Check if required files exist
                        artifact_paths = {}
                        for artifact in artifacts:
                            artifact_paths[artifact] = os.path.join(item_path, f"{artifact}.pt")
                        
                        # Always include config and masks paths for validation
                        artifact_paths['config'] = config_path
                        artifact_paths['masks'] = os.path.join(item_path, "masks.pt")
                        
                        # Check if core files exist (not surrogate which might not exist yet)
                        core_files = [artifact_paths['activations'], artifact_paths['masks']]
                        if all(os.path.exists(p) for p in core_files):
                            candidates.append({
                                'path': item_path,
                                'config': cached_config,
                                'timestamp': cached_config.get('unique_id', ''),
                                'artifact_paths': artifact_paths
                            })
                except (json.JSONDecodeError, KeyError):
                    # Skip invalid config files
                    continue
    
    if not candidates:
        # Fall back to old behavior if no new-style candidates found
        folder_dir = generate_llm_dataset_folder_dir(cfg)
        artifact_paths = {}
        for artifact in artifacts:
            artifact_paths[artifact] = os.path.join(folder_dir, f"{artifact}.pt")
        
        masks_path = os.path.join(folder_dir, "masks.pt")
        core_files = [artifact_paths['activations'], masks_path]
        
        if all(os.path.exists(p) for p in core_files):
            loaded_artifacts = {}
            
            # Check whether all masks are 1
            with open(masks_path, "rb") as f:
                masks_BP = th.load(f, weights_only=False)
            assert th.all(masks_BP == 1)
            
            # Load requested artifacts
            for artifact in artifacts:
                if os.path.exists(artifact_paths[artifact]):
                    with open(artifact_paths[artifact], "rb") as f:
                        if artifact == 'activations':
                            acts_LBPD = th.load(f, weights_only=False)
                            loaded_artifacts[artifact] = acts_LBPD[0]
                        else:
                            loaded_artifacts[artifact] = th.load(f, weights_only=False)
            
            return loaded_artifacts, folder_dir
        else:
            raise FileNotFoundError(f"No cached activations found matching the configuration")
    
    # Sort candidates by timestamp (newest first)
    candidates.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # If multiple candidates, print warning
    if len(candidates) > 1:
        print(f"Warning: Found {len(candidates)} cached activation candidates matching the configuration:")
        for i, candidate in enumerate(candidates):
            print(f"  {i+1}. {os.path.basename(candidate['path'])}")
        print(f"\nSelected newest candidate: {os.path.basename(candidates[0]['path'])}")
        print("Config of selected candidate:")
        for key, value in candidates[0]['config'].items():
            print(f"  {key}: {value}")
    
    # Use the newest candidate
    selected = candidates[0]
    
    # Check whether all masks are 1
    with open(selected['artifact_paths']['masks'], "rb") as f:
        masks_BP = th.load(f, weights_only=False)
    assert th.all(masks_BP == 1)

    # Load requested artifacts
    loaded_artifacts = {}
    for artifact in artifacts:
        artifact_path = selected['artifact_paths'][artifact]
        if os.path.exists(artifact_path):
            with open(artifact_path, "rb") as f:
                if artifact == 'activations':
                    acts_LBPD = th.load(f, weights_only=False)
                    loaded_artifacts[artifact] = acts_LBPD[0]
                else:
                    loaded_artifacts[artifact] = th.load(f, weights_only=False)
        elif artifact == 'surrogate':
            # Surrogate might not exist yet, that's OK
            pass
        else:
            raise FileNotFoundError(f"Required artifact '{artifact}' not found at {artifact_path}")

    return loaded_artifacts, selected['path']


def load_experiment_result(cfg: BaseConfig) -> dict:
    """Load experiment results that match the configuration parameters."""
    results_dir = cfg.results_dir
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    candidates = []
    
    # Scan all files in results directory for JSON files
    for item in os.listdir(results_dir):
        if item.endswith('.json'):
            item_path = os.path.join(results_dir, item)
            if os.path.isfile(item_path):
                try:
                    with open(item_path, "r") as f:
                        result_data = json.load(f)
                    
                    # Check if this result matches the required parameters
                    if 'config' not in result_data:
                        continue
                        
                    cached_config = result_data['config']
                    matches = True
                    match_params = ['llm_name', 'layer_idx', 'dataset_name', 'num_sequences', 'context_length']
                    
                    for param in match_params:
                        if getattr(cfg, param) != cached_config.get(param):
                            matches = False
                            break
                    
                    if matches:
                        candidates.append({
                            'path': item_path,
                            'config': cached_config,
                            'timestamp': cached_config.get('unique_id', ''),
                            'data': result_data
                        })
                except (json.JSONDecodeError, KeyError):
                    # Skip invalid JSON files
                    continue
    
    if not candidates:
        raise FileNotFoundError(f"No experiment results found matching the configuration")
    
    # Sort candidates by timestamp (newest first)
    candidates.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # If multiple candidates, print warning
    if len(candidates) > 1:
        print(f"Warning: Found {len(candidates)} experiment result candidates matching the configuration:")
        for i, candidate in enumerate(candidates):
            print(f"  {i+1}. {os.path.basename(candidate['path'])}")
        print(f"\nSelected newest candidate: {os.path.basename(candidates[0]['path'])}")
        print("Config of selected candidate:")
        for key, value in candidates[0]['config'].items():
            print(f"  {key}: {value}")
    
    # Return all results as a dictionary keyed by filename
    results = {}
    for candidate in candidates:
        filename = os.path.basename(candidate['path']).replace('.json', '')
        results[filename] = candidate['data']
    
    return results


def get_p_indices(cfg):
    """Compute p_indices for subsampling based on configuration."""
    if cfg.p_end is None:
        p_end = cfg.num_tokens_per_story
    else:
        p_end = cfg.p_end
        
    if cfg.do_log_p:
        p_indices = th.logspace(
            th.log10(th.tensor(float(max(1, cfg.p_start)))),
            th.log10(th.tensor(float(p_end - 1))),
            steps=cfg.num_p,
        ).long()
    else:
        p_indices = th.linspace(cfg.p_start, p_end - 1, cfg.num_p).long()
    
    return p_indices

def subsample_p_indices(x_BPD, cfg):
    p_indices = get_p_indices(cfg)
    x_subsample_BpD = x_BPD[:, p_indices, :]
    return x_subsample_BpD, p_indices

def load_subsampled_act_surr(cfg):
    artifacts_dict, artifact_dir = load_llm_artifacts(cfg, ['activations', 'surrogate'])
    act_BPD = artifacts_dict['activations']
    surr_BPD = artifacts_dict['surrogate']
    act_sub_BpD, p_indices = subsample_p_indices(act_BPD, cfg)
    surr_sub_BpD, _ = subsample_p_indices(surr_BPD, cfg)
    return act_sub_BpD, surr_sub_BpD, p_indices

def get_exp_name(cfg):
    return "ustatistic" \
            f"_{cfg.llm_name.split('/')[-1]}" \
            f"_{cfg.dataset.name.split('/')[-1].split(".")[0]}" \
            f"_seq{cfg.num_total_stories}" \
            f"_ctx{cfg.num_tokens_per_story}" \
            f"_normalize{cfg.do_normalize}"


def generate_filename(cfg, sweep_params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = getattr(cfg, 'experiment_name', 'experiment')
    
    param_strs = []
    for param in sweep_params.keys():
        value = getattr(cfg, param)
        if isinstance(value, str):
            value = value.split('/')[-1].replace('-', '_')
        param_strs.append(f"{param}-{value}")
    
    param_str = "_".join(param_strs)
    filename = f"{exp_name}_{param_str}_{timestamp}"
    return filename


def create_sweep_configs(base_cfg, sweep_params):
    configs = []
    param_combinations = []
    for combination in itertools.product(*sweep_params.values()):
        cfg = deepcopy(base_cfg)
        param_combo = {}
        for param, value in zip(sweep_params.keys(), combination):
            setattr(cfg, param, value)
            param_combo[param] = value
        
        # Generate filename and unique_id for this configuration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.unique_id = timestamp
        cfg.filename = generate_filename(cfg, sweep_params)
        
        configs.append(cfg)
        param_combinations.append(param_combo)
    return configs, param_combinations


def run_parameter_sweep(base_cfg, sweep_params, run_single_experiment):
    from tqdm import tqdm
    import gc
    import torch as th
    
    configs, param_combinations = create_sweep_configs(base_cfg, sweep_params)
    
    for cfg, param_combo in tqdm(zip(configs, param_combinations), total=len(configs), desc="Running sweep"):
        param_str = ", ".join([f"{k}: {v}" for k, v in param_combo.items()])
        print(f"Running: {param_str}")
        
        run_single_experiment(cfg)
        
        # Aggressive cleanup between sweep iterations
        th.cuda.empty_cache()
        gc.collect()
        
        print(f"Completed: {param_str}")