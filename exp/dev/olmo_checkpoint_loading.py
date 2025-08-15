#%%
import re
import numpy as np
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/OLMo-2-1124-7B")
branches = [b.name for b in out.branches]

stage1_branches = []
for b in branches:
    if re.search(r'stage1', b):
        stage1_branches.append(b)

# Extract step numbers
steps = []
for branch in stage1_branches:
    match = re.search(r'stage1-step(\d+)', branch)
    if match:
        steps.append(int(match.group(1)))

if steps:
    steps_sorted = sorted(steps)
    
    print(f"All stage1 branches ({len(stage1_branches)}):")
    for branch in sorted(stage1_branches):
        print(branch)
    
    print(f"\nAll step numbers: {steps_sorted}")
    
    # Select 5 evenly distributed steps in log space from the actual list
    min_step = steps_sorted[0]
    max_step = steps_sorted[-1]
    
    # Generate 5 log-spaced target values
    log_targets = np.logspace(np.log10(max(min_step, 1)), np.log10(max_step), 5)
    
    # Find closest actual steps to each log-spaced target
    selected_steps = []
    for target in log_targets:
        closest_step = min(steps_sorted, key=lambda x: abs(x - target))
        if closest_step not in selected_steps:
            selected_steps.append(closest_step)
    
    # If we don't have 5 unique steps, fill with evenly spaced indices
    if len(selected_steps) < 5:
        n = len(steps_sorted)
        indices = [0, n//4, n//2, 3*n//4, n-1]
        selected_steps = [steps_sorted[i] for i in indices]
    
    print(f"\n5 evenly distributed steps from available list:")
    for step in selected_steps:
        # Find the corresponding branch name
        matching_branch = None
        for branch in stage1_branches:
            if f"step{step}" in branch:
                matching_branch = branch
                break
        if matching_branch:
            print(f'revision="{matching_branch}"')
        else:
            print(f"step{step}")
else:
    print("No stage1 branches with step numbers found")
# %%
