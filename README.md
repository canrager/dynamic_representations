# Your-project

A Python project using transformers, PyTorch, and OpenAI.

## Setup

### Using uv only
```bash
pip install uv
uv venv --python 3.12.8
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
uv pip install -e .
```

### Using conda & uv
Create a virtual environment (default: using conda)
```bash
conda create -n dyn_rep python=3.12.8
conda activate dyn_rep
```

Setup dependencies (default: using uv)
```bash
pip install uv
uv pip install -e .
```

## Project Structure

- `src/`: Source code directory
- `exp/`: Experiments directory
- `artifacts/`: Model outputs and artifacts
- `scripts/`: Utility scripts

The `src/project_config.py` contains folder directories, please adapt them to your compute setup.