# Dynamic Representations

Analysis of intrinsic dimensionality and temporal dynamics in LLM activations.

## Quick Start

1. **Set up virtual environment**
   ```bash
   conda create -n dyn_rep python=3.12.8
   conda activate dyn_rep
   pip install uv && uv pip install -e .
   ```

2. **Check HF cache directory** in `src/project_config.py` (adapt to your setup)

3. **Download activation artifacts**
   ```bash
   ./scripts/download_artifacts_from_hf.sh
   ```

4. **Adapt experiment config** in `exp/u_stat.py`:
   - Uncomment the correct `llm` line to match your downloaded activations
   - Set `num_total_stories` (N) and `num_tokens_per_story` (T) to match the activation filename pattern: `activations_u-stat_{model}_{N}N_{T}T.pt`

5. **Run experiment**
   ```bash
   python exp/u_stat.py
   ```

**Pro tip:** Once completed, the figure path will be printed in the terminal - click to see results!

## Project Structure

- `src/`: Source code directory
- `exp/`: Experiments directory
- `artifacts/`: Model outputs and artifacts
- `scripts/`: Utility scripts for HF Hub sync