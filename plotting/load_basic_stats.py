"""Load all basic_stats JSON files into a dictionary."""

import json
import os
from pathlib import Path
import numpy as np


def load_basic_stats_results(results_dir: str = "artifacts/results") -> dict:
    """
    Load all basic_stats JSON files from the results directory.

    Args:
        results_dir: Path to the results directory

    Returns:
        Dictionary mapping filenames to their JSON contents
    """
    results_path = Path(results_dir)
    basic_stats_files = sorted(results_path.glob("basic_stats_*.json"))

    results = {}
    for filepath in basic_stats_files:
        with open(filepath, "r") as f:
            results[filepath.name] = json.load(f)

    print(f"Loaded {len(results)} basic_stats files")
    return results


def compute_mean_fve_per_file(results: dict) -> dict:
    """
    Compute mean fraction variance explained score per file.

    Args:
        results: Dictionary of loaded basic_stats results

    Returns:
        Dictionary mapping filenames to their mean FVE scores
    """
    mean_fve = {}

    for filename, data in results.items():
        for x in data.keys():
            if "code" in x:
                print(x)
                if "l0" in data[x]:
                    print(data[x])
                    fve_scores = data[x]["l0"]["score"]
                    mean_fve[filename + x] = (
                        np.mean(fve_scores).item(),
                        data["config"]["sae"]["name"],
                        data["config"]["data"]["name"],
                        x,
                    )
                    # mean_fve[data["config"]["sae"]["name"]] = np.mean(fve_scores)

    return mean_fve


def main():
    results = load_basic_stats_results()

    # Print summary
    print("\nLoaded files:")
    for name in results.keys():
        print(f"  {name}")

    # Compute mean FVE per file
    mean_fve = compute_mean_fve_per_file(results)

    print("\nMean Fraction Variance Explained per file:")
    for name, fve in sorted(mean_fve.items()):
        # print(f"  {name}: {fve}")
        print(fve)


if __name__ == "__main__":
    main()
