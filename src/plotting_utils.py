import os
import matplotlib.pyplot as plt

def savefig(plot_name: str, plots_dir: str = "artifacts/plots", suffix=".pdf"):
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{plot_name}{suffix}")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"saved figure to: {plot_path}")
    plt.close()