import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Font family
plt.rcParams['font.family'] = 'Montserrat'

def render_legend():
    """
    Render a legend showing:
    1. Empty circle marker (user tokens)
    2. X marker (model tokens)
    3. Black to white colorbar (gradient)
    """
    fig = plt.figure(figsize=(8, 3), dpi=150, facecolor='white')

    # Create two subplots: one for markers, one for colorbar
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    ax_markers = fig.add_subplot(gs[0, 0])
    ax_colorbar = fig.add_subplot(gs[0, 1])

    # --- Left panel: Markers ---
    ax_markers.set_xlim(0, 10)
    ax_markers.set_ylim(0, 10)
    ax_markers.set_aspect('equal')

    # User token: empty circle
    ax_markers.scatter(2, 7, s=300, marker='o', facecolors='none',
                      edgecolors='black', linewidths=2.0, label='User token')
    ax_markers.text(4, 7, 'User token', fontsize=14, va='center')

    # Model token: x marker
    ax_markers.scatter(2, 3, s=300, marker='x', c='black',
                      linewidths=2.0, label='Model token')
    ax_markers.text(4, 3, 'Model token', fontsize=14, va='center')

    # Remove axes
    ax_markers.set_xticks([])
    ax_markers.set_yticks([])
    for spine in ax_markers.spines.values():
        spine.set_visible(False)
    ax_markers.set_title('Token Type', fontsize=16, pad=10)

    # --- Right panel: Colorbar ---
    # Create black to white colormap
    cmap = LinearSegmentedColormap.from_list('black_to_white',
                                             [(0, 0, 0, 1), (1, 1, 1, 1)], N=256)

    # Create gradient image
    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    ax_colorbar.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])
    ax_colorbar.set_xlim(0, 1)
    ax_colorbar.set_ylim(0, 1)

    # Add labels
    ax_colorbar.set_xticks([0, 1])
    ax_colorbar.set_xticklabels(['Start', 'End'], fontsize=12)
    ax_colorbar.set_yticks([])
    ax_colorbar.set_title('Position in Sequence', fontsize=16, pad=10)

    # Add border
    for spine in ax_colorbar.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()

    # Save figure
    output_path = '/home/can/dynamic_representations/legend.pdf'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved legend to {output_path}")
    plt.close()


if __name__ == "__main__":
    render_legend()
