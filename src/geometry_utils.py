import torch
import torch.nn.functional as F
from umap import UMAP

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Font family
plt.rcParams['font.family'] = 'Montserrat'

# =========================
# Trace data functions
# =========================
def get_trace_data(plot_activations, centering=True, normalize=True, n_components=3):
    # Center and normalize
    plot_activations = plot_activations - plot_activations.mean(dim=1, keepdim=True) if centering else plot_activations
    plot_activations = F.normalize(plot_activations, p=2, dim=-1) if normalize else plot_activations

    # Flatten for UMAP
    B, L, D = plot_activations.shape
    X = plot_activations.reshape(B * L, D).float().cpu().numpy()

    reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1,
                metric="cosine" if normalize else "euclidean",
                random_state=42)
    X_umap = reducer.fit_transform(X)      # shape: (B*L, n_components)

    # Reshape for plot
    traj = torch.from_numpy(X_umap).reshape(B, L, n_components)
    embedding = traj  # (B, L, n_components)
    pos_labels = np.arange(L).reshape(1, L).repeat(B, axis=0)  # (B, L)

    return embedding, pos_labels


# ==========================
# Geometric features
# ==========================
def tortuosity(X, ord=2, eps=1e-12, return_SC=False):
    """
    Compute tortuosity τ = S / C for a batch of trajectories.

    Args:
        X: np.ndarray, shape (N, L, D). N trajectories of length L in D dims.
        ord: norm order (2 = Euclidean).
        eps: small threshold for numerical stability.
        return_SC: if True, also return (S, C) arrays.

    Returns:
        tau: shape (N,), tortuosity per trajectory.
        (optional) S, C: shape (N,), path length and chord length.
    """
    if X.ndim != 3:
        raise ValueError("X must have shape (N, L, D)")

    # Segment lengths
    diffs = np.diff(X, axis=1)                               # (N, L-1, D)
    seg_lengths = np.linalg.norm(diffs, ord=ord, axis=2)     # (N, L-1)
    S = seg_lengths.sum(axis=1)                              # (N,)

    # End-to-end chord lengths
    C = np.linalg.norm(X[:, -1, :] - X[:, 0, :], ord=ord, axis=1)  # (N,)

    # τ = S / C with careful edge handling
    tau = np.full_like(S, np.inf, dtype=float)
    mask = C > eps
    tau[mask] = S[mask] / C[mask]

    # If S≈0 and C≈0 (degenerate path), define τ=1
    degenerate = (~mask) & (S <= eps)
    tau[degenerate] = 1.0

    if return_SC:
        return tau, S, C
    return tau
    
    
# =========================
# Styling helpers for 3D plots
# =========================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def rgba_str_to_tuple(s):
    # e.g. "rgba(239,158,114,0.5)"
    s = s.strip().lower().replace("rgba(", "").replace(")", "")
    r, g, b, a = [float(v) for v in s.split(",")]
    return (r/255.0, g/255.0, b/255.0, a)

def hex_to_rgba(h, a=1.0):
    h = h.strip()
    if len(h) == 4 and h[0] == "#":  # #abc -> #aabbcc
        h = "#" + "".join([c*2 for c in h[1:]])
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4)) + (a,)

# def two_color_cmap(start_hex, end_hex, name="two", alpha=1.0):
#     c0 = hex_to_rgba(start_hex, 0.1); c1 = hex_to_rgba(end_hex, 1.0)
#     return LinearSegmentedColormap.from_list(name, [c0, c1], N=256)

def two_color_cmap(start_hex, end_hex, name="two", alpha=1.0):
    # Convert end color to RGB
    end_rgb = hex_to_rgba(end_hex, 1.0)[:3]
    start_rgb = hex_to_rgba(start_hex, 0.1)[:3]
    
    # Create whitened version of end color for start
    # Blend with white (1, 1, 1) - adjust blend factor as needed
    white_blend = 0.3  # 0.9 = very white, 0.5 = medium, 0.0 = no white
    start_rgb = tuple(c * (1 - white_blend) + white_blend for c in start_rgb)
    
    c0 = start_rgb + (1.0,)
    c1 = end_rgb + (1.0,)
    return LinearSegmentedColormap.from_list(name, [c0, c1], N=256)

def compute_percentile_bounds(X, lo=5.0, hi=99.0):
    # X: (N, 3)
    lo_v = np.percentile(X, lo, axis=0)
    hi_v = np.percentile(X, hi, axis=0)
    return lo_v, hi_v

def break_line_with_mask(coords, mask):
    # Insert NaNs where mask is False so segments break
    out = coords.copy()
    out[~mask] = np.nan
    return out

def add_floor(ax, bounds_xy, z_plane, color="#FFFFFF", alpha=1.0):
    xmin, xmax, ymin, ymax = bounds_xy
    verts = [[(xmin, ymin, z_plane),
              (xmax, ymin, z_plane),
              (xmax, ymax, z_plane),
              (xmin, ymax, z_plane)]]
    poly = Poly3DCollection(verts, facecolors=color, edgecolor="none", alpha=alpha)
    poly.set_zsort('min')  # draw behind if it's below the data
    ax.add_collection3d(poly)

def style_axes(ax, floor_color="#FFFFFF", wall_strength=0.15, tick_strength=0.55, nticks=5):
    # Pane (wall) colors
    floor_color = "#FFFFFF"
    wall = np.array(to_rgba(floor_color))
    wall = tuple(wall)
    try:
        ax.xaxis.set_pane_color(wall)
        ax.yaxis.set_pane_color(wall)
        ax.zaxis.set_pane_color(wall)
    except Exception:
        pass

    # Ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nticks))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=nticks))
    # Remove axis titles to match your soft theme; keep ticks
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["tick"]["inward_factor"] = 0.
        axis._axinfo["tick"]["outward_factor"] = 0.4
        axis.set_tick_params(colors=(0,0,0,tick_strength))
        axis._axinfo["axisline"]["linewidth"] = 0.0
        axis._axinfo["axisline"]["color"] = (0, 0, 0, 0)  # fully transparent RGBA
        # pane border
        axis.line.set_visible(False)
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.pane.set_linewidth(0.0)
        
    ax.grid(False)


# =========================
# Panel drawing functions
# =========================
def find_optimal_viewing_angle(embedding, is_user_labels, n_samples=36):
    """
    Find the azimuth and elevation angles that maximally separate user and model points.

    Args:
        embedding: Tensor of shape (B, L, 3) - UMAP embeddings
        is_user_labels: Tensor of shape (B, L) - boolean labels (True=user, False=model)
        n_samples: Number of angles to sample (default: 36 for 10-degree increments)

    Returns:
        best_elev, best_azim: Optimal viewing angles in degrees
    """
    E = _to_numpy(embedding)
    is_user = _to_numpy(is_user_labels).astype(bool)

    # Flatten
    points = E.reshape(-1, 3)  # (B*L, 3)
    labels = is_user.reshape(-1)  # (B*L,)

    # Get centroids for user and model points
    user_points = points[labels]
    model_points = points[~labels]

    if len(user_points) == 0 or len(model_points) == 0:
        # If only one class, return default angles
        return 10, -30

    user_centroid = user_points.mean(axis=0)
    model_centroid = model_points.mean(axis=0)

    # Direction vector from user to model centroid
    sep_vector = model_centroid - user_centroid
    sep_vector = sep_vector / (np.linalg.norm(sep_vector) + 1e-8)

    # Convert to spherical coordinates to get optimal viewing angle
    # The optimal view is perpendicular to the separation vector
    x, y, z = sep_vector

    # Azimuth: angle in xy-plane
    azim_sep = np.degrees(np.arctan2(y, x))
    # Elevation: angle from xy-plane
    elev_sep = np.degrees(np.arcsin(z))

    # View from perpendicular direction (90 degrees rotated)
    best_azim = (azim_sep + 90) % 360
    if best_azim > 180:
        best_azim -= 360

    # Use moderate elevation for good 3D perspective
    best_elev = 15

    return best_elev, best_azim


def draw_panel(
    ax, embedding, pos_labels, start_hex, end_hex, title,
    connect_sequences=(0,), base_marker=36, glow_increase=12, glow_alpha=0.10,
    outlier_lo=5.0, outlier_hi=99.0, floor_hex="#FFFFFF", elev=10, azim=-30,
    emphasize=None, label_min=None, label_max=None, show_colorbar=False, colorbar_ax=None,
    lines_over_points=False, is_user_labels=None, force_black_markers=False
):

    # Shapes: embedding [B, L, 3], pos_labels [B, L]
    E = _to_numpy(embedding); P = _to_numpy(pos_labels).astype(float)
    B, L, _ = E.shape

    # Flatten for scatter
    flat = E.reshape(-1, 3)
    pos_flat = P.reshape(-1)

    # Normalize labels for a shared colorbar across panels
    if label_min is None: label_min = float(np.min(pos_flat))
    if label_max is None: label_max = float(np.max(pos_flat))
    norm = Normalize(vmin=label_min, vmax=label_max, clip=True)
    cmap = two_color_cmap(start_hex, end_hex, alpha=1.0)

    # --- Outlier clipping (markers define bounds), keep_if="all" ---
    lo_v, hi_v = compute_percentile_bounds(flat, lo=outlier_lo, hi=outlier_hi)
    inlier_mask = np.all((flat >= lo_v) & (flat <= hi_v), axis=1)

    # Soft floor extent (use actual inlier min Z and a bigger gap)
    xmin, ymin, zmin = lo_v
    xmax, ymax, zmax = hi_v
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    # strictly below all plotted points
    z_inlier_min = np.nanmin(flat[inlier_mask, 2]) if np.any(inlier_mask) else zmin
    z_range = max(1e-9, zmax - zmin)
    gap = 0.3 * z_range  # increase if you want more separation
    z_plane = z_inlier_min - gap

    # add the floor first
    add_floor(ax, (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y),
            z_plane, color=floor_hex, alpha=1.0)

    # now lock axes so the plane stays below the points
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_zlim(z_plane, zmax + 0.05 * z_range)

    # --- Main points (on top of lines) ---
    # Detect user/model labeling mode: if is_user_labels is provided,
    # we use different markers for user vs model but color by position
    is_user_model_mode = (is_user_labels is not None)

    pt_z = 5 if not lines_over_points else 3
    sizes = np.full(inlier_mask.sum(), base_marker)

    if not is_user_model_mode:
        # Standard gradient coloring by position
        sc = ax.scatter(
            flat[inlier_mask,0], flat[inlier_mask,1], flat[inlier_mask,2],
            s=sizes, c=pos_flat[inlier_mask], cmap=cmap, norm=norm,
            depthshade=False, alpha=0.60, linewidths=0.0, zorder=pt_z)
    else:
        # User/model mode: split into two groups with different markers
        # Color is based on pos_labels (position in sequence), creating gradient black -> sae_type_color
        # Marker shape is based on is_user_labels
        is_user_flat = _to_numpy(is_user_labels).astype(bool).reshape(-1)
        pos_inlier = pos_flat[inlier_mask]
        is_user_inlier = is_user_flat[inlier_mask]

        # Get colors based on position gradient (or force black for trajectory plots)
        if force_black_markers:
            # Trajectory plots: all markers black
            user_edge_color = 'black'
            model_color = 'black'
        else:
            # Population plots: colored by position
            colors_inlier = cmap(norm(pos_inlier))
            user_edge_color = colors_inlier[is_user_inlier] if np.any(is_user_inlier) else 'black'
            model_color = colors_inlier[~is_user_inlier] if np.any(~is_user_inlier) else 'black'

        # User points: empty circles (marker='o', facecolors='none')
        if np.any(is_user_inlier):
            ax.scatter(
                flat[inlier_mask][is_user_inlier, 0],
                flat[inlier_mask][is_user_inlier, 1],
                flat[inlier_mask][is_user_inlier, 2],
                s=sizes[is_user_inlier],
                marker='o',
                facecolors='none',
                edgecolors=user_edge_color,
                linewidths=0.8,
                depthshade=False,
                alpha=0.60,
                zorder=pt_z
            )

        # Model points: x markers
        if np.any(~is_user_inlier):
            ax.scatter(
                flat[inlier_mask][~is_user_inlier, 0],
                flat[inlier_mask][~is_user_inlier, 1],
                flat[inlier_mask][~is_user_inlier, 2],
                s=sizes[~is_user_inlier],
                marker='x',
                c=model_color,
                linewidths=0.8,
                depthshade=False,
                alpha=0.60,
                zorder=pt_z
            )

    # --- Lines (sequence connections), then halo/thickening ---
    for b in connect_sequences:
        if b < B:
            coords = E[b]  # (L, 3)
            m = np.all((coords >= lo_v) & (coords <= hi_v), axis=1)
            xs = break_line_with_mask(coords[:,0].copy(), m)
            ys = break_line_with_mask(coords[:,1].copy(), m)
            zs = break_line_with_mask(coords[:,2].copy(), m)

            halo_z = 2 if not lines_over_points else 8
            line_z = 3 if not lines_over_points else 9

            if emphasize is not None:
                line_col = emphasize["line_color"] if emphasize is not None else "#1a1a1a"
                line_col = line_col.split('(')[1].split(')')[0].split(',')
                line_col = [float(c) for c in line_col]
                line_col[:3] = [c / 255.0 for c in line_col[:3]]
                line_col = tuple(line_col)
                line, = ax.plot(xs, ys, zs, lw=emphasize.get("line_width", 4) if emphasize else 3,
                        color=line_col, alpha=line_col[-1], solid_capstyle="round", zorder=line_z, 
                        marker='o', markersize=4)
                hline, = ax.plot(xs, ys, zs, lw=emphasize.get("line_width",4)+emphasize.get("extra_width",2),
                                 color=line_col, alpha=line_col[-1], solid_capstyle="round", zorder=line_z,
                                 marker='o', markersize=4)

            # Arrow at end
            if emphasize is not None and emphasize.get("arrow", True):
                # last two finite points
                finite = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
                idx = np.where(finite)[0]
                if idx.size >= 2:
                    i, j = idx[-2], idx[-1]
                    u, v, w = xs[j]-xs[i], ys[j]-ys[i], zs[j]-zs[i]
                    q = ax.quiver(xs[j], ys[j], zs[j], u, v, w,
                              length=1.0, normalize=False,
                              arrow_length_ratio=0.35*emphasize.get("arrow_size", 1.0),
                              color=emphasize.get("arrow_color", "#1a1a1a"),
                              linewidths=0.0, zorder=line_z)

    # --- Point glow (underlay) ---
    # Skip glow effect in user/model mode as it doesn't work well with different markers
    if not is_user_model_mode:
        sizes = np.full(inlier_mask.sum(), base_marker + glow_increase)
        glow = ax.scatter(
            flat[inlier_mask,0], flat[inlier_mask,1], flat[inlier_mask,2],
            s=sizes, c=cmap(norm(pos_flat[inlier_mask])),
            depthshade=False, alpha=glow_alpha, edgecolors="none", zorder=pt_z)

    # optional shared colorbar
    if show_colorbar:
        assert colorbar_ax is not None, "Provide colorbar_ax when show_colorbar=True"
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = plt.colorbar(mappable, cax=colorbar_ax, orientation="vertical")
        cb.set_label("Position")

        # shrink + thin the colorbar
        bbox = colorbar_ax.get_position()
        colorbar_ax.set_position([
            bbox.x0,                      # same x
            bbox.y0 + 0.18*bbox.height,   # move up a bit
            bbox.width * 0.6,             # thinner
            bbox.height * 0.64            # shorter
        ])

    # consistent box and view
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=elev, azim=azim)
    style_axes(ax, floor_color=floor_hex, wall_strength=0.15, tick_strength=0.0, nticks=1)
    # ax.set_title(title, pad=8)



# =========================
# Plotting functions
# =========================
def make_matplotlib_umap_grid(
    panels, figsize=(15,5), dpi=400, floor_hex="#FFFFFF",
    elev=10, azim=-30, connect_idx=0,
    title_pad=2,                # smaller title padding
    hspace=0.01,                # very small gap between rows
    base_marker=36,
    viz_title=True,
    two_rows=False              # If True, split into population (top) and trajectory (bottom) rows
):

    if two_rows:
        # Two rows: top = population only, bottom = trajectory only
        fig = plt.figure(figsize=(figsize[0], figsize[1]*2), dpi=dpi, facecolor="white", constrained_layout=False)

        npanels = len(panels)
        gs = fig.add_gridspec(nrows=2, ncols=npanels, width_ratios=[1]*npanels,
                              wspace=0.01, hspace=0.15)

        # Create axes for both rows
        axes_top = [fig.add_subplot(gs[0, i], projection="3d") for i in range(npanels)]
        axes_bottom = [fig.add_subplot(gs[1, i], projection="3d") for i in range(npanels)]

        all_labels = np.concatenate([_to_numpy(p['pos_labels']).reshape(-1) for p in panels])
        label_min, label_max = float(np.min(all_labels)), float(np.max(all_labels))

        for i, (ax_top, ax_bottom, p) in enumerate(zip(axes_top, axes_bottom, panels)):
            panel_elev = p.get('elev', elev)
            panel_azim = p.get('azim', azim)

            # Top row: Population only (no trajectory line)
            draw_panel(
                ax_top,
                p['embedding'], p['pos_labels'],
                p['start'], p['end'], p['title'],
                connect_sequences=(),  # No trajectory line
                base_marker=base_marker,
                glow_increase=12, glow_alpha=0.10,
                outlier_lo=5.0, outlier_hi=99.0,
                floor_hex=floor_hex,
                elev=panel_elev, azim=panel_azim,
                emphasize=None,  # No trajectory emphasis
                label_min=label_min, label_max=label_max,
                show_colorbar=False,
                lines_over_points=False,
                is_user_labels=p.get('is_user', None)  # Pass user/model labels
            )
            ax_top.set_box_aspect((1,1,1))
            ax_top.view_init(elev=panel_elev, azim=panel_azim)
            if viz_title and i == 0:
                ax_top.set_ylabel("Population", fontsize=14, labelpad=15)
            if viz_title:
                ax_top.set_title(p['title'], pad=title_pad, fontsize=16)

            # Bottom row: Single trajectory only (no population)
            # Extract only the connected sequence
            E = _to_numpy(p['embedding'])
            P = _to_numpy(p['pos_labels'])

            # Also extract is_user if present
            single_is_user = None
            if 'is_user' in p and p['is_user'] is not None:
                U = _to_numpy(p['is_user'])
                single_is_user = U[connect_idx:connect_idx+1] if connect_idx is not None and connect_idx < U.shape[0] else p['is_user']

            if connect_idx is not None and connect_idx < E.shape[0]:
                # Get single trajectory
                single_embedding = E[connect_idx:connect_idx+1]  # Keep batch dim: (1, L, 3)
                single_pos_labels = P[connect_idx:connect_idx+1]  # (1, L)
            else:
                # Fallback to full if no valid connect_idx
                single_embedding = p['embedding']
                single_pos_labels = p['pos_labels']

            draw_panel(
                ax_bottom,
                single_embedding, single_pos_labels,
                p['start'], p['end'], "",  # No title on bottom row
                connect_sequences=(0,),  # Connect the first (and only) sequence
                base_marker=base_marker,
                glow_increase=12, glow_alpha=0.10,
                outlier_lo=5.0, outlier_hi=99.0,
                floor_hex=floor_hex,
                elev=panel_elev, azim=panel_azim,
                emphasize=p['emph'],
                label_min=label_min, label_max=label_max,
                show_colorbar=False,
                lines_over_points=True,  # Trajectory on top
                is_user_labels=single_is_user,  # Pass user/model labels
                force_black_markers=False  # Color markers by position for trajectory plot
            )
            ax_bottom.set_box_aspect((1,1,1))
            ax_bottom.view_init(elev=panel_elev, azim=panel_azim)
            if viz_title and i == 0:
                ax_bottom.set_ylabel("Trajectory", fontsize=14, labelpad=15)

        fig.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.05, hspace=0.15, wspace=0.05)
        return fig, (axes_top, axes_bottom)

    else:
        # Original single row behavior
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white", constrained_layout=False)

        npanels = len(panels)
        gs = fig.add_gridspec(nrows=1, ncols=npanels, width_ratios=[1]*npanels, wspace=0.01)

        # top row: 4 scenes + optional colorbar
        axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(npanels)]

        # ----- draw panels (unchanged) -----
        all_labels = np.concatenate([_to_numpy(p['pos_labels']).reshape(-1) for p in panels])
        label_min, label_max = float(np.min(all_labels)), float(np.max(all_labels))

        for i, (ax, p) in enumerate(zip(axes, panels)):
            # Use per-panel angles if provided, otherwise use global angles
            panel_elev = p.get('elev', elev)
            panel_azim = p.get('azim', azim)

            draw_panel(
                ax,
                p['embedding'], p['pos_labels'],
                p['start'], p['end'], p['title'],
                connect_sequences=(connect_idx,),
                base_marker=base_marker,
                glow_increase=12, glow_alpha=0.10,
                outlier_lo=5.0, outlier_hi=99.0,
                floor_hex=floor_hex,
                elev=panel_elev, azim=panel_azim,
                emphasize=p['emph'],
                label_min=label_min, label_max=label_max,
                show_colorbar=False,
                lines_over_points=(i==npanels-1),
                is_user_labels=p.get('is_user', None)  # Pass user/model labels
            )
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=panel_elev, azim=panel_azim)
            if viz_title:
                ax.set_title(p['title'], pad=title_pad, fontsize=16)

        fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.12, hspace=hspace, wspace=0.05)

        return fig, axes

# ---------- convenience wrapper that matches interactive plot style ----------
def make_fig_like_plotly(acts_embeddings, acts_pos_labels,
                         sae1_embeddings, sae1_pos_labels,
                         sae2_embeddings, sae2_pos_labels,
                         sae3_embeddings, sae3_pos_labels,
                         novel_embeddings, novel_pos_labels,
                         pred_embeddings, pred_pos_labels,
                         elev=10, azim=-30, connect_idx=None,
                         base_marker=36, viz_title=True):
    panels = [
        # Activations
        dict(embedding=acts_embeddings,   pos_labels=acts_pos_labels,   start="#000000", end="#EF9E72",
             title="Activations", emph=dict(
                 halo_color="rgba(239,158,114,0.3)", line_color="rgba(239,158,114,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # ReLU SAE
        dict(embedding=sae1_embeddings, pos_labels=sae1_pos_labels, start="#000000", end="#A855F7",
             title="ReLU", emph=dict(
                 halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # TopK SAE
        dict(embedding=sae2_embeddings, pos_labels=sae2_pos_labels, start="#000000", end="#A855F7",
             title="TopK", emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # BatchTopK SAE
        dict(embedding=sae3_embeddings, pos_labels=sae3_pos_labels, start="#000000", end="#A855F7",
             title="BatchTopK", emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # Temporal (Novel)
        dict(embedding=novel_embeddings,  pos_labels=novel_pos_labels,  start="#000000", end="#06B6D4",
             title="Temporal (Novel)", emph=dict(
                 halo_color="rgba(6,182,212,0.3)", line_color="rgba(6,182,212,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),
        # Temporal (Pred)
        dict(embedding=pred_embeddings,   pos_labels=pred_pos_labels,   start="#000000", end="#AA1010",
             title="Temporal (Pred)", emph=dict(
                 halo_color="rgba(255,180,80,0.4)", line_color="rgba(255,180,80,0.4)",
                 extra_width=1, line_width=3, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),
    ]

    fig, axes = make_matplotlib_umap_grid(panels, figsize=(15, 5), dpi=400, floor_hex="#DFDFDF",
                                          elev=elev, azim=azim, connect_idx=connect_idx, base_marker=base_marker,
                                          viz_title=viz_title)
    # Camera/view already set; nothing else to tweak.
    return fig, axes


def make_fig_like_plotly_with_user_labels(
    acts_embeddings, acts_pos_labels, acts_is_user,
    sae1_embeddings, sae1_pos_labels, sae1_is_user,
    sae2_embeddings, sae2_pos_labels, sae2_is_user,
    sae3_embeddings, sae3_pos_labels, sae3_is_user,
    novel_embeddings, novel_pos_labels, novel_is_user,
    pred_embeddings, pred_pos_labels, pred_is_user,
    elev=10, azim=-30, connect_idx=None,
    base_marker=36, viz_title=True, optimize_angles=True
):
    """
    Create figure with binary coloring based on user/model labels.
    Black for user tokens, end color for model tokens.

    Args:
        *_embeddings: Embedding tensors of shape (B, P, 3)
        *_pos_labels: Position labels of shape (B, P)
        *_is_user: Boolean tensors of shape (B, P) where True=user, False=model
        elev, azim: Default camera angles (used if optimize_angles=False)
        connect_idx: Which sequence to connect with lines
        base_marker: Base marker size
        viz_title: Whether to show panel titles
        optimize_angles: If True, compute optimal viewing angle per panel for max separation

    Returns:
        fig, axes: Matplotlib figure and axes
    """

    # Convert is_user labels to binary colors
    # user=True -> 0.0 (start of colormap, black)
    # model=False -> 1.0 (end of colormap, end color)
    def user_to_color_labels(is_user_BP, pos_labels_BP):
        """Convert boolean user labels to color values (0=user/black, 1=model/end_color)"""
        is_user_np = _to_numpy(is_user_BP).astype(float)
        # Invert: True (user) -> 0.0, False (model) -> 1.0
        color_labels = 1.0 - is_user_np
        return color_labels

    # Compute optimal viewing angles for each panel if requested
    if optimize_angles:
        acts_elev, acts_azim = find_optimal_viewing_angle(acts_embeddings, acts_is_user)
        sae1_elev, sae1_azim = find_optimal_viewing_angle(sae1_embeddings, sae1_is_user)
        sae2_elev, sae2_azim = find_optimal_viewing_angle(sae2_embeddings, sae2_is_user)
        sae3_elev, sae3_azim = find_optimal_viewing_angle(sae3_embeddings, sae3_is_user)
        novel_elev, novel_azim = find_optimal_viewing_angle(novel_embeddings, novel_is_user)
        pred_elev, pred_azim = find_optimal_viewing_angle(pred_embeddings, pred_is_user)

        print("Optimal viewing angles:")
        print(f"  Activations: elev={acts_elev:.1f}, azim={acts_azim:.1f}")
        print(f"  ReLU: elev={sae1_elev:.1f}, azim={sae1_azim:.1f}")
        print(f"  TopK: elev={sae2_elev:.1f}, azim={sae2_azim:.1f}")
        print(f"  BatchTopK: elev={sae3_elev:.1f}, azim={sae3_azim:.1f}")
        print(f"  Temporal (Novel): elev={novel_elev:.1f}, azim={novel_azim:.1f}")
        print(f"  Temporal (Pred): elev={pred_elev:.1f}, azim={pred_azim:.1f}")
    else:
        # Use default angles for all panels
        acts_elev = sae1_elev = sae2_elev = sae3_elev = novel_elev = pred_elev = elev
        acts_azim = sae1_azim = sae2_azim = sae3_azim = novel_azim = pred_azim = azim

    # Create panels with binary coloring and optimal angles
    panels = [
        # Activations
        dict(embedding=acts_embeddings,
             pos_labels=user_to_color_labels(acts_is_user, acts_pos_labels),
             start="#000000", end="#EF9E72",  # Black to orange
             title="Activations",
             elev=acts_elev, azim=acts_azim,
             emph=dict(
                 halo_color="rgba(239,158,114,0.3)", line_color="rgba(239,158,114,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # ReLU SAE
        dict(embedding=sae1_embeddings,
             pos_labels=user_to_color_labels(sae1_is_user, sae1_pos_labels),
             start="#000000", end="#A855F7",  # Black to purple
             title="ReLU",
             elev=sae1_elev, azim=sae1_azim,
             emph=dict(
                 halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # TopK SAE
        dict(embedding=sae2_embeddings,
             pos_labels=user_to_color_labels(sae2_is_user, sae2_pos_labels),
             start="#000000", end="#A855F7",  # Black to purple
             title="TopK",
             elev=sae2_elev, azim=sae2_azim,
             emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # BatchTopK SAE
        dict(embedding=sae3_embeddings,
             pos_labels=user_to_color_labels(sae3_is_user, sae3_pos_labels),
             start="#000000", end="#A855F7",  # Black to purple
             title="BatchTopK",
             elev=sae3_elev, azim=sae3_azim,
             emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # Temporal (Novel)
        dict(embedding=novel_embeddings,
             pos_labels=user_to_color_labels(novel_is_user, novel_pos_labels),
             start="#000000", end="#06B6D4",  # Black to cyan
             title="Temporal (Novel)",
             elev=novel_elev, azim=novel_azim,
             emph=dict(
                 halo_color="rgba(6,182,212,0.3)", line_color="rgba(6,182,212,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # Temporal (Pred)
        dict(embedding=pred_embeddings,
             pos_labels=user_to_color_labels(pred_is_user, pred_pos_labels),
             start="#000000", end="#AA1010",  # Black to red
             title="Temporal (Pred)",
             elev=pred_elev, azim=pred_azim,
             emph=dict(
                 halo_color="rgba(255,180,80,0.4)", line_color="rgba(255,180,80,0.4)",
                 extra_width=1, line_width=3, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),
    ]

    fig, axes = make_matplotlib_umap_grid(
        panels, figsize=(15, 5), dpi=400, floor_hex="#DFDFDF",
        elev=None, azim=None, connect_idx=connect_idx,
        base_marker=base_marker, viz_title=viz_title,
        two_rows=True  # Split into population and trajectory rows
    )

    return fig, axes


# =========================
# Flexible plotting functions
# =========================
def make_fig_like_plotly_flexible(available_data, elev=10, azim=-30, connect_idx=None,
                                   base_marker=36, viz_title=True):
    """
    Create figure from list of available activation data (flexible version).

    Args:
        available_data: List of dicts with keys 'type', 'embeddings', 'pos_labels'
        elev, azim: Camera viewing angles
        connect_idx: Which sequence to connect
        base_marker: Base marker size
        viz_title: Whether to show titles
    """
    # Mapping from activation type to display config
    type_configs = {
        "activations": {
            "title": "Activations",
            "start": "#000000",
            "end": "#EF9E72",
            "emph": dict(halo_color="rgba(239,158,114,0.3)", line_color="rgba(239,158,114,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "relu/codes": {
            "title": "ReLU",
            "start": "#000000",
            "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "topk/codes": {
            "title": "TopK",
            "start": "#000000",
            "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "batchtopk/codes": {
            "title": "BatchTopK",
            "start": "#000000",
            "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/novel_codes": {
            "title": "Temporal (Novel)",
            "start": "#000000",
            "end": "#06B6D4",
            "emph": dict(halo_color="rgba(6,182,212,0.3)", line_color="rgba(6,182,212,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/pred_codes": {
            "title": "Temporal (Pred)",
            "start": "#000000",
            "end": "#AA1010",
            "emph": dict(halo_color="#AA1010", line_color="#AA1010",
                        extra_width=1, line_width=3, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
    }

    # Build panels only for available data
    panels = []
    for data in available_data:
        act_type = data['type']
        config = type_configs.get(act_type)
        if config:
            panels.append(dict(
                embedding=data['embeddings'],
                pos_labels=data['pos_labels'],
                start=config['start'],
                end=config['end'],
                title=config['title'],
                emph=config['emph']
            ))

    if not panels:
        raise ValueError("No valid panels to plot")

    fig, axes = make_matplotlib_umap_grid(panels, figsize=(15, 5), dpi=400, floor_hex="#DFDFDF",
                                          elev=elev, azim=azim, connect_idx=connect_idx, base_marker=base_marker,
                                          viz_title=viz_title)
    return fig, axes


def make_matplotlib_umap_grid_2d(
    panels, figsize=(15, 3), dpi=400, floor_color="#FFFFFF",
    connect_idx=0, base_marker=36, viz_title=True, two_rows=False
):
    """
    Create 2D UMAP grid visualization.

    Args:
        panels: List of panel dicts with 'embedding', 'pos_labels', 'start', 'end', 'title', 'emph'
        figsize: Figure size
        dpi: Figure DPI
        floor_color: Background color
        connect_idx: Which sequence to connect with lines
        base_marker: Base marker size
        viz_title: Whether to show panel titles
        two_rows: If True, split into population (top) and trajectory (bottom) rows

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    if two_rows:
        # Two rows: top = population only, bottom = trajectory only
        fig = plt.figure(figsize=(figsize[0], figsize[1]*2), dpi=dpi, facecolor=floor_color)
        npanels = len(panels)
        gs = fig.add_gridspec(nrows=2, ncols=npanels, width_ratios=[1]*npanels,
                              wspace=0.05, hspace=0.15)

        axes_top = [fig.add_subplot(gs[0, i]) for i in range(npanels)]
        axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(npanels)]

        all_labels = np.concatenate([_to_numpy(p['pos_labels']).reshape(-1) for p in panels])
        label_min, label_max = float(np.min(all_labels)), float(np.max(all_labels))

        for i, (ax_top, ax_bottom, p) in enumerate(zip(axes_top, axes_bottom, panels)):
            # Top row: Population only
            _draw_panel_2d(ax_top, p, connect_sequences=(), base_marker=base_marker,
                          label_min=label_min, label_max=label_max,
                          is_user_labels=p.get('is_user', None))
            if viz_title:
                ax_top.set_title(p['title'], pad=8, fontsize=16)
            if viz_title and i == 0:
                ax_top.set_ylabel("Population", fontsize=14)
            ax_top.set_box_aspect(1)
            # Add axis labels to leftmost panel
            if i == 0:
                ax_top.set_ylabel("UMAP 2", fontsize=10, labelpad=5)
            # Add axis labels to bottom panel
            if i == npanels - 1:
                pass  # Will add x-label to bottom row instead

            # Bottom row: Single trajectory only
            E = _to_numpy(p['embedding'])
            P = _to_numpy(p['pos_labels'])

            # Also extract is_user if present
            single_is_user = None
            if 'is_user' in p and p['is_user'] is not None:
                U = _to_numpy(p['is_user'])
                single_is_user = torch.from_numpy(U[connect_idx:connect_idx+1]) if connect_idx is not None and connect_idx < U.shape[0] else p['is_user']

            if connect_idx is not None and connect_idx < E.shape[0]:
                single_embedding = torch.from_numpy(E[connect_idx:connect_idx+1])
                single_pos_labels = P[connect_idx:connect_idx+1]
            else:
                single_embedding = p['embedding']
                single_pos_labels = p['pos_labels']

            p_bottom = dict(p, embedding=single_embedding, pos_labels=single_pos_labels, is_user=single_is_user)
            _draw_panel_2d(ax_bottom, p_bottom, connect_sequences=(0,), base_marker=base_marker,
                          label_min=label_min, label_max=label_max, lines_over_points=True,
                          is_user_labels=single_is_user, force_black_markers=False)
            if viz_title and i == 0:
                ax_bottom.set_ylabel("Trajectory", fontsize=14)
            ax_bottom.set_box_aspect(1)
            # Add axis labels to leftmost and bottom panels
            if i == 0:
                ax_bottom.set_ylabel("UMAP 2", fontsize=10, labelpad=5)
            ax_bottom.set_xlabel("UMAP 1", fontsize=10, labelpad=5)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, hspace=0.15, wspace=0.05)
        return fig, (axes_top, axes_bottom)
    else:
        # Single row
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=floor_color)
        npanels = len(panels)
        gs = fig.add_gridspec(nrows=1, ncols=npanels, width_ratios=[1]*npanels, wspace=0.05)
        axes = [fig.add_subplot(gs[0, i]) for i in range(npanels)]

        all_labels = np.concatenate([_to_numpy(p['pos_labels']).reshape(-1) for p in panels])
        label_min, label_max = float(np.min(all_labels)), float(np.max(all_labels))

        for i, (ax, p) in enumerate(zip(axes, panels)):
            _draw_panel_2d(ax, p, connect_sequences=(connect_idx,), base_marker=base_marker,
                          label_min=label_min, label_max=label_max,
                          is_user_labels=p.get('is_user', None))
            if viz_title:
                ax.set_title(p['title'], pad=8, fontsize=16)
            ax.set_box_aspect(1)
            # Add axis labels to leftmost and bottom panels
            if i == 0:
                ax.set_ylabel("UMAP 2", fontsize=10, labelpad=5)
            ax.set_xlabel("UMAP 1", fontsize=10, labelpad=5)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.05)
        return fig, axes


def _draw_panel_2d(ax, panel, connect_sequences=(0,), base_marker=36,
                   label_min=None, label_max=None, lines_over_points=False, is_user_labels=None, force_black_markers=False):
    """
    Draw a single 2D panel.

    Args:
        ax: Matplotlib axis
        panel: Dict with 'embedding', 'pos_labels', 'start', 'end', 'emph'
        connect_sequences: Tuple of sequence indices to connect with lines
        base_marker: Base marker size
        label_min, label_max: Label normalization range
        lines_over_points: Whether to draw lines over points
    """
    E = _to_numpy(panel['embedding'])  # (B, L, 2)
    P = _to_numpy(panel['pos_labels']).astype(float)  # (B, L)
    B, L, _ = E.shape

    # Flatten for scatter
    flat = E.reshape(-1, 2)
    pos_flat = P.reshape(-1)

    # Normalize labels
    if label_min is None:
        label_min = float(np.min(pos_flat))
    if label_max is None:
        label_max = float(np.max(pos_flat))
    norm = Normalize(vmin=label_min, vmax=label_max, clip=True)
    cmap = two_color_cmap(panel['start'], panel['end'], alpha=1.0)

    # Draw lines first if needed
    line_z = 2 if lines_over_points else 1
    for b in connect_sequences:
        if b < B:
            coords = E[b]  # (L, 2)
            xs, ys = coords[:, 0], coords[:, 1]

            if panel.get('emph'):
                emph = panel['emph']
                line_col = emph.get("line_color", "#1a1a1a")
                # Parse rgba string
                if 'rgba' in line_col:
                    line_col = line_col.split('(')[1].split(')')[0].split(',')
                    line_col = [float(c) for c in line_col]
                    line_col[:3] = [c / 255.0 for c in line_col[:3]]
                    line_col = tuple(line_col)
                else:
                    line_col = line_col

                ax.plot(xs, ys, lw=emph.get("line_width", 2), color=line_col,
                       alpha=line_col[-1] if isinstance(line_col, tuple) else 0.5,
                       solid_capstyle="round", zorder=line_z, marker='o', markersize=3)

                # Arrow at end
                if emph.get("arrow", True) and len(xs) >= 2:
                    dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
                    arrow_len = np.sqrt(dx**2 + dy**2)
                    if arrow_len > 0:
                        ax.arrow(xs[-1], ys[-1], dx*0.3, dy*0.3,
                                head_width=arrow_len*0.15*emph.get("arrow_size", 1.0),
                                head_length=arrow_len*0.2*emph.get("arrow_size", 1.0),
                                fc=emph.get("arrow_color", "#1a1a1a"),
                                ec=emph.get("arrow_color", "#1a1a1a"),
                                zorder=line_z+1)

    # Draw points
    pt_z = 3 if lines_over_points else 2

    # Detect user/model mode: if is_user_labels is provided
    is_user_model_mode = (is_user_labels is not None)

    if is_user_model_mode:
        # User/model mode: split into two groups with different markers
        # Color by position, marker by user/model
        is_user_flat = _to_numpy(is_user_labels).astype(bool).reshape(-1)

        # Get colors based on position gradient (or force black for trajectory plots)
        if force_black_markers:
            # Trajectory plots: all markers black
            user_edge_color = 'black'
            model_color = 'black'
        else:
            # Population plots: colored by position
            colors_flat = cmap(norm(pos_flat))
            user_edge_color = colors_flat[is_user_flat] if np.any(is_user_flat) else 'black'
            model_color = colors_flat[~is_user_flat] if np.any(~is_user_flat) else 'black'

        # User points: empty circles
        if np.any(is_user_flat):
            ax.scatter(flat[is_user_flat, 0], flat[is_user_flat, 1],
                      s=base_marker, marker='o', facecolors='none',
                      edgecolors=user_edge_color, linewidths=0.8,
                      alpha=0.7, zorder=pt_z)

        # Model points: x markers
        if np.any(~is_user_flat):
            ax.scatter(flat[~is_user_flat, 0], flat[~is_user_flat, 1],
                      s=base_marker, marker='x', c=model_color,
                      linewidths=0.8, alpha=0.7, zorder=pt_z)
    else:
        # Standard mode: gradient coloring
        ax.scatter(flat[:, 0], flat[:, 1], s=base_marker, c=pos_flat,
                  cmap=cmap, norm=norm, alpha=0.7, linewidths=0.0, zorder=pt_z)

    # Style
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Black borders for all spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)


def make_fig_like_plotly_with_user_labels_flexible(
    available_data, elev=10, azim=-30, connect_idx=None,
    base_marker=50, viz_title=True, optimize_angles=True
):
    """
    Create figure with binary coloring based on user/model labels (flexible version).
    Automatically detects 2D vs 3D based on embedding dimensions.
    Black for user tokens, end color for model tokens.

    Args:
        available_data: List of dicts with keys 'type', 'embeddings', 'pos_labels', 'is_user'
        elev, azim: Default camera angles (used if optimize_angles=False, 3D only)
        connect_idx: Which sequence to connect with lines
        base_marker: Base marker size
        viz_title: Whether to show panel titles
        optimize_angles: If True, compute optimal viewing angle per panel for max separation (3D only)

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Detect dimensionality from first embedding
    if not available_data:
        raise ValueError("No data provided")

    first_embedding = available_data[0]['embeddings']
    E = _to_numpy(first_embedding)
    n_components = E.shape[-1]

    if n_components == 2:
        return _make_fig_2d_with_user_labels_flexible(
            available_data, connect_idx=connect_idx,
            base_marker=base_marker, viz_title=viz_title
        )
    elif n_components == 3:
        return _make_fig_3d_with_user_labels_flexible(
            available_data, elev=elev, azim=azim, connect_idx=connect_idx,
            base_marker=base_marker, viz_title=viz_title, optimize_angles=optimize_angles
        )
    else:
        raise ValueError(f"Unsupported number of components: {n_components}. Expected 2 or 3.")


def _make_fig_2d_with_user_labels_flexible(
    available_data, connect_idx=None, base_marker=36, viz_title=True
):
    """2D version of make_fig_like_plotly_with_user_labels_flexible."""
    type_configs = {
        "activations": {
            "title": "Activations", "start": "#000000", "end": "#EF9E72",
            "emph": dict(line_color="rgba(239,158,114,0.5)", line_width=2,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "relu/codes": {
            "title": "ReLU", "start": "#000000", "end": "#A855F7",
            "emph": dict(line_color="rgba(168,85,247,0.5)", line_width=2,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "topk/codes": {
            "title": "TopK", "start": "#000000", "end": "#A855F7",
            "emph": dict(line_color="rgba(168,85,247,0.5)", line_width=2,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "batchtopk/codes": {
            "title": "BatchTopK", "start": "#000000", "end": "#A855F7",
            "emph": dict(line_color="rgba(168,85,247,0.5)", line_width=2,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/novel_codes": {
            "title": "Temporal (Novel)", "start": "#000000", "end": "#06B6D4",
            "emph": dict(line_color="rgba(6,182,212,0.5)", line_width=2,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/pred_codes": {
            "title": "Temporal (Pred)", "start": "#000000", "end": "#CF0303",
            "emph": dict(line_color="rgba(207,3,3,0.6)", line_width=3,
                        arrow=False, arrow_size=1.0, arrow_color="#1a1a1a")
        },
    }

    panels = []
    for data in available_data:
        act_type = data['type']
        config = type_configs.get(act_type)
        if config:
            panels.append(dict(
                embedding=data['embeddings'],
                pos_labels=data['pos_labels'],  # Keep actual position labels for coloring
                is_user=data['is_user'],  # Pass user/model labels separately
                start=config['start'],
                end=config['end'],
                title=config['title'],
                emph=config['emph']
            ))

    if not panels:
        raise ValueError("No valid panels to plot")

    fig, axes = make_matplotlib_umap_grid_2d(
        panels, figsize=(15, 3), dpi=400, floor_color="#FFFFFF",
        connect_idx=connect_idx, base_marker=base_marker,
        viz_title=viz_title, two_rows=True
    )

    return fig, axes


def _make_fig_3d_with_user_labels_flexible(
    available_data, elev=10, azim=-30, connect_idx=None,
    base_marker=36, viz_title=True, optimize_angles=True
):
    """3D version of make_fig_like_plotly_with_user_labels_flexible."""
    type_configs = {
        "activations": {
            "title": "Activations", "start": "#000000", "end": "#EF9E72",
            "emph": dict(halo_color="rgba(239,158,114,0.3)", line_color="rgba(239,158,114,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "relu/codes": {
            "title": "ReLU", "start": "#000000", "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "topk/codes": {
            "title": "TopK", "start": "#000000", "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "batchtopk/codes": {
            "title": "BatchTopK", "start": "#000000", "end": "#A855F7",
            "emph": dict(halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/novel_codes": {
            "title": "Temporal (Novel)", "start": "#000000", "end": "#06B6D4",
            "emph": dict(halo_color="rgba(6,182,212,0.3)", line_color="rgba(6,182,212,0.3)",
                        extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
        "temporal/pred_codes": {
            "title": "Temporal (Pred)", "start": "#000000", "end": "#AA1010",
            "emph": dict(halo_color="rgba(170,16,16,0.4)", line_color="rgba(170,16,16,0.4)",
                        extra_width=1, line_width=3, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")
        },
    }

    panels = []
    for data in available_data:
        act_type = data['type']
        config = type_configs.get(act_type)
        if config:
            if optimize_angles:
                panel_elev, panel_azim = find_optimal_viewing_angle(data['embeddings'], data['is_user'])
                print(f"  {config['title']}: elev={panel_elev:.1f}, azim={panel_azim:.1f}")
            else:
                panel_elev, panel_azim = elev, azim

            panels.append(dict(
                embedding=data['embeddings'],
                pos_labels=data['pos_labels'],  # Keep actual position labels for coloring
                is_user=data['is_user'],  # Pass user/model labels separately
                start=config['start'],
                end=config['end'],
                title=config['title'],
                emph=config['emph'],
                elev=panel_elev,
                azim=panel_azim
            ))

    if not panels:
        raise ValueError("No valid panels to plot")

    if optimize_angles:
        print("Optimal viewing angles:")

    fig, axes = make_matplotlib_umap_grid(
        panels, figsize=(15, 5), dpi=400, floor_hex="#DFDFDF",
        elev=None, azim=None, connect_idx=connect_idx,
        base_marker=base_marker, viz_title=viz_title, two_rows=True
    )

    return fig, axes