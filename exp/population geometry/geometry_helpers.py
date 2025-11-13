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
def get_trace_data(plot_activations, centering=True, normalize=True):
    # Center and normalize
    plot_activations = plot_activations - plot_activations.mean(dim=1, keepdim=True) if centering else plot_activations
    plot_activations = F.normalize(plot_activations, p=2, dim=-1) if normalize else plot_activations

    # Flatten for UMAP
    B, L, D = plot_activations.shape
    X = plot_activations.reshape(B * L, D).float().cpu().numpy()

    reducer = UMAP(n_components=3, n_neighbors=30, min_dist=0.05,
                metric="cosine" if normalize else "euclidean",
                random_state=42)
    X_umap = reducer.fit_transform(X)      # shape: (B*L, 3)

    # Reshape for 3D plot
    traj_3d = torch.from_numpy(X_umap).reshape(B, L, 3)
    embedding = traj_3d  # (B, L, n_components)
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
def draw_panel(
    ax, embedding, pos_labels, start_hex, end_hex, title,
    connect_sequences=(0,), base_marker=36, glow_increase=12, glow_alpha=0.10,
    outlier_lo=5.0, outlier_hi=99.0, floor_hex="#FFFFFF", elev=10, azim=-30,
    emphasize=None, label_min=None, label_max=None, show_colorbar=False, colorbar_ax=None,
    lines_over_points=False
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
    pt_z = 5 if not lines_over_points else 3
    sizes = np.full(inlier_mask.sum(), base_marker)
    sc = ax.scatter(
        flat[inlier_mask,0], flat[inlier_mask,1], flat[inlier_mask,2],
        s=sizes, c=pos_flat[inlier_mask], cmap=cmap, norm=norm,
        depthshade=False, alpha=0.60, linewidths=0.0, zorder=pt_z)

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
    viz_title=True
):
    
    # fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white", constrained_layout=False)

    npanels = len(panels)
    gs = fig.add_gridspec(nrows=1, ncols=npanels, width_ratios=[1]*npanels, wspace=0.01)

    # top row: 4 scenes + optional colorbar
    axes = [fig.add_subplot(gs[0, i], projection="3d") for i in range(npanels)]

    # ----- draw panels (unchanged) -----
    all_labels = np.concatenate([_to_numpy(p['pos_labels']).reshape(-1) for p in panels])
    label_min, label_max = float(np.min(all_labels)), float(np.max(all_labels))

    for i, (ax, p) in enumerate(zip(axes, panels)):
        draw_panel(
            ax,
            p['embedding'], p['pos_labels'],
            p['start'], p['end'], p['title'],
            connect_sequences=(connect_idx,),
            base_marker=base_marker,
            glow_increase=12, glow_alpha=0.10,
            outlier_lo=5.0, outlier_hi=99.0,
            floor_hex=floor_hex,
            emphasize=p['emph'],
            label_min=label_min, label_max=label_max,
            show_colorbar=False,
            lines_over_points=(i==npanels-1)
        )
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev, azim=azim)
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
        dict(embedding=acts_embeddings,   pos_labels=acts_pos_labels,   start="#3D2F22", end="#EF9E72",
             title="Activations", emph=dict(
                 halo_color="rgba(239,158,114,0.3)", line_color="rgba(239,158,114,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # ReLU SAE
        dict(embedding=sae1_embeddings, pos_labels=sae1_pos_labels, start="#3D2F22", end="#A855F7",
             title="ReLU", emph=dict(
                 halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # TopK SAE
        dict(embedding=sae2_embeddings, pos_labels=sae2_pos_labels, start="#3D2F22", end="#A855F7",
             title="TopK", emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # BatchTopK SAE
        dict(embedding=sae3_embeddings, pos_labels=sae3_pos_labels, start="#3D2F22", end="#A855F7",
             title="BatchTopK", emph=dict(
                halo_color="rgba(168,85,247,0.3)", line_color="rgba(168,85,247,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),

        # Temporal (Novel)
        dict(embedding=novel_embeddings,  pos_labels=novel_pos_labels,  start="#3D2F22", end="#06B6D4",
             title="Temporal (Novel)", emph=dict(
                 halo_color="rgba(6,182,212,0.3)", line_color="rgba(6,182,212,0.3)",
                 extra_width=1, line_width=1, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),
        # Temporal (Pred)
        dict(embedding=pred_embeddings,   pos_labels=pred_pos_labels,   start="#3D2F22", end="#AA1010",
             title="Temporal (Pred)", emph=dict(
                 halo_color="rgba(255,180,80,0.4)", line_color="rgba(255,180,80,0.4)",
                 extra_width=1, line_width=3, arrow=True, arrow_size=1.0, arrow_color="#1a1a1a")),
    ]

    fig, axes = make_matplotlib_umap_grid(panels, figsize=(15, 5), dpi=400, floor_hex="#DFDFDF", 
                                          elev=elev, azim=azim, connect_idx=connect_idx, base_marker=base_marker,
                                          viz_title=viz_title)
    # Camera/view already set; nothing else to tweak.
    return fig, axes