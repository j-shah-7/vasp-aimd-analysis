"""Drop-in animation helpers for an ALREADY RUN hbn_aimd_local_analysis notebook.

Read this file into the live notebook namespace, then call
animate_hbn_with_lattice(trajectories, OUTPUT_DIR, prefix_function=prefix_for).
No raw movie, OUTCAR, MSD fit or temperature/energy processing is performed.
B/N dots are static reference sites; the metal moves in the corrected frame.
"""
from pathlib import Path
from itertools import product
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _bn_sites_in_view(points, cell, basis, xlim, ylim):
    projected = (np.asarray(points, float) @ basis.T)[:, :2]
    if not len(projected):
        return np.empty((0, 2))
    if cell is None:
        return projected
    lattice = (np.asarray(cell, float)[:2] @ basis.T)[:, :2]
    inverse = np.linalg.inv(lattice)
    corners = np.asarray(list(product(xlim, ylim))) @ inverse
    fractions = projected @ inverse
    lower = np.floor(corners.min(axis=0) - fractions.max(axis=0)).astype(int) - 1
    upper = np.ceil(corners.max(axis=0) - fractions.min(axis=0)).astype(int) + 1
    if np.prod(upper - lower + 1) > 10000:
        raise ValueError("The path spans too many lattice tiles for this animation.")
    pieces = []
    for i, j in product(range(lower[0], upper[0] + 1), range(lower[1], upper[1] + 1)):
        tile = projected + np.asarray([i, j]) @ lattice
        inside = ((tile[:, 0] >= xlim[0]) & (tile[:, 0] <= xlim[1])
                  & (tile[:, 1] >= ylim[0]) & (tile[:, 1] <= ylim[1]))
        if np.any(inside):
            pieces.append(tile[inside])
    return np.unique(np.round(np.concatenate(pieces), 9), axis=0) if pieces else np.empty((0, 2))


def make_xy_animation_with_lattice(traj, frames=300, tail_points=1000,
                                   padding_A=2.5, fps=25, lattice_alpha=.45,
                                   lattice_dot_size=40, line_width=4.0):
    """Fixed square view, true cached B/N sites, and no trail across segment gaps.

    This h-BN notebook's projected_A retains the initial substrate-frame origin;
    snapshot_A must therefore use that same origin (not the batch delta-x frame).
    The cached first/reference geometry already has this convention.
    """
    t = np.asarray(traj["time_ps"], float)
    p = np.asarray(traj["projected_A"], float)
    seg = np.asarray(traj.get("segment_id", np.zeros(len(t), int)))
    if not len(t) or p.shape != (len(t), 3) or seg.shape != t.shape:
        raise ValueError("The loaded trajectory has missing/inconsistent time, position or segment arrays.")
    if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Animation requires finite coordinates and increasing saved times.")
    if frames < 1 or int(frames) != frames or fps <= 0 or padding_A <= 0:
        raise ValueError("frames must be a positive integer; fps and padding_A must be positive.")
    if tail_points is not None and (tail_points < 1 or int(tail_points) != tail_points):
        raise ValueError("tail_points must be a positive integer, or None for the whole current segment.")
    snapshot, symbols = traj.get("snapshot_A"), traj.get("symbols")
    if snapshot is None or symbols is None:
        raise ValueError("The loaded trajectory lacks snapshot_A/symbols. Supply the small reference POSCAR geometry; no full MD reread is needed. See docs/lattice_and_animation.md.")
    snapshot, symbols = np.asarray(snapshot, float), np.asarray(symbols)
    if snapshot.shape != (len(symbols), 3) or not np.isin(symbols, ("B", "N")).any():
        raise ValueError("No valid B/N reference geometry was found in this loaded trajectory.")
    basis = np.asarray(traj.get("basis", np.eye(3)), float)
    if basis.shape != (3, 3):
        raise ValueError("basis must be the notebook's 3-by-3 substrate projection matrix.")
    center = (p[:, :2].min(axis=0) + p[:, :2].max(axis=0)) / 2
    span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1]))) + 2 * padding_A
    xlim, ylim = (center[0] - span/2, center[0] + span/2), (center[1] - span/2, center[1] + span/2)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for element, color in (("B", "#249B59"), ("N", "#3659C9")):
        sites = _bn_sites_in_view(snapshot[symbols == element], traj.get("cell"), basis, xlim, ylim)
        if len(sites):
            ax.scatter(sites[:, 0], sites[:, 1], s=lattice_dot_size, color=color,
                       alpha=lattice_alpha, edgecolors="none", zorder=0, label=element)
    metal = traj.get("metal", traj.get("label", "Metal"))
    color = {"Au": "#C49A08", "Ag": "#85898F", "Cu": "#B86C32"}.get(metal, "black")
    ax.scatter(*p[0, :2], s=110, facecolors="white", edgecolors="black",
               linewidths=2, zorder=4, label="Start")
    line, = ax.plot([], [], color=color, lw=line_width, zorder=3)
    dot, = ax.plot([], [], "o", color=color, ms=10, markeredgecolor="black",
                   markeredgewidth=1, zorder=5, label=metal)
    clock = ax.text(.025, .97, "", va="top", transform=ax.transAxes, fontsize=16,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"}, zorder=8)
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"$x$ (Å)", ylabel=r"$y$ (Å)")
    ax.set_title(f"{traj.get('label', metal)}: B/N reference lattice", fontsize=20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    ids = np.unique(np.rint(np.linspace(0, len(t)-1, min(int(frames), len(t)))).astype(int))
    multiple_segments = np.unique(seg).size > 1
    def update(i):
        lo = 0 if tail_points is None else max(0, i-int(tail_points)+1)
        take = np.arange(lo, i+1)
        take = take[seg[take] == seg[i]]
        line.set_data(p[take, 0], p[take, 1])
        dot.set_data([p[i, 0]], [p[i, 1]])
        clock.set_text(f"{t[i]:.2f} ps" + (f"   segment {seg[i]}" if multiple_segments else ""))
        return line, dot, clock
    return FuncAnimation(fig, update, frames=ids, interval=1000/fps,
                         blit=True, repeat=True, cache_frame_data=False)


def animate_hbn_with_lattice(trajectories, output_dir, label="all",
                             prefix_function=None, save_gif=True, show_inline=True,
                             frames=300, fps=25, tail_points=1000, dpi=100):
    """Animate one label or all loaded cases, reusing only in-memory arrays.

    Keep the returned dictionary assigned in the notebook. It retains the
    animation objects as required by Matplotlib. GIF playback is compressed
    visualization time; the clock displays actual saved MD time.
    """
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("No trajectories are loaded in this notebook kernel.")
    chosen = trajectories if label in (None, "all", "ALL") else [r for r in trajectories if r["label"] == label]
    if not chosen:
        raise ValueError(f"No loaded label {label!r}. Available: {[r['label'] for r in trajectories]}")
    if save_gif:
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise ImportError("Install Pillow in this notebook kernel: %pip install pillow") from None
    output_dir = Path(output_dir)
    if save_gif:
        output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for traj in chosen:
        run_label = traj["label"]
        animation = make_xy_animation_with_lattice(traj, frames=frames, fps=fps, tail_points=tail_points)
        path = None
        if save_gif:
            prefix = prefix_function(run_label) if prefix_function else run_label
            prefix = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(prefix))
            path = output_dir / f"{prefix}_trajectory_BN_lattice.gif"
            print(f"Saving {run_label}: {path}", flush=True)
            animation.save(str(path), writer=PillowWriter(fps=fps), dpi=dpi)
        if show_inline:
            from IPython.display import display, Image, HTML
            if path is not None:
                display(Image(filename=str(path), width=700))
            else:
                display(HTML(animation.to_jshtml(fps=fps)))
        plt.close(animation._fig)
        results[run_label] = {"animation": animation, "gif_path": path}
    return results
