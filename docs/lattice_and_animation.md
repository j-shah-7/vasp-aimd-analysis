# All-metal migration plots and B/N lattice animations

## Corrected raw batch workflow

Use the updated `aimd_batch_analysis.py` or the self-contained
`aimd_batch_analysis.ipynb`. The old notebook's preview cell preferred Au; it
did not imply that Ag/Cu figures had not been saved. The new preview shows all
successful cases by default, including multiple conditions of the same metal.
If a metal is absent, check the discovery table and `BATCH["errors"]`/`errors.csv`.
The selected `ROOT` must contain all the intended metal subfolders.

The corrected plots include:

- Per-case `xy_trajectory.png/.pdf/.svg` with B green and N blue.
- `comparison_xy_trajectories.png/.pdf/.svg` in the timestamped report root.
  All cases have separate panels, ordered Au, Ag, Cu, with a shared time scale.
- The existing thermodynamics, coordinates, MSD, scaling and comparison figures.

B/N coordinates come from the cached reference structure, not a synthetic
honeycomb. They are transformed using the same substrate center-of-mass
convention and in-plane basis as the trajectory, then shifted to its initial
metal origin. Periodic copies fill the displayed unwrapped view. The plots
retain missing sites in a vacancy structure. They use a **static reference
lattice**, not instantaneous substrate vibrations.

The raw extraction-cache version is unchanged: these plotting changes reuse
matching existing caches. Keep the same ROOT, raw inputs and extraction settings;
leave `force_reextract=False`.

Run a script as before:

```powershell
python .\aimd_batch_analysis.py
```

Or set ROOT in the corrected notebook and run its cells. In an already running
batch notebook, retain BATCH and your ROOT/settings, replace/rerun only the
large processing-functions cell from the new notebook, then run this to refresh
figures from memory, without any extraction or MSD calculation:

```python
PLOT_SETTINGS = dict(DEFAULTS)
PLOT_SETTINGS.update(BATCH.get("settings", {}))
PLOT_SETTINGS.update(SETTINGS)
for result in BATCH["results"]:
    plot_case(result, PLOT_SETTINGS)
plot_comparisons(BATCH["results"], BATCH["report_root"], PLOT_SETTINGS)
```

Then run the new View figures cell. Alternatively set its
`REFRESH_FIGURES_FROM_MEMORY=True`; this performs those same plotting calls.
Plot refresh replaces figures in the existing report folder. Numeric outputs
and raw caches are left alone.

## Replace only the animation cell in the h-BN notebook

Download `hbn_animation_with_lattice.py` into the existing **DATA_ROOT** folder
(normally `bvac-diff`). Keep the current notebook kernel open: `trajectories`,
`DATA_ROOT`, `OUTPUT_DIR` and `prefix_for` must still be defined.

Replace the last animation code cell with:

```python
exec((DATA_ROOT / "hbn_animation_with_lattice.py").read_text(encoding="utf-8"))

BN_ANIMATIONS = animate_hbn_with_lattice(
    trajectories, OUTPUT_DIR,
    label="all",                  # Or "Au", "Ag", or "Cu" exactly as a loaded label.
    prefix_function=prefix_for,
    save_gif=True,
    show_inline=True,
    frames=300,
    fps=25,
    tail_points=1000,              # Saved data points; None shows the whole current segment.
)
```

Run **only that cell**. It uses the already loaded time/coordinate/reference
arrays; it does not read raw MD, recompute MSD, or regenerate other analyses.
`label="all"` makes one GIF for every loaded trajectory; `label="Ag"` makes
only the Ag GIF. Files appear in OUTPUT_DIR as
`<existing-prefix>_trajectory_BN_lattice.gif`. Original GIF filenames are retained.
The saved GIF is also displayed inline when requested.

The metal and its trail move over fixed green B and blue N reference sites.
The view is square with padding, so a narrow metal path no longer collapses
the lattice to a thin strip. Flagged segments are never connected by a trail.
The clock shows the saved MD time; GIF playback speed is a display setting.
No new trajectory statistics are calculated from the selected display frames.

If Pillow is missing, run `%pip install pillow` in that notebook kernel, then
rerun only the animation cell. Keep BN_ANIMATIONS assigned so the animation
objects remain alive.

If a trajectory was originally loaded from a metal-only CSV and has no
`snapshot_A`/`symbols`, add its **initial reference POSCAR** in the same coordinate
frame. The existing notebook already has the small POSCAR reader:

```python
r = next(r for r in trajectories if r["label"] == "Ag")
reference = read_poscar(Path(r"D:\actual\Ag\initial\POSCAR"))
r["snapshot_A"] = reference["positions_A"]
r["symbols"] = reference["symbols"]
r["cell"] = reference["cell"]
```

Then rerun the animation cell. Do not substitute a final metal-only structure
or change the projection basis of an already prepared trajectory.

## Software references — ready to copy

1. Matplotlib Development Team. *matplotlib.animation.FuncAnimation*. Matplotlib documentation. https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html (accessed 11 November 2026).
2. Matplotlib Development Team. *matplotlib.animation.PillowWriter*. Matplotlib documentation. https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html (accessed 11 November 2026).
