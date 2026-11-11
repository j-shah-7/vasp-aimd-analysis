# VASP AIMD analysis

Local postprocessing of existing VASP molecular-dynamics runs: raw-file
extraction, combined trajectories, temperature/energy, metal migration plots,
time-origin-averaged MSD, apparent diffusion slopes and vacancy residence.

Developed for Jameela Fatheema's Au/Ag/Cu diffusion studies on pristine,
B-vacancy and N-vacancy h-BN. Two complementary workflows are included.

## Choose a workflow

| Workflow | Entry point | Best use |
|---|---|---|
| Raw batch analysis | `aimd_batch_analysis.py` or `aimd_batch_analysis.ipynb` | Start from raw numbered runs in `nvac-diff`, `pristine`, or another metal/condition tree. No Perl/Bash preprocessing. |
| h-BN figure and capture analysis | `hbn_aimd_local_analysis.ipynb` | The existing B-vacancy dataset, lattice-background trajectories, capture/crossing analysis and cached figure revisions. |
| Importable h-BN functions | `hbn_aimd_analysis.py` | Functions matching the current embedded h-BN notebook module. This module is imported by Python; it is not a standalone CLI. |
| Historical utilities | `legacy/` | Earlier CSV/XYZ conversion and combining scripts; see `legacy/README.md` before use. |

## Install

Python 3.10 or newer is recommended; validation used Python 3.12.

```powershell
python -m pip install -r requirements.txt
```

For local Jupyter/VS Code notebook use:

```powershell
python -m pip install -r requirements-notebook.txt
```

Select that Python environment as the notebook kernel. The raw batch program
itself needs NumPy and Matplotlib; pandas, openpyxl and Pillow support the
earlier h-BN table/Excel/animation workflow.

## Raw runs: quick start

Copy `aimd_batch_analysis.py` into `nvac-diff`, beside `au-ps`, `aag-ps` and
`cu-ps`, then run it. Repeat with another copy in `pristine`:

```powershell
python .\aimd_batch_analysis.py
```

The script defaults to its own parent folder even when PowerShell is elsewhere.
It can also stay in this checkout with an explicit data root:

```powershell
python aimd_batch_analysis.py --root "D:\research\nvac-diff"
python aimd_batch_analysis.py --root "D:\research\pristine"
```

The notebook is self-contained: set `ROOT` in its first settings cell and run
the cells in order. Check the discovery table before processing. It does not
require its companion `.py` file.

Your folder names are accepted without a mapping: `au-ps`, `aag-ps` (or
`ag-ps`), and `cu-ps`. The element is read from structure contents. The
notebook discovery table previews both folder and metal identity before processing.
Expected inputs include `au-ps/1/XDATCAR`, `au-ps/2/XDATCAR`, etc.; an extra
condition level such as `gold/au-100ps/1` and raw files directly inside a metal
folder also work. The reader uses XDATCAR, OSZICAR/stdout, INCAR and
POSCAR/CONTCAR, with OUTCAR fallbacks. Compressed `.gz`, `.bz2` and `.xz` files
are supported. Metal identity comes from the structure. Multiple metal atoms
require a selected 1-based atom index.

Physical time uses the observed completed steps, POTIM and saved-coordinate
stride (NBLOCK or an explicit override). It is not forced to 90 ps [3]. Review
the run audit if timing defaults were needed. Fixed cells and consistent atom
ordering are required. The default restart policy checks CONTCAR→POSCAR
geometry; `strict` additionally requires saved ionic velocity matching, while
`split` excludes every cross-restart MSD pair.

## Run order and connection to NEB

1. Analyze one `nvac-diff` dataset with `aimd_batch_analysis.ipynb` and check its
   discovery table before running the extraction.
2. Repeat with a separate copy in `pristine` and that dataset's `ROOT` path.
3. Use `neb_extract_and_review.ipynb` from the NEB repository inside each NEB
   parent folder containing `00`, `01`, ... . Set its `NEB_ROOT` accordingly.

The NEB and MD notebooks process separate existing calculations. Neither needs
the other's output to run. NEB describes the static migration energy landscape;
MD describes the observed temperature, motion, capture and residence over time.
Each notebook contains its own processing code, so running its companion `.py`
file as well is optional. The earlier `hbn_aimd_local_analysis.ipynb` is for the
existing B-vacancy figure workflow and is not needed to preprocess these raw runs.

## Outputs and settings

Each data root gets an `AIMD_processed` folder. Raw exports and compact caches
are in `raw_cache/<case>/<signature>/`. New analysis reports go into
`reports/<timestamp>/`, with per-case and cross-metal results:

- Combined temperature/energy CSV, per-run CSVs, combined XDATCAR and XYZ movie.
- Full tracked-metal coordinates, vacancy distance/height and restart audit.
- Temperature/energy, x/y/height/r, XY trajectory, MSD and scaling plots.
- Whole/equilibrated/post-residence MSD CSVs, apparent slope summaries and block diagnostics.
- Capture records, settings, warnings and processing errors.

Figures are exported as PNG/PDF/SVG, with Au gold, Ag gray and Cu brown. The
first pass can take several minutes and full movies can be large. Changing
only MSD windows, fits or styles reuses compatible extraction caches.

Edit the `DEFAULTS` dictionary or notebook `SETTINGS`; an optional JSON file
can be passed with `--config`. Examples:

```powershell
python aimd_batch_analysis.py --root "D:\research\nvac-diff" --fit-lag 2 8 --end-ps 90
python aimd_batch_analysis.py --root "D:\research\nvac-diff" --restart-policy strict
python aimd_batch_analysis.py --root "D:\research\nvac-diff" --no-movies
```

These are usage examples, not recommended final fit limits. See
[the detailed quick start](docs/AIMD_batch_quick_start.md) for settings and
output filenames.

## Whole-trajectory and adsorbed-state MSD

For stationary two-dimensional normal diffusion,

$$
D_{\mathrm{app}}=\frac14\frac{d\,\mathrm{MSD}_{xy}(\tau)}{d\tau},
\qquad 1\;\mathrm{\mathring A^2/ps}=10^{-8}\;\mathrm{m^2/s}.
$$

The code unwraps periodic crossings before removing substrate translation.
Time-origin averages use only valid pairs within continuous segments. A
10-ps lag axis still draws time origins from throughout the selected ~90-ps
trajectory. The default fit interval must be fully available.

The raw workflow reports whole, after-equilibration and, where identified,
first persistent vacancy-residence windows. A manual `post_start_ps` selects a
known adsorption time. The 1 ps equilibration exclusion and 1–5 ps fit interval
are initial settings requiring inspection. A single missing B/N is classified
as a vacancy candidate and its three neighbors are checked geometrically.
Automatic persistent entry is a geometric criterion, not proof of chemical
adsorption. Pristine systems do not get a fabricated vacancy-capture time.

A whole trajectory may mix mobile and trapped periods. A bound-state plateau
describes confinement; a small fitted slope is not automatically a valid
surface diffusion coefficient. Apparent slopes, log–log exponents and R² are
diagnostics. Correlated MSD samples make ordinary line-fit standard errors
unreliable [1,2]; block spread is descriptive rather than a formal confidence
interval. One observed capture is not an ensemble mean capture rate. Residence
without observed escape is right-censored. Electrical switching energy/time
is not directly measured by the thermostat energy or a single atomic trajectory.

VASP energy quantities remain explicit: `E` is the MD extended-system total,
`F/TOTEN` the electronic free energy, and `E0` the sigma→0 extrapolation [4].

## Existing B-vacancy plotting notebook

`hbn_aimd_local_analysis.ipynb` retains the previously developed manuscript
figure workflow. Set `DATA_ROOT`, `FOLDERS` and independent thermodynamics
paths to the actual data. Its saved configuration initially expects existing
`AIMD_results` caches (`REPLOT_ONLY=True`). For a fresh XYZ/OUTCAR/CSV/Excel
read, set `REPLOT_ONLY=False` and configure `SOURCE` and file paths. It does
not replace the raw batch converter.

The optional common 12–50 ps post-capture comparison is specific to the earlier
B-vacancy analysis; choose an appropriate continuous interval for other data
or disable `COMPARE_POST_CAPTURE_MSD`. Its vacancy-neighbor indices are
**zero-based**, while the raw batch workflow's explicit atom overrides are
**one-based**. Notebook plots retain the requested 4-point line width,
20-point ticks/legends and 26-point axis labels. Inspect axis limits and the
exported out-of-range table when reusing its temperature/energy styling.

## Validation and scope

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic VASP-style inputs and an independently calculated MSD,
including a 90-ps random walk and confined motion. Both raw batch notebook
folder layouts were executed during preparation. Actual hard-drive datasets
require local timing, atom-order, restart and vacancy-geometry checks.

All repository notebook execution outputs are cleared. Raw research datasets,
calculation outputs and VASP pseudopotentials are excluded. A redistribution
license has not yet been selected; existing historical script provenance is
retained. The companion static-path workflow is
[j-shah-7/vasp-neb-analysis](https://github.com/j-shah-7/vasp-neb-analysis).

## References — ready to copy

1. Bullerjahn, J. T.; von Bülow, S.; Hummer, G. Optimal estimates of self-diffusion coefficients from molecular dynamics simulations. *Journal of Chemical Physics* **153**, 024116 (2020). https://doi.org/10.1063/5.0008312.
2. McCluskey, A. R.; Coles, S. W.; Morgan, B. J. Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation. *Journal of Chemical Theory and Computation* **21**, 79–87 (2025). https://doi.org/10.1021/acs.jctc.4c01249.
3. VASP Wiki. NBLOCK. https://vasp.at/wiki/NBLOCK.
4. VASP Wiki. OSZICAR. https://vasp.at/wiki/OSZICAR.
5. VASP Wiki. Molecular-dynamics calculations. https://vasp.at/wiki/Molecular-dynamics_calculations.
