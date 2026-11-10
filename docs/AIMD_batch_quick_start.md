# Analyze raw Au, Ag and Cu AIMD runs

Use `aimd_batch_analysis.py` **or** `aimd_batch_analysis.ipynb`. Each is self-contained.
No Perl/Bash conversion or existing temperature/energy CSV is required.

## Run the Python script

1. Copy the Python file into `nvac-diff`, alongside the `gold`, `silver` and `copper` folders.
2. Run it with your Python 3 environment in VS Code, or open a terminal in that folder and run:

   ```powershell
   python .\aimd_batch_analysis.py
   ```

3. Copy the same file into `pristine` and run it there as well.

The script defaults to **its own location**, even if PowerShell is still in
`C:\Users\jfath`. An explicit root also works:

```powershell
python aimd_batch_analysis.py --root "D:\YOUR_DATA_PATH\nvac-diff"
```

If required, install NumPy and Matplotlib in the same Python environment:

```powershell
python -m pip install numpy matplotlib
```

## Run the notebook

Copy the notebook into each system folder. Set `ROOT` in the first code cell to
your actual `nvac-diff` or `pristine` folder, select the correct Python kernel,
and run the cells in order. Check the discovered paths before the processing
cell. It contains its own processing code; the `.py` file is optional.

## Input layouts and automatic processing

The reader finds numeric restart folders such as `gold/1`, `gold/2`, etc., also
under an extra condition folder such as `gold/au-100ps/1`. It sorts runs
numerically. Raw files directly inside a metal folder also work.

| Raw file | Used for |
|---|---|
| XDATCAR | Full saved trajectory, combined XDATCAR, XYZ and tracked-metal coordinates |
| OSZICAR or stdout | MD temperature, extended total E, electronic free energy F and E0 |
| INCAR / OUTCAR header | POTIM, NBLOCK and calculation parameters |
| POSCAR / CONTCAR | Element/atom identification, reference geometry and restart continuity |
| OUTCAR | Position/thermodynamics fallback if preferred records are unavailable |

Compressed `.gz`, `.bz2` and `.xz` files can be read without unpacking. If OUTCAR
is the only trajectory source, provide a readable POSCAR or CONTCAR as the initial
structure. Fixed cell vectors and consistent atom ordering are required.

Metal identity comes from the structure, not the folder spelling. The code
tracks one Au/Ag/Cu/Li atom. With multiple metal atoms, select its 1-based index
in the notebook/settings. N/B vacancy classification uses composition and a
three-neighbor geometric check; inspect the reported atom numbers. Pristine
has no automatic vacancy-capture window.

The first raw pass can take several minutes and the full movies can be large.
Changing only MSD fits, analysis windows or plotting style reuses cached data.
Raw files are read without modification; new reports get a new timestamp.

## Results

Everything is written below `AIMD_processed` in the selected input folder.

| Output location | Results |
|---|---|
| `raw_cache/<case>/<signature>/` | `combined-temp-energy.csv`, `trajectory.csv`, `run_audit.csv`, `movie-combined.xyz`, `XDATCAR_combined`, compact cache and per-run `runs/file-N.csv` |
| `reports/<timestamp>/<case>/` | Temperature/energy, x/y/height/r, XY migration, MSD, local-scaling plots; per-window MSD and block CSVs; capture record; `diffusion_summary.csv` |
| `reports/<timestamp>/` | `all_diffusion_summaries.csv`, Au/Ag/Cu MSD comparisons, `errors.csv`, settings and `METHODS_AND_REFERENCES.txt` |

Figures are saved as PNG, PDF and SVG. Au is gold, Ag gray and Cu brown.
The combined XDATCAR uses saved-frame ordinal labels; use `trajectory.csv` for
physical timestamps. XYZ comments include physical time.

## Whole-run and post-adsorption analysis

The notebook calculates time-origin-averaged lateral MSD and an apparent slope:

\[
D_{\mathrm{app}}=\frac{1}{4}\frac{d\mathrm{MSD}_{xy}}{d\tau},
\quad 1\;\mathrm{\mathring A^2/ps}=10^{-8}\;\mathrm{m^2/s}.
\]

Default windows are **whole**, **after 1 ps equilibration**, and, if identified,
the **first persistent vacancy residence after capture**. An independent
adsorption-time assignment can instead define `post_selected`.

The full-run result may mix mobile and trapped periods. A post-binding plateau
describes confinement. The code labels slope estimates as apparent and records
the MSD exponent and fit R²; a screening pass is not automatic proof of normal,
stationary diffusion. Correlated time-origin/lag samples also mean an ordinary
line-fit error is not a valid diffusion uncertainty [1,2]. Block spread is
descriptive, not a formal confidence interval.

A 10-ps MSD lag axis still uses origins throughout the selected ~90-ps run.
The actual time comes from observed output with POTIM and saved-coordinate
stride, not a presumed 90-ps duration [3]. Unverified restarts/gaps are split
for MSD pairs. A trajectory can span 90 ps yet lack enough continuous data for
a 1–5 ps fit: inspect `run_audit.csv` in that case.

The default restart policy joins matching CONTCAR→POSCAR geometry within 0.02 Å.
Unverified velocities are flagged. `strict` additionally requires saved velocity
matching; `split` prevents every cross-restart MSD pair. A saved velocity match
does not certify all thermostat/integrator state.

Automatic capture means lateral distance ≤1.5 Å and absolute height ≤3 Å for at
least 0.5 ps at the vacancy. These are adjustable **geometric** criteria, not
proof of chemical adsorption. The post-capture window starts after confirmation
and buffer and ends before the first persistent exit or at the continuous
segment end. A residence with no observed escape is right-censored.

## Change settings

Edit the notebook settings or the `DEFAULTS` dictionary near the top of the
Python file. Useful command-line alternatives:

```powershell
python .\aimd_batch_analysis.py --fit-lag 2 8 --end-ps 90
python .\aimd_batch_analysis.py --restart-policy strict
python .\aimd_batch_analysis.py --no-movies
```

An optional `--config settings.json` accepts the same settings as a JSON object.
For instance, a **measured** post-adsorption time can be passed with
`--post-start Au=YOUR_TIME_IN_PS`, replacing the placeholder with a number.
In a notebook, use `SETTINGS["post_start_ps"] = {"Au": measured_Au_time}`.
Exact case IDs/relative condition paths can replace element keys for multiple
series of the same metal.

The 1 ps equilibration exclusion and 1–5 ps fit are starting selections for
inspection. Check temperature, energy and MSD before choosing final intervals.
`energy_field="E"` plots extended-system total E; choose `F` for electronic free
energy or `E0` for the extrapolated energy [4]. These quantities are distinct from
electrical switching energy.

Validation used synthetic raw VASP-style datasets, an independent direct MSD
calculation, and a 90-ps random walk/confined trajectory. Your hard-drive data
have not been analyzed here; inspect the first real run's timing, warnings and
vacancy-neighbor identification before using its values in the manuscript.

## References — ready to copy

1. Bullerjahn, J. T.; von Bülow, S.; Hummer, G. Optimal estimates of self-diffusion coefficients from molecular dynamics simulations. *Journal of Chemical Physics* **153**, 024116 (2020). https://doi.org/10.1063/5.0008312

2. McCluskey, A. R.; Coles, S. W.; Morgan, B. J. Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation. *Journal of Chemical Theory and Computation* **21**, 79–87 (2025). https://doi.org/10.1021/acs.jctc.4c01249

3. VASP Wiki. NBLOCK. https://vasp.at/wiki/NBLOCK (accessed November 10, 2026).

4. VASP Wiki. OSZICAR. https://vasp.at/wiki/OSZICAR (accessed November 10, 2026).

5. VASP Wiki. Molecular-dynamics calculations. https://vasp.at/wiki/Molecular-dynamics_calculations (accessed November 10, 2026).
