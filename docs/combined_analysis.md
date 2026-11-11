# Combine the saved AIMD and NEB results

Put `combine_neb_aimd_results.py` in your main folder. Put the saved MD results
below `folder-md`, and the saved NEB/M1–M4 results below `folderm1m2m3m4`.
The script reads the exported CSV/JSON files recursively, so both one shared
results tree and separately named metal/condition folders work.

| Example path below `mainfolder` | What belongs there |
|---|---|
| `combine_neb_aimd_results.py` | This standalone script |
| `folder-md/6nn-au-100ps/AIMD_processed/` | Saved Au AIMD batch outputs |
| `folder-md/7nn-ag-100ps/AIMD_processed/` | Saved Ag AIMD batch outputs |
| `folder-md/7nn-cu-100ps/AIMD_processed/` | Saved Cu AIMD batch outputs |
| `folderm1m2m3m4/Au/M1-M2/NEB_extracted/` | Saved Au NEB outputs for one pathway |
| `folderm1m2m3m4/Au/M2-M3/NEB_extracted/` | Another pathway, kept separate |
| `folderm1m2m3m4/Ag/...` and `folderm1m2m3m4/Cu/...` | Saved Ag/Cu NEB outputs |

If a single `AIMD_processed` folder already contains all three metals, copy that
one folder into `folder-md` instead. The names above are examples, not a required
metal-folder mapping. The alternative sibling name `folder-m1m2m3m4` is also
recognized. No raw OUTCAR/XDATCAR/XYZ movies need to be copied for this stage.

## First run

Open a terminal in `mainfolder` and run:

```powershell
python .\combine_neb_aimd_results.py
```

Or use VS Code's Run Python File button. The script defaults to **its own
folder**, even if PowerShell is elsewhere. It can also live in the repository:

```powershell
python combine_neb_aimd_results.py --root "D:\research\mainfolder"
```

The standard library is sufficient for the tables. For plots, use the same
Python environment as your notebooks, or install Matplotlib:

```powershell
python -m pip install matplotlib
```

The original data and existing workflow outputs are read without modification.
Each run creates a fresh `Combined_analysis/<timestamp>/` folder.

## Files to retain from the notebook workflows

For raw batch AIMD, retain the `reports` tree with its per-case
`diffusion_summary.csv`, `capture_result.json`, `analysis_settings.json` and
`msd_*.csv` files. A copied `all_diffusion_summaries.csv` alone can supply the
summary rows, with fewer diagnostics and no MSD curves. Optional
`raw_cache/.../cache_manifest.json` files supply composition/source metadata;
large trajectory caches and movies are not read by this script.

For NEB, retain each selected result folder's `neb_summary.json`,
`neb_images.csv` and `neb_local_frequencies.csv`. Copy the whole small
`NEB_extracted` folder if that is easiest.

The earlier h-BN notebook's per-metal `*_diffusion_summary.csv`,
`*_capture_and_residence.csv` and `*_MSD_data.csv` files are also supported.
Its combined tables are skipped when matching per-metal exports are present.

## What you get automatically

| Output | Contents |
|---|---|
| `comparison_summary.csv` | Main side-by-side table: metal/condition/window, NEB energy rises, frequency ranges, formal waiting-time ranges, AIMD apparent D, capture and residence |
| `comparison_details.csv` | Additional pairing, screening and source information |
| `model_rates_by_fit.csv` | Each initial/final frequency fit separately, its formal rate/waiting time, and scientific screening flags |
| `aimd_windows_all.csv` | Every selected MD time window, kept separate |
| `neb_profiles_all.csv` | Selected NEB energy-profile and atomic-path rows |
| `neb_frequencies_all.csv` | The original frequency fits and their diagnostics |
| `selected_cases.csv` | Case IDs and their source locations |
| `input_inventory.csv` | Selected reports, older duplicates, incompatible pairs and unmatched cases |
| `input_errors.csv` | Malformed files or missing information that prevented reading a result |
| `case_settings_template.csv` | Optional settings for an explicitly justified hopping model |
| `METHODS_AND_REFERENCES.md` | Definitions, assumptions and copy-ready references |

Figures are saved as PNG, PDF and SVG:

- `NEB_profile_comparison`: relative profiles for each metal; pathways keep their labels.
- `AIMD_MSD_<window>`: the exported MSD curves for each analysis window, when available.
- `AIMD_capture_and_residence`: observed entry and first-residence durations; ≥ marks a lower bound.
- `NEB_formal_waiting_times`: frequency-window ranges of formal single-channel waiting times; gray markers fail the screening checks.

The latest timestamped report is selected separately for each calculation.
Different NEB energy conventions remain separate. To retain every rerun, use
`--all-reports`. Au 6nn and Ag/Cu 7nn remain different starting conditions; their
arrival times are not automatically interpreted as intrinsic diffusivity ranks.

## How the rate calculation is labelled

The formal one-dimensional calculation is

\[
r(T)=\nu_{\mathrm{est}}\exp[-E_a/(k_BT)],\qquad \tau=1/r.
\]

Using ν in THz gives r in ps⁻¹ and τ in ps. The initial-state frequency is used
for the forward pathway and the final-state frequency for the reverse pathway.
The default temperature is **300 K**; change the setting or use
`--temperature 300` as appropriate for your MD conditions. The program does not
infer a thermodynamic temperature from instantaneous thermal fluctuations.

These are **formal approximate model numbers**, not measured capture/escape
rates or full Vineyard-prefactor calculations. Every fitted window is retained.
The program flags missing frequencies, a maximum at an endpoint rather than a
resolved interior saddle, multiple internal energy basins, low barriers, and
the extractor's curvature-fit flags.
The default low-barrier flag is `Ea < 3 kBT`; this is a heuristic screen, not
a validity theorem. Passing the screen also requires physical inspection.
Fit-window ranges are not statistical confidence intervals. Downhill capture
and nearly barrierless Ag/Cu paths need particular care.

## Optional analytical hopping diffusivity

The first run combines the outputs without asking you to assign a hopping
network. `D_hop_model` stays blank unless you explicitly describe a suitable
equivalent surface hop in the generated settings CSV:

1. Open `case_settings_template.csv` from the new output folder.
2. For a genuinely equivalent lateral hopping path, set `process_kind` to
   `surface_hop`, `equivalent_sites` to `True`, and `z` to its number of equivalent
   outgoing directions. Leave other paths as `unspecified`/`capture`/`vertical`.
3. Optionally select one `md_id` from `selected_cases.csv`. A blank `md_id` keeps
   all compatible MD conditions alongside that pathway.
4. Run again using `--case-settings` with the actual edited file path.

For an eligible forward path the program uses `D_hop = z*a²*r/4`. The hop length
`a` is the NEB endpoint lateral displacement, not the cumulative curved path
length. An explicit positive `hop_length_A` can replace it. The rate must pass
the screening checks. Heuristic screens also reject endpoint energy differences
above 0.01 eV and vertical displacements above 0.5 Å. These checks cannot establish
site equivalence by themselves. This is a simple xy-plane uncorrelated-hop model, not a
trap-network or multistep-path diffusivity. A capture path into a B vacancy is
not automatically an equivalent surface hop. The program does not generate
an effective diffusion constant by combining M1–M4 segments.

## Interpretation

MD apparent D and fit diagnostics are copied from the notebook, not refitted.
A confined post-capture MSD plateau does not establish a free-surface D of zero.
One observed first arrival is not a mean capture rate. No escape supplies an
observed residence lower bound, while entry at the first frame is left-censored.
Geometric thresholds differ between old and new workflows and are preserved.

Pairing checks metal and known pristine/vacancy/substrate identity. It does not
prove matching charge, supercell, NEB convergence or that the MD event follows
that exact pathway. Each NEB pathway is kept independent. Inspect these physical
conditions before interpreting a model time against an AIMD event time. Thermal
atomic times are distinct from electrical-device switching/retention times.

## References — ready to copy

1. Ratsch, C.; Scheffler, M. Density-functional theory calculations of hopping rates of surface diffusion. *Physical Review B* **58**, 13163–13166 (1998). https://doi.org/10.1103/PhysRevB.58.13163
2. Vineyard, G. H. Frequency factors and isotope effects in solid state rate processes. *Journal of Physics and Chemistry of Solids* **3**, 121–127 (1957). https://doi.org/10.1016/0022-3697(57)90059-8
3. McCluskey, A. R.; Coles, S. W.; Morgan, B. J. Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation. *Journal of Chemical Theory and Computation* **21**, 79–87 (2025). https://doi.org/10.1021/acs.jctc.4c01249
4. Synopsys QuantumATK. *HTSTEvent documentation*. https://docs.quantumatk.com/manual/Types/HTSTEvent/HTSTEvent.html
