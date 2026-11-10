# Historical preprocessing utilities

These are the earlier scripts supplied/developed during the MD analysis.
Use the root `aimd_batch_analysis.py` or its notebook for new raw-file analysis.
These copies preserve the earlier workflow and are not required by the new one.

| Script | Earlier role / assumption |
|---|---|
| `extract-temp-energy.py` | Extract temperature and energy from stdout; its older regex expects a particular output format. |
| `combine-temp-energy.py` | Combine already extracted `file-N.csv` rows with an explicitly chosen saved-row timestep. |
| `combine_gold_csv_1_to_38.py` | Earlier Au-specific folder, runs 1–38, with known zero-frame run 28 excluded. Edit its path/settings before reuse. |
| `count_md_frames.py` | Count existing XDATCAR/XYZ frames. |
| `xdat-combine.py` | Earlier combined XDATCAR creation; coordinate-equality deduplication can omit a genuinely stationary later frame. |
| `xdatcartoxyz-combinedmovie.py` | Convert an already combined XDATCAR to a movie. |
| `xdat2xyz(1).pl` | Supplied historical Perl converter; needs Unix utilities and can unpack/recompress inputs. Its atom-label indexing needs inspection. |
| `process_md_runs.sh` | Earlier Bash orchestration; expects the other tools beside the data and a converter named `xdat2xyz.pl`. Rename a working copy if reproducing that workflow. |

The Bash/Perl utilities require a Unix-like environment (for example WSL),
unlike the new Windows-compatible Python batch workflow. Do not assume 0.5 fs
between saved frames without checking POTIM and NBLOCK. A missing run with no
trajectory does not by itself establish that zero physical MD time elapsed.
No new correctness guarantee is made for these historical copies.
