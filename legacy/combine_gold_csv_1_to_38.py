#!/usr/bin/env python3
"""Combine file-1.csv through file-38.csv in numeric order.

Edit the settings below, then run this script or use %run in Jupyter.
The inputs must already contain temperature/energy data extracted from VASP.
Each retained CSV row is treated as one saved frame.
Run 28 is explicitly excluded because it failed and saved zero images.
"""

from pathlib import Path
import csv
import math
import sys


# SETTINGS: this is the gold folder configured in your analysis notebook.
FOLDER = Path(r"D:\pathfinder-cades\md-calc\bvac-diff\6nn\gold\au-100ps")
FIRST_RUN = 1
LAST_RUN = 38
SKIP_RUNS = {28}  # Confirmed failed run with ZERO saved images.
DT_FS = 0.5  # Time BETWEEN saved CSV rows, in fs.
START_AT_ZERO = False  # False: frame 1 is at 0.5 fs; True: frame 1 is at 0 fs.


def read_csv_file(path):
    """Read comma, semicolon, or tab separated data without changing values."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(handle, dialect=dialect)
        headers = next((row for row in reader if any(v.strip() for v in row)), None)
        if headers is None:
            raise ValueError(f"{path.name}: no CSV header found.")
        headers = [h.strip() for h in headers]
        if any(not h for h in headers) or len(set(headers)) != len(headers):
            raise ValueError(f"{path.name}: blank or duplicate column names: {headers}")
        rows = []
        for values in reader:
            if not any(v.strip() for v in values):
                continue
            if len(values) != len(headers):
                raise ValueError(
                    f"{path.name}, line {reader.line_num}: expected {len(headers)} "
                    f"columns, found {len(values)}."
                )
            rows.append(dict(zip(headers, values)))
    if not rows:
        raise ValueError(f"{path.name}: no data rows found.")
    return headers, rows


def combine_csv_files(folder, first_run=1, last_run=38, dt_fs=0.5,
                      start_at_zero=False, skip_runs=()):
    """Validate all inputs, then write the combined CSV and per-run summary."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist. Edit FOLDER:\n{folder}")
    if first_run < 1 or last_run < first_run:
        raise ValueError("Use 1 <= FIRST_RUN <= LAST_RUN.")
    if not math.isfinite(dt_fs) or dt_fs <= 0:
        raise ValueError("DT_FS must be a positive finite number.")
    skip_runs = set(skip_runs)
    expected_runs = set(range(first_run, last_run + 1))
    if skip_runs - expected_runs:
        raise ValueError("SKIP_RUNS must be within FIRST_RUN to LAST_RUN.")
    if skip_runs == expected_runs:
        raise ValueError("No input runs remain after applying SKIP_RUNS.")

    # Explicit numeric order: 1, 2, 3, ..., 38 (rather than 1, 10, 11, ...).
    inputs = [(run, folder / f"file-{run}.csv")
              for run in range(first_run, last_run + 1)]
    missing = [(run, path) for run, path in inputs
               if run not in skip_runs and not path.is_file()]
    if missing:
        details = [
            f"Input folder: {folder}",
            "Missing input CSVs: " + ", ".join(path.name for _, path in missing),
        ]
        available = sorted((p for p in folder.iterdir() if p.is_file()),
                           key=lambda p: p.name.casefold())
        for run, path in missing:
            matches = [p.name for p in available if str(run) in p.name]
            if matches:
                details.append(
                    f"Filenames containing '{run}' (check for a naming mismatch): "
                    + ", ".join(matches)
                )
            else:
                details.append(f"No filenames containing '{run}' found in this folder.")
        details.append(
            "Restore the missing CSV, or correct its filename if it already exists. "
            "A missing run must be resolved before a continuous time axis can be built."
        )
        raise FileNotFoundError("\n".join(details))

    # Validate every file before opening either output for writing.
    file_data, all_headers = [], []
    for run, path in inputs:
        if run in skip_runs:
            file_data.append((run, path, []))
            continue
        headers, rows = read_csv_file(path)
        file_data.append((run, path, rows))
        for header in headers:
            if header not in all_headers:
                all_headers.append(header)

    metadata = ["run", "source_file", "local_frame", "global_frame",
                "time_fs", "time_ps"]
    reserved = set(metadata) | set(all_headers)
    rename = {}
    for header in all_headers:
        new_header = header
        if header in metadata:
            while new_header in reserved:
                new_header = "input_" + new_header
            reserved.add(new_header)
        rename[header] = new_header

    output = folder / "combined-temp-energy.csv"
    summary_output = folder / "combined-temp-energy-summary.csv"
    summary, global_frame = [], 0
    offset = 1 if start_at_zero else 0

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata + list(rename.values()))
        writer.writeheader()
        for run, path, rows in file_data:
            first_global = global_frame + 1
            for local_frame, row in enumerate(rows, 1):
                global_frame += 1
                time_fs = (global_frame - offset) * dt_fs
                record = {
                    "run": run, "source_file": path.name,
                    "local_frame": local_frame, "global_frame": global_frame,
                    "time_fs": f"{time_fs:.6f}",
                    "time_ps": f"{time_fs / 1000:.9f}",
                }
                record.update({rename[h]: row.get(h, "") for h in all_headers})
                writer.writerow(record)
            summary.append({
                "run": run, "source_file": path.name, "frames": len(rows),
                "start_global_frame": first_global if rows else "",
                "end_global_frame": global_frame if rows else "",
                "start_time_ps": f"{(first_global - offset) * dt_fs / 1000:.9f}" if rows else "",
                "end_time_ps": f"{(global_frame - offset) * dt_fs / 1000:.9f}" if rows else "",
                "status": "combined" if rows else "skipped: failed run, zero saved images",
            })

    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        if row["frames"]:
            print(f"{row['source_file']:>12}: {row['frames']:>8,} rows | "
                  f"{row['start_time_ps']} to {row['end_time_ps']} ps")
        else:
            print(f"{row['source_file']:>12}: {row['status']}")
    print(f"\nCombined {len(inputs) - len(skip_runs)} CSVs: {global_frame:,} frames")
    print(f"Time between saved rows: {dt_fs} fs")
    print(f"Wrote: {output}")
    print(f"Wrote: {summary_output}")
    return output, summary_output


if __name__ == "__main__":
    try:
        combine_csv_files(FOLDER, FIRST_RUN, LAST_RUN, DT_FS, START_AT_ZERO, SKIP_RUNS)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
