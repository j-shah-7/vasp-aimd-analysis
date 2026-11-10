#!/usr/bin/env python3
from pathlib import Path
import csv
import re
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Combine file-1.csv, file-2.csv, etc. and add global frame, time_fs, and time_ps."
)

parser.add_argument(
    "--pattern",
    default="file-*.csv",
    help="Input CSV pattern. Default: file-*.csv"
)

parser.add_argument(
    "--dt-fs",
    type=float,
    default=0.5,
    help="Time per frame in fs. Default: 0.5 fs"
)

parser.add_argument(
    "--output",
    default="combined-temp-energy.csv",
    help="Output combined CSV filename. Default: combined-temp-energy.csv"
)

parser.add_argument(
    "--summary",
    default="combined-temp-energy-summary.csv",
    help="Output summary CSV filename. Default: combined-temp-energy-summary.csv"
)

parser.add_argument(
    "--start-at-zero",
    action="store_true",
    help="Use time = 0 ps for global_frame 1. Default is global_frame 1 = 0.5 fs."
)

args = parser.parse_args()

ROOT = Path(".")
files = list(ROOT.glob(args.pattern))

def file_sort_key(path):
    m = re.search(r"file-(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return path.name

files = sorted(files, key=file_sort_key)

if not files:
    print(f"ERROR: No files found matching pattern: {args.pattern}")
    sys.exit(1)

def get_run_number(path):
    m = re.search(r"file-(\d+)\.csv$", path.name)
    if m:
        return m.group(1)
    return path.stem

def read_csv_file(path):
    with path.open("r", newline="", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)

        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        rows = []
        for row in reader:
            # Skip fully empty rows
            if not any((v or "").strip() for v in row.values()):
                continue
            rows.append(row)

        return reader.fieldnames, rows

# Read all files first to collect all column names
all_input_headers = []
file_data = []

for file in files:
    headers, rows = read_csv_file(file)
    file_data.append((file, headers, rows))

    for h in headers:
        if h not in all_input_headers:
            all_input_headers.append(h)

meta_cols = [
    "run",
    "source_file",
    "local_frame",
    "global_frame",
    "time_fs",
    "time_ps"
]

# Avoid name collision if input already has these column names
output_input_headers = []
rename_map = {}

for h in all_input_headers:
    if h in meta_cols:
        new_h = f"input_{h}"
    else:
        new_h = h
    rename_map[h] = new_h
    output_input_headers.append(new_h)

output_headers = meta_cols + output_input_headers

global_frame = 0
summary_rows = []

with open(args.output, "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=output_headers)
    writer.writeheader()

    for file, headers, rows in file_data:
        run = get_run_number(file)

        start_global = global_frame + 1
        start_time_ps = None
        end_time_ps = None

        for local_frame, row in enumerate(rows, start=1):
            global_frame += 1

            if args.start_at_zero:
                time_fs = (global_frame - 1) * args.dt_fs
            else:
                time_fs = global_frame * args.dt_fs

            time_ps = time_fs / 1000.0

            if start_time_ps is None:
                start_time_ps = time_ps
            end_time_ps = time_ps

            out_row = {
                "run": run,
                "source_file": file.name,
                "local_frame": local_frame,
                "global_frame": global_frame,
                "time_fs": f"{time_fs:.6f}",
                "time_ps": f"{time_ps:.9f}",
            }

            for h in all_input_headers:
                out_row[rename_map[h]] = row.get(h, "")

            writer.writerow(out_row)

        end_global = global_frame

        summary_rows.append({
            "run": run,
            "source_file": file.name,
            "frames": len(rows),
            "start_global_frame": start_global if rows else "",
            "end_global_frame": end_global if rows else "",
            "start_time_ps": f"{start_time_ps:.9f}" if start_time_ps is not None else "",
            "end_time_ps": f"{end_time_ps:.9f}" if end_time_ps is not None else "",
        })

with open(args.summary, "w", newline="") as out:
    fieldnames = [
        "run",
        "source_file",
        "frames",
        "start_global_frame",
        "end_global_frame",
        "start_time_ps",
        "end_time_ps",
    ]

    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print()
print("Combined temperature/energy data")
print("=" * 70)
for row in summary_rows:
    print(
        f"{row['source_file']:>12s} : "
        f"{int(row['frames']):>8d} frames, "
        f"global {row['start_global_frame']} to {row['end_global_frame']}, "
        f"time {row['start_time_ps']} to {row['end_time_ps']} ps"
    )

print("-" * 70)
print(f"Total frames: {global_frame}")
print(f"Time step: {args.dt_fs} fs/frame = {args.dt_fs / 1000.0} ps/frame")

if args.start_at_zero:
    print("Time convention: global_frame 1 = 0 ps")
else:
    print(f"Time convention: global_frame 1 = {args.dt_fs} fs = {args.dt_fs / 1000.0} ps")

print()
print(f"Wrote: {args.output}")
print(f"Wrote: {args.summary}")
