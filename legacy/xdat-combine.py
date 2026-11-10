#!/usr/bin/env python3
import os
import sys
import re

# Usage:
#   python xdat-combine.py
#   python xdat-combine.py 1 2 3 4 5 6

if len(sys.argv) > 1:
    folders = sys.argv[1:]
else:
    folders = sorted(
        [d for d in os.listdir(".") if os.path.isdir(d) and re.fullmatch(r"\d+", d)],
        key=lambda x: int(x)
    )

outdir = "combined_xdatcar"
os.makedirs(outdir, exist_ok=True)

combined_xdatcar = os.path.join(outdir, "XDATCAR_combined")


def is_config_line(line):
    s = line.strip()
    return s.startswith("Direct configuration=") or s.startswith("Cartesian configuration=")


def read_header_and_first_config(f, xdatcar_file):
    header = []

    for line in f:
        if is_config_line(line):
            if len(header) < 7:
                raise ValueError(f"Header too short in {xdatcar_file}")

            try:
                counts = header[6].split()
                natoms = sum(int(x) for x in counts)
            except Exception:
                raise ValueError(f"Could not read atom counts from header line 7 in {xdatcar_file}")

            return header, natoms, line

        header.append(line)

    raise ValueError(f"No Direct/Cartesian configuration blocks found in {xdatcar_file}")


def frame_iterator(f, natoms, first_config_line, xdatcar_file):
    config_line = first_config_line

    while config_line is not None:
        frame = []

        for _ in range(natoms):
            line = f.readline()

            if not line:
                print(f"Warning: incomplete frame at end of {xdatcar_file}; skipping final incomplete frame.")
                return

            frame.append(line.rstrip("\n"))

        yield frame

        config_line = None

        for line in f:
            if is_config_line(line):
                config_line = line
                break
            elif line.strip() == "":
                continue
            else:
                # Usually this should not happen in XDATCAR.
                # Continue searching rather than crashing.
                continue


def coords_equal(frame1, frame2, tol=1e-8):
    if len(frame1) != len(frame2):
        return False

    for a, b in zip(frame1, frame2):
        try:
            xa = [float(x) for x in a.split()[:3]]
            xb = [float(x) for x in b.split()[:3]]
        except Exception:
            return False

        for va, vb in zip(xa, xb):
            if abs(va - vb) > tol:
                return False

    return True


header_ref = None
natoms_ref = None
last_frame = None
total_frames = 0
written_frames = 0

print(f"Folders to combine: {' '.join(folders)}")

with open(combined_xdatcar, "w") as out:
    for folder in folders:
        xdatcar_file = os.path.join(folder, "XDATCAR")

        if not os.path.isfile(xdatcar_file):
            print(f"Warning: {xdatcar_file} not found. Skipping.")
            continue

        with open(xdatcar_file, "r", errors="ignore") as f:
            header, natoms, first_config_line = read_header_and_first_config(f, xdatcar_file)

            if header_ref is None:
                header_ref = header
                natoms_ref = natoms

                for line in header_ref:
                    out.write(line if line.endswith("\n") else line + "\n")

            else:
                if natoms != natoms_ref:
                    print(f"Error: atom count mismatch in {xdatcar_file}")
                    print(f"Expected {natoms_ref}, found {natoms}")
                    sys.exit(1)

            folder_frames = 0
            folder_written = 0

            for frame in frame_iterator(f, natoms, first_config_line, xdatcar_file):
                folder_frames += 1
                total_frames += 1

                if last_frame is not None and folder_written == 0:
                    if coords_equal(last_frame, frame):
                        print(f"Duplicate first frame detected in folder {folder}; skipping it.")
                        continue

                written_frames += 1
                folder_written += 1

                out.write(f"Direct configuration= {written_frames}\n")
                for coord_line in frame:
                    out.write(coord_line + "\n")

                last_frame = frame

            print(f"Read {folder_frames} frames from {xdatcar_file}; wrote {folder_written} frames.")

if header_ref is None:
    print("Error: no XDATCAR files were found.")
    sys.exit(1)

print(f"Wrote {combined_xdatcar}")
print(f"Total frames read: {total_frames}")
print(f"Total frames written: {written_frames}")
print(f"Atoms per frame: {natoms_ref}")
