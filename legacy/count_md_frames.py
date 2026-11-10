#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(".")

def numeric_sort_key(p):
    return int(p.name) if p.name.isdigit() else p.name

def count_xdatcar(path):
    if not path.is_file():
        return 0, None, "missing"

    frames = 0
    header = []

    with path.open("r", errors="ignore") as f:
        before_first_config = True

        for line in f:
            s = line.strip()

            if s.startswith("Direct configuration=") or s.startswith("Cartesian configuration="):
                frames += 1
                before_first_config = False
            elif before_first_config:
                header.append(line)

    natoms = None
    if len(header) >= 7:
        try:
            natoms = sum(int(x) for x in header[6].split())
        except Exception:
            natoms = None

    return frames, natoms, "ok"

def count_xyz(path):
    if not path.is_file():
        return 0, None, "missing"

    frames = 0
    natoms_ref = None

    with path.open("r", errors="ignore") as f:
        while True:
            line = f.readline()

            if not line:
                break

            if not line.strip():
                continue

            try:
                natoms = int(line.split()[0])
            except Exception:
                return frames, natoms_ref, f"bad atom-count line near frame {frames + 1}"

            if natoms_ref is None:
                natoms_ref = natoms
            elif natoms != natoms_ref:
                return frames, natoms_ref, f"atom-count changed near frame {frames + 1}"

            comment = f.readline()
            if not comment:
                return frames, natoms_ref, f"truncated after atom count near frame {frames + 1}"

            for _ in range(natoms):
                coord = f.readline()
                if not coord:
                    return frames, natoms_ref, f"truncated coordinates near frame {frames + 1}"

            frames += 1

    return frames, natoms_ref, "ok"

folders = sorted(
    [p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)],
    key=numeric_sort_key
)

print()
print("Per-folder frame count")
print("=" * 80)
print(f"{'Folder':<8} {'XDATCAR frames':>15} {'Atoms':>8} {'movie-file.xyz frames':>22} {'XYZ status':>18}")
print("-" * 80)

total_xdat = 0
total_xyz = 0

for folder in folders:
    xdat_path = folder / "XDATCAR"
    xdat_frames, xdat_atoms, xdat_status = count_xdatcar(xdat_path)

    movie_path = ROOT / f"movie-file-{folder.name}.xyz"
    xyz_frames, xyz_atoms, xyz_status = count_xyz(movie_path)

    total_xdat += xdat_frames
    total_xyz += xyz_frames

    print(
        f"{folder.name:<8} "
        f"{xdat_frames:>15} "
        f"{str(xdat_atoms):>8} "
        f"{xyz_frames:>22} "
        f"{xyz_status:>18}"
    )

print("-" * 80)
print(f"{'TOTAL':<8} {total_xdat:>15} {'':>8} {total_xyz:>22}")
print("=" * 80)

combined_xdat = ROOT / "combined_xdatcar" / "XDATCAR_combined"
combined_xyz_main = ROOT / "movie-combined.xyz"
combined_xyz_inside = ROOT / "combined_xdatcar" / "movie-combined.xyz"

cx_frames, cx_atoms, cx_status = count_xdatcar(combined_xdat)
m1_frames, m1_atoms, m1_status = count_xyz(combined_xyz_main)
m2_frames, m2_atoms, m2_status = count_xyz(combined_xyz_inside)

print()
print("Combined-file check")
print("=" * 80)
print(f"combined_xdatcar/XDATCAR_combined : {cx_frames} frames, atoms = {cx_atoms}, status = {cx_status}")
print(f"movie-combined.xyz in main folder  : {m1_frames} frames, atoms = {m1_atoms}, status = {m1_status}")
print(f"movie-combined.xyz inside combined_xdatcar/ : {m2_frames} frames, atoms = {m2_atoms}, status = {m2_status}")
print("=" * 80)

if cx_frames != total_xdat:
    print()
    print("WARNING: Combined XDATCAR frame count does not match sum of folder XDATCARs.")

if m1_frames and m1_frames != cx_frames:
    print()
    print("WARNING: Main movie-combined.xyz does not match combined XDATCAR frame count.")

if m2_frames and m2_frames != cx_frames:
    print()
    print("WARNING: combined_xdatcar/movie-combined.xyz does not match combined XDATCAR frame count.")
