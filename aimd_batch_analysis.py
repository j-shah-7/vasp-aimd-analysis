#!/usr/bin/env python3
"""Raw fixed-cell VASP AIMD processing for gold/silver/copper folder trees.

Place in nvac-diff or pristine and run this file. No Perl, Bash, or pre-extracted
CSV/XYZ files are required. Large raw movies are streamed; compact results are
cached for subsequent plotting and MSD analysis. See the accompanying notebook
for methods, settings, and references.
"""
from pathlib import Path
from itertools import product, chain
from datetime import datetime
import argparse
import bz2
import contextlib
import csv
import gzip
import hashlib
import json
import lzma
import math
import os
import re
import sys
import time

try:
    import numpy as np
except ImportError:
    raise SystemExit("Install dependencies: python -m pip install numpy matplotlib")

VERSION = "1.0"


NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


NUM_RE = re.compile(NUMBER)


MASSES = {
    "H": 1.00794, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.9984, "Na": 22.98977, "Mg": 24.305,
    "Al": 26.98154, "Si": 28.085, "P": 30.97376, "S": 32.06,
    "Cl": 35.45, "K": 39.0983, "Ca": 40.078, "Ti": 47.867,
    "V": 50.9415, "Cr": 51.9961, "Mn": 54.93804, "Fe": 55.845,
    "Co": 58.93319, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.92160, "Se": 78.971,
    "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95, "Ru": 101.07,
    "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Hf": 178.49, "Ta": 180.9479, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.96657, "Hg": 200.592, "Pb": 207.2, "Bi": 208.9804,
}


HOST_ELEMENTS = {"B", "N", "C", "Mo", "W", "S", "Se", "Te"}


def number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def open_text(path):
    path = Path(path)
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix, open)
    return opener(path, "rt", encoding="utf-8-sig", errors="replace")


def existing_file(folder, stem):
    return next((Path(folder) / name for name in
                 (stem, stem + ".gz", stem + ".bz2", stem + ".xz")
                 if (Path(folder) / name).is_file()), None)


def read_incar(path):
    if not Path(path).is_file():
        return {}
    result = {}
    with open_text(path) as handle:
        for line in handle:
            line = re.split(r"[#!]", line, maxsplit=1)[0]
            for item in line.split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    result[key.strip().upper()] = value.strip()
    return result


def read_poscar(path, species_hint=None):
    with open_text(path) as handle:
        lines = handle.readlines()
    if len(lines) < 8:
        raise ValueError(f"Empty or incomplete structure: {path}")
    scales = np.asarray([number(v) for v in lines[1].split()], float)
    base = np.asarray([[number(v) for v in line.split()[:3]] for line in lines[2:5]])
    if base.shape != (3, 3) or abs(np.linalg.det(base)) < 1e-12:
        raise ValueError(f"Invalid lattice: {path}")
    if len(scales) == 1:
        factor = scales[0]
        if factor < 0:
            factor = (abs(factor) / abs(np.linalg.det(base))) ** (1 / 3)
        if factor <= 0:
            raise ValueError(f"Invalid scale factor: {path}")
        scale_xyz = np.repeat(factor, 3)
    elif len(scales) == 3 and np.all(scales > 0):
        scale_xyz = scales
    else:
        raise ValueError(f"Unsupported POSCAR scale factors: {path}")
    cell = base * scale_xyz[None, :]
    tokens = lines[5].split()
    if tokens and all(v.isdigit() for v in tokens):
        counts = [int(v) for v in tokens]
        species = list(species_hint or [])
        cursor = 6
    else:
        species = tokens
        counts = [int(v) for v in lines[6].split()]
        cursor = 7
    if len(species) != len(counts):
        raise ValueError(f"VASP-4 structure needs element names from an OUTCAR: {path}")
    if lines[cursor].strip().lower().startswith("s"):
        cursor += 1
    mode = lines[cursor].strip().lower()
    cursor += 1
    if not mode.startswith(("d", "c", "k")):
        raise ValueError(f"Expected Direct or Cartesian coordinates: {path}")
    natoms = sum(counts)
    coordinates = np.asarray([[number(v) for v in line.split()[:3]]
                              for line in lines[cursor:cursor + natoms]], float)
    if coordinates.shape != (natoms, 3):
        raise ValueError(f"Incomplete atom positions: {path}")
    positions = coordinates @ cell if mode.startswith("d") else coordinates * scale_xyz
    if not np.isfinite(positions).all():
        raise ValueError(f"Nonfinite atom positions: {path}")
    return {"cell": cell, "positions": positions, "species": species, "counts": counts,
            "symbols": np.repeat(species, counts), "source": str(path), "quality": "structure_file"}


def minimum_image(displacements, cell, pbc=(True, True, False)):
    """Nearest image, including neighboring translations for hexagonal cells."""
    shape = np.asarray(displacements).shape
    fractional = np.asarray(displacements).reshape(-1, 3) @ np.linalg.inv(cell)
    periodic = np.asarray(pbc, bool)
    fractional[:, periodic] -= np.rint(fractional[:, periodic])
    best = fractional @ cell
    distance2 = np.sum(best ** 2, axis=1)
    for shift in product(*[(-1, 0, 1) if flag else (0,) for flag in pbc]):
        candidate = (fractional + shift) @ cell
        trial = np.sum(candidate ** 2, axis=1)
        improve = trial < distance2
        best[improve], distance2[improve] = candidate[improve], trial[improve]
    return best.reshape(shape)


# EDITABLE DEFAULTS. Notebook settings can override any of these.
DEFAULTS = {
    "dt_fs_override": None,
    "dt_fs_fallback": 0.5,
    "saved_frame_stride_override": None,
    "energy_field": "E",  # E: extended total; F: electronic free energy; E0: sigma->0.
    "equilibration_ps": 1.0,
    "analysis_end_ps": None,  # Set 90.0 to select the first 90 ps explicitly.
    "post_start_ps": {},  # e.g. {"Au": 5.0, "Ag": 8.0, "Cu": 4.0}.
    "metal_atom_index_1based": {},
    "vacancy_neighbors_1based": {},
    "capture_radius_A": 1.5,
    "capture_height_A": 3.0,
    "capture_dwell_ps": 0.5,
    "post_capture_buffer_ps": 0.5,
    "bond_cutoff_A": 1.9,
    "restart_policy": "geometry",  # geometry, strict (also velocities), or split.
    "restart_tolerance_A": 0.02,
    "max_adjacent_metal_jump_A": 2.0,
    "analysis_stride": 10,
    "msd_max_lag_ps": 10.0,
    "msd_max_fraction": 0.25,
    "fit_lag_ps": [1.0, 5.0],
    "block_min_duration_ps": 20.0,
    "write_xyz": True,
    "write_combined_xdatcar": True,
    "movie_stride": 1,
    "progress_every_frames": 10000,
    "force_reextract": False,
    "make_plots": True,
    "figure_formats": ["png", "pdf", "svg"],
    "font_size": 20,
    "axis_size": 26,
    "tick_size": 20,
    "line_width": 4.0,
}
COLORS = {"Au": "#C49A08", "Ag": "#85898F", "Cu": "#B86C32", "Li": "#7A49A5"}
IGNORE_DIRS = {"AIMD_processed", "AIMD_results", "AIMD_results_updated", "NEB_extracted",
               "combined_xdatcar", "postprocess-logs", "__pycache__"}
RAW_STEMS = ("XDATCAR", "OUTCAR", "OSZICAR", "stdout", "vasp.out")
CFG_RE = re.compile(r"^\s*(Direct|Cartesian)\s+configuration\s*=\s*(\d+)", re.I)
STEP_T_RE = re.compile(r"(?<!\d)(\d+)\s+T\s*=\s*(" + NUMBER + ")")
FIELD_RES = {key: re.compile(r"(?<![A-Za-z0-9_])" + key + r"\s*=\s*(" + NUMBER + ")")
             for key in ("E", "F", "E0")}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(safe_json(value), handle, indent=2, ensure_ascii=False, allow_nan=False)


def save_csv(path, rows, fields=None):
    rows = list(rows)
    if not fields:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_json(row.get(key)) for key in fields})


def lookup(mapping, case, default=None):
    for key in (case.get("id"), case.get("relative"), case.get("metal")):
        if key in mapping:
            return mapping[key]
    return default


def raw_present(folder):
    return any(existing_file(folder, stem) is not None for stem in RAW_STEMS)


def numbered_runs(folder):
    values = sorted((p for p in Path(folder).iterdir() if p.is_dir() and p.name.isdecimal()),
                    key=lambda p: int(p.name))
    if len({int(p.name) for p in values}) != len(values):
        raise ValueError(f"Duplicate numeric run labels in {folder}.")
    return values


def discover_cases(root, max_depth=4):
    root = Path(root).resolve()
    cases = []
    for directory, subdirs, files in os.walk(root):
        path = Path(directory)
        depth = len(path.relative_to(root).parts)
        subdirs[:] = [d for d in subdirs if d not in IGNORE_DIRS and not d.startswith(".")]
        runs = numbered_runs(path)
        if runs and any(raw_present(run) for run in runs):
            cases.append({"folder": path, "runs": runs})
            subdirs[:] = []
        elif raw_present(path) and not runs:
            cases.append({"folder": path, "runs": [path]})
            subdirs[:] = []
        elif depth >= max_depth:
            subdirs[:] = []
    for case in cases:
        relative = case["folder"].relative_to(root).as_posix()
        case["relative"] = relative if relative != "." else root.name
        case["id"] = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["relative"])
    return sorted(cases, key=lambda c: c["relative"].lower())


def read_outcar_header(path):
    metadata, titles, counts = {}, [], []
    if path is None:
        return metadata
    tags = ("POTIM", "NBLOCK", "NSW", "IBRION", "ISIF", "TEBEG", "TEEND", "MDALGO", "SMASS", "ML_OUTBLOCK")
    with open_text(path) as handle:
        stream = iter(handle)
        for index, line in enumerate(stream):
            if index > 8000 or "POSITION" in line and "TOTAL-FORCE" in line:
                break
            if "TITEL" in line and "=" in line:
                tokens = line.split("=", 1)[1].split()
                if len(tokens) > 1:
                    match = re.match(r"[A-Z][a-z]?", tokens[1])
                    if match:
                        titles.append(match.group())
            if "ions per type" in line and "=" in line:
                counts = [int(v) for v in line.split("=", 1)[1].split()]
            for tag in tags:
                match = re.search(r"\b" + tag + r"\s*=\s*(" + NUMBER + ")", line)
                if match:
                    metadata[tag] = number(match.group(1))
            if "direct lattice vectors" in line.lower():
                matrix = [[number(v) for v in next(stream).split()[:3]] for _ in range(3)]
                metadata["cell"] = np.asarray(matrix, float)
    if counts and len(titles) >= len(counts):
        metadata.update(species=titles[:len(counts)], counts=counts)
    return metadata


def run_metadata(run, case_folder, settings):
    incar = read_incar(Path(case_folder) / "INCAR")
    incar.update(read_incar(Path(run) / "INCAR"))
    outcar = existing_file(run, "OUTCAR")
    actual = read_outcar_header(outcar)
    values = {}
    for key, value in incar.items():
        try:
            values[key] = number(value.split()[0])
        except (ValueError, IndexError):
            values[key] = value
    values.update(actual)
    if "IBRION" in values and int(values["IBRION"]) != 0:
        raise ValueError(f"{run}: IBRION={values['IBRION']}; expected an MD run with IBRION=0.")
    dt = settings["dt_fs_override"]
    dt_source = "explicit override"
    if dt is None:
        dt = values.get("POTIM", settings["dt_fs_fallback"])
        dt_source = "POTIM" if "POTIM" in values else "configured fallback; verify"
    if dt is None or not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError(f"{run}: no valid POTIM; set dt_fs_override or dt_fs_fallback.")
    stride = settings["saved_frame_stride_override"]
    stride_source = "explicit saved-frame override"
    if stride is None:
        stride = int(values.get("NBLOCK", 1))
        stride_source = "NBLOCK" if "NBLOCK" in values else "VASP default NBLOCK=1; verify"
    if int(stride) != stride or int(stride) < 1:
        raise ValueError("The saved-frame stride must be a positive integer.")
    if values.get("ML_OUTBLOCK", 1) > 1 and settings["saved_frame_stride_override"] is None:
        raise ValueError(f"{run}: ML_OUTBLOCK changes output frequency; specify saved_frame_stride_override explicitly.")
    return {"dt_fs": float(dt), "dt_source": dt_source, "stride": int(stride),
            "stride_source": stride_source, "parameters": values}


def read_thermo_text(path):
    rows, duplicates = [], 0
    with open_text(path) as handle:
        for line in handle:
            match = STEP_T_RE.search(line)
            if match is None:
                continue
            row = {"step": int(match.group(1)), "temperature_K": number(match.group(2))}
            for key, pattern in FIELD_RES.items():
                value = pattern.search(line)
                row[key] = number(value.group(1)) if value else np.nan
            if not np.isfinite(row["E"]) and not np.isfinite(row["F"]):
                continue
            if rows and row["step"] <= rows[-1]["step"]:
                if row["step"] == rows[-1]["step"] and all(
                    np.isclose(row[key], rows[-1][key], equal_nan=True) for key in ("temperature_K", "E", "F", "E0")):
                    duplicates += 1
                    continue
                raise ValueError(f"{path}: MD steps reset/repeat inside one file. Separate concatenated jobs before processing.")
            rows.append(row)
    return rows, duplicates


def outcar_md_records(path, metadata, fallback_structure=None, issues=None):
    """Stream complete position blocks and the associated MD thermodynamics."""
    issues = issues if issues is not None else []
    params = metadata["parameters"]
    counts = params.get("counts", fallback_structure.get("counts") if fallback_structure else None)
    species = params.get("species", fallback_structure.get("species") if fallback_structure else None)
    cell = params.get("cell", fallback_structure.get("cell") if fallback_structure else None)
    if not counts or not species or cell is None:
        raise ValueError(f"{path}: missing structure header for OUTCAR trajectory fallback.")
    natoms = sum(counts)
    current, record = None, {}
    temp_re = re.compile(r"temperature\s*=?\s*(" + NUMBER + r")\s*K", re.I)
    etotal_re = re.compile(r"total\s+energy\s+ETOTAL\s*=\s*(" + NUMBER + ")", re.I)
    toten_re = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(" + NUMBER + ")", re.I)
    e0_re = re.compile(r"energy\s*\(\s*sigma\s*->\s*0\s*\)\s*=\s*(" + NUMBER + ")", re.I)

    def complete(rec):
        return (rec.get("positions") is not None
                and np.isfinite(rec.get("temperature_K", np.nan)))

    with open_text(path) as handle:
        stream = iter(handle)
        for line in stream:
            match = re.search(r"Iteration\s+(\d+)\s*\(\s*\d+\s*\)", line)
            if match:
                step = int(match.group(1))
                if current is not None and step != current:
                    if complete(record):
                        yield record
                    record = {}
                current = step
                record["step"] = step
            if "direct lattice vectors" in line.lower():
                vectors = np.asarray([[number(v) for v in next(stream).split()[:3]] for _ in range(3)])
                if not np.allclose(vectors, cell, atol=1e-5, rtol=1e-6):
                    raise ValueError("Variable-cell MD detected; this fixed-cell diffusion workflow does not apply.")
            if "POSITION" in line and "TOTAL-FORCE" in line:
                if record.get("positions") is not None:
                    if complete(record):
                        yield record
                    record = {"step": current}
                lines = []
                for body in stream:
                    if not body.strip() or set(body.strip()) <= {"-"}:
                        if not lines:
                            continue
                        break
                    values = body.split()
                    try:
                        xyz = [number(v) for v in values[:3]]
                    except ValueError:
                        break
                    if len(xyz) != 3:
                        break
                    lines.append(xyz)
                    if len(lines) == natoms:
                        break
                if len(lines) == natoms:
                    record["positions"] = np.asarray(lines, float)
                else:
                    issues.append("Incomplete final OUTCAR position block ignored.")
            for key, pattern in (("temperature_K", temp_re), ("E", etotal_re), ("F", toten_re), ("E0", e0_re)):
                match = pattern.search(line)
                if match:
                    record[key] = number(match.group(1))
        if complete(record):
            yield record


def read_thermodynamics(run, metadata, fallback_structure=None):
    candidates = []
    for name in ("OSZICAR", "stdout", "vasp.out", "vasp.stdout", "stdout.out"):
        path = existing_file(run, name)
        if path is not None and path not in candidates:
            candidates.append(path)
    if not candidates:
        candidates = sorted(Path(run).glob("slurm-*.out"))
        if len(candidates) > 1:
            raise ValueError(f"{run}: multiple slurm logs. Rename the intended MD log to stdout or supply OSZICAR.")
    for path in candidates:
        rows, duplicates = read_thermo_text(path)
        if rows:
            return rows, str(path), duplicates
    outcar = existing_file(run, "OUTCAR")
    if outcar is not None:
        rows = []
        for rec in outcar_md_records(outcar, metadata, fallback_structure):
            rows.append({key: rec.get(key, np.nan) for key in ("step", "temperature_K", "E", "F", "E0")})
        if rows:
            return rows, str(outcar), 0
    return [], "", 0


def xdat_header(handle, species_hint=None):
    title = handle.readline()
    if not title:
        raise ValueError("Empty XDATCAR.")
    scales = np.asarray([number(v) for v in handle.readline().split()])
    base = np.asarray([[number(v) for v in handle.readline().split()[:3]] for _ in range(3)])
    if base.shape != (3, 3) or abs(np.linalg.det(base)) < 1e-12:
        raise ValueError("Invalid XDATCAR lattice.")
    if len(scales) == 1:
        factor = scales[0]
        if factor < 0:
            factor = (abs(factor) / abs(np.linalg.det(base))) ** (1 / 3)
        scale_xyz = np.repeat(factor, 3)
    elif len(scales) == 3 and np.all(scales > 0):
        scale_xyz = scales
    else:
        raise ValueError("Unsupported XDATCAR scale factors.")
    tokens = handle.readline().split()
    if tokens and all(v.isdigit() for v in tokens):
        counts = [int(v) for v in tokens]
        species = list(species_hint or [])
    else:
        species = tokens
        counts = [int(v) for v in handle.readline().split()]
    if len(species) != len(counts):
        raise ValueError("XDATCAR element names are missing or inconsistent.")
    return {"title": title.strip(), "cell": base * scale_xyz[None, :],
            "scale_xyz": scale_xyz, "species": species, "counts": counts,
            "symbols": np.repeat(species, counts)}


def iter_xdatcar(path, issues=None, species_hint=None):
    issues = issues if issues is not None else []
    with open_text(path) as handle:
        header = xdat_header(handle, species_hint)
        natoms, previous = sum(header["counts"]), None
        for line in handle:
            if not line.strip():
                continue
            match = CFG_RE.match(line)
            if not match:
                raise ValueError(f"{path}: unexpected content between frames; check repeated headers or variable-cell output.")
            index = int(match.group(2))
            if previous is not None and index <= previous:
                raise ValueError(f"{path}: configuration labels reset or repeat inside one run.")
            body = [handle.readline() for _ in range(natoms)]
            if any(not item for item in body):
                issues.append("Incomplete final XDATCAR frame ignored.")
                return
            text = " ".join(body).replace("D", "E").replace("d", "e")
            values = np.fromstring(text, sep=" ")
            if values.size != natoms * 3 or not np.isfinite(values).all():
                raise ValueError(f"{path}: malformed coordinates in configuration {index}.")
            coordinates = values.reshape(natoms, 3)
            direct = match.group(1).lower() == "direct"
            positions = coordinates @ header["cell"] if direct else coordinates * header["scale_xyz"]
            fractional = coordinates if direct else positions @ np.linalg.inv(header["cell"])
            yield {"index": index, "positions": positions, "fractional": fractional,
                   "header": header, "raw_direct_lines": body if direct else None}
            previous = index


def first_structure(case):
    for run in case["runs"]:
        meta = read_outcar_header(existing_file(run, "OUTCAR"))
        for name in ("POSCAR", "CONTCAR"):
            path = existing_file(run, name)
            if path:
                try:
                    return read_poscar(path, meta.get("species"))
                except (ValueError, IndexError):
                    pass
        path = existing_file(run, "XDATCAR")
        if path:
            try:
                frame = next(iter_xdatcar(path, species_hint=meta.get("species")))
                return dict(frame["header"], positions=frame["positions"], source=str(path))
            except (ValueError, StopIteration):
                pass
    raise ValueError(f"No readable POSCAR, CONTCAR, or complete XDATCAR frame in {case['folder']}.")


def identify_system(case, reference, settings):
    symbols = reference["symbols"]
    candidates = sorted(set(symbols) & {"Au", "Ag", "Cu", "Li"})
    if len(candidates) != 1:
        raise ValueError(f"{case['folder']}: expected one metal species among Au/Ag/Cu/Li; found {candidates}.")
    case["metal"] = str(candidates[0])
    requested = lookup(settings["metal_atom_index_1based"], case)
    indices = np.flatnonzero(symbols == case["metal"])
    if requested is None and len(indices) != 1:
        raise ValueError(f"{case['folder']}: multiple metal atoms. Set metal_atom_index_1based for this case.")
    selected = int(requested) - 1 if requested is not None else int(indices[0])
    if not 0 <= selected < len(symbols) or symbols[selected] != case["metal"]:
        raise ValueError("The selected 1-based atom index does not match the detected metal.")
    counts = {str(s): int(np.sum(symbols == s)) for s in set(symbols)}
    environment = "unclassified"
    neighbors = lookup(settings["vacancy_neighbors_1based"], case)
    if neighbors is not None:
        neighbors = np.asarray(neighbors, int) - 1
        if len(neighbors) != 3 or np.any(neighbors < 0) or np.any(neighbors >= len(symbols)):
            raise ValueError("vacancy_neighbors_1based must contain three valid atom numbers.")
        environment = "user_defined_vacancy"
    elif "B" in counts and "N" in counts:
        if counts["B"] == counts["N"]:
            environment = "pristine_stoichiometry"
        elif abs(counts["B"] - counts["N"]) == 1:
            environment = "B_vacancy_candidate" if counts["B"] < counts["N"] else "N_vacancy_candidate"
            adjacent = "N" if counts["B"] < counts["N"] else "B"
            opposite = "B" if adjacent == "N" else "N"
            atoms = np.flatnonzero(symbols == adjacent)
            other = np.flatnonzero(symbols == opposite)
            disp = reference["positions"][atoms, None, :] - reference["positions"][other][None, :, :]
            distances = np.linalg.norm(minimum_image(disp, reference["cell"]), axis=2)
            coordination = np.sum(distances < settings["bond_cutoff_A"], axis=1)
            undercoordinated = atoms[coordination == 2]
            if len(undercoordinated) == 3:
                tri = reference["positions"][undercoordinated].copy()
                tri[1:] = tri[0] + minimum_image(tri[1:] - tri[0], reference["cell"])
                if np.max(np.linalg.norm(tri[:, None] - tri[None, :], axis=2)) < 4.0:
                    neighbors = undercoordinated
    host = np.asarray([s in HOST_ELEMENTS for s in symbols])
    if not np.any(host):
        raise ValueError("No supported substrate atoms were identified.")
    case.update(metal_index0=selected, environment=environment)
    return {"metal": case["metal"], "metal_index0": selected, "environment": environment,
            "composition": counts, "host_mask": host, "vacancy_neighbors0": neighbors}


def velocities_from_poscar(path):
    """Read a conventional ionic velocity block when one is present."""
    if path is None:
        return None
    with open_text(path) as handle:
        lines = handle.readlines()
    try:
        tokens = lines[5].split()
        old = all(v.isdigit() for v in tokens)
        counts = [int(v) for v in (tokens if old else lines[6].split())]
        cursor = 6 if old else 7
        if lines[cursor].strip().lower().startswith("s"):
            cursor += 1
        cursor += 1 + sum(counts)
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and lines[cursor].strip().lower().startswith(("c", "d")):
            cursor += 1
        values = np.asarray([[number(v) for v in line.split()[:3]] for line in lines[cursor:cursor + sum(counts)]])
        return values if values.shape == (sum(counts), 3) else None
    except (IndexError, ValueError):
        return None


def audit_restart(previous_run, following_run, reference, settings):
    if previous_run is None:
        return False, "first available segment", None, False
    if settings["restart_policy"] == "split":
        return False, "split at every restart by configuration", None, False
    old = existing_file(previous_run, "CONTCAR")
    new = existing_file(following_run, "POSCAR")
    if old is None or new is None:
        return False, "CONTCAR/POSCAR unavailable: continuity not established", None, False
    try:
        a, b = read_poscar(old, reference["species"]), read_poscar(new, reference["species"])
        if not np.array_equal(a["symbols"], b["symbols"]) or not np.allclose(a["cell"], b["cell"], atol=1e-5):
            return False, "restart cell or element order changed", None, False
        deviation = float(np.max(np.linalg.norm(minimum_image(a["positions"] - b["positions"], a["cell"]), axis=1)))
        va, vb = velocities_from_poscar(old), velocities_from_poscar(new)
        velocity_match = va is not None and vb is not None and np.allclose(va, vb, atol=1e-8, rtol=1e-6)
        if deviation > settings["restart_tolerance_A"]:
            return False, "restart positions disagree", deviation, velocity_match
        if settings["restart_policy"] == "strict" and not velocity_match:
            return False, "positions agree; velocity continuation unverified", deviation, False
        return True, "positions and velocities agree" if velocity_match else "positions agree; velocities unverified", deviation, velocity_match
    except (ValueError, IndexError) as exc:
        return False, f"restart comparison failed: {exc}", None, False


def input_signature(case, settings):
    files = []
    for run in case["runs"]:
        for stem in RAW_STEMS + ("POSCAR", "CONTCAR", "INCAR", "vasp.stdout", "stdout.out"):
            path = existing_file(run, stem)
            if path:
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        for path in sorted(Path(run).glob("slurm-*.out")):
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    parent = Path(case["folder"]) / "INCAR"
    if parent.is_file():
        stat = parent.stat()
        files.append((str(parent), stat.st_size, stat.st_mtime_ns))
    raw_keys = ("dt_fs_override", "dt_fs_fallback", "saved_frame_stride_override",
                "metal_atom_index_1based", "vacancy_neighbors_1based", "bond_cutoff_A",
                "restart_policy", "restart_tolerance_A", "max_adjacent_metal_jump_A",
                "write_xyz", "write_combined_xdatcar", "movie_stride")
    payload = {"version": VERSION, "files": files, "settings": {key: settings[key] for key in raw_keys}}
    return hashlib.sha256(json.dumps(safe_json(payload), sort_keys=True).encode()).hexdigest()[:18]


def load_case_cache(directory):
    with (directory / "cache_manifest.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(directory / "trajectory_cache.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    return {"data": arrays, "metadata": metadata, "cache_dir": directory}


def prepare_case(case, output_root, settings):
    signature = input_signature(case, settings)
    cache_parent = Path(output_root) / "raw_cache" / case["id"]
    cache_parent.mkdir(parents=True, exist_ok=True)
    if not settings["force_reextract"]:
        candidates = sorted(cache_parent.glob(signature + "*"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for candidate in candidates:
            if (candidate / "cache_manifest.json").is_file() and (candidate / "trajectory_cache.npz").is_file():
                try:
                    cached = load_case_cache(candidate)
                    case["metal"] = cached["metadata"]["metal"]
                    print(f"{case['relative']}: using saved raw extraction")
                    return cached
                except (ValueError, OSError, KeyError):
                    pass
    cache_dir = cache_parent / signature
    if cache_dir.exists():
        cache_dir = cache_parent / (signature + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    cache_dir.mkdir()
    per_run_dir = cache_dir / "runs"
    per_run_dir.mkdir()
    reference = first_structure(case)
    system = identify_system(case, reference, settings)
    symbols, cell = reference["symbols"], reference["cell"]
    host, metal_index = system["host_mask"], system["metal_index0"]
    neighbors = system["vacancy_neighbors0"]
    masses = np.asarray([MASSES.get(str(s), np.nan) for s in symbols])
    if not np.isfinite(masses).all():
        raise ValueError("Unknown substrate atomic mass.")
    e1 = cell[0] / np.linalg.norm(cell[0])
    normal0 = np.cross(cell[0], cell[1]); normal0 /= np.linalg.norm(normal0)
    e2 = np.cross(normal0, e1)
    basis = np.asarray([e1, e2, normal0])
    trace, thermo_trace, audits, warnings = [], [], [], []
    time_offset_ps, total_steps = 0.0, 0
    previous_wrapped, previous_unwrapped, previous_valid_run = None, None, None
    previous_time, previous_dt, previous_duration_known = None, None, True
    segment, global_frame = -1, 0
    empty_numbers = set()
    issues = []
    thermo_fields = ["run", "source_file", "raw_step", "local_step", "global_step", "time_fs", "time_ps",
                     "temperature_K", "energy_eV", "E_total_eV", "F_TOTEN_eV", "E0_eV"]
    trace_fields = ["global_frame", "run_index", "local_frame", "local_step", "time_ps", "segment_id",
                    "x_relative_A", "y_relative_A", "z_relative_A", "height_A", "rho_vacancy_A"]
    print(f"{case['relative']}: {system['metal']}, {len(case['runs'])} run folder(s), {system['environment']}")
    if "vacancy" in system["environment"] and neighbors is None:
        warnings.append("A vacancy is suggested by composition, but its three neighbors were not identified. Set vacancy_neighbors_1based if needed; automatic capture is unavailable.")
    start_clock = time.monotonic()
    with contextlib.ExitStack() as stack:
        thermo_handle = stack.enter_context((cache_dir / "combined-temp-energy.csv").open("w", newline="", encoding="utf-8"))
        thermo_writer = csv.DictWriter(thermo_handle, fieldnames=thermo_fields); thermo_writer.writeheader()
        trace_handle = stack.enter_context((cache_dir / "trajectory.csv").open("w", newline="", encoding="utf-8"))
        trace_writer = csv.DictWriter(trace_handle, fieldnames=trace_fields); trace_writer.writeheader()
        xyz_handle = stack.enter_context((cache_dir / "movie-combined.xyz").open("w", encoding="utf-8")) if settings["write_xyz"] else None
        xdat_handle = stack.enter_context((cache_dir / "XDATCAR_combined").open("w", encoding="utf-8")) if settings["write_combined_xdatcar"] else None
        if xdat_handle:
            xdat_handle.write(f"Combined AIMD: {case['id']}\n1.0\n")
            np.savetxt(xdat_handle, cell, fmt="%.12f")
            xdat_handle.write(" ".join(reference["species"]) + "\n")
            xdat_handle.write(" ".join(map(str, reference["counts"])) + "\n")
        lattice_string = " ".join(f"{v:.10f}" for v in cell.ravel())
        for run_index, run in enumerate(case["runs"], 1):
            metadata = run_metadata(run, case["folder"], settings)
            dt_fs, stride = metadata["dt_fs"], metadata["stride"]
            thermodynamics, thermo_source, duplicates = read_thermodynamics(run, metadata, reference)
            base_step = thermodynamics[0]["step"] - 1 if thermodynamics else 0
            md_steps = thermodynamics[-1]["step"] - base_step if thermodynamics else 0
            if thermodynamics and thermodynamics[0]["step"] != 1:
                warnings.append(f"Run {run.name}: thermodynamic records start at step {thermodynamics[0]['step']}; earlier elapsed time is not inferred.")
            part_rows = []
            for row in thermodynamics:
                local_step = row["step"] - base_step
                t = time_offset_ps + local_step * dt_fs / 1000
                output_row = {"run": run.name, "source_file": thermo_source, "raw_step": row["step"],
                              "local_step": local_step, "global_step": total_steps + local_step,
                              "time_fs": f"{t * 1000:.6f}", "time_ps": f"{t:.9f}",
                              "temperature_K": row["temperature_K"], "energy_eV": row["E"],
                              "E_total_eV": row["E"], "F_TOTEN_eV": row["F"], "E0_eV": row["E0"]}
                thermo_writer.writerow({key: safe_json(v) for key, v in output_row.items()})
                part_rows.append(output_row)
                thermo_trace.append((t, row["temperature_K"], row["E"], row["F"], row["E0"]))
            save_csv(per_run_dir / f"file-{run.name}.csv", part_rows, thermo_fields)
            join, reason, deviation, velocity_match = audit_restart(previous_valid_run, run, reference, settings)
            if previous_valid_run is not None and run.name.isdecimal() and previous_valid_run.name.isdecimal():
                absent = set(range(int(previous_valid_run.name) + 1, int(run.name))) - empty_numbers
                if absent:
                    join, reason = False, "missing run numbers with unknown elapsed time: " + ",".join(map(str, sorted(absent)))
                    warnings.append(f"Before run {run.name}: {reason}. Global time describes observed runs only.")
            if not previous_duration_known:
                join, reason = False, "previous run duration not independently established by MD summaries"
            if join and not velocity_match:
                warnings.append(f"Before run {run.name}: positions match, but velocity/thermostat-state continuation is not certified. Use restart_policy='strict' or 'split' for a conservative check.")
            run_issues = []
            xdatcar = existing_file(run, "XDATCAR")
            if xdatcar and xdatcar.stat().st_size:
                iterator = iter_xdatcar(xdatcar, run_issues, reference["species"])
                source = str(xdatcar)
            elif xdatcar:
                iterator, source = iter(()), str(xdatcar)
            elif existing_file(run, "OUTCAR"):
                source = str(existing_file(run, "OUTCAR"))
                def outcar_frames():
                    for record in outcar_md_records(Path(source), metadata, reference, run_issues):
                        positions = record["positions"]
                        yield {"index": int(record["step"]), "positions": positions,
                               "fractional": positions @ np.linalg.inv(cell),
                               "header": {"cell": cell, "symbols": symbols}, "raw_direct_lines": None}
                iterator = outcar_frames()
            else:
                iterator, source = iter(()), ""
            head = []
            for _ in range(2):
                try:
                    head.append(next(iterator))
                except StopIteration:
                    break
            if not head and xdatcar is not None and existing_file(run, "OUTCAR") is not None:
                # Some interrupted jobs leave an empty/header-only XDATCAR while
                # complete MD records remain in OUTCAR.
                source = str(existing_file(run, "OUTCAR"))
                def fallback_outcar_frames():
                    for record in outcar_md_records(Path(source), metadata, reference, run_issues):
                        positions = record["positions"]
                        yield {"index": int(record["step"]), "positions": positions,
                               "fractional": positions @ np.linalg.inv(cell),
                               "header": {"cell": cell, "symbols": symbols}, "raw_direct_lines": None}
                iterator = fallback_outcar_frames()
                for _ in range(2):
                    try:
                        head.append(next(iterator))
                    except StopIteration:
                        break
                if head:
                    run_issues.append("XDATCAR had no complete frames; complete OUTCAR MD frames were used.")
                xdatcar = None
            mapping = "no coordinates"
            if head:
                first_label = head[0]["index"]
                label_step = head[1]["index"] - first_label if len(head) > 1 else None
                use_md_labels = (xdatcar is None or
                    (stride == 1 and first_label <= (thermodynamics[-1]["step"] if thermodynamics else first_label)) or
                    (stride > 1 and label_step == stride and first_label >= stride))
                mapping = "configuration labels as MD steps" if use_md_labels else "configuration ordinals times saved-frame stride"
                coordinate_stride = stride if xdatcar is not None else max(1, label_step or 1)
                def local_step_for(frame):
                    if use_md_labels:
                        return frame["index"] - base_step
                    ordinal = frame["index"] - first_label + (0 if first_label == 0 else 1)
                    return ordinal * stride
            else:
                local_step_for = lambda frame: 0
                coordinate_stride = stride
            count, last_coordinate_step, first_coordinate_time = 0, 0, None
            for local_frame, frame in enumerate(chain(head, iterator), 1):
                if (not np.array_equal(frame["header"]["symbols"], symbols)
                        or not np.allclose(frame["header"]["cell"], cell, rtol=1e-6, atol=1e-5)):
                    raise ValueError(f"{run}: cell or element order changed. Do not combine structurally different systems.")
                local_step = local_step_for(frame)
                if local_step < 0:
                    raise ValueError(f"{run}: configuration labels precede the local MD time origin; inspect the raw files.")
                t = time_offset_ps + local_step * dt_fs / 1000
                saved_dt = dt_fs * coordinate_stride / 1000
                wrapped = frame["positions"]
                displacement = None if previous_wrapped is None else minimum_image(wrapped - previous_wrapped, cell)
                temporal_duplicate = (previous_time is not None and np.isclose(t, previous_time, atol=1e-12, rtol=0))
                if temporal_duplicate:
                    if displacement is not None and np.max(np.linalg.norm(displacement, axis=1)) < 1e-8:
                        run_issues.append("Identical coordinate record at the same physical time was not counted twice.")
                        continue
                    if local_frame != 1 or join:
                        raise ValueError(f"{run}: different coordinate frames share the same physical time.")
                if previous_time is not None and t < previous_time:
                    raise ValueError(f"{run}: reconstructed coordinate time moves backwards.")
                new_segment = previous_time is None or count == 0 and not join
                if previous_time is not None and not np.isclose(t - previous_time, saved_dt, atol=1e-10, rtol=1e-6):
                    new_segment = True
                if previous_dt is not None and not np.isclose(saved_dt, previous_dt, atol=1e-10, rtol=1e-6):
                    new_segment = True
                if displacement is not None and np.linalg.norm(displacement[metal_index]) > settings["max_adjacent_metal_jump_A"]:
                    new_segment = True
                    run_issues.append(f"Large adjacent-frame metal displacement before local frame {local_frame}; MSD pairs are split here.")
                if new_segment:
                    segment += 1
                unwrapped = wrapped.copy() if previous_unwrapped is None else previous_unwrapped + displacement
                center = np.average(unwrapped[host], axis=0, weights=masses[host])
                relative = (unwrapped[metal_index] - center) @ basis.T
                height, rho = float(relative[2]), np.nan
                if neighbors is not None:
                    triangle = unwrapped[neighbors].copy()
                    triangle[1:] = triangle[0] + minimum_image(triangle[1:] - triangle[0], cell)
                    local_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                    if np.linalg.norm(local_normal) > 1e-8:
                        local_normal /= np.linalg.norm(local_normal)
                        if np.dot(local_normal, normal0) < 0:
                            local_normal *= -1
                        delta = minimum_image(unwrapped[metal_index] - triangle.mean(axis=0), cell)
                        height = float(np.dot(delta, local_normal))
                        rho = float(np.sqrt(max(0.0, np.dot(delta, delta) - height ** 2)))
                global_frame += 1; count += 1
                if first_coordinate_time is None:
                    first_coordinate_time = t
                trace.append((t, relative[0], relative[1], relative[2], height, rho,
                              segment, run_index, local_step, local_frame))
                trace_writer.writerow(dict(zip(trace_fields, [global_frame, run_index, local_frame, local_step,
                    f"{t:.9f}", segment, relative[0], relative[1], relative[2], height,
                    rho if np.isfinite(rho) else ""])))
                if xyz_handle and (global_frame - 1) % settings["movie_stride"] == 0:
                    xyz_handle.write(f'{len(symbols)}\nLattice="{lattice_string}" Properties=species:S:1:pos:R:3 Time_ps={t:.9f} run={run.name} frame={local_frame} pbc="T T F"\n')
                    xyz_handle.writelines(f"{str(symbol):2s} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}\n" for symbol, xyz in zip(symbols, wrapped))
                if xdat_handle:
                    xdat_handle.write(f"Direct configuration= {global_frame}\n")
                    if frame["raw_direct_lines"] is not None:
                        xdat_handle.writelines(frame["raw_direct_lines"])
                    else:
                        np.savetxt(xdat_handle, frame["fractional"], fmt="%.10f")
                previous_wrapped, previous_unwrapped = wrapped.copy(), unwrapped
                previous_time, previous_dt = t, saved_dt
                last_coordinate_step = max(last_coordinate_step, local_step)
                every = settings["progress_every_frames"]
                if every and global_frame % every == 0:
                    print(f"  {case['id']}: {global_frame:,} saved frames, t={t:.3f} ps, elapsed {time.monotonic()-start_clock:.1f} s", flush=True)
            duration_steps = max(int(md_steps), int(last_coordinate_step))
            duration_ps = duration_steps * dt_fs / 1000
            duration_known = bool(thermodynamics and thermodynamics[0]["step"] == 1 and md_steps >= last_coordinate_step)
            if count and md_steps:
                expected = md_steps / coordinate_stride
                if abs(count - expected) > 1.01:
                    run_issues.append(f"Coordinate count {count} differs from MD steps/output stride ({expected:g}). No time rescaling was applied; inspect missing/truncated output and stride metadata.")
            if not count and md_steps:
                run_issues.append("Completed MD steps exist without saved coordinates; their elapsed time is retained as a gap.")
            if not count and not md_steps and run.name.isdecimal():
                empty_numbers.add(int(run.name))
            audit = {"run": run.name, "folder": str(run), "coordinate_source": source,
                     "thermo_source": thermo_source, "coordinate_frames": count,
                     "thermo_rows": len(thermodynamics), "completed_steps_observed": md_steps,
                     "run_duration_ps": duration_ps, "time_offset_ps": time_offset_ps,
                     "dt_fs": dt_fs, "dt_source": metadata["dt_source"], "saved_stride": coordinate_stride,
                     "stride_source": metadata["stride_source"], "coordinate_step_mapping": mapping,
                     "first_coordinate_time_ps": first_coordinate_time,
                     "last_coordinate_time_ps": previous_time if count else None,
                     "restart_join_requested": join, "restart_reason": reason,
                     "restart_max_position_difference_A": deviation, "restart_velocities_match": velocity_match,
                     "duration_confirmed_by_md_summaries": duration_known,
                     "duplicate_thermo_lines_ignored": duplicates, "notes": " | ".join(dict.fromkeys(run_issues))}
            audits.append(audit)
            warnings.extend(f"Run {run.name}: {message}" for message in dict.fromkeys(run_issues))
            if count:
                previous_valid_run = run
                previous_duration_known = duration_known
            elif duration_steps:
                previous_duration_known = False
            total_steps += duration_steps
            time_offset_ps += duration_ps
            print(f"  run {run.name}: {count:,} coordinate frames; {len(thermodynamics):,} thermo rows; {duration_ps:.4f} ps")
    if not trace:
        raise ValueError(f"{case['folder']}: no complete trajectory frames were found.")
    trace = np.asarray(trace, float)
    thermo_trace = np.asarray(thermo_trace, float).reshape(-1, 5)
    data = {"time_ps": trace[:, 0], "positions_A": trace[:, 1:4], "height_A": trace[:, 4],
            "rho_vacancy_A": trace[:, 5], "segment_ids": trace[:, 6].astype(int),
            "run_index": trace[:, 7].astype(int), "local_step": trace[:, 8].astype(int),
            "local_frame": trace[:, 9].astype(int), "thermo_time_ps": thermo_trace[:, 0],
            "temperature_K": thermo_trace[:, 1], "thermo_E": thermo_trace[:, 2],
            "thermo_F": thermo_trace[:, 3], "thermo_E0": thermo_trace[:, 4],
            "reference_positions_A": reference["positions"], "cell_A": cell, "symbols": symbols}
    manifest = {"version": VERSION, "signature": signature, "case_id": case["id"],
                "case_folder": str(case["folder"]), "relative": case["relative"],
                "metal": system["metal"], "metal_atom_index_1based": metal_index + 1,
                "environment": system["environment"], "composition": system["composition"],
                "vacancy_neighbors_1based": (neighbors + 1).tolist() if neighbors is not None else None,
                "coordinate_frames": len(trace), "thermo_rows": len(thermo_trace),
                "first_time_ps": trace[0, 0], "last_time_ps": trace[-1, 0],
                "sampled_span_ps": trace[-1, 0] - trace[0, 0],
                "number_of_continuous_segments": int(trace[-1, 6]) + 1,
                "observed_run_duration_ps": time_offset_ps, "run_audit": audits,
                "warnings": list(dict.fromkeys(warnings)),
                "csv_energy_eV_quantity": "E: VASP extended-system total energy",
                "combined_xdatcar_numbering": "Saved-frame ordinal; use trajectory.csv for physical MD time.",
                "source_settings": {key: settings[key] for key in ("restart_policy", "dt_fs_override", "saved_frame_stride_override", "write_xyz", "write_combined_xdatcar", "movie_stride")}}
    np.savez_compressed(cache_dir / "trajectory_cache.npz", **data)
    save_csv(cache_dir / "run_audit.csv", audits)
    save_json(cache_dir / "cache_manifest.json", manifest)
    return {"data": data, "metadata": manifest, "cache_dir": cache_dir}


def first_persistent_region(data, settings):
    time_ps, segments = data["time_ps"], data["segment_ids"]
    inside = ((data["rho_vacancy_A"] <= settings["capture_radius_A"])
              & (np.abs(data["height_A"]) <= settings["capture_height_A"]))
    if settings["analysis_end_ps"] is not None:
        inside &= time_ps <= settings["analysis_end_ps"]
    start = None
    capture = None
    for i, flag in enumerate(inside):
        if not flag or i and segments[i] != segments[i - 1]:
            start = None
        if flag:
            start = i if start is None else start
            if time_ps[i] - time_ps[start] >= settings["capture_dwell_ps"] - 1e-12:
                capture = {"time_ps": float(time_ps[start]), "index": int(start),
                           "confirmed_at_ps": float(time_ps[i]), "segment_id": int(segments[start])}
                break
    if capture is None:
        return None
    end = np.flatnonzero(segments == capture["segment_id"])[-1]
    outside_start, escape_index = None, None
    for j in range(capture["index"] + 1, end + 1):
        if settings["analysis_end_ps"] is not None and time_ps[j] > settings["analysis_end_ps"]:
            end = j - 1
            break
        if not inside[j]:
            outside_start = j if outside_start is None else outside_start
            if time_ps[j] - time_ps[outside_start] >= settings["capture_dwell_ps"] - 1e-12:
                escape_index = outside_start
                break
        else:
            outside_start = None
    stop = end if escape_index is None else escape_index
    capture.update(observed_residence_ps=float(time_ps[stop] - time_ps[capture["index"]]),
                   observed_residence_end_ps=float(time_ps[stop]),
                   residence_right_censored=escape_index is None,
                   geometric_escape_time_ps=float(time_ps[escape_index]) if escape_index is not None else None)
    return capture


def fft_msd_sums(positions, max_lag):
    """Exact time-origin sums for one uniformly sampled continuous segment."""
    positions = np.asarray(positions, float)
    n = len(positions)
    max_lag = min(int(max_lag), n - 1)
    centered = positions - np.mean(positions, axis=0)
    length = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=length, axis=0)
    autocorrelation = np.fft.irfft(transformed.conj() * transformed, n=length, axis=0)[:max_lag + 1]
    squared = centered ** 2
    cumulative = np.vstack([np.zeros((1, squared.shape[1])), np.cumsum(squared, axis=0)])
    lag = np.arange(max_lag + 1)
    sums = cumulative[n] - cumulative[lag] + cumulative[n - lag] - 2 * autocorrelation
    sums = np.maximum(sums, 0.0)
    sums[0] = 0.0
    return sums, n - lag


def window_segments(data, start_ps, end_ps, settings):
    time_ps = data["time_ps"]
    mask = time_ps >= start_ps
    if end_ps is not None:
        mask &= time_ps <= end_ps
    raw_segments = []
    for identifier in np.unique(data["segment_ids"][mask]):
        indices = np.flatnonzero(mask & (data["segment_ids"] == identifier))
        if len(indices) >= 3:
            dt = float(np.median(np.diff(time_ps[indices])))
            if dt > 0 and np.allclose(np.diff(time_ps[indices]), dt, atol=1e-9, rtol=1e-5):
                raw_segments.append((indices, dt))
    if not raw_segments:
        return [], None, mask
    target_dt = max(dt for _, dt in raw_segments) * int(settings["analysis_stride"])
    result = []
    for indices, dt in raw_segments:
        step = int(round(target_dt / dt))
        if step < 1 or not np.isclose(step * dt, target_dt, atol=1e-9, rtol=1e-5):
            raise ValueError("Saved-frame intervals are not commensurate within this case. Analyze the different timestep series separately.")
        chosen = indices[::step]
        if len(chosen) >= 3:
            result.append({"time_ps": time_ps[chosen], "xy_A": data["positions_A"][chosen, :2]})
    return result, target_dt, mask


def pool_msd(segments, dt, settings):
    if not segments or dt is None:
        return None
    maximum = max(len(segment["xy_A"]) for segment in segments)
    max_lag = min(int(settings["msd_max_lag_ps"] / dt),
                  int((maximum - 1) * settings["msd_max_fraction"]))
    if max_lag < 2:
        return None
    sums, pairs = np.zeros((max_lag + 1, 2)), np.zeros(max_lag + 1, np.int64)
    for segment in segments:
        n = len(segment["xy_A"])
        limit = min(max_lag, int((n - 1) * settings["msd_max_fraction"]))
        if limit < 1:
            continue
        local_sum, local_pairs = fft_msd_sums(segment["xy_A"], limit)
        sums[:limit + 1] += local_sum
        pairs[:limit + 1] += local_pairs
    valid = pairs > 0
    lags = np.arange(max_lag + 1)[valid] * dt
    components = sums[valid] / pairs[valid, None]
    return {"lag_ps": lags, "msd_xy_A2": components.sum(axis=1),
            "msd_x_A2": components[:, 0], "msd_y_A2": components[:, 1],
            "time_origin_pairs": pairs[valid]}


def fit_msd(curve, fit_lag_ps):
    start, stop = map(float, fit_lag_ps)
    empty = {"fit_status": "insufficient_lag_coverage", "D_apparent_A2_per_ps": None,
             "D_apparent_m2_per_s": None, "slope_A2_per_ps": None, "intercept_A2": None,
             "R2": None, "alpha_fit": None, "linear_screen_pass": False,
             "requested_fit_start_ps": start, "requested_fit_end_ps": stop,
             "actual_fit_start_ps": None, "actual_fit_end_ps": None}
    if curve is None or len(curve["lag_ps"]) < 5:
        return empty
    t, y = curve["lag_ps"], curve["msd_xy_A2"]
    dt = float(np.median(np.diff(t)))
    if t[-1] < stop - dt * 0.05:
        return empty
    mask = (t >= start - 1e-10) & (t <= stop + 1e-10) & (t > 0)
    if np.sum(mask) < 5:
        return empty
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    predicted = slope * t[mask] + intercept
    variance = np.sum((y[mask] - np.mean(y[mask])) ** 2)
    r2 = 1 - float(np.sum((y[mask] - predicted) ** 2) / variance) if variance > 1e-24 else None
    positive = mask & (y > 1e-20)
    alpha = float(np.polyfit(np.log(t[positive]), np.log(y[positive]), 1)[0]) if np.sum(positive) >= 5 else None
    passing = slope > 0 and r2 is not None and r2 >= 0.98 and alpha is not None and 0.8 <= alpha <= 1.2
    empty.update(fit_status="candidate_linear_regime_review_required" if passing else "apparent_slope_only_not_diffusive_by_screen",
                 D_apparent_A2_per_ps=float(slope / 4), D_apparent_m2_per_s=float(slope / 4 * 1e-8),
                 slope_A2_per_ps=float(slope), intercept_A2=float(intercept), R2=r2, alpha_fit=alpha,
                 linear_screen_pass=bool(passing), actual_fit_start_ps=float(t[mask][0]), actual_fit_end_ps=float(t[mask][-1]))
    return empty


def block_fits(segments, dt, settings):
    duration = max(settings["block_min_duration_ps"], 5 * settings["fit_lag_ps"][1])
    count = max(10, int(math.ceil(duration / dt))) if dt else 0
    values = []
    if not count:
        return values
    for segment_index, segment in enumerate(segments):
        for start in range(0, len(segment["xy_A"]) - count + 1, count):
            block = {key: value[start:start + count] for key, value in segment.items()}
            curve = pool_msd([block], dt, settings)
            fit = fit_msd(curve, settings["fit_lag_ps"])
            if fit["D_apparent_A2_per_ps"] is not None:
                values.append({"segment": segment_index, "start_time_ps": float(block["time_ps"][0]),
                               "end_time_ps": float(block["time_ps"][-1]), **fit})
    return values


def scaling_curve(curve):
    if curve is None:
        return np.array([]), np.array([])
    t, y = curve["lag_ps"], curve["msd_xy_A2"]
    good = (t > 0) & (y > 1e-20)
    t, y = t[good], y[good]
    if len(t) < 8:
        return np.array([]), np.array([])
    centers = np.unique(np.geomspace(2, len(t) - 3, min(80, len(t) - 4)).astype(int))
    lags, exponents = [], []
    for center in centers:
        radius = max(2, int(center * 0.10))
        selection = slice(max(0, center - radius), min(len(t), center + radius + 1))
        lags.append(t[center])
        exponents.append(np.polyfit(np.log(t[selection]), np.log(y[selection]), 1)[0])
    return np.asarray(lags), np.asarray(exponents)


def analyze_case(case, extracted, report_dir, settings):
    data, metadata = extracted["data"], extracted["metadata"]
    case["metal"] = metadata["metal"]
    report_dir.mkdir(parents=True, exist_ok=True)
    capture = first_persistent_region(data, settings) if metadata.get("vacancy_neighbors_1based") else None
    post = lookup(settings["post_start_ps"], case)
    end_ps = settings["analysis_end_ps"]
    windows = [("whole", 0.0, end_ps), ("equilibrated", float(settings["equilibration_ps"]), end_ps)]
    if post is not None:
        windows.append(("post_selected", float(post), end_ps))
    elif capture is not None:
        # Keep this first-residence window free of any later persistent escape.
        # The whole window still includes all observed migration and escape.
        residence_end = capture["observed_residence_end_ps"]
        if not capture["residence_right_censored"]:
            residence_end = np.nextafter(residence_end, -np.inf)
        windows.append(("post_capture", max(capture["confirmed_at_ps"],
                       capture["time_ps"] + settings["post_capture_buffer_ps"]), residence_end))
    summaries, curves = [], {}
    for name, start, stop in windows:
        segments, dt, mask = window_segments(data, start, stop, settings)
        curve = pool_msd(segments, dt, settings)
        fit = fit_msd(curve, settings["fit_lag_ps"])
        blocks = block_fits(segments, dt, settings)
        block_values = [row["D_apparent_A2_per_ps"] for row in blocks]
        nframes = int(np.sum(mask))
        longest = max(segments, key=lambda s: len(s["xy_A"])) if segments else None
        radius = float(np.sqrt(np.mean(np.sum((longest["xy_A"] - np.mean(longest["xy_A"], axis=0)) ** 2, axis=1)))) if longest else None
        summary = {"case": case["id"], "metal": metadata["metal"], "environment": metadata["environment"],
                   "window": name, "selected_start_ps": start, "selected_end_ps": stop,
                   "first_saved_time_ps": float(data["time_ps"][mask][0]) if nframes else None,
                   "last_saved_time_ps": float(data["time_ps"][mask][-1]) if nframes else None,
                   "frames_in_window": nframes, "continuous_segments_used": len(segments),
                   "analysis_sample_dt_ps": dt,
                   "largest_msd_lag_ps": float(curve["lag_ps"][-1]) if curve is not None else None,
                   "rms_lateral_extent_longest_segment_A": radius,
                   "block_count": len(block_values),
                   "block_D_mean_A2_per_ps": float(np.mean(block_values)) if block_values else None,
                   "block_D_std_A2_per_ps": float(np.std(block_values, ddof=1)) if len(block_values) > 1 else None,
                   "geometric_capture_time_ps": capture["time_ps"] if capture else None,
                   "observed_residence_ps": capture["observed_residence_ps"] if capture else None,
                   "residence_right_censored": capture["residence_right_censored"] if capture else None,
                   "raw_cache_folder": str(extracted["cache_dir"]), **fit}
        summaries.append(summary)
        curves[name] = {"curve": curve, "fit": fit, "summary": summary}
        if curve is not None:
            save_csv(report_dir / f"msd_{name}.csv", [dict(zip(curve, values)) for values in zip(*curve.values())])
            lag, alpha = scaling_curve(curve)
            save_csv(report_dir / f"scaling_{name}.csv", [{"lag_ps": t, "alpha_local": a} for t, a in zip(lag, alpha)], ["lag_ps", "alpha_local"])
        save_csv(report_dir / f"blocks_{name}.csv", blocks)
    save_csv(report_dir / "diffusion_summary.csv", summaries)
    save_json(report_dir / "analysis_settings.json", settings)
    save_json(report_dir / "capture_result.json", {"definition": "persistent geometric entry into the vacancy region; not proof of chemical adsorption",
              "radius_A": settings["capture_radius_A"], "height_A": settings["capture_height_A"],
              "dwell_ps": settings["capture_dwell_ps"], "capture": capture,
              "post_start_override_ps": post})
    result = {"case": case, "extracted": extracted, "report_dir": report_dir,
              "summaries": summaries, "curves": curves, "capture": capture}
    return result


def set_plot_style(settings):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": settings["font_size"], "axes.labelsize": settings["axis_size"],
                         "axes.titlesize": settings["axis_size"], "xtick.labelsize": settings["tick_size"],
                         "ytick.labelsize": settings["tick_size"], "legend.fontsize": settings["font_size"],
                         "lines.linewidth": settings["line_width"], "axes.linewidth": 1.5})


def save_figure(fig, folder, stem, settings):
    for extension in settings["figure_formats"]:
        fig.savefig(Path(folder) / (stem + "." + extension), dpi=300, bbox_inches="tight")


def draw_segmented(ax, time_ps, values, segments, **kwargs):
    first = True
    label = kwargs.pop("label", None)
    for identifier in np.unique(segments):
        mask = segments == identifier
        ax.plot(time_ps[mask], values[mask], label=label if first else None, **kwargs)
        first = False


def plot_case(result, settings, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    set_plot_style(settings)
    data = result["extracted"]["data"]
    metadata = result["extracted"]["metadata"]
    folder, metal = result["report_dir"], metadata["metal"]
    color = COLORS.get(metal, "tab:blue")
    figs = []
    thermo_mask = np.ones(len(data["thermo_time_ps"]), bool)
    if settings["analysis_end_ps"] is not None:
        thermo_mask &= data["thermo_time_ps"] <= settings["analysis_end_ps"]
    thermo_time = data["thermo_time_ps"][thermo_mask]
    temperature = data["temperature_K"][thermo_mask]
    energy_key = "thermo_" + settings["energy_field"]
    energy = data[energy_key][thermo_mask]
    labels = {"E": "E: extended-system total energy", "F": "F/TOTEN: electronic free energy", "E0": "E0: sigma→0 energy"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for audit in metadata["run_audit"]:
        mask = (thermo_time > audit["time_offset_ps"] - 1e-12) & (thermo_time <= audit["time_offset_ps"] + audit["run_duration_ps"] + 1e-12)
        if np.any(mask):
            axes[0].plot(thermo_time[mask], temperature[mask], color=color, lw=max(1.0, settings["line_width"] / 2))
            baseline = energy[np.flatnonzero(np.isfinite(energy))[0]] if np.isfinite(energy).any() else 0
            axes[1].plot(thermo_time[mask], energy[mask] - baseline, color=color, lw=max(1.0, settings["line_width"] / 2))
    axes[0].axhline(300, color="black", ls="--", lw=1.2)
    axes[0].set(ylabel="Temperature (K)", title=f"{metal}: {result['case']['relative']}")
    axes[1].set(xlabel="Time (ps)", ylabel="ΔE (eV)", title=labels[settings["energy_field"]])
    if not len(thermo_time):
        axes[0].text(.5, .5, "No temperature/energy MD records found", transform=axes[0].transAxes, ha="center")
    fig.tight_layout(); save_figure(fig, folder, "temperature_energy", settings); figs.append(fig)

    mask = np.ones(len(data["time_ps"]), bool)
    if settings["analysis_end_ps"] is not None:
        mask &= data["time_ps"] <= settings["analysis_end_ps"]
    t, segments = data["time_ps"][mask], data["segment_ids"][mask]
    relative = data["positions_A"][mask] - data["positions_A"][0]
    height = data["height_A"][mask]
    radius = np.linalg.norm(relative[:, :2], axis=1)
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    for ax, values, label in zip(axes[:3], (relative[:, 0], relative[:, 1], height),
                                 ("Δx (Å)", "Δy (Å)", "Height (Å)")):
        draw_segmented(ax, t, values, segments, color=color)
        ax.set_ylabel(label)
    draw_segmented(axes[3], t, radius, segments, color=color, label="From initial position")
    rho = data["rho_vacancy_A"][mask]
    if np.isfinite(rho).any():
        draw_segmented(axes[3], t, rho, segments, color="tab:blue", label="To nearest vacancy image")
        axes[3].axhline(settings["capture_radius_A"], color="gray", lw=1.2, ls=":")
    for ax in axes:
        if result["capture"]:
            ax.axvline(result["capture"]["time_ps"], color="black", lw=1.3, ls="--")
    axes[0].set_title(f"{metal}: trajectory coordinates in the substrate frame")
    axes[3].set(ylabel="Lateral r (Å)", xlabel="Time (ps)")
    axes[3].legend(fontsize=max(12, settings["font_size"] - 4))
    fig.tight_layout(); save_figure(fig, folder, "coordinates_and_r", settings); figs.append(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    normalization = Normalize(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0] + 1))
    collection = None
    for identifier in np.unique(segments):
        idx = np.flatnonzero(segments == identifier)
        # Display thinning only; raw exports/MSD use their independently configured sampling.
        idx = idx[::max(1, int(math.ceil(len(idx) / 25000)))]
        if len(idx) < 2:
            continue
        points = relative[idx, :2]
        lines = np.stack([points[:-1], points[1:]], axis=1)
        collection = LineCollection(lines, cmap="viridis", norm=normalization, linewidth=2)
        collection.set_array(t[idx][:-1]); ax.add_collection(collection)
    ax.scatter(relative[0, 0], relative[0, 1], color="green", s=60, label="Start", zorder=5)
    ax.scatter(relative[-1, 0], relative[-1, 1], color="red", marker="x", s=80, label="End", zorder=5)
    ax.autoscale(); ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="Δx (Å)", ylabel="Δy (Å)", title=f"{metal}: lateral migration")
    ax.legend(fontsize=14)
    if collection is not None:
        colorbar = fig.colorbar(collection, ax=ax, pad=.03); colorbar.set_label("Time (ps)", fontsize=20)
    fig.tight_layout(); save_figure(fig, folder, "xy_trajectory", settings); figs.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = {"whole": color, "equilibrated": "tab:blue", "post_capture": "tab:green", "post_selected": "tab:green"}
    for name, item in result["curves"].items():
        curve, fit = item["curve"], item["fit"]
        if curve is None:
            continue
        lag, msd = curve["lag_ps"], curve["msd_xy_A2"]
        label = name.replace("_", " ")
        axes[0].plot(lag, msd, color=palette[name], label=label)
        valid = (lag > 0) & (msd > 0)
        axes[1].loglog(lag[valid], msd[valid], color=palette[name], label=label)
        if fit["slope_A2_per_ps"] is not None:
            selected = (lag >= fit["actual_fit_start_ps"]) & (lag <= fit["actual_fit_end_ps"])
            axes[0].plot(lag[selected], fit["slope_A2_per_ps"] * lag[selected] + fit["intercept_A2"], color=palette[name], ls="--", lw=2)
    for ax in axes:
        ax.set(xlabel="Lag time (ps)", ylabel="MSD$_{xy}$ (Å$^2$)")
        if ax.lines:
            ax.legend(fontsize=14)
    axes[0].set_title(f"{metal}: time-origin averaged MSD")
    axes[1].set_title("Log–log view")
    fig.tight_layout(); save_figure(fig, folder, "msd_windows", settings); figs.append(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, item in result["curves"].items():
        lag, alpha = scaling_curve(item["curve"])
        if len(lag):
            ax.semilogx(lag, alpha, label=name.replace("_", " "), color=palette[name])
    for value, linestyle in ((0, ":"), (1, "--"), (2, ":")):
        ax.axhline(value, color="gray", lw=1.2, ls=linestyle)
    ax.set(xlabel="Lag time (ps)", ylabel="Local MSD exponent α", title=f"{metal}: scaling diagnostic", ylim=(-1, 3))
    if any(item["curve"] is not None for item in result["curves"].values()):
        ax.legend(fontsize=14)
    fig.tight_layout(); save_figure(fig, folder, "msd_scaling", settings); figs.append(fig)
    for fig in figs:
        if show:
            plt.show()
        plt.close(fig)


def plot_comparisons(results, report_root, settings):
    import matplotlib.pyplot as plt
    set_plot_style(settings)
    for group in ("whole", "equilibrated", "post"):
        fig, ax = plt.subplots(figsize=(10, 7))
        plotted = 0
        for result in results:
            candidates = [key for key in result["curves"] if key == group or group == "post" and key.startswith("post_")]
            if not candidates:
                continue
            curve = result["curves"][candidates[0]]["curve"]
            if curve is None:
                continue
            metal = result["extracted"]["metadata"]["metal"]
            ax.plot(curve["lag_ps"], curve["msd_xy_A2"], color=COLORS.get(metal), label=f"{metal}: {result['case']['relative']}")
            plotted += 1
        if plotted:
            ax.set(xlabel="Lag time (ps)", ylabel="MSD$_{xy}$ (Å$^2$)", title=f"Across metals: {group.replace('_', ' ')} window")
            ax.legend(fontsize=14)
            fig.tight_layout(); save_figure(fig, report_root, f"comparison_msd_{group}", settings)
        plt.close(fig)


METHODS_TEXT = """AIMD processing and diffusion analysis

Run this program separately in nvac-diff and pristine. It discovers individual
metal/condition series and numeric restart folders. It reads XDATCAR (OUTCAR
positions are a fallback), OSZICAR/stdout (OUTCAR thermodynamics are a fallback),
POSCAR/CONTCAR, and INCAR. Perl/Bash and pre-extracted files are not required.

Physical time comes from POTIM times the saved-coordinate stride (NBLOCK or an
explicit override). Configuration numbers can be MD-step labels or saved-frame
ordinals; the mapping used for each run appears in run_audit.csv. For NBLOCK>1,
ordinal frames are assigned to block-completion steps. This preserves the saved
spacing; inspect the first-frame phase if sub-output-interval timing matters.
No total duration is forced to be 90 ps. Missing/truncated coordinates and
unverified durations are reported. Planned NSW is never substituted for actual
completed steps. Coordinates at identical later times are not deduplicated just
because an atom stayed still. A duplicate initial record is omitted only when
both its physical timestamp and all its positions match the preceding record.

All atoms are unwrapped before a mass-weighted host center is calculated. In-
plane minimum images include neighboring translations for hexagonal cells.
Analysis uses orthonormal coordinates in the substrate plane. Variable cells
or changes in atom/element ordering are rejected. A gap, timestep change, large
adjacent displacement, or failed restart check breaks MSD continuity.

restart_policy='geometry' joins position-matched restarts, with a warning if
velocities are unverified. 'strict' additionally requires matching saved ionic
velocities; 'split' excludes all cross-restart MSD pairs. Even a velocity match
does not prove all thermostat/internal integrator state was preserved.

For a single missing B or N in h-BN, the program looks for three neighboring
undercoordinated opposite-sublattice atoms. Vacancy distance/height use their
instantaneous triangle and the nearest periodic image of the vacancy. Persistent
geometric entry is defined by configurable lateral radius, height, and dwell.
This is a geometric state definition, not proof of chemical adsorption. Manual
post_start_ps and neighbor atom-number overrides are available. Pristine systems
have a separate after-equilibration window rather than a fabricated capture time.
The automatic post_capture window begins after entry has been confirmed and
after the configured buffer. It stops before the first persistent geometric
escape or at the end of that continuous segment. Thus a later escape is included
in the whole-run MSD, but not in this first-residence MSD. A manual post_start_ps
instead selects from that time to analysis_end_ps (or the available run end).

MSD is time-origin averaged using exact FFT sums, pooling only valid pairs
within continuous, uniformly sampled segments. Analysis decimation is performed
after unwrapping. Whole-run analysis uses all selected trajectory times as time
origins: a 10-ps lag axis does not mean only the first 10 ps were analyzed.
The requested fit interval must be fully available; it is not silently clipped.
D_apparent = fitted lateral-MSD slope / 4, with 1 A^2/ps = 1e-8 m^2/s.
The alpha/R2 screen is a diagnostic, not proof of normal stationary diffusion.
No ordinary least-squares confidence interval is claimed. Nonoverlapping long
blocks provide a descriptive spread of apparent estimates when enough data
exist; this spread is not automatically a formal confidence interval.

Whole, equilibrated, and post-capture/selected windows answer different questions.
An MSD plateau after binding describes confinement; it does not establish a
free-surface diffusion coefficient of zero. A single capture is an observed
first-entry time, not a mean capture rate. Residence without escape is censored
at the end of the continuous observed segment, not an infinite lifetime.

Energy columns are explicit: E is VASP extended-system total energy, F/TOTEN is
electronic free energy, and E0 is the sigma->0 value. energy_eV in the raw combined
CSV retains E for compatibility; energy_field selects the plotted column only.
Energy shifts are not electrical switching energy. Temperature is not inferred
from position fluctuations. No thermodynamic equilibration time is detected
automatically; the configurable default exclusion of 1 ps needs inspection.

Full XYZ and combined XDATCAR exports are optional and can be large. XYZ comments
include physical time; combined XDATCAR labels are saved-frame ordinals, with
physical time in trajectory.csv. Raw extraction is cached by source path, size,
mtime, and extraction settings. Changing only MSD windows/styles reuses caches.
XY display can be thinned to 25,000 points per segment; quantitative analyses
use analysis_stride and full-resolution exports remain available. The alpha
figure displays -1 to 3; its complete numerical values are exported separately.

References
1. Bullerjahn, J. T.; von Buelow, S.; Hummer, G. Optimal estimates of self-diffusion
coefficients from molecular dynamics simulations. J. Chem. Phys. 153, 024116
(2020). https://doi.org/10.1063/5.0008312
2. McCluskey, A. R.; Coles, S. W.; Morgan, B. J. Accurate estimation of diffusion
coefficients and their uncertainties from computer simulation. J. Chem. Theory
Comput. 21, 79-87 (2025). https://doi.org/10.1021/acs.jctc.4c01249
3. VASP Wiki. NBLOCK. https://vasp.at/wiki/NBLOCK
4. VASP Wiki. OSZICAR. https://vasp.at/wiki/OSZICAR
5. VASP Wiki. Molecular-dynamics calculations.
https://vasp.at/wiki/Molecular-dynamics_calculations
"""


def process_root(root, settings=None, show_plots=False):
    root = Path(root).expanduser().resolve()
    configuration = dict(DEFAULTS)
    if settings:
        configuration.update(settings)
    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if configuration["restart_policy"] not in ("geometry", "strict", "split"):
        raise ValueError("restart_policy must be geometry, strict, or split.")
    if configuration["analysis_stride"] < 1 or configuration["movie_stride"] < 1:
        raise ValueError("Sampling strides must be positive integers.")
    if int(configuration["analysis_stride"]) != configuration["analysis_stride"] or int(configuration["movie_stride"]) != configuration["movie_stride"]:
        raise ValueError("Sampling strides must be integers.")
    if not 0 < configuration["msd_max_fraction"] <= 0.5:
        raise ValueError("msd_max_fraction must be in (0, 0.5].")
    start, stop = configuration["fit_lag_ps"]
    if start < 0 or stop <= start:
        raise ValueError("fit_lag_ps must satisfy 0 <= start < end.")
    if configuration["energy_field"] not in ("E", "F", "E0"):
        raise ValueError("energy_field must be E, F, or E0.")
    cases = discover_cases(root)
    if not cases:
        raise ValueError(f"No raw MD series found below {root}. Expected gold/silver/copper with numbered run folders, or raw files directly in each metal folder.")
    output = root / "AIMD_processed"
    report_root = output / "reports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_root.mkdir(parents=True)
    (report_root / "METHODS_AND_REFERENCES.txt").write_text(METHODS_TEXT, encoding="utf-8")
    results, errors, rows = [], [], []
    print(f"Dataset root: {root}\nDiscovered {len(cases)} condition/metal series.", flush=True)
    if configuration["make_plots"]:
        try:
            import matplotlib.pyplot
        except ImportError:
            configuration["make_plots"] = False
            print("Matplotlib is unavailable: numeric exports will be created. Install with python -m pip install matplotlib.")
    for case in cases:
        try:
            extracted = prepare_case(case, output, configuration)
            result = analyze_case(case, extracted, report_root / case["id"], configuration)
            if configuration["make_plots"]:
                plot_case(result, configuration, show=show_plots)
            results.append(result); rows.extend(result["summaries"])
            metadata = extracted["metadata"]
            print(f"{case['relative']}: {metadata['coordinate_frames']:,} frames; sampled time {metadata['first_time_ps']:.6f}–{metadata['last_time_ps']:.6f} ps")
            for summary in result["summaries"]:
                value = summary["D_apparent_A2_per_ps"]
                estimate = "not fitted" if value is None else f"apparent D={value:.6g} A^2/ps"
                print(f"  {summary['window']}: {estimate}; {summary['fit_status']}")
        except (ValueError, OSError, KeyError, IndexError) as exc:
            error = {"case": case["id"], "folder": str(case["folder"]), "error": str(exc)}
            errors.append(error)
            print(f"ERROR in {case['relative']}: {exc}", file=sys.stderr, flush=True)
    save_csv(report_root / "all_diffusion_summaries.csv", rows)
    save_csv(report_root / "errors.csv", errors, ["case", "folder", "error"])
    save_json(report_root / "batch_settings.json", configuration)
    save_json(report_root / "batch_manifest.json", {"root": str(root), "successful_cases": len(results),
              "failed_cases": errors, "report_folder": str(report_root),
              "raw_cache_folders": [str(r["extracted"]["cache_dir"]) for r in results]})
    if results and configuration["make_plots"]:
        plot_comparisons(results, report_root, configuration)
    print(f"\nCompleted {len(results)} series; {len(errors)} series with errors.\nReports: {report_root}", flush=True)
    return {"results": results, "errors": errors, "report_root": report_root, "settings": configuration}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="nvac-diff or pristine; default is the folder containing this script.")
    parser.add_argument("--config", type=Path, help="Optional JSON object overriding DEFAULTS.")
    parser.add_argument("--post-start", action="append", default=[], metavar="METAL=PS", help="Example: --post-start Au=5 --post-start Ag=8")
    parser.add_argument("--fit-lag", type=float, nargs=2, metavar=("START_PS", "END_PS"))
    parser.add_argument("--end-ps", type=float, help="Optional absolute end of the analyzed trajectory, e.g. 90.")
    parser.add_argument("--restart-policy", choices=("geometry", "strict", "split"))
    parser.add_argument("--no-movies", action="store_true", help="Skip combined XYZ/XDATCAR exports; direct analysis still runs.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-extract raw files instead of reusing a matching cache.")
    args = parser.parse_args(argv)
    settings = {}
    try:
        if args.config:
            settings.update(json.loads(args.config.read_text(encoding="utf-8-sig")))
        if args.post_start:
            mapping = dict(settings.get("post_start_ps", {}))
            for item in args.post_start:
                key, value = item.split("=", 1); mapping[key] = float(value)
            settings["post_start_ps"] = mapping
        if args.fit_lag:
            settings["fit_lag_ps"] = args.fit_lag
        if args.end_ps is not None:
            settings["analysis_end_ps"] = args.end_ps
        if args.restart_policy:
            settings["restart_policy"] = args.restart_policy
        if args.no_movies:
            settings.update(write_xyz=False, write_combined_xdatcar=False)
        if args.no_plots:
            settings["make_plots"] = False
        if args.force:
            settings["force_reextract"] = True
        root = args.root if args.root else Path(__file__).resolve().parent
        batch = process_root(root, settings)
        return 0 if batch["results"] and not batch["errors"] else 2
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
