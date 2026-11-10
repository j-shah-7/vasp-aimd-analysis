"""Local h-BN AIMD postprocessing. Read-only inputs; results go to output_dir.

All distances are Angstrom and all times are ps. No ASE or VASP license is
needed to read an existing OUTCAR. The notebook embeds this entire module.
"""

from pathlib import Path
from itertools import product
import gzip
import bz2
import lzma
import re
import json
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import fftconvolve
from scipy.stats import linregress

COLORS = {"Au": "#C49A08", "Ag": "#85898F", "Cu": "#B86C32"}
FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def minimum_image(displacements, cell, pbc=(True, True, False)):
    """Nearest image for a conventional slab cell, including a hexagonal cell.

    Search the neighboring lattice translations after fractional rounding;
    fractional rounding alone is not a nearest-image rule for skew cells.
    """
    a = np.asarray(displacements, float)
    if cell is None:
        return a.copy()
    original_shape = a.shape
    matrix = np.asarray(cell, float)
    f = a.reshape(-1, 3) @ np.linalg.inv(matrix)
    periodic = np.array(pbc, bool)
    f[:, periodic] -= np.rint(f[:, periodic])
    best = f @ matrix
    best2 = np.sum(best * best, axis=1)
    choices = [(-1, 0, 1) if flag else (0,) for flag in periodic]
    for shift in product(*choices):
        if not any(shift):
            continue
        candidate = (f + np.asarray(shift)) @ matrix
        d2 = np.sum(candidate * candidate, axis=1)
        improve = d2 < best2
        best[improve] = candidate[improve]
        best2[improve] = d2[improve]
    return best.reshape(original_shape)


def unwrap_positions(positions, cell, pbc=(True, True, False)):
    positions = np.asarray(positions, float)
    if cell is None or len(positions) < 2:
        return positions.copy()
    steps = minimum_image(np.diff(positions, axis=0), cell, pbc)
    return np.vstack([positions[0], positions[0] + np.cumsum(steps, axis=0)])


def text_file(path):
    path = Path(path)
    opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix, open)
    return opener(path, "rt", encoding="utf-8", errors="replace")


def read_poscar(path):
    """Read a VASP-5/6 POSCAR, including Cartesian/Direct/selective dynamics."""
    with text_file(path) as f:
        title = next(f).strip()
        scales = np.array([number(v) for v in next(f).split()])
        base = np.array([[number(v) for v in next(f).split()[:3]] for _ in range(3)])
        if len(scales) == 1:
            factor = scales[0]
            if factor < 0:
                factor = (abs(factor)/abs(np.linalg.det(base)))**(1/3)
            scale_xyz = np.repeat(factor, 3)
        elif len(scales) == 3 and np.all(scales > 0):
            scale_xyz = scales
        else:
            raise ValueError("Unsupported POSCAR scaling factors.")
        cell = base * scale_xyz[None, :]
        species = next(f).split()
        if all(v.isdigit() for v in species):
            raise ValueError("This VASP-4 POSCAR lacks element names; use a VASP-5/6 POSCAR.")
        counts = [int(v) for v in next(f).split()]
        if len(counts) != len(species):
            raise ValueError("POSCAR element/count mismatch.")
        line = next(f).strip()
        if line.lower().startswith("s"):
            line = next(f).strip()
        direct = line.lower().startswith("d")
        if not direct and not line.lower().startswith(("c", "k")):
            raise ValueError("Expected Direct or Cartesian POSCAR coordinates.")
        positions = np.array([[number(v) for v in next(f).split()[:3]] for _ in range(sum(counts))])
        positions = positions @ cell if direct else positions * scale_xyz
    return {"title": title, "cell": cell, "symbols": np.repeat(species, counts),
            "positions_A": positions}


def attach_structure(raw, structure, vacancy_neighbors=None):
    """Supply a static cell/reference plane to spreadsheet or metal-only XYZ data."""
    if structure is None:
        return raw
    raw = dict(raw)
    cell, snapshot, symbols = structure["cell"], structure["positions_A"], structure["symbols"]
    if raw.get("cell") is not None and not np.allclose(raw["cell"], cell, atol=1e-5):
        raise ValueError("POSCAR cell differs from trajectory cell; verify the folder mapping.")
    if raw.get("snapshot_A") is not None:
        if not np.array_equal(raw["symbols"], symbols):
            raise ValueError("POSCAR and trajectory element ordering differ; verify the selected POSCAR.")
        offset = np.linalg.norm(minimum_image(raw["snapshot_A"]-snapshot, cell), axis=1).max()
        raw["first_frame_poscar_max_offset_A"] = float(offset)
        if offset > 0.25:
            warnings.warn("First trajectory frame differs from the chosen POSCAR by more than 0.25 A; "
                          "check the initial run and N-neighbor indices.")
    raw["cell"] = cell
    raw.setdefault("snapshot_A", snapshot)
    raw.setdefault("symbols", symbols)
    normal = np.cross(cell[0], cell[1]); normal /= np.linalg.norm(normal)
    e1 = cell[0]/np.linalg.norm(cell[0])
    raw.setdefault("basis", np.vstack([e1, np.cross(normal, e1), normal]))
    if raw.get("height_A") is None:
        center = snapshot[np.isin(symbols, ("B", "N"))].mean(axis=0)
        raw["height_A"] = (raw["positions_A"]-center) @ normal
        raw["height_reference"] = "static POSCAR plane"
    if vacancy_neighbors is not None and raw.get("vacancy_centers_A") is None:
        ids = np.asarray(vacancy_neighbors, int)
        tri = snapshot[ids].copy()
        tri[1:] = tri[0]+minimum_image(tri[1:]-tri[0], cell)
        raw["vacancy_centers_A"] = np.tile(tri.mean(axis=0), (len(raw["time_ps"]), 1))
        raw["vacancy_reference"] = "static POSCAR N-neighbor centroid"
    return raw


def read_xyz_movie(path, metal, structure=None, metal_index=None, frame_dt_fs=0.5,
                   vacancy_neighbors=None, xyz_time_unit=None, progress_every=25000):
    """Stream a multi-frame XYZ/extXYZ movie; retain selected coordinates only.

    Standard XYZ: integer atom count, one comment line, then 'Element x y z'.
    A POSCAR supplies the cell when ordinary XYZ has no Lattice metadata.
    frame_dt_fs is the time BETWEEN SAVED FRAMES, not necessarily POTIM.
    Time= comments are used only when xyz_time_unit is explicitly 'fs' or 'ps'.
    """
    if frame_dt_fs is None or frame_dt_fs <= 0:
        raise ValueError("Set frame_dt_fs to the spacing between saved XYZ frames.")
    t, positions, drifts, heights, centers, normals = [], [], [], [], [], []
    cell = None if structure is None else np.asarray(structure["cell"])
    symbols0 = first = previous_full = full_u = initial_center = basis = None
    selected = metal_index
    previous_comment_time = None
    with text_file(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                n = int(line.strip())
            except ValueError as exc:
                raise ValueError("Expected XYZ atom count. Select an .xyz/.extxyz coordinate movie.") from exc
            comment = f.readline()
            if not comment:
                warnings.warn("Ignored incomplete final XYZ frame.")
                break
            lattice = re.search(r'Lattice="([^"]+)"', comment)
            if lattice:
                lattice_cell = np.array([number(v) for v in lattice.group(1).split()]).reshape(3, 3)
                if cell is None:
                    cell = lattice_cell
                elif not np.allclose(cell, lattice_cell, atol=1e-5):
                    raise ValueError("XYZ cell changed; variable-cell trajectories need a different analysis.")
            props = re.search(r"Properties=([^\s]+)", comment)
            species_col, pos_col = 0, 1
            if props:
                entries, offset = props.group(1).split(":"), 0
                for j in range(0, len(entries), 3):
                    name, count = entries[j], int(entries[j+2])
                    if name in ("species", "element"):
                        species_col = offset
                    if name in ("pos", "positions"):
                        pos_col = offset
                    offset += count
            rows, symbols = [], []
            for _ in range(n):
                row = f.readline().split()
                if len(row) < pos_col+3:
                    break
                symbols.append(row[species_col])
                rows.append([number(v) for v in row[pos_col:pos_col+3]])
            if len(rows) != n:
                warnings.warn("Ignored incomplete final XYZ frame.")
                break
            full = np.array(rows)
            symbols = np.array(symbols)
            substrate = np.isin(symbols, ("B", "N"))
            if first is None:
                first, symbols0 = full.copy(), symbols.copy()
                idx = np.flatnonzero(symbols == metal)
                if selected is None:
                    if len(idx) != 1:
                        raise ValueError(f"Found {len(idx)} {metal} atoms; set metal_index.")
                    selected = int(idx[0])
                if not 0 <= selected < n or symbols[selected] != metal:
                    raise ValueError("metal_index does not identify the requested element.")
                if cell is None and np.any(substrate):
                    raise ValueError("Provide POSCAR or extXYZ Lattice metadata for full-atom XYZ.")
                normal = np.array([0., 0., 1.]) if cell is None else np.cross(cell[0], cell[1])
                normal /= np.linalg.norm(normal)
                e1 = np.array([1., 0., 0.]) if cell is None else cell[0]/np.linalg.norm(cell[0])
                basis = np.vstack([e1, np.cross(normal, e1), normal])
                full_u = full.copy()
                if np.any(substrate):
                    initial_center = full[substrate].mean(axis=0)
            else:
                if not np.array_equal(symbols, symbols0):
                    raise ValueError("Atom ordering or atom count changed within the XYZ movie.")
                if cell is None:
                    full_u += full-previous_full
                else:
                    step = (full-previous_full) @ np.linalg.inv(cell)
                    step[:, :2] -= np.rint(step[:, :2])
                    full_u += step @ cell
            comment_time = re.search(r"\b[Tt]ime\s*=\s*("+FLOAT+")", comment)
            if comment_time and xyz_time_unit in ("fs", "ps"):
                value = number(comment_time.group(1))*(0.001 if xyz_time_unit == "fs" else 1)
                if not t:
                    first_comment_time = value
                if previous_comment_time is not None and value <= previous_comment_time:
                    raise ValueError("XYZ comment times reset; split the restarted movies.")
                time = value-first_comment_time
                previous_comment_time = value
            else:
                time = len(t)*frame_dt_fs/1000
            t.append(time)
            positions.append(full[selected].copy())
            if np.any(substrate):
                center = full_u[substrate].mean(axis=0)
                drifts.append(center-initial_center)
                heights.append(np.dot(full_u[selected]-center, basis[2]))
            if vacancy_neighbors is not None and len(full) > 1:
                ids = np.asarray(vacancy_neighbors, int)
                if len(ids) != 3 or np.any(symbols[ids] != "N"):
                    raise ValueError("Vacancy neighbors must be three N indices in the XYZ ordering.")
                tri = full[ids].copy()
                tri[1:] = tri[0]+minimum_image(tri[1:]-tri[0], cell)
                normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
                normal /= np.linalg.norm(normal)
                if np.dot(normal, basis[2]) < 0:
                    normal = -normal
                centers.append(tri.mean(axis=0))
                normals.append(normal)
            previous_full = full.copy()
            if progress_every and len(t) % progress_every == 0:
                print(f"Read {len(t):,} XYZ frames ({time:.3f} ps)")
    if not t:
        raise ValueError("No complete XYZ movie frames read.")
    raw = {"time_ps": np.array(t), "positions_A": np.array(positions), "cell": cell,
           "metal": metal, "source": str(path), "basis": basis, "metal_index": selected}
    if len(first) > 1:
        raw.update(symbols=symbols0, snapshot_A=first)
    if drifts:
        raw.update(substrate_drift_A=np.array(drifts), height_A=np.array(heights),
                   height_reference="instantaneous B/N plane")
    if centers:
        raw.update(vacancy_centers_A=np.array(centers), vacancy_normals=np.array(normals),
                   vacancy_reference="instantaneous three-N centroid")
    return attach_structure(raw, structure, vacancy_neighbors)


def read_excel_trajectory(workbook, sheet, metal, cache_dir=None):
    path = Path(workbook).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Set WORKBOOK to your local Excel file: {path}")
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = f"{path}|{path.stat().st_size}|{path.stat().st_mtime_ns}|{sheet}"
        cache = cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:18] + ".npz")
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            time, positions = z["time_ps"], z["positions_A"]
    else:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        required = ["time_ps", "x_A", "y_A", "z_A"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"{sheet}: expected columns {required}; found {list(df.columns)}")
        time = df["time_ps"].to_numpy(float)
        positions = df[["x_A", "y_A", "z_A"]].to_numpy(float)
        if cache is not None:
            np.savez_compressed(cache, time_ps=time, positions_A=positions)
    return {"time_ps": time, "positions_A": positions, "metal": metal,
            "cell": None, "source": str(path), "sheet": sheet}


def read_csv_trajectory(path, metal, column_map=None):
    """Read a combined metal-coordinate CSV, with explicit ps/Angstrom columns."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    mapping = {name: name for name in ("time_ps", "x_A", "y_A", "z_A")}
    mapping.update(column_map or {})
    missing = [name for name, original in mapping.items() if original not in frame.columns]
    if missing:
        raise ValueError(f"{path}: expected columns {mapping}; found {list(frame.columns)}. "
                         "Set CSV_COLUMN_MAP explicitly if your headers differ.")
    return {"time_ps": frame[mapping["time_ps"]].to_numpy(float),
            "positions_A": frame[[mapping[v] for v in ("x_A", "y_A", "z_A")]].to_numpy(float),
            "metal": metal, "cell": None, "source": str(path)}


def read_thermodynamics_csv(path, column_map=None, timestep_fs=None):
    """Read a combined thermo CSV, keeping its own time axis and energy meaning.

    Mapping keys: time_ps OR time_fs OR step, temperature_K, energy_eV.
    Generic 'Energy' is retained as an unspecified energy, never called TOTEN.
    Step-based input requires an explicit MD timestep and nonresetting steps.
    """
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    norm = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    columns = {norm(c): c for c in frame.columns}
    mapping = dict(column_map or {})
    aliases = {
        "time_ps": ("time_ps", "time (ps)", "global_time_ps", "cumulative_time_ps"),
        "time_fs": ("time_fs", "time (fs)", "global_time_fs"),
        "step": ("global_step", "global_md_step", "cumulative_step", "step", "md_step", "ionic_step"),
        "temperature_K": ("temperature_K", "temperature (K)", "temperature", "temp_K", "temp", "T(K)", "T"),
        "energy_eV": ("TOTEN", "free_energy_eV", "potential_energy_eV", "potential energy (eV)",
                      "energy_eV", "energy (eV)", "energy", "total_energy_eV", "total_energy"),
    }
    for key, candidates in aliases.items():
        if key not in mapping:
            mapping[key] = next((columns[norm(c)] for c in candidates if norm(c) in columns), None)
    for key in ("temperature_K", "energy_eV"):
        if mapping[key] is None or mapping[key] not in frame:
            raise ValueError(f"{path}: cannot identify {key}; headers are {list(frame.columns)}. "
                             "Set THERMO_COLUMN_MAP explicitly, or use THERMO_SOURCE='outcar'.")
    explicit_times = [key for key in ("time_ps", "time_fs", "step") if key in (column_map or {})]
    time_key = explicit_times[0] if explicit_times else next(
        (key for key in ("time_ps", "time_fs", "step") if mapping[key] is not None), None)
    if time_key is None or mapping[time_key] not in frame:
        raise ValueError(f"{path}: supply a ps/fs time column or a global MD-step column. "
                         "A generic 'Time' column needs an explicit unit mapping.")
    time = pd.to_numeric(frame[mapping[time_key]], errors="coerce").to_numpy(float)
    if time_key == "time_fs":
        time *= 0.001
    elif time_key == "step":
        if timestep_fs is None or timestep_fs <= 0:
            raise ValueError("A step-based thermo CSV requires THERMO_TIMESTEP_FS.")
        time *= timestep_fs / 1000
    if len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: thermodynamic times/steps must increase globally. "
                         "Use ordered OUTCARs if this CSV resets at each restart.")
    time -= time[0]
    temperature = pd.to_numeric(frame[mapping["temperature_K"]], errors="coerce").to_numpy(float)
    energy = pd.to_numeric(frame[mapping["energy_eV"]], errors="coerce").to_numpy(float)
    if not np.isfinite(temperature).any() or not np.isfinite(energy).any():
        raise ValueError(f"{path}: no numeric temperature or energy values found.")
    energy_header = str(mapping["energy_eV"])
    quantity = "TOTEN" if "toten" in norm(energy_header) else energy_header
    return {"time_ps": time, "temperature_K": temperature, "energy_eV": energy,
            "source_ids": np.zeros(len(time), int), "source": str(path),
            "energy_quantity": quantity, "column_mapping": mapping}


def read_outcar_thermodynamics(paths, timestep_fs=None):
    """Stream ionic-step temperature and electronic free energy TOTEN only.

    Coordinates are not retained. Restart files must already be in numeric order.
    Time at each file boundary advances by one POTIM; no missing elapsed time
    can be inferred from a concatenation. File boundaries remain in source_ids.
    """
    times, temperatures, energies, source_ids, ionic_steps = [], [], [], [], []
    paths = [Path(p) for p in paths]
    iteration_re = re.compile(r"Iteration\s+(\d+)\s*\(\s*\d+\s*\)")
    potim_re = re.compile(r"POTIM\s*=\s*(" + FLOAT + ")")
    energy_re = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(" + FLOAT + ")")
    temp_re = re.compile(r"temperature\s*=?\s*(" + FLOAT + r")\s*K", re.I)
    for source_id, path in enumerate(paths):
        dt_fs, step, record, previous_step = timestep_fs, None, {}, None
        emitted = 0
        def emit(rec):
            nonlocal previous_step, emitted
            if not rec.get("has_position") or not ("temperature" in rec or "energy" in rec):
                return
            if dt_fs is None or dt_fs <= 0:
                raise ValueError(f"{path}: missing POTIM; set the thermo timestep explicitly.")
            current = rec.get("step")
            delta = current - previous_step if emitted and current is not None and previous_step is not None else 1
            if delta <= 0:
                raise ValueError(f"{path}: ionic-step numbering resets within this file.")
            times.append(0.0 if not times else times[-1] + delta*dt_fs/1000)
            temperatures.append(rec.get("temperature", np.nan))
            energies.append(rec.get("energy", np.nan))
            source_ids.append(source_id)
            ionic_steps.append(current if current is not None else np.nan)
            previous_step, emitted = current, emitted + 1
        with text_file(path) as handle:
            for line in handle:
                if "POTIM" in line and timestep_fs is None:
                    match = potim_re.search(line)
                    if match:
                        dt_fs = number(match.group(1))
                if "IBRION" in line:
                    match = re.search(r"IBRION\s*=\s*(-?\d+)", line)
                    if match and int(match.group(1)) != 0:
                        raise ValueError(f"{path}: IBRION is not zero; expected an MD OUTCAR.")
                if "Iteration" in line:
                    match = iteration_re.search(line)
                    if match:
                        current = int(match.group(1))
                        if step is not None and current != step:
                            emit(record)
                            record = {}
                        step = current
                        record["step"] = step
                if "POSITION" in line and "TOTAL-FORCE" in line:
                    if record.get("has_position"):
                        emit(record)
                        record = {"step": step}
                    record["has_position"] = True
                if "TOTEN" in line:
                    match = energy_re.search(line)
                    if match:
                        record["energy"] = number(match.group(1))
                if "temperature" in line.lower():
                    match = temp_re.search(line)
                    if match:
                        record["temperature"] = number(match.group(1))
            emit(record)
        if not emitted:
            raise ValueError(f"{path}: no ionic MD thermodynamics found.")
    if not times or not np.isfinite(temperatures).any() or not np.isfinite(energies).any():
        raise ValueError("No usable MD temperature and TOTEN values were found.")
    return {"time_ps": np.array(times), "temperature_K": np.array(temperatures),
            "energy_eV": np.array(energies), "source_ids": np.array(source_ids),
            "ionic_steps": np.array(ionic_steps), "source": [str(p) for p in paths],
            "energy_quantity": "TOTEN: electronic free energy"}


def load_thermodynamics(cfg, trajectory=None, source="auto", cache_dir=None, use_cache=True):
    """Independent thermo input: existing OUTCAR trajectory, combined CSV, or OUTCARs."""
    if source == "off":
        return None
    if source not in ("auto", "csv", "outcar"):
        raise ValueError("THERMO_SOURCE must be auto, csv, outcar, or off.")
    if (source in ("auto", "outcar") and cfg["kind"] == "outcar" and trajectory is not None
            and np.isfinite(trajectory.get("temperature_K", [np.nan])).any()
            and np.isfinite(trajectory.get("energy_eV", [np.nan])).any()):
        return {key: trajectory[key] for key in ("time_ps", "temperature_K", "energy_eV", "source_ids", "source")} | {
            "label": cfg["label"], "metal": cfg["metal"], "energy_quantity": "TOTEN: electronic free energy"}
    candidates = ("csv", "outcar") if source == "auto" else (source,)
    errors = []
    for kind in candidates:
        try:
            paths = [Path(cfg["thermo_csv"])] if kind == "csv" else [Path(p) for p in cfg["outcars"]]
            if not paths or any(not p.is_file() for p in paths):
                raise FileNotFoundError(f"{kind} input not available")
            if kind == "outcar" and cfg.get("missing_outcar_runs"):
                raise ValueError(f"Missing OUTCAR runs {cfg['missing_outcar_runs']}; elapsed time would be incomplete.")
            settings = {"kind": kind, "mapping": cfg.get("thermo_column_map"),
                        "timestep_fs": cfg.get("thermo_timestep_fs")}
            files = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
            digest = hashlib.sha256(json.dumps({"version": 1, "settings": settings, "files": files}, default=str).encode()).hexdigest()[:16]
            cache = None if cache_dir is None else Path(cache_dir)/f"{cfg['metal']}_thermo_{digest}.npz"
            if use_cache and cache is not None and cache.is_file():
                with np.load(cache, allow_pickle=False) as saved:
                    raw = json.loads(str(saved["_metadata"]))
                    raw.update({k: saved[k].copy() for k in saved.files if k != "_metadata"})
            else:
                print(f"{cfg['label']}: reading thermodynamics from {kind}")
                raw = read_thermodynamics_csv(paths[0], cfg.get("thermo_column_map"), cfg.get("thermo_timestep_fs")) if kind == "csv" else read_outcar_thermodynamics(paths, cfg.get("thermo_timestep_fs"))
                if use_cache and cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    arrays = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
                    metadata = {k: v for k, v in raw.items() if k not in arrays}
                    np.savez_compressed(cache, **arrays, _metadata=np.array(json.dumps(metadata)))
            raw.update(label=cfg["label"], metal=cfg["metal"])
            return raw
        except (ValueError, FileNotFoundError) as exc:
            errors.append(f"{kind}: {exc}")
            if source == "auto" and kind == "csv" and paths and paths[0].is_file():
                warnings.warn(f"{cfg['label']}: thermo CSV could not be interpreted; trying OUTCARs. {exc}")
    raise ValueError(f"{cfg['label']}: no thermodynamics loaded. " + " | ".join(errors))


def discover_run_folder(folder):
    """Lightweight inventory: numeric folders, first POSCAR, and existing parts."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Metal folder does not exist: {folder}. Edit FOLDERS/DATA_ROOT.")
    parts = sorted((p for p in folder.iterdir() if p.is_dir() and p.name.isdecimal()),
                   key=lambda p: int(p.name))
    def first_existing(parent, names):
        return next((parent/name for name in names if (parent/name).is_file()), None)
    poscar = first_existing(folder, ("POSCAR", "POSCAR.gz", "poscar"))
    rows, outcars = [], []
    for p in parts:
        outcar = first_existing(p, ("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2", "OUTCAR.xz"))
        pc = first_existing(p, ("POSCAR", "POSCAR.gz", "poscar"))
        if poscar is None and pc is not None:
            poscar = pc
        if outcar is not None:
            outcars.append(outcar)
        movie = folder/f"movie-file-{p.name}.xyz"
        temp = folder/f"file-{p.name}.csv"
        rows.append({"run_number": int(p.name), "folder": str(p),
                     "OUTCAR": str(outcar) if outcar else None,
                     "POSCAR": str(pc) if pc else None,
                     "CONTCAR": str(p/"CONTCAR") if (p/"CONTCAR").is_file() else None,
                     "part_movie_exists": movie.is_file(),
                     "part_movie_bytes": movie.stat().st_size if movie.is_file() else np.nan,
                     "part_temperature_csv_exists": temp.is_file()})
    if not parts:
        outcar = first_existing(folder, ("OUTCAR", "OUTCAR.gz", "OUTCAR.bz2", "OUTCAR.xz"))
        if outcar is not None:
            outcars = [outcar]
    return {"folder": folder, "poscar": poscar, "outcars": outcars,
            "parts": parts, "part_table": pd.DataFrame(rows),
            "missing_outcar_runs": [r["run_number"] for r in rows if r["OUTCAR"] is None]}


def restart_geometry_audit(layout, metal, tolerance_A=0.02):
    """Compare CONTCAR(n) to POSCAR(n+1); positions only, not velocities/state."""
    rows = []
    for previous, following in zip(layout["parts"][:-1], layout["parts"][1:]):
        row = {"previous_run": int(previous.name), "next_run": int(following.name),
               "nonconsecutive_run_numbers": int(following.name)-int(previous.name) != 1}
        final, initial = previous/"CONTCAR", following/"POSCAR"
        if not final.is_file() or not initial.is_file():
            rows.append({**row, "geometry_status": "CONTCAR/POSCAR unavailable"})
            continue
        a, b = read_poscar(final), read_poscar(initial)
        if not np.array_equal(a["symbols"], b["symbols"]) or not np.allclose(a["cell"], b["cell"], atol=1e-5):
            rows.append({**row, "geometry_status": "cell or atom order changed"})
            continue
        d = minimum_image(b["positions_A"]-a["positions_A"], a["cell"])
        distances = np.linalg.norm(d, axis=1)
        ids = np.flatnonzero(a["symbols"] == metal)
        rows.append({**row, "max_position_change_A": float(distances.max()),
                     "metal_position_change_A": float(distances[ids[0]]) if len(ids) == 1 else np.nan,
                     "geometry_status": "matches" if distances.max() <= tolerance_A else "position mismatch",
                     "velocities_checked": False})
    return pd.DataFrame(rows)


def load_run(cfg, workbook=None, cache_dir=None, use_cache=True):
    """Load one local run, optionally caching extracted arrays after streaming."""
    kind = cfg["kind"]
    poscar = cfg.get("poscar")
    structure = read_poscar(poscar) if poscar is not None and Path(poscar).is_file() else None
    neighbors = cfg.get("vacancy_neighbors")
    if isinstance(neighbors, str) and neighbors == "auto":
        proposals = [] if structure is None else suggest_b_vacancies(
            structure["positions_A"], structure["symbols"], structure["cell"])
        neighbors = proposals[0]["N_indices_zero_based"] if len(proposals) == 1 else None
        print(f"{cfg['label']}: {len(proposals)} candidate B-vacancy neighbor triplets; "
              f"selected {neighbors}. Inspect the proposed site in the trajectory plot.")
    if kind == "excel":
        paths = [Path(workbook)]
    elif kind == "csv":
        paths = [Path(cfg["csv"])]
    elif kind == "outcar":
        paths = [Path(p) for p in cfg["outcars"]]
        if not paths:
            raise FileNotFoundError(f"{cfg['label']}: no OUTCAR files were discovered.")
        if cfg.get("missing_outcar_runs") and not cfg.get("allow_missing_outcars", False):
            raise ValueError(f"Missing OUTCARs in runs {cfg['missing_outcar_runs']}. "
                             "Use the combined movie/CSV or resolve the missing files first.")
    elif kind == "xyz":
        paths = [Path(cfg["movie"])]
    else:
        raise ValueError("kind must be excel, csv, outcar, or xyz.")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"{cfg['label']}: edit the settings for missing local file {p}")
    fingerprint_paths = paths + ([Path(poscar)] if structure is not None else [])
    fingerprint = {"version": 2, "settings": cfg,
                   "files": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in fingerprint_paths]}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:18]
    cache = None
    if use_cache and cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(cache_dir)/(key+"_raw.npz")
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            raw = json.loads(str(z["metadata_json"]))
            raw.update({k: z[k] for k in z.files if k != "metadata_json"})
        print(f"{cfg['label']}: loaded extracted coordinate cache")
        return raw
    print(f"{cfg['label']}: reading {kind} from local storage")
    if kind == "excel":
        raw = read_excel_trajectory(workbook, cfg["sheet"], cfg["metal"])
    elif kind == "csv":
        raw = read_csv_trajectory(cfg["csv"], cfg["metal"], cfg.get("csv_column_map"))
    elif kind == "outcar":
        raw = read_outcars(cfg["outcars"], cfg["metal"], cfg.get("metal_index"),
                           cfg.get("timestep_fs"), cfg.get("frame_interval_steps", 1), neighbors)
    elif kind == "xyz":
        raw = read_xyz_movie(cfg["movie"], cfg["metal"], structure,
                            cfg.get("metal_index"), cfg.get("frame_dt_fs", 0.5), neighbors,
                            cfg.get("xyz_time_unit"))
    else:
        raise ValueError("kind must be excel, csv, outcar, or xyz.")
    raw = attach_structure(raw, structure, neighbors)
    if cache is not None:
        arrays = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
        metadata = {k: v for k, v in raw.items() if k not in arrays}
        np.savez_compressed(cache, metadata_json=json.dumps(metadata, default=str), **arrays)
    return raw


def read_outcars(paths, metal, metal_index=None, timestep_fs=None,
                 frame_interval_steps=1, vacancy_neighbors=None, progress_every=25000):
    """Stream POSITION/TOTAL-FORCE blocks and their ionic-step thermodynamics.

    paths must be in chronological order and have the same atom ordering/cell.
    Atom indices are zero-based. Repeated boundary frames are retained and flagged
    by the diagnostics, not silently deleted. Variable-cell MD is not supported.
    """
    paths = [Path(p).expanduser() for p in paths]
    times, positions, drifts, heights, temperatures, energies = [], [], [], [], [], []
    centers, normals, source_ids, ionic_steps = [], [], [], []
    cell0 = None
    symbols0 = None
    first_positions = None
    previous_full = None
    unwrapped_full = None
    substrate0 = None
    selected_index = None
    species_seen = []
    normal0 = None
    basis = None

    for source_id, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        species, counts, cell = [], None, None
        dt_fs = timestep_fs
        current_step, record = None, {}
        previous_step_file = None
        emitted_file = 0
        with text_file(path) as handle:
            def emit(rec):
                nonlocal previous_full, unwrapped_full, first_positions, substrate0
                nonlocal cell0, symbols0, normal0, basis, selected_index
                nonlocal previous_step_file, emitted_file
                if "positions" not in rec:
                    return
                if cell is None or counts is None or len(species) < len(counts):
                    raise ValueError(f"{path}: could not identify cell, species, and atom counts.")
                if dt_fs is None or dt_fs <= 0:
                    raise ValueError(f"{path}: POTIM unavailable; set timestep_fs explicitly.")
                sym = np.repeat(species[:len(counts)], counts)
                full = rec["positions"]
                if first_positions is None:
                    cell0, symbols0 = np.array(cell), np.array(sym)
                    first_positions = full.copy()
                    idx = np.flatnonzero(sym == metal)
                    if metal_index is None:
                        if len(idx) != 1:
                            raise ValueError(f"Found {len(idx)} {metal} atoms; set metal_index explicitly.")
                        selected_index = int(idx[0])
                    else:
                        selected_index = int(metal_index)
                        if not 0 <= selected_index < len(sym) or sym[selected_index] != metal:
                            raise ValueError("metal_index does not point to the requested metal.")
                    substrate = np.isin(sym, ("B", "N"))
                    if not np.any(substrate):
                        raise ValueError("No B/N substrate atoms found.")
                    normal0 = np.cross(cell0[0], cell0[1])
                    normal0 /= np.linalg.norm(normal0)
                    e1 = cell0[0] / np.linalg.norm(cell0[0])
                    basis = np.vstack([e1, np.cross(normal0, e1), normal0])
                    unwrapped_full = full.copy()
                    substrate0 = unwrapped_full[substrate].mean(axis=0)
                else:
                    if not np.array_equal(sym, symbols0) or not np.allclose(cell, cell0, atol=1e-5):
                        raise ValueError("Atom order/cell changed. Analyze these runs separately.")
                    # For small MD increments, fractional image counters remove wrapping.
                    fstep = (full - previous_full) @ np.linalg.inv(cell0)
                    fstep[:, :2] -= np.rint(fstep[:, :2])
                    unwrapped_full += fstep @ cell0
                substrate = np.isin(sym, ("B", "N"))
                sheet_center = unwrapped_full[substrate].mean(axis=0)
                drift = sheet_center - substrate0
                height = np.dot(unwrapped_full[selected_index] - sheet_center, normal0)
                step = rec.get("ionic_step")
                if not times:
                    time = 0.0
                elif emitted_file and step is not None and previous_step_file is not None:
                    delta_steps = step - previous_step_file
                    if delta_steps <= 0:
                        raise ValueError("Ionic step reset inside one OUTCAR; split it at the restart.")
                    time = times[-1] + delta_steps * dt_fs / 1000
                else:
                    time = times[-1] + frame_interval_steps * dt_fs / 1000
                times.append(time)
                positions.append(full[selected_index].copy())
                drifts.append(drift)
                heights.append(height)
                temperatures.append(rec.get("temperature", np.nan))
                energies.append(rec.get("energy", np.nan))
                source_ids.append(source_id)
                ionic_steps.append(step if step is not None else np.nan)
                if vacancy_neighbors is not None:
                    ids = np.asarray(vacancy_neighbors, int)
                    if len(ids) != 3 or np.any(symbols0[ids] != "N"):
                        raise ValueError("vacancy_neighbors must contain three zero-based N indices.")
                    tri = full[ids].copy()
                    tri[1:] = tri[0] + minimum_image(tri[1:] - tri[0], cell0)
                    center = tri.mean(axis=0)
                    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                    n /= np.linalg.norm(n)
                    if np.dot(n, normal0) < 0:
                        n = -n
                    centers.append(center)
                    normals.append(n)
                previous_full = full.copy()
                previous_step_file = step
                emitted_file += 1
                if progress_every and len(times) % progress_every == 0:
                    print(f"Read {len(times):,} frames ({time:.3f} ps)")

            for line in handle:
                match = re.search(r"VRHFIN\s*=\s*([A-Za-z]{1,2})\s*:", line)
                if match:
                    species.append(match.group(1))
                if "ions per type" in line:
                    counts = [int(v) for v in line.split("=", 1)[1].split()]
                match = re.search(r"POTIM\s*=\s*(" + FLOAT + ")", line)
                if match and timestep_fs is None:
                    dt_fs = number(match.group(1))
                match = re.search(r"IBRION\s*=\s*(-?\d+)", line)
                if match and int(match.group(1)) != 0:
                    raise ValueError(f"{path}: IBRION is not zero; this is not an MD OUTCAR.")
                if "direct lattice vectors" in line:
                    cell = np.array([[number(v) for v in next(handle).split()[:3]] for _ in range(3)])
                match = re.search(r"Iteration\s+(\d+)\s*\(\s*\d+\s*\)", line)
                if match:
                    step = int(match.group(1))
                    if current_step is not None and step != current_step:
                        emit(record)
                        record = {}
                    current_step = step
                    record["ionic_step"] = step
                match = re.search(r"free\s+energy\s+TOTEN\s*=\s*(" + FLOAT + ")", line)
                if match:
                    record["energy"] = number(match.group(1))
                match = re.search(r"temperature\s*=?\s*(" + FLOAT + r")\s*K", line, re.I)
                if match:
                    record["temperature"] = number(match.group(1))
                if "POSITION" in line and "TOTAL-FORCE" in line:
                    if counts is None:
                        raise ValueError("Cannot read positions without ions-per-type metadata.")
                    if "positions" in record:
                        emit(record)
                        record = {"ionic_step": current_step}
                    block = []
                    for coordinate_line in handle:
                        values = coordinate_line.split()
                        if not block and (not values or set(coordinate_line.strip()) == {"-"}):
                            continue
                        try:
                            row = [number(v) for v in values[:3]]
                        except ValueError:
                            break
                        if len(row) != 3:
                            break
                        block.append(row)
                        if len(block) == sum(counts):
                            break
                    if len(block) != sum(counts):
                        warnings.warn(f"Ignored incomplete final coordinate block in {path}")
                        break
                    record["positions"] = np.array(block)
            emit(record)
        if emitted_file == 0:
            raise ValueError(f"No complete POSITION blocks found in {path}.")
    if not times:
        raise ValueError("No trajectory frames read.")
    result = {"time_ps": np.array(times), "positions_A": np.array(positions),
              "substrate_drift_A": np.array(drifts), "height_A": np.array(heights),
              "temperature_K": np.array(temperatures), "energy_eV": np.array(energies),
              "cell": cell0, "basis": basis, "symbols": symbols0,
              "snapshot_A": first_positions, "metal_index": selected_index,
              "source_ids": np.array(source_ids), "ionic_steps": np.array(ionic_steps),
              "metal": metal, "source": [str(p) for p in paths]}
    if centers:
        result.update(vacancy_centers_A=np.array(centers), vacancy_normals=np.array(normals))
    return result


def suggest_b_vacancies(snapshot, symbols, cell, bond_cutoff_A=1.85):
    """Propose three undercoordinated N neighbors; inspect before accepting."""
    b = np.flatnonzero(np.asarray(symbols) == "B")
    n = np.flatnonzero(np.asarray(symbols) == "N")
    candidates = []
    for i in n:
        distance = np.linalg.norm(minimum_image(snapshot[b] - snapshot[i], cell), axis=1)
        if np.sum(distance < bond_cutoff_A) == 2:
            candidates.append(int(i))
    proposals, used = [], set()
    for i in candidates:
        near = [j for j in candidates if np.linalg.norm(minimum_image(snapshot[j] - snapshot[i], cell)) < 3.2]
        if len(near) == 3 and tuple(sorted(near)) not in used:
            ids = tuple(sorted(near))
            used.add(ids)
            tri = snapshot[list(ids)].copy()
            tri[1:] = tri[0] + minimum_image(tri[1:] - tri[0], cell)
            proposals.append({"N_indices_zero_based": ids, "center_A": tri.mean(axis=0).tolist()})
    return proposals


def prepare_trajectory(raw, cfg, jump_speed_A_ps=50.0):
    result = dict(raw)
    t = np.asarray(raw["time_ps"], float)
    p = np.asarray(raw["positions_A"], float)
    if len(t) < 3 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
        raise ValueError("Need at least three finite trajectory frames.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time is not strictly increasing; preserve/split the original restart files.")
    cell = cfg.get("cell_A")
    if cell is None:
        cell = raw.get("cell")
    if cell is not None:
        cell = np.asarray(cell, float)
        if cell.shape != (3, 3) or abs(np.linalg.det(cell)) < 1e-9:
            raise ValueError("cell_A must be three noncoplanar row vectors in Angstrom.")
    unwrapped = unwrap_positions(p, cell)
    drift = raw.get("substrate_drift_A", np.zeros_like(p))
    relative = unwrapped - drift
    basis = raw.get("basis", np.eye(3))
    projected = relative @ np.asarray(basis).T
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=1) / np.diff(t)
    breaks = np.flatnonzero(speed > jump_speed_A_ps) + 1
    # Explicit source boundaries protect fits even when a restart gap looks small.
    if "source_ids" in raw and cfg.get("split_restart_boundaries", True):
        breaks = np.union1d(breaks, np.flatnonzero(np.diff(raw["source_ids"]) != 0) + 1)
    segment = np.zeros(len(t), int)
    segment[np.asarray(breaks, int)] = 1
    segment = np.cumsum(segment)
    vacancy = cfg.get("vacancy_cart_A")
    if vacancy is None and cfg.get("vacancy_frac") is not None:
        if cell is None:
            raise ValueError("A cell is required for vacancy_frac.")
        vacancy = np.asarray(cfg["vacancy_frac"], float) @ cell
    centers = raw.get("vacancy_centers_A")
    if centers is None and vacancy is not None:
        # A fixed site follows the measured substrate translation, if available.
        centers = np.asarray(vacancy, float)[None, :] + drift
    distance = height = local_height = None
    if centers is not None:
        delta = minimum_image(unwrapped - centers, cell)
        normal = np.asarray(basis)[2]
        height_along_normal = delta @ normal
        lateral = delta - height_along_normal[:, None] * normal
        distance = np.linalg.norm(lateral, axis=1)
        if raw.get("vacancy_normals") is not None:
            local_height = np.sum(delta * raw["vacancy_normals"], axis=1)
    height = raw.get("height_A")
    if local_height is not None:
        height = local_height
    elif height is None and cfg.get("sheet_plane_z_A") is not None:
        height = p[:, 2] - float(cfg["sheet_plane_z_A"])
    result.update(label=cfg["label"], condition=cfg.get("condition", "unassigned"),
                  replicate=cfg.get("replicate", "1"), cell=cell, projected_A=projected,
                  corrected_A=relative, segment_id=segment, break_indices=np.asarray(breaks, int),
                  jump_speed_A_ps=speed, distance_vacancy_A=distance, height_A=height,
                  drift_corrected=("substrate_drift_A" in raw))
    if centers is not None:
        result["vacancy_centers_A"] = centers
    if local_height is not None:
        result["height_reference"] = "instantaneous local three-N plane"
    return result


def trajectory_table(traj):
    t = traj["time_ps"]
    df = pd.DataFrame({"time_ps": t, "segment": traj["segment_id"]})
    for j, name in enumerate(("x", "y", "z")):
        df[name + "_raw_A"] = traj["positions_A"][:, j]
        df[name + "_corrected_A"] = traj["corrected_A"][:, j]
    for key, name in [("height_A", "height_A"), ("distance_vacancy_A", "r_vacancy_xy_A"),
                      ("temperature_K", "temperature_K"), ("energy_eV", "potential_energy_eV")]:
        if traj.get(key) is not None:
            df[name] = traj[key]
    return df


def inventory(trajectories):
    return pd.DataFrame([{
        "label": r["label"], "condition": r["condition"], "metal": r["metal"],
        "frames": len(r["time_ps"]), "duration_ps": float(np.ptp(r["time_ps"])),
        "dt_fs": 1000 * float(np.median(np.diff(r["time_ps"]))),
        "segments": int(r["segment_id"][-1] + 1), "cell_known": r["cell"] is not None,
        "substrate_drift_removed": r["drift_corrected"],
        "height_available": r.get("height_A") is not None,
        "height_reference": r.get("height_reference", "instantaneous B/N plane" if r["drift_corrected"] else "unavailable"),
        "vacancy_distance_available": r.get("distance_vacancy_A") is not None,
    } for r in trajectories])


def discontinuities(traj):
    rows = []
    for i in traj["break_indices"]:
        rows.append({"label": traj["label"], "frame_after": int(i),
                     "time_before_ps": traj["time_ps"][i-1], "time_after_ps": traj["time_ps"][i],
                     "step_A": np.linalg.norm(traj["corrected_A"][i] - traj["corrected_A"][i-1]),
                     "speed_A_ps": traj["jump_speed_A_ps"][i-1]})
    return pd.DataFrame(rows)


def choose_window(traj, time_window=(None, None), segment_id=None, stride=10):
    t = traj["time_ps"]
    low = t[0] if time_window[0] is None else time_window[0]
    high = t[-1] if time_window[1] is None else time_window[1]
    valid = (t >= low) & (t <= high)
    if segment_id is None:
        ids, counts = np.unique(traj["segment_id"][valid], return_counts=True)
        if len(ids) == 0:
            raise ValueError("No frames in the analysis window.")
        segment_id = int(ids[np.argmax(counts)])
    indices = np.flatnonzero(valid & (traj["segment_id"] == segment_id))[::int(stride)]
    if len(indices) < 20:
        raise ValueError("Too few frames in the selected continuous segment/window.")
    dt = np.diff(t[indices])
    if not np.allclose(dt, np.median(dt), rtol=1e-4, atol=1e-9):
        raise ValueError("MSD needs uniformly spaced frames; analyze a uniformly sampled section.")
    return indices


def msd_fft(positions):
    """Time-origin averaged component MSD, O(N log N), without crossing gaps."""
    p = np.asarray(positions, float)
    p = p - p.mean(axis=0)
    n = len(p)
    lag = np.arange(n)
    result = np.empty_like(p)
    for j in range(p.shape[1]):
        x = p[:, j]
        prefix = np.r_[0.0, np.cumsum(x*x)]
        ac = fftconvolve(x, x[::-1], mode="full")[n-1:]
        result[:, j] = (prefix[n-lag] + prefix[n] - prefix[lag] - 2*ac) / (n-lag)
    result[0] = 0.0
    return np.maximum(result, 0.0)


def msd_analysis(traj, time_window=(None, None), segment_id=None, stride=10, max_lag_ps=10.0):
    ids = choose_window(traj, time_window, segment_id, stride)
    p = traj["projected_A"][ids]
    t = traj["time_ps"][ids]
    dt = np.median(np.diff(t))
    # At most one quarter of the selected duration; long-lag samples are sparse.
    last = min(int(max_lag_ps/dt), (len(t)-1)//4)
    msd = msd_fft(p)[:last+1]
    tau = np.arange(last+1)*dt
    result = pd.DataFrame({"lag_ps": tau, "MSD_x_A2": msd[:, 0],
                           "MSD_y_A2": msd[:, 1], "MSD_z_A2": msd[:, 2],
                           "MSD_xy_A2": msd[:, 0]+msd[:, 1], "time_origins": len(t)-np.arange(last+1)})
    info = {"label": traj["label"], "analysis_start_ps": t[0], "analysis_end_ps": t[-1],
            "segment_id": int(traj["segment_id"][ids[0]]), "sample_dt_ps": dt,
            "selected_frames": len(t)}
    return result, info, ids


def fit_diffusion(msd, lag_window_ps, trajectory=None, selected_indices=None):
    """A candidate coefficient from a user-selected stationary diffusive window.

    R2 is descriptive. Correlated MSD points do not supply independent fit errors.
    The caller must assess stationarity and normal (alpha~1) diffusion.
    """
    low, high = lag_window_ps
    selected = (msd.lag_ps >= low) & (msd.lag_ps <= high) & (msd.lag_ps > 0)
    if selected.sum() < 5 or high > msd.lag_ps.max() + 1e-9:
        raise ValueError("Choose a fit interval within the computed MSD range, with >=5 points.")
    x = msd.loc[selected, "lag_ps"].to_numpy()
    y = msd.loc[selected, "MSD_xy_A2"].to_numpy()
    fit = linregress(x, y)
    exponent = linregress(np.log(x), np.log(np.maximum(y, 1e-30))).slope
    dx = linregress(x, msd.loc[selected, "MSD_x_A2"]).slope/2
    dy = linregress(x, msd.loc[selected, "MSD_y_A2"]).slope/2
    candidate = fit.slope / 4
    checks = []
    if candidate <= 0:
        checks.append("nonpositive slope")
    if not 0.8 <= exponent <= 1.2:
        checks.append("MSD exponent outside 0.8-1.2")
    if fit.rvalue**2 < 0.95:
        checks.append("R2 below 0.95")
    if trajectory is not None and trajectory["cell"] is None:
        checks.append("periodic cell not supplied")
    if trajectory is not None and not trajectory["drift_corrected"]:
        checks.append("substrate drift not available")
    return {"D_candidate_A2_ps": candidate, "D_candidate_m2_s": candidate*1e-8,
            "D_candidate_cm2_s": candidate*1e-4, "Dx_A2_ps": dx, "Dy_A2_ps": dy,
            "fit_intercept_A2": fit.intercept, "fit_slope_A2_ps": fit.slope,
            "R2": fit.rvalue**2, "alpha_fit": exponent,
            "fit_start_lag_ps": low, "fit_end_lag_ps": high,
            "assessment": "; ".join(checks) if checks else "candidate: verify stationarity and block consistency"}


def block_diffusion_estimates(traj, indices, lag_window_ps, blocks=4):
    """Descriptive variability across nonoverlapping time blocks, not a CI."""
    rows = []
    for j, ids in enumerate(np.array_split(indices, blocks)):
        if len(ids) < 20:
            continue
        time = traj["time_ps"][ids]
        dt = np.median(np.diff(time))
        last = min((len(ids)-1)//4, int(lag_window_ps[1]/dt)+1)
        if last*dt < lag_window_ps[1]:
            continue
        components = msd_fft(traj["projected_A"][ids])[:last+1]
        frame = pd.DataFrame({"lag_ps": np.arange(last+1)*dt,
                              "MSD_x_A2": components[:, 0], "MSD_y_A2": components[:, 1],
                              "MSD_xy_A2": components[:, 0]+components[:, 1]})
        f = fit_diffusion(frame, lag_window_ps, traj)
        rows.append({"label": traj["label"], "block": j+1, "start_ps": time[0], "end_ps": time[-1], **f})
    return pd.DataFrame(rows)


def characteristic_diffusion_time(D_A2_ps, length_A):
    """2D normal-diffusion timescale L^2/(4D); not a vacancy-capture rate."""
    if D_A2_ps <= 0 or length_A <= 0:
        return np.nan
    return length_A**2 / (4*D_A2_ps)


def persistent_runs(mask, time, segments, dwell_ps):
    mask = np.asarray(mask, bool)
    boundaries = np.r_[0, np.flatnonzero((mask[1:] != mask[:-1]) | (segments[1:] != segments[:-1]))+1, len(mask)]
    return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])
            if mask[a] and time[b-1] - time[a] >= dwell_ps]


def capture_metrics(traj, radius_A=1.5, dwell_ps=0.5):
    distance = traj.get("distance_vacancy_A")
    if distance is None:
        return {"capture_status": "vacancy position not supplied"}
    t, segments = traj["time_ps"], traj["segment_id"]
    visits = persistent_runs(distance <= radius_A, t, segments, dwell_ps)
    if not visits:
        gapped = len(traj["break_indices"]) > 0
        return {"capture_status": "no entry in observed pieces; trajectory has gaps" if gapped else "no persistent entry observed",
                "capture_observed": False,
                "capture_censor_time_ps": np.nan if gapped else float(t[-1]-t[0]), "capture_radius_A": radius_A}
    first, end = visits[0]
    if first == 0:
        return {"capture_status": "already inside at first frame; entry is left-censored",
                "capture_observed": False, "first_persistent_entry_ps": np.nan,
                "capture_radius_A": radius_A}
    gap_before = bool(np.any(traj["break_indices"] <= first))
    reaches_boundary = end == len(t) or (end < len(t) and segments[end] != segments[end-1])
    inside_durations = [float(t[b-1]-t[a]) for a, b in visits]
    return {"capture_status": "entry after a data gap; timing uncertain" if gap_before else "persistent entry observed",
            "capture_observed": not gap_before,
            "first_persistent_entry_ps": float(t[first]-t[0]),
            "entry_confirmed_ps": float(t[first]-t[0]+dwell_ps),
            "capture_radius_A": radius_A, "capture_dwell_ps": dwell_ps,
            "first_residence_observed_ps": float(t[end-1]-t[first]),
            "first_residence_right_censored": reaches_boundary,
            "persistent_visits": len(visits), "inside_time_fraction": float(np.mean(distance <= radius_A)),
            "observed_visit_durations_ps": inside_durations}


def crossing_events(traj, deadband_A=0.25, dwell_ps=0.1):
    height = traj.get("height_A")
    if height is None:
        return pd.DataFrame()
    t, segments = traj["time_ps"], traj["segment_id"]
    episodes = []
    for side, mask in [(1, height >= deadband_A), (-1, height <= -deadband_A)]:
        episodes += [(a, b, side) for a, b in persistent_runs(mask, t, segments, dwell_ps)]
    episodes.sort()
    previous = None
    events = []
    for a, b, side in episodes:
        if previous is not None and previous[2] != side and segments[a] == segments[previous[0]]:
            events.append({"label": traj["label"], "time_ps": float(t[a]),
                           "direction": "above to below" if side < 0 else "below to above",
                           "r_vacancy_xy_A": float(traj["distance_vacancy_A"][a]) if traj.get("distance_vacancy_A") is not None else np.nan,
                           "deadband_A": deadband_A, "dwell_ps": dwell_ps})
        previous = (a, b, side)
    return pd.DataFrame(events)


def exponential_capture_tau(event_times_ps, event_observed):
    """Right-censored exponential MLE for independently prepared equivalent runs.

    Do not combine different metals or starting-distance groups into one rate.
    """
    t, e = np.asarray(event_times_ps, float), np.asarray(event_observed, bool)
    if len(t) < 3:
        raise ValueError("Supply at least three independent comparable trajectories; more are preferable.")
    if np.any(t < 0) or len(t) != len(e) or not np.all(np.isfinite(t)):
        raise ValueError("Supply finite nonnegative durations and matching event flags.")
    events = int(e.sum())
    return {"replicates": len(t), "capture_events": events,
            "tau_exponential_ps": float(t.sum()/events) if events else np.nan,
            "rate_per_ps": float(events/t.sum()) if t.sum() else np.nan}


def periodic_sites_in_view(points, cell, basis, xlim, ylim):
    """Project/tile real reference coordinates into an unwrapped path's view."""
    xy = np.asarray(points) @ np.asarray(basis).T
    if cell is None or not len(xy):
        return xy[:, :2]
    lattice = (np.asarray(cell)[:2] @ np.asarray(basis).T)[:, :2]
    inverse = np.linalg.inv(lattice)
    corners = np.array(list(product(xlim, ylim))) @ inverse
    fractional_sites = xy[:, :2] @ inverse
    lo = np.floor(corners.min(axis=0)-fractional_sites.max(axis=0)).astype(int)-1
    hi = np.ceil(corners.max(axis=0)-fractional_sites.min(axis=0)).astype(int)+1
    if np.prod(hi-lo+1) > 10000:
        raise ValueError("Path spans too many periodic images for a lattice background; disable SHOW_LATTICE.")
    tiles = []
    for i, j in product(range(lo[0], hi[0]+1), range(lo[1], hi[1]+1)):
        tile = xy[:, :2] + np.array([i, j]) @ lattice
        inside = ((tile[:, 0] >= xlim[0]) & (tile[:, 0] <= xlim[1])
                  & (tile[:, 1] >= ylim[0]) & (tile[:, 1] <= ylim[1]))
        tiles.append(tile[inside])
    return np.concatenate(tiles) if tiles else np.empty((0, 2))


def draw_migration_xy(trajectories, show_lattice=True, lattice_colors=None,
                      lattice_alpha=0.40, lattice_dot_size=25, cmap="viridis",
                      padding_A=2.5, line_width=None, max_plot_points=12000,
                      time_limits_ps=None, show_colorbar=False):
    colors = lattice_colors or {"B": "#249B59", "N": "#3659C9"}
    count = len(trajectories)
    fig, axes = plt.subplots(1, count, figsize=(7.6*count, 8.5), squeeze=False)
    bounds = time_limits_ps if time_limits_ps is not None else (0, max(r["time_ps"][-1] for r in trajectories))
    norm = Normalize(*bounds)
    width = plot_line_width() if line_width is None else float(line_width)
    for ax, r in zip(axes[0], trajectories):
        p, t = r["projected_A"], r["time_ps"]
        center = (p[:, :2].min(axis=0)+p[:, :2].max(axis=0))/2
        span = max(float(np.ptp(p[:, 0])), float(np.ptp(p[:, 1])))+2*padding_A
        xlim, ylim = (center[0]-span/2, center[0]+span/2), (center[1]-span/2, center[1]+span/2)
        basis = r.get("basis", np.eye(3))
        if show_lattice and r.get("snapshot_A") is not None:
            symbols = np.asarray(r["symbols"])
            for element in ("B", "N"):
                points = r["snapshot_A"][symbols == element]
                if len(points):
                    xy = periodic_sites_in_view(points, r["cell"], basis, xlim, ylim)
                    ax.scatter(xy[:, 0], xy[:, 1], s=lattice_dot_size, c=colors[element],
                               alpha=lattice_alpha, edgecolors="none", zorder=0,
                               label=element, rasterized=True)
        for sid in np.unique(r["segment_id"]):
            ids = np.flatnonzero(r["segment_id"] == sid)
            stride = max(1, int(np.ceil(len(ids)/max_plot_points)))
            ids = np.unique(np.r_[ids[::stride], ids[-1]])
            if len(ids) > 1:
                xy = p[ids, :2]
                lines = LineCollection(np.stack([xy[:-1], xy[1:]], axis=1),
                                       cmap=cmap, norm=norm, linewidth=width, zorder=3)
                lines.set_array(t[ids[:-1]])
                ax.add_collection(lines)
        ax.scatter(*p[0, :2], s=180, marker="o", facecolors="white", edgecolors="black",
                   linewidth=width, zorder=6, label="Start")
        ax.scatter(*p[-1, :2], s=130, marker="s", c=COLORS.get(r["metal"], "black"),
                   edgecolors="black", linewidth=width, zorder=7, label="End")
        if r.get("vacancy_centers_A") is not None:
            xy = periodic_sites_in_view(np.asarray(r["vacancy_centers_A"])[0:1], r["cell"], basis, xlim, ylim)
            ax.scatter(xy[:, 0], xy[:, 1], s=160, marker="x", c="#882255", linewidth=width,
                       zorder=5, label=r"$V_\mathrm{B}$")
        ax.set(xlim=xlim, ylim=ylim, xlabel=r"$x$ (Å)", ylabel=r"$y$ (Å)", title=r["label"])
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0),
               ncol=3 if count == 1 else len(labels), frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    fig.subplots_adjust(left=0.18 if count == 1 else 0.075, right=0.97,
                        bottom=0.15 if show_colorbar else 0.10, top=0.81, wspace=0.40)
    if show_colorbar:
        from matplotlib.cm import ScalarMappable
        cax = fig.add_axes([0.28, 0.025, 0.5, 0.025])
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal").set_label("Time (ps)")
    return fig



def draw_height_history(trajectories, max_plot_points=10000):
    fig, axes = plt.subplots(1, len(trajectories), figsize=(7*len(trajectories), 5.7), squeeze=False)
    for ax, r in zip(axes[0], trajectories):
        height = r.get("height_A")
        y = height if height is not None else r["projected_A"][:, 2]-r["projected_A"][0, 2]
        for sid in np.unique(r["segment_id"]):
            ids = np.flatnonzero(r["segment_id"] == sid)
            ids = ids[::max(1, int(np.ceil(len(ids)/max_plot_points)))]
            ax.plot(r["time_ps"][ids], y[ids], color=COLORS.get(r["metal"]), lw=plot_line_width())
        if height is not None:
            ax.axhline(0, color="#999999", ls="--", lw=plot_line_width())
        ax.set(xlabel="Time (ps)", ylabel="Height (Å)" if height is not None else r"$z-z(0)$ (Å)", title=r["label"])
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig



def draw_coordinate_history(trajectories, max_plot_points=10000):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for r in trajectories:
        p = r["projected_A"]-r["projected_A"][0]
        for sid in np.unique(r["segment_id"]):
            ids = np.flatnonzero(r["segment_id"] == sid)
            ids = ids[::max(1, int(np.ceil(len(ids)/max_plot_points)))]
            for k, ax in enumerate(axes):
                ax.plot(r["time_ps"][ids], p[ids, k], lw=plot_line_width(), c=COLORS.get(r["metal"]),
                        label=r["label"] if sid == r["segment_id"][0] else None)
    for ax, coordinate in zip(axes, ("x", "y", "z")):
        ax.set_ylabel(f"${coordinate}-{coordinate}(0)$ (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=len(trajectories), fontsize=plt.rcParams["axes.titlesize"])
    axes[-1].set_xlabel("Time (ps)")
    fig.tight_layout(pad=1.2)
    return fig



def draw_trajectory_overview(trajectories, font_scale=1.4, max_plot_points=6000):
    fig, axes = plt.subplots(2, len(trajectories), figsize=(4.5*len(trajectories), 7.2), squeeze=False)
    norm = Normalize(0, max(r["time_ps"][-1] for r in trajectories))
    artist = None
    for col, r in enumerate(trajectories):
        ax, az = axes[0, col], axes[1, col]
        t, p = r["time_ps"], r["projected_A"]
        if r.get("snapshot_A") is not None:
            fixed = r["snapshot_A"] @ np.asarray(r.get("basis", np.eye(3))).T
            for element, color in [("B", "#99BC90"), ("N", "#AEA0CB")]:
                ids = np.asarray(r["symbols"]) == element
                ax.scatter(fixed[ids, 0], fixed[ids, 1], s=13, c=color, alpha=0.45, zorder=0)
        for sid in np.unique(r["segment_id"]):
            ids = np.flatnonzero(r["segment_id"] == sid)
            ids = ids[::max(1, int(np.ceil(len(ids)/max_plot_points)))]
            if len(ids) > 1:
                xy = p[ids, :2]
                pieces = np.stack([xy[:-1], xy[1:]], axis=1)
                artist = LineCollection(pieces, cmap="viridis", norm=norm, linewidth=plot_line_width())
                artist.set_array(t[ids[:-1]])
                ax.add_collection(artist)
            az.plot(t[ids], (r["height_A"] if r.get("height_A") is not None else p[:, 2]-p[0, 2])[ids],
                    c=COLORS.get(r["metal"], "#444444"), lw=plot_line_width())
        ax.scatter(p[0, 0], p[0, 1], s=65, marker="o", facecolors="none", edgecolors="black", zorder=5)
        ax.scatter(p[-1, 0], p[-1, 1], s=40, marker="s", c=COLORS.get(r["metal"], "black"), zorder=5)
        if r.get("vacancy_centers_A") is not None:
            center = r["vacancy_centers_A"][0]
            # Plot the vacancy image nearest the region the corrected atom explores.
            target = r["corrected_A"].mean(axis=0)
            center = target + minimum_image(center-target, r["cell"])
            xy_center = center @ np.asarray(r.get("basis", np.eye(3))).T
            ax.scatter(xy_center[0], xy_center[1], marker="x", s=70, color="#882255", lw=plot_line_width(), zorder=7)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        # Restrict a lattice background to the actual region traversed by the atom.
        if r.get("snapshot_A") is not None:
            ax.set_xlim(p[:, 0].min()-2, p[:, 0].max()+2)
            ax.set_ylim(p[:, 1].min()-2, p[:, 1].max()+2)
        ax.set_title(r["label"])
        ax.set_xlabel(r"$x$ (Å)")
        ax.set_ylabel(r"$y$ (Å)")
        az.set_xlabel("Time (ps)")
        az.set_ylabel("Height (Å)" if r.get("height_A") is not None else r"$z-z(0)$ (Å)")
        if r.get("height_A") is not None:
            az.axhline(0, color="#888888", ls="--", lw=plot_line_width())
        for a in (ax, az):
            a.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.10, top=0.92, hspace=0.35, wspace=0.35)
    if artist is not None:
        bar = fig.colorbar(artist, cax=fig.add_axes([0.925, 0.57, 0.012, 0.30]))
        bar.set_label("Time (ps)")
    return fig


def draw_msd(results, fits=None):
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.2))
    for label, bundle in results.items():
        msd, metal = bundle[0], bundle[3]
        color = COLORS.get(metal)
        axes[0].plot(msd.lag_ps, msd.MSD_xy_A2, label=label, color=color, lw=plot_line_width())
        select = (msd.lag_ps > 0) & (msd.MSD_xy_A2 > 0)
        axes[1].loglog(msd.loc[select, "lag_ps"], msd.loc[select, "MSD_xy_A2"],
                       label=label, color=color, lw=plot_line_width())
        if fits and label in fits:
            f = fits[label]
            x = np.array([f["fit_start_lag_ps"], f["fit_end_lag_ps"]])
            axes[0].plot(x, f["fit_intercept_A2"]+x*f["fit_slope_A2_ps"],
                         "--", color=color, lw=plot_line_width())
    for ax in axes:
        ax.set_xlabel("Lag time (ps)")
        ax.set_ylabel(r"In-plane MSD (Å$^2$)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=plt.rcParams["axes.titlesize"])
    fig.tight_layout(pad=1.3, w_pad=2)
    return fig



def draw_vacancy_distances(trajectories, radius_A=1.5):
    usable = [r for r in trajectories if r.get("distance_vacancy_A") is not None]
    if not usable:
        return None
    fig, axes = plt.subplots(len(usable), 1, figsize=(12, 4.8*len(usable)), squeeze=False, sharex=True)
    for ax, r in zip(axes[:, 0], usable):
        for sid in np.unique(r["segment_id"]):
            ids = np.flatnonzero(r["segment_id"] == sid)
            ax.plot(r["time_ps"][ids], r["distance_vacancy_A"][ids], lw=plot_line_width(),
                    c=COLORS.get(r["metal"], "black"))
        ax.axhline(radius_A, c="#882255", ls="--", lw=plot_line_width())
        ax.set_ylabel(r"$r_{V_B,xy}$ (Å)")
        ax.set_title(r["label"])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Time (ps)")
    fig.tight_layout(pad=1.2)
    return fig



def draw_thermodynamics(trajectories, energy_reference="initial", reference_temperature_K=300,
                        max_plot_points=10000, temperature_ylim=None, energy_ylims=None):
    usable = [r for r in trajectories if np.isfinite(r.get("temperature_K", [np.nan])).any()
              and np.isfinite(r.get("energy_eV", [np.nan])).any()]
    if not usable:
        return None
    if energy_reference not in ("initial", "absolute"):
        raise ValueError("THERMO_ENERGY_REFERENCE must be initial or absolute.")
    tlim = temperature_ylim if temperature_ylim is not None else globals().get("TEMPERATURE_YLIM", (200, 500))
    elims = energy_ylims if energy_ylims is not None else globals().get("ENERGY_YLIMS", {"Au": (-1, 1), "Ag": (-3, 1), "Cu": (-1, 1)})
    fig, axes = plt.subplots(2, len(usable), figsize=(6.8*len(usable), 9.0), squeeze=False, sharex="col")
    for column, r in enumerate(usable):
        t, energy = r["time_ps"], np.asarray(r["energy_eV"], float).copy()
        if energy_reference == "initial":
            energy -= energy[np.flatnonzero(np.isfinite(energy))[0]]
        source_ids = r.get("source_ids", np.zeros(len(t), int))
        for sid in np.unique(source_ids):
            ids = np.flatnonzero(source_ids == sid)
            ids = ids[::max(1, int(np.ceil(len(ids)/max_plot_points)))]
            axes[0, column].plot(t[ids], r["temperature_K"][ids], lw=plot_line_width(), c=COLORS.get(r["metal"]))
            axes[1, column].plot(t[ids], energy[ids], lw=plot_line_width(), c=COLORS.get(r["metal"]))
        if reference_temperature_K is not None:
            axes[0, column].axhline(reference_temperature_K, c="#999999", ls="--", lw=plot_line_width())
        axes[0, column].set(title=r["label"], ylabel="Temperature (K)", ylim=tlim)
        axes[0, column].set_yticks([200, 300, 400, 500] if tuple(tlim) == (200, 500) else np.linspace(*tlim, 4))
        axes[1, column].set(ylabel=r"$E-E_0$ (eV)" if energy_reference == "initial" else "Energy (eV)", xlabel="Time (ps)")
        if energy_reference == "initial" and r["metal"] in elims:
            axes[1, column].set_ylim(*elims[r["metal"]])
            low, high = elims[r["metal"]]
            axes[1, column].set_yticks(np.arange(low, high+0.01, 1))
        for ax in axes[:, column]:
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.3, w_pad=2, h_pad=1.2)
    return fig



def make_xy_animation(traj, frames=300, tail_points=1000):
    """Return a notebook-ready animation; never connect across flagged gaps."""
    from matplotlib.animation import FuncAnimation
    t, p, seg = traj["time_ps"], traj["projected_A"], traj["segment_id"]
    ids = np.unique(np.linspace(0, len(t)-1, min(frames, len(t))).astype(int))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(p[:, 0].min()-0.5, p[:, 0].max()+0.5)
    ax.set_ylim(p[:, 1].min()-0.5, p[:, 1].max()+0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$ (Å)")
    ax.set_ylabel(r"$y$ (Å)")
    line, = ax.plot([], [], lw=plot_line_width(), c=COLORS.get(traj["metal"], "black"))
    dot, = ax.plot([], [], "o", c=COLORS.get(traj["metal"], "black"))
    clock = ax.text(0.03, 0.95, "", transform=ax.transAxes)
    def update(i):
        lo = max(0, i-tail_points)
        take = np.arange(lo, i+1)
        take = take[seg[take] == seg[i]]
        line.set_data(p[take, 0], p[take, 1])
        dot.set_data([p[i, 0]], [p[i, 1]])
        clock.set_text(f"{traj['label']}   {t[i]:.2f} ps")
        return line, dot, clock
    return FuncAnimation(fig, update, frames=ids, interval=40, blit=True)

def plot_line_width():
    return float(globals().get("LINE_WIDTH", 4.0))



def apply_plot_style():
    sizes = globals().get("FONT_SIZES", {"base": 20, "axes": 26, "ticks": 20, "title": 26, "legend": 20})
    scale = float(globals().get("FONT_SCALE", 1.0))
    width = plot_line_width()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": scale*sizes["base"],
        "axes.labelsize": scale*sizes["axes"], "axes.titlesize": scale*sizes["title"],
        "xtick.labelsize": scale*sizes["ticks"], "ytick.labelsize": scale*sizes["ticks"],
        "legend.fontsize": scale*sizes["legend"], "axes.linewidth": width,
        "lines.linewidth": width, "patch.linewidth": width,
        "xtick.major.width": width, "ytick.major.width": width,
        "xtick.minor.width": width, "ytick.minor.width": width,
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })



def load_saved_raw(results_dir, metal, explicit_cache=None):
    """Use the existing full-geometry cache for styling, without reading a movie."""
    paths = [Path(explicit_cache)] if explicit_cache else sorted((Path(results_dir)/"cache").glob("*_raw.npz"))
    matches = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data:
                continue
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("metal") == metal:
                matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{metal}: found {len(matches)} matching saved raw caches. "
                         "Set REPLOT_RAW_CACHE[metal] to the intended cache, or REPLOT_ONLY=False.")
    with np.load(matches[0], allow_pickle=False) as data:
        raw = json.loads(str(data["metadata_json"]))
        raw.update({key: data[key].copy() for key in data.files if key != "metadata_json"})
    print(f"{metal}: using saved coordinate arrays ({len(raw['time_ps']):,} frames)")
    return raw



def load_saved_thermo(results_dir, prefix, metal):
    path = Path(results_dir)/f"{prefix}_temperature_energy_data.csv"
    data = pd.read_csv(path)
    quantity = "unspecified energy from saved CSV"
    reports = list(Path(results_dir).glob("*_thermodynamic_sources.csv"))
    if len(reports) == 1:
        report = pd.read_csv(reports[0])
        row = report[report.metal == metal]
        if len(row) == 1:
            quantity = str(row.iloc[0].energy_quantity)
    return {"time_ps": data.time_ps.to_numpy(float),
            "temperature_K": data.temperature_K.to_numpy(float),
            "energy_eV": data.energy_eV.to_numpy(float),
            "source_ids": data.source_segment.to_numpy(int) if "source_segment" in data else np.zeros(len(data), int),
            "label": metal, "metal": metal, "source": str(path), "energy_quantity": quantity}



def draw_migration_timebar(time_limits_ps, cmap="viridis"):
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(9.5, 2.2))
    cax = fig.add_axes([0.09, 0.52, 0.84, 0.23])
    bar = fig.colorbar(ScalarMappable(norm=Normalize(*time_limits_ps), cmap=cmap),
                       cax=cax, orientation="horizontal")
    bar.locator = MaxNLocator(nbins=5)
    bar.update_ticks()
    bar.set_label("Time (ps)", labelpad=10)
    bar.outline.set_linewidth(plot_line_width())
    return fig
