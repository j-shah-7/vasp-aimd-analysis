#!/usr/bin/env python3
"""Combine saved NEB and AIMD notebook results without rereading raw VASP files.

Copy this file into mainfolder beside folder-md and folderm1m2m3m4, then run it.
The current VASP NEB extractor and raw AIMD batch outputs are supported, along
with the earlier h-BN per-metal diffusion/capture CSVs. Python's standard library
is sufficient for tables; Matplotlib is optional for the comparison plots.
"""
from pathlib import Path, PurePosixPath
from collections import defaultdict
from datetime import datetime
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys


# FIRST-RUN SETTINGS: normally leave these unchanged.
MD_FOLDER = "folder-md"
NEB_FOLDER = "folderm1m2m3m4"
MODEL_TEMPERATURE_K = 300.0
MIN_BARRIER_KBT = 3.0  # A screening heuristic, not a proof of HTST validity.
FIGURE_FORMATS = ("png", "pdf", "svg")
VERSION = "1.0"
KB_EV_K = 8.617333262145e-5
LN10 = math.log(10.0)
COLORS = {"Au": "#C49A08", "Ag": "#85898F", "Cu": "#B86C32", "Li": "#7A49A5"}
STAMP_RE = re.compile(r"\d{8}_\d{6}(?:_\d+)?")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "Combined_analysis", "NEB_AIMD_combined"}
FIT_WARNING_FLAGS = {"nonparabolic_window", "free_fit_has_endpoint_slope",
                     "broad_fraction_of_path", "fit_window_sensitive"}
CASE_FIELDS = ["neb_id", "md_id", "process_kind", "equivalent_sites", "z",
               "hop_length_A", "temperature_K"]


def numeric(value, default=None):
    try:
        result = float(str(value).replace("D", "E").replace("d", "e"))
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def portable(value):
    return str(value or "").replace("\\", "/").rstrip("/")


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "case"


def unique_id(metal, label, identity):
    digest = hashlib.sha256(str(identity).encode()).hexdigest()[:8]
    return f"{metal}_{slug(label)[-70:]}_{digest}"


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path, default=None):
    if not Path(path).is_file():
        return default
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for key in fields:
                value = clean_json(row.get(key))
                values[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            writer.writerow(values)


def scan(folder):
    found = []
    for directory, subdirs, filenames in os.walk(folder):
        subdirs[:] = [name for name in subdirs if name not in SKIP_DIRS and not name.startswith(".")]
        found.extend(Path(directory) / name for name in filenames if name.endswith((".csv", ".json")))
    return found


def relative_context(path, root, marker=None):
    parts = list(Path(path).relative_to(root).parts[:-1])
    if marker in parts:
        parts = parts[:parts.index(marker)]
    elif any(STAMP_RE.fullmatch(item) for item in parts):
        parts = parts[:next(i for i, item in enumerate(parts) if STAMP_RE.fullmatch(item))]
    return "/".join(item for item in parts if item not in {"reports", "NEB_extracted"})


def report_rank(path):
    stamps = [item for item in Path(path).parts if STAMP_RE.fullmatch(item)]
    return (stamps[-1] if stamps else "", Path(path).stat().st_mtime_ns, str(path))


def environment(value=None, composition=None):
    token = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    if token in {"vb", "bv", "bvac", "bvacancy", "bvacancycandidate"}:
        return "B_vacancy"
    if token in {"vn", "nv", "nvac", "nvacancy", "nvacancycandidate"}:
        return "N_vacancy"
    if token in {"pristine", "pristinestoichiometry"}:
        return "pristine"
    composition = composition or {}
    b, n = numeric(composition.get("B")), numeric(composition.get("N"))
    if b is not None and n is not None:
        if b == n:
            return "pristine"
        if b == n - 1:
            return "B_vacancy"
        if n == b - 1:
            return "N_vacancy"
    return "unknown"


def host_elements(composition):
    return tuple(sorted(str(k) for k, v in (composition or {}).items()
                        if k not in {"Au", "Ag", "Cu", "Li"} and (numeric(v) or 0) > 0))


def best_reports(records, inventory, all_reports=False):
    groups = defaultdict(list)
    for record in records:
        groups[record["identity"]].append(record)
    selected = []
    for group in groups.values():
        group.sort(key=lambda r: report_rank(r["source"]))
        keep = group if all_reports else group[-1:]
        for record in group:
            chosen = any(record is item for item in keep)
            inventory.append({"kind": record["kind"], "source": str(record["source"]),
                              "case_label": record["label"], "metal": record["metal"],
                              "selected": chosen, "note": "selected" if chosen else "older report of the same calculation"})
        selected.extend(keep)
    for item in selected:
        tag = str(item["source"]) if all_reports else ""
        item["id"] = unique_id(item["metal"], item["label"], repr(item["identity"]) + tag)
    return selected


def legacy_capture(folder, metal, prefix):
    preferred = folder / (prefix + "_capture_and_residence.csv")
    paths = ([preferred] if preferred.is_file() else []) + sorted(folder.glob("*_capture_and_residence.csv"))
    for path in dict.fromkeys(paths):
        for row in read_csv(path):
            if row.get("metal", row.get("label")) == metal:
                return row
    return {}


def load_md(root, files, inventory, errors, all_reports=False):
    caches, candidates = [], []
    for path in files:
        if path.name == "cache_manifest.json":
            try:
                meta = read_json(path)
                if isinstance(meta, dict) and "case_id" in meta:
                    caches.append((path.parent, meta))
            except (OSError, ValueError) as exc:
                errors.append({"source": str(path), "error": str(exc)})
    per_case = [p for p in files if p.name == "diffusion_summary.csv" or p.name.endswith("_diffusion_summary.csv")]
    aggregates = [p for p in files if p.name == "all_diffusion_summaries.csv"]
    paths = per_case + [p for p in aggregates if not any(p.parent in other.parents for other in per_case)]
    for path in aggregates:
        if path not in paths:
            inventory.append({"kind": "MD", "source": str(path), "selected": False,
                              "note": "aggregate duplicates available per-case reports"})
    for path in paths:
        try:
            rows = read_csv(path)
            buckets = defaultdict(list)
            for row in rows:
                metal = row.get("metal") or row.get("label")
                if metal not in COLORS:
                    raise ValueError(f"Unrecognized/missing metal in {path.name}: {metal}")
                case = row.get("case") or row.get("label") or metal
                buckets[(case, metal)].append(row)
            for (case, metal), windows in buckets.items():
                legacy = "window" not in windows[0]
                prefix = path.name.removesuffix("_diffusion_summary.csv")
                # The earlier notebook exports combined AND per-metal tables.
                if legacy and len(buckets) > 1:
                    individual = [p for p in paths if p.parent == path.parent and p != path
                                  and p.name.startswith(metal + "_")
                                  and len(re.findall(r"(?:^|_)(Au|Ag|Cu|Li)(?:_|$)", p.name)) == 1]
                    if individual:
                        inventory.append({"kind": "MD", "source": str(path), "metal": metal,
                                          "selected": False, "note": "combined legacy row duplicates a per-metal file"})
                        continue
                recorded_cache = portable(windows[0].get("raw_cache_folder"))
                signature = PurePosixPath(recorded_cache).name if recorded_cache else ""
                matches = [(folder, meta) for folder, meta in caches if signature and folder.name == signature
                           and meta.get("case_id") == case and meta.get("metal") == metal]
                if not matches:
                    matches = [(folder, meta) for folder, meta in caches if meta.get("case_id") == case
                               and meta.get("metal") == metal]
                cache_folder, meta = matches[0] if len(matches) == 1 else (None, {})
                context = relative_context(path, root, "AIMD_processed")
                capture_json = read_json(path.parent / "capture_result.json", {}) if not legacy else {}
                capture = legacy_capture(path.parent, metal, prefix) if legacy else capture_json.get("capture") or {}
                capture_details_available = bool(capture_json) if not legacy else bool(capture)
                if not legacy and not capture_json:
                    first = windows[0]
                    capture = {"time_ps": first.get("geometric_capture_time_ps"),
                               "observed_residence_ps": first.get("observed_residence_ps"),
                               "residence_right_censored": first.get("residence_right_censored")}
                env = environment(windows[0].get("environment") or capture.get("condition"), meta.get("composition"))
                original = portable(meta.get("case_folder"))
                if original:
                    identity_base = original
                elif recorded_cache and "/AIMD_processed/" in recorded_cache:
                    identity_base = recorded_cache.split("/AIMD_processed/", 1)[0] + "/" + case
                else:
                    identity_base = context + "/" + (prefix if legacy and len(buckets) == 1 else case)
                label = str(meta.get("relative") or (prefix if legacy else case))
                if context and context not in label:
                    label = context + "/" + label
                candidates.append({"kind": "MD", "source": path, "metal": metal, "label": label,
                    "identity": (identity_base, metal, env), "environment": env,
                    "composition": meta.get("composition", {}), "metadata": meta,
                    "cache_folder": cache_folder, "windows": windows, "legacy": legacy,
                    "capture": capture, "capture_definition": capture_json, "case": case,
                    "capture_details_available": capture_details_available,
                    "curve_prefix": prefix, "settings": read_json(path.parent / "analysis_settings.json", {})})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"source": str(path), "error": str(exc)})
    return best_reports(candidates, inventory, all_reports)


def load_neb(root, files, inventory, errors, all_reports=False):
    candidates = []
    for path in files:
        if path.name != "neb_summary.json":
            continue
        try:
            summary = read_json(path)
            metal = summary.get("metal")
            if metal not in COLORS:
                raise ValueError("NEB metal is missing or unsupported; resolve metal selection in the NEB extractor first.")
            origin = portable(summary.get("neb_folder"))
            context = relative_context(path, root, "NEB_extracted")
            label = context or "/".join(PurePosixPath(origin).parts[-2:]) or path.parent.name
            env = environment(composition=summary.get("composition"))
            energy = summary.get("energy_type", "unknown")
            images = read_csv(path.parent / "neb_images.csv") if (path.parent / "neb_images.csv").is_file() else []
            fits = read_csv(path.parent / "neb_local_frequencies.csv") if (path.parent / "neb_local_frequencies.csv").is_file() else []
            displacement, vertical = None, None
            energies = [numeric(row.get("selected_energy_eV")) for row in images]
            energy_resolution = numeric(summary.get("energy_tolerance_for_curvature_eV"), 0.001)
            multiple_basins = any(energies[i] is not None and energies[i - 1] is not None
                                  and energies[i + 1] is not None
                                  and energies[i] < min(energies[i - 1], energies[i + 1]) - energy_resolution
                                  for i in range(1, len(energies) - 1))
            if len(images) >= 2:
                coordinates = [[numeric(row.get("metal_" + axis + "_A")) for axis in "xyz"]
                               for row in (images[0], images[-1])]
                if all(value is not None for xyz in coordinates for value in xyz):
                    displacement = math.hypot(coordinates[1][0] - coordinates[0][0],
                                              coordinates[1][1] - coordinates[0][1])
                    vertical = abs(coordinates[1][2] - coordinates[0][2])
            candidates.append({"kind": "NEB", "source": path, "metal": metal, "label": label,
                "identity": (origin or context, metal, energy, env), "environment": env,
                "composition": summary.get("composition", {}), "summary": summary,
                "images": images, "fits": fits, "hop_xy_A": displacement, "vertical_A": vertical,
                "multiple_basins": multiple_basins})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"source": str(path), "error": str(exc)})
    return best_reports(candidates, inventory, all_reports)


def compatible(md, neb):
    if md["metal"] != neb["metal"]:
        return False, "different metals"
    if md["environment"] != "unknown" and neb["environment"] != "unknown" and md["environment"] != neb["environment"]:
        return False, "different pristine/vacancy environments"
    a, b = host_elements(md["composition"]), host_elements(neb["composition"])
    if a and b and a != b:
        return False, "different substrate elements"
    if md["environment"] == "unknown" or neb["environment"] == "unknown":
        return True, "same metal; environment partly unknown; pathway matching needs inspection"
    return True, "same metal and environment; charge, cell, initial state and event matching still need inspection"


def case_settings(path, neb_cases, md_cases):
    settings = {}
    if path is None:
        return settings
    neb_ids, md_ids = {x["id"] for x in neb_cases}, {x["id"] for x in md_cases}
    for row in read_csv(path):
        identifier = row.get("neb_id", "").strip()
        if not identifier:
            continue
        if identifier not in neb_ids:
            raise ValueError(f"Unknown neb_id in case settings: {identifier}. Use the new inventory/template.")
        md_id = row.get("md_id", "").strip()
        if md_id and md_id not in md_ids:
            raise ValueError(f"Unknown md_id in case settings: {md_id}")
        kind = row.get("process_kind", "unspecified").strip() or "unspecified"
        if kind not in {"unspecified", "surface_hop", "capture", "escape", "vertical"}:
            raise ValueError(f"Unknown process_kind: {kind}")
        if (identifier, md_id) in settings:
            raise ValueError(f"Duplicate case-settings row: {identifier}, {md_id}")
        clean = dict(row, process_kind=kind, md_id=md_id,
                     equivalent_sites=boolean(row.get("equivalent_sites")))
        for key in ("z", "hop_length_A", "temperature_K"):
            raw = row.get(key, "").strip()
            clean[key] = numeric(raw)
            if raw and (clean[key] is None or clean[key] <= 0):
                raise ValueError(f"{key} must be positive and finite for {identifier}.")
        if clean["z"] is not None and int(clean["z"]) != clean["z"]:
            raise ValueError("z must be an integer number of equivalent hop directions.")
        settings[(identifier, md_id)] = clean
    return settings


def settings_for(settings, neb_id, md_id):
    return settings.get((neb_id, md_id), settings.get((neb_id, ""), {}))


def exp10_safe(log_value):
    if log_value is None or not -300 <= log_value <= 300:
        return None
    return 10.0 ** log_value


def model_rates(neb, md, cfg, temperature, min_barrier_kbt):
    temperature = cfg.get("temperature_K") or temperature
    results = []
    summary = neb["summary"]
    for endpoint, direction, field in (("initial", "forward", "sampled_forward_energy_rise_eV"),
                                        ("final", "reverse", "sampled_reverse_energy_rise_eV")):
        fits = [row for row in neb["fits"] if row.get("endpoint") == endpoint] or [{}]
        for fit in fits:
            barrier, nu = numeric(summary.get(field)), numeric(fit.get("frequency_THz"))
            flags = set(item.strip() for item in str(fit.get("flags", "")).split(";") if item.strip())
            checks = []
            if not boolean(summary.get("complete_energy_profile")):
                checks.append("incomplete_energy_profile")
            if not boolean(summary.get("maximum_is_interior")):
                checks.append("maximum_is_endpoint_or_unresolved_saddle")
            if neb.get("multiple_basins"):
                checks.append("multiple_energy_basins_on_path")
            if barrier is None or barrier < 0:
                checks.append("barrier_unavailable")
            ratio = barrier / (KB_EV_K * temperature) if barrier is not None else None
            if ratio is not None and ratio < min_barrier_kbt:
                checks.append("low_barrier_relative_to_kBT")
            if nu is None or nu <= 0:
                checks.append("frequency_unavailable")
            if fit.get("status") != "exploratory_1d_only":
                checks.append("frequency_fit_not_available")
            checks.extend(sorted(flags & FIT_WARNING_FLAGS))
            log_tau = (barrier / (KB_EV_K * temperature) / LN10 - math.log10(nu)
                       if nu is not None and nu > 0 and barrier is not None and barrier >= 0 else None)
            passed = not checks
            row = {"md_id": md["id"], "neb_id": neb["id"], "metal": neb["metal"],
                   "neb_path": neb["label"], "direction": direction, "frequency_endpoint": endpoint,
                   "temperature_K": temperature, "barrier_eV": barrier, "barrier_over_kBT": ratio,
                   "frequency_THz": nu, "fit_n_images": fit.get("n_images"),
                   "fit_status": fit.get("status", "unavailable"), "fit_flags": fit.get("flags", ""),
                   "formal_rate_ps_inv": exp10_safe(-log_tau) if log_tau is not None else None,
                   "formal_single_channel_tau_ps": exp10_safe(log_tau),
                   "formal_log10_tau_ps": log_tau,
                   "activated_screen_pass": passed,
                   "model_status": "approximate_1d_review_required" if passed else "formal_only_or_unavailable",
                   "model_screen_flags": "; ".join(checks),
                   "rate_interpretation": "single pathway, 1D prefactor, transmission coefficient assumed one; not a measured event rate"}
            d_flags = []
            if direction != "forward":
                d_flags.append("D_uses_selected_forward_surface_hop_only")
            if cfg.get("process_kind") != "surface_hop" or not cfg.get("equivalent_sites"):
                d_flags.append("equivalent_surface_hop_not_declared")
            delta = numeric(summary.get("endpoint_energy_change_eV"))
            if delta is not None and abs(delta) > 0.01:
                d_flags.append("endpoint_energies_differ_by_more_than_0.01_eV_equivalence_screen")
            z = cfg.get("z")
            if z is None:
                d_flags.append("number_of_equivalent_directions_not_supplied")
            hop = cfg.get("hop_length_A") or neb["hop_xy_A"]
            if hop is None or hop <= 0:
                d_flags.append("lateral_hop_displacement_unavailable")
            if neb["vertical_A"] is not None and neb["vertical_A"] > 0.5:
                d_flags.append("endpoint_vertical_displacement_exceeds_0.5_A_screen")
            if not passed:
                d_flags.append("activated_rate_screen_failed")
            d_log = (math.log10(z * hop * hop / 4) - log_tau if not d_flags else None)
            row.update(process_kind=cfg.get("process_kind", "unspecified"), equivalent_sites=cfg.get("equivalent_sites", False),
                       equivalent_directions=z, hop_displacement_xy_A=hop,
                       hop_displacement_source="explicit setting" if cfg.get("hop_length_A") else "NEB endpoint xy displacement",
                       D_hop_model_A2_per_ps=exp10_safe(d_log),
                       D_hop_model_m2_per_s=exp10_safe(d_log - 8) if d_log is not None else None,
                       D_hop_status="conditional_uncorrelated_equivalent_hop_model" if not d_flags else "; ".join(d_flags))
            results.append(row)
    return results


def capture_fields(md):
    capture, legacy = md["capture"], md["legacy"]
    if legacy:
        time = numeric(capture.get("first_persistent_entry_ps"))
        residence = numeric(capture.get("first_residence_observed_ps"))
        censor_value = capture.get("first_residence_right_censored")
        censored = boolean(censor_value) if residence is not None and censor_value not in (None, "") else None
        status = capture.get("capture_status", "capture file unavailable")
        observed = boolean(capture.get("capture_observed"))
        absolute = None
    else:
        absolute = numeric(capture.get("time_ps"))
        first = numeric(md["metadata"].get("first_time_ps"))
        if first is None:
            first = min((v for row in md["windows"] if (v := numeric(row.get("first_saved_time_ps"))) is not None), default=None)
        residence = numeric(capture.get("observed_residence_ps"))
        censor_value = capture.get("residence_right_censored")
        censored = boolean(censor_value) if residence is not None and censor_value not in (None, "") else None
        index = numeric(capture.get("index"))
        if absolute is None:
            status, observed, time = "no persistent entry available", False, None
        elif index == 0 or first is not None and abs(absolute - first) < 1e-9:
            status, observed, time = "already inside at first saved frame; entry is left-censored", False, None
        elif not md.get("capture_details_available", True):
            status, observed, time = "summary-only geometric entry; continuity details unavailable", None, None
        elif numeric(capture.get("segment_id"), 0) > 0:
            status, observed, time = "entry after a continuity break; arrival timing uncertain", False, None
        else:
            status, observed = "persistent geometric entry observed", True
            time = absolute - first if first is not None else None
    return {"geometric_entry_status": status, "entry_observed_without_prior_gap": observed,
            "geometric_entry_absolute_ps": absolute, "entry_delay_from_first_saved_frame_ps": time,
            "first_residence_observed_ps": residence, "residence_right_censored": censored,
            "residence_lower_bound_ps": residence if censored else None,
            "capture_definition": md["capture_definition"].get("definition", "earlier h-BN geometric-distance criterion" if legacy else "geometric entry"),
            "capture_radius_A": md["capture_definition"].get("radius_A", capture.get("capture_radius_A")),
            "capture_height_A": md["capture_definition"].get("height_A"),
            "capture_dwell_ps": md["capture_definition"].get("dwell_ps", capture.get("capture_dwell_ps"))}


def md_window(md, source):
    legacy = md["legacy"]
    return {"md_id": md["id"], "metal": md["metal"], "md_case": md["label"],
            "md_environment": md["environment"], "md_window": source.get("window", "selected_legacy"),
            "first_time_ps": numeric(source.get("analysis_start_ps" if legacy else "first_saved_time_ps")),
            "last_time_ps": numeric(source.get("analysis_end_ps" if legacy else "last_saved_time_ps")),
            "frames": numeric(source.get("selected_frames" if legacy else "frames_in_window")),
            "D_MD_apparent_A2_per_ps": numeric(source.get("D_candidate_A2_ps" if legacy else "D_apparent_A2_per_ps")),
            "D_MD_apparent_m2_per_s": numeric(source.get("D_candidate_m2_s" if legacy else "D_apparent_m2_per_s")),
            "MD_fit_status": source.get("assessment" if legacy else "fit_status", ""),
            "MD_linear_screen_pass": (str(source.get("assessment", "")).startswith("candidate:") if legacy else boolean(source.get("linear_screen_pass"))),
            "MD_alpha_fit": numeric(source.get("alpha_fit")), "MD_fit_R2": numeric(source.get("R2")),
            "MD_rms_lateral_extent_A": numeric(source.get("rms_lateral_extent_longest_segment_A")),
            "MD_continuous_segments": numeric(source.get("continuous_segments_used")),
            "MD_report": str(md["source"]), **capture_fields(md)}


def value_range(rows, key):
    values = [value for row in rows if (value := numeric(row.get(key))) is not None]
    return (min(values), max(values)) if values else (None, None)


def comparison_rows(md, neb, rates, basis, cfg):
    summary = neb["summary"]
    base = {"neb_id": neb["id"], "neb_path": neb["label"], "neb_environment": neb["environment"],
            "NEB_energy_type": summary.get("energy_type"),
            "NEB_forward_energy_rise_eV": numeric(summary.get("sampled_forward_energy_rise_eV")),
            "NEB_reverse_energy_rise_eV": numeric(summary.get("sampled_reverse_energy_rise_eV")),
            "NEB_final_minus_initial_eV": numeric(summary.get("endpoint_energy_change_eV")),
            "NEB_complete_profile": summary.get("complete_energy_profile"),
            "NEB_maximum_is_interior": summary.get("maximum_is_interior"),
            "NEB_endpoint_displacement_xy_A": neb["hop_xy_A"],
            "NEB_endpoint_displacement_z_A": neb["vertical_A"],
            "model_temperature_K": rates[0]["temperature_K"],
            "pairing_basis": basis, "process_kind": cfg.get("process_kind", "unspecified"),
            "NEB_report": str(neb["source"]), "NEB_warnings": summary.get("warnings", [])}
    for direction in ("forward", "reverse"):
        subset = [row for row in rates if row["direction"] == direction]
        for source, label in (("frequency_THz", "frequency_THz"),
                              ("formal_rate_ps_inv", "formal_rate_ps_inv"),
                              ("formal_single_channel_tau_ps", "formal_tau_ps"),
                              ("D_hop_model_A2_per_ps", "D_hop_model_A2_per_ps")):
            low, high = value_range(subset, source)
            base[f"{direction}_{label}_min"], base[f"{direction}_{label}_max"] = low, high
        base[f"{direction}_any_activated_screen_pass"] = any(row["activated_screen_pass"] for row in subset)
        base[f"{direction}_all_fits_activated_screen_pass"] = bool(subset) and all(row["activated_screen_pass"] for row in subset)
        base[f"{direction}_screen_notes"] = " | ".join(sorted({row["model_screen_flags"] for row in subset if row["model_screen_flags"]}))
    # Explicit provenance: model time and observed first-arrival time are different observables.
    base["comparison_interpretation"] = "side-by-side context; no automatic validation of rates, D, or device switching time"
    return [{**md_window(md, row), **base} for row in md["windows"]]


def load_msd_curves(md_cases):
    curves = []
    for md in md_cases:
        folder = md["source"].parent
        for window in md["windows"]:
            name = window.get("window", "selected_legacy")
            path = (folder / (md["curve_prefix"] + "_MSD_data.csv") if md["legacy"] else folder / f"msd_{name}.csv")
            if not path.is_file() and md["legacy"]:
                matches = [p for p in folder.glob(md["metal"] + "_*MSD_data.csv") if "post_capture" not in p.name]
                path = matches[0] if len(matches) == 1 else path
            if path.is_file():
                rows = read_csv(path)
                xy = [(numeric(row.get("lag_ps")), numeric(row.get("msd_xy_A2", row.get("MSD_xy_A2")))) for row in rows]
                xy = [(t, v) for t, v in xy if t is not None and v is not None]
                if xy:
                    curves.append({"md": md, "window": name, "path": path, "xy": xy})
    return curves


def make_plots(output, md_cases, neb_cases, rates, formats):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Tables saved. Install matplotlib for plots: python -m pip install matplotlib")
        return []
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 18, "axes.titlesize": 18,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "lines.linewidth": 2.5})
    written = []

    def save(fig, name):
        for extension in formats:
            path = output / (name + "." + extension)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            written.append(str(path))
        plt.close(fig)

    metals = [metal for metal in COLORS if any(case["metal"] == metal for case in neb_cases)]
    if metals:
        fig, axes = plt.subplots(1, len(metals), figsize=(6 * len(metals), 5), squeeze=False)
        for ax, metal in zip(axes[0], metals):
            for i, neb in enumerate(case for case in neb_cases if case["metal"] == metal):
                # Break the line at missing image indices; no interpolation across missing data.
                points = []
                for row in neb["images"]:
                    index, energy = numeric(row.get("image_index")), numeric(row.get("relative_to_initial_eV"))
                    if index is not None:
                        points.append((index, energy if energy is not None else float("nan")))
                if not points:
                    continue
                span = max(x for x, _ in points) - min(x for x, _ in points)
                if span <= 0:
                    continue
                x, y, previous = [], [], None
                for index, energy in points:
                    if previous is not None and index - previous > 1:
                        x.append(float("nan")); y.append(float("nan"))
                    x.append((index - points[0][0]) / span); y.append(energy); previous = index
                ax.plot(x, y, marker="o", ms=4, color=COLORS[metal], ls=("-", "--", ":", "-.")[i % 4],
                        alpha=max(.4, 1 - .08 * i), label=neb["label"][-55:] + " / " + str(neb["summary"].get("energy_type")))
            ax.set(xlabel="Fraction of image-index range", ylabel="Energy relative to initial (eV)", title=metal)
            if ax.lines:
                ax.legend(fontsize=9)
        fig.tight_layout(); save(fig, "NEB_profile_comparison")

    curves = load_msd_curves(md_cases)
    for window in sorted({curve["window"] for curve in curves}):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, curve in enumerate(c for c in curves if c["window"] == window):
            time, values = zip(*curve["xy"])
            ax.plot(time, values, color=COLORS[curve["md"]["metal"]], ls=("-", "--", ":", "-.")[i % 4],
                    label=curve["md"]["label"][-65:])
        ax.set(xlabel="Lag time (ps)", ylabel="MSD$_{xy}$ (Å$^2$)", title=window.replace("_", " ") + ": exported AIMD MSD")
        ax.legend(fontsize=10); fig.tight_layout(); save(fig, "AIMD_MSD_" + slug(window))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for i, md in enumerate(md_cases):
        capture = capture_fields(md)
        entry, residence = capture["entry_delay_from_first_saved_frame_ps"], capture["first_residence_observed_ps"]
        color = COLORS[md["metal"]]
        if entry is not None and capture["entry_observed_without_prior_gap"]:
            axes[0].barh(i, entry, color=color)
        else:
            axes[0].text(0, i, "entry unavailable / censored", va="center", fontsize=9)
        if residence is not None:
            axes[1].barh(i, residence, color=color)
            if capture["residence_right_censored"]:
                axes[1].annotate("≥", (residence, i), xytext=(4, -4), textcoords="offset points", fontsize=18)
    for ax in axes:
        ax.set_yticks(range(len(md_cases)), [md["label"][-50:] for md in md_cases])
    axes[0].set(xlabel="Observed entry delay (ps)", title="First persistent geometric entry")
    axes[1].set(xlabel="Observed first residence (ps)", title="Residence: ≥ marks a lower bound")
    fig.tight_layout(); save(fig, "AIMD_capture_and_residence")

    groups = defaultdict(list)
    for row in rates:
        if row["formal_log10_tau_ps"] is not None:
            groups[(row["neb_id"], row["direction"], row["temperature_K"])].append(row)
    if groups:
        fig, ax = plt.subplots(figsize=(11, max(5, .5 * len(groups) + 2)))
        labels = []
        for i, group in enumerate(groups.values()):
            low, high = value_range(group, "formal_log10_tau_ps")
            middle = (low + high) / 2
            passed = all(row["activated_screen_pass"] for row in group)
            ax.errorbar(middle, i, xerr=[[middle - low], [high - middle]], fmt="o" if passed else "x",
                        color=COLORS[group[0]["metal"]] if passed else "gray", capsize=4)
            labels.append(f"{group[0]['metal']} {group[0]['neb_path'][-40:]} {group[0]['direction']} ({group[0]['temperature_K']:g} K)")
        ax.set_yticks(range(len(labels)), labels, fontsize=10)
        ax.set(xlabel="log$_{10}$(formal single-channel waiting time / ps)",
               title="1D model: fit-window ranges; gray × fails the screening checks")
        fig.tight_layout(); save(fig, "NEB_formal_waiting_times")
    return written


METHODS = r"""# Combined NEB and AIMD analysis

This program reads existing notebook CSV/JSON exports. It does not read raw
OUTCAR, XDATCAR, XYZ movies, trajectory NPZ arrays or rerun the original analysis.

## Discovery and pairing

The latest report is selected separately for each MD case and NEB pathway;
timestamp directory names are preferred over copy-modified times. Different
NEB energy conventions remain separate. --all-reports retains every report.
The inventory records selected/older/duplicate reports and excluded pairings.
Copied data can be used: absolute source paths are provenance, not instructions
to read data outside the selected input folders. Cache manifests are optional.

Metals must agree. Known pristine/B-vacancy/N-vacancy environments and substrate
elements must agree. Compatible cases are compared without combining their
starting distances or their MD time windows. Same-metal pairing is context,
not verification that NEB and MD describe the same event, charge or geometry.
Every NEB path is independent: M1-M2 and M2-M3 outputs are not stitched into a
common-energy multistep profile or a diffusion network.

## Rates and diffusion

Formal 1D model: r[ps^-1] = nu[THz] exp(-Ea/(kB*T)); tau[ps] = 1/r. 1 THz is
one inverse ps. Initial-state frequency is used for the forward path; the final
state frequency is used for the reverse path. Every fitted frequency window is
retained. Ranges describe fit-window sensitivity, not confidence intervals.
Log waiting times remain available when ordinary numbers overflow/underflow.

The numeric columns are explicitly FORMAL quantities. A screen flags missing
data, endpoint maxima/unresolved saddles, multiple internal energy basins,
barriers <3 kBT by default, and the
NEB extractor's broad/nonparabolic/slope/window-sensitivity flags. These are
heuristics. Passing the screen does not establish convergence, metastability,
transmission coefficient, harmonicity or accuracy of a prefactor. The source
frequencies are constrained 1D estimates, not full Vineyard prefactors. Near-
barrierless/downhill capture especially requires inspection of the actual MD.

D_hop = z*a^2*r/4 is exported only when optional case settings explicitly declare
a surface_hop, equivalent_sites=true, positive integer z, a positive endpoint
hop displacement, and the forward rate passes screening. Additional heuristic
screens reject endpoint energy differences >0.01 eV and endpoint vertical
displacements >0.5 A. These checks cannot establish site equivalence by themselves.
The model assumes an xy-plane, uncorrelated equivalent hops and no net drift.
It does not apply directly to capture at an isolated trap or a multistep path.
a is endpoint lateral displacement, never cumulative NEB arc length. A supplied
hop_length_A overrides this geometric value. 1 A^2/ps = 1e-8 m^2/s.

MD apparent D, alpha, R2 and fit-status fields are copied, not recomputed or
upgraded into validated material diffusion coefficients. A post-binding MSD
plateau describes confinement. No averaging of NEB and MD D is performed.
First geometric entry is not a mean capture time; first-frame entry is left-
censored and entry after a continuity break is marked uncertain. Residence
without escape is right-censored, with the observed span reported as a lower
bound. Geometric capture criteria can differ between original workflows.

The model temperature is an explicit setting (default 300 K), not an inference
from one instantaneous MD temperature. A calculated single-hop/escape waiting
time is not the first-passage travel time from 6nn/7nn or an electrical-device
switching/retention time. Match lateral barriers to lateral dynamics and vertical
barriers to through-plane events. Confirm compatible charge/cell/DFT conditions.

## References — ready to copy

1. Ratsch, C.; Scheffler, M. Density-functional theory calculations of hopping rates of surface diffusion. Physical Review B 58, 13163-13166 (1998). https://doi.org/10.1103/PhysRevB.58.13163
2. Vineyard, G. H. Frequency factors and isotope effects in solid state rate processes. Journal of Physics and Chemistry of Solids 3, 121-127 (1957). https://doi.org/10.1016/0022-3697(57)90059-8
3. McCluskey, A. R.; Coles, S. W.; Morgan, B. J. Accurate Estimation of Diffusion Coefficients and their Uncertainties from Computer Simulation. Journal of Chemical Theory and Computation 21, 79-87 (2025). https://doi.org/10.1021/acs.jctc.4c01249
4. Synopsys QuantumATK. HTSTEvent documentation. https://docs.quantumatk.com/manual/Types/HTSTEvent/HTSTEvent.html
"""


SUMMARY_FIELDS = ["metal", "md_case", "md_environment", "md_window", "neb_path", "NEB_energy_type",
    "NEB_forward_energy_rise_eV", "NEB_reverse_energy_rise_eV", "NEB_final_minus_initial_eV",
    "forward_frequency_THz_min", "forward_frequency_THz_max", "reverse_frequency_THz_min", "reverse_frequency_THz_max",
    "model_temperature_K", "forward_formal_tau_ps_min", "forward_formal_tau_ps_max",
    "reverse_formal_tau_ps_min", "reverse_formal_tau_ps_max",
    "forward_all_fits_activated_screen_pass", "reverse_all_fits_activated_screen_pass",
    "D_MD_apparent_A2_per_ps", "MD_fit_status", "MD_alpha_fit", "MD_fit_R2",
    "forward_D_hop_model_A2_per_ps_min", "forward_D_hop_model_A2_per_ps_max",
    "entry_delay_from_first_saved_frame_ps", "geometric_entry_status",
    "first_residence_observed_ps", "residence_right_censored", "residence_lower_bound_ps",
    "MD_rms_lateral_extent_A", "first_time_ps", "last_time_ps", "process_kind", "pairing_basis",
    "forward_screen_notes", "reverse_screen_notes", "md_id", "neb_id", "MD_report", "NEB_report"]


def resolve_input(root, requested, aliases):
    path = Path(requested).expanduser()
    path = path if path.is_absolute() else root / path
    if path.is_dir():
        return path.resolve()
    if requested in {MD_FOLDER, NEB_FOLDER}:
        matches = [root / name for name in aliases if (root / name).is_dir()]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(f"Input folder does not exist: {path}. Edit the folder setting or use --md-folder / --neb-folder.")


def combine(root, md_folder=MD_FOLDER, neb_folder=NEB_FOLDER, temperature_K=MODEL_TEMPERATURE_K,
            settings_path=None, min_barrier_kbt=MIN_BARRIER_KBT, all_reports=False,
            plots=True, formats=FIGURE_FORMATS):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Main folder does not exist: {root}")
    if not math.isfinite(temperature_K) or temperature_K <= 0 or not math.isfinite(min_barrier_kbt) or min_barrier_kbt < 0:
        raise ValueError("Temperature must be positive; the barrier/kBT screen must be nonnegative.")
    md_root = resolve_input(root, md_folder, ("folder-md", "folder_md"))
    neb_root = resolve_input(root, neb_folder, ("folderm1m2m3m4", "folder-m1m2m3m4", "folder_m1m2m3m4"))
    if md_root == neb_root:
        raise ValueError("Choose separate MD and NEB result folders.")
    output = root / "Combined_analysis" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    (output / "METHODS_AND_REFERENCES.md").write_text(METHODS, encoding="utf-8")
    inventory, errors = [], []
    print(f"MD results: {md_root}\nNEB results: {neb_root}\nModel temperature: {temperature_K:g} K", flush=True)
    md_cases = load_md(md_root, scan(md_root), inventory, errors, all_reports)
    neb_cases = load_neb(neb_root, scan(neb_root), inventory, errors, all_reports)
    write_csv(output / "input_errors.csv", errors, ["source", "error"])
    if not md_cases or not neb_cases:
        write_csv(output / "input_inventory.csv", inventory)
        missing = "MD diffusion_summary.csv/all_diffusion_summaries.csv" if not md_cases else "NEB neb_summary.json"
        raise ValueError(f"No usable {missing} found. Copy the saved notebook result folders, including CSV/JSON files. Diagnostics: {output}")
    settings = case_settings(settings_path, neb_cases, md_cases)
    comparisons, rates, matched = [], [], set()
    for neb in neb_cases:
        explicit_md_ids = {md_id for (neb_id, md_id) in settings if neb_id == neb["id"] and md_id}
        for md in md_cases:
            if explicit_md_ids and md["id"] not in explicit_md_ids:
                continue
            allowed, basis = compatible(md, neb)
            if not allowed:
                if md["metal"] == neb["metal"]:
                    inventory.append({"kind": "pair", "source": str(neb["source"]), "case_label": md["label"],
                                      "metal": md["metal"], "selected": False, "note": basis})
                if md["id"] in explicit_md_ids:
                    raise ValueError(f"Explicit pairing is incompatible: {neb['id']} / {md['id']}: {basis}")
                continue
            cfg = settings_for(settings, neb["id"], md["id"])
            pair_rates = model_rates(neb, md, cfg, temperature_K, min_barrier_kbt)
            rates.extend(pair_rates)
            comparisons.extend(comparison_rows(md, neb, pair_rates, basis, cfg))
            matched.add((md["id"], neb["id"]))
    md_rows = [md_window(md, row) for md in md_cases for row in md["windows"]]
    write_csv(output / "comparison_summary.csv", comparisons, SUMMARY_FIELDS)
    write_csv(output / "comparison_details.csv", comparisons)
    write_csv(output / "model_rates_by_fit.csv", rates)
    write_csv(output / "aimd_windows_all.csv", md_rows)
    write_csv(output / "neb_profiles_all.csv", [{"neb_id": neb["id"], "neb_path": neb["label"], "metal": neb["metal"], **row}
                                               for neb in neb_cases for row in neb["images"]])
    write_csv(output / "neb_frequencies_all.csv", [{"neb_id": neb["id"], "neb_path": neb["label"], "metal": neb["metal"], **row}
                                                  for neb in neb_cases for row in neb["fits"]])
    write_csv(output / "case_settings_template.csv", [{"neb_id": neb["id"], "md_id": "", "process_kind": "unspecified",
               "equivalent_sites": False, "z": "", "hop_length_A": "", "temperature_K": temperature_K} for neb in neb_cases], CASE_FIELDS)
    write_csv(output / "selected_cases.csv", [{"kind": "MD", "id": md["id"], "metal": md["metal"], "label": md["label"],
               "environment": md["environment"], "source": str(md["source"])} for md in md_cases] +
              [{"kind": "NEB", "id": neb["id"], "metal": neb["metal"], "label": neb["label"],
                "environment": neb["environment"], "source": str(neb["source"])} for neb in neb_cases])
    for cases, index in ((md_cases, 0), (neb_cases, 1)):
        for case in cases:
            if not any(pair[index] == case["id"] for pair in matched):
                inventory.append({"kind": case["kind"], "source": str(case["source"]), "case_label": case["label"],
                                  "metal": case["metal"], "selected": True, "note": "no compatible opposite-workflow case; source still exported"})
    write_csv(output / "input_inventory.csv", inventory)
    figures = make_plots(output, md_cases, neb_cases, rates, formats) if plots else []
    manifest = {"version": VERSION, "main_folder": str(root), "md_root": str(md_root), "neb_root": str(neb_root),
                "selected_MD_cases": len(md_cases), "selected_NEB_paths": len(neb_cases),
                "comparison_pairs": len(matched), "comparison_rows": len(comparisons), "input_errors": errors,
                "model_temperature_K": temperature_K, "min_barrier_kBT_screen": min_barrier_kbt,
                "all_reports": all_reports, "case_settings": str(settings_path) if settings_path else None,
                "output_folder": str(output), "figures": figures}
    write_json(output / "combined_manifest.json", manifest)
    print(f"Selected {len(md_cases)} MD cases and {len(neb_cases)} NEB paths; {len(matched)} comparison pairs.", flush=True)
    print(f"Saved: {output / 'comparison_summary.csv'}\nAll outputs: {output}", flush=True)
    print("Formal rates are approximate 1D calculations; inspect screening flags. D_hop needs explicit equivalent-hop settings.")
    return {"output": output, "md_cases": md_cases, "neb_cases": neb_cases,
            "comparisons": comparisons, "rates": rates, "errors": errors, "manifest": manifest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Main folder; default is the folder containing this script.")
    parser.add_argument("--md-folder", default=MD_FOLDER)
    parser.add_argument("--neb-folder", default=NEB_FOLDER)
    parser.add_argument("--temperature", type=float, default=MODEL_TEMPERATURE_K)
    parser.add_argument("--case-settings", type=Path, help="Optional edited case_settings_template.csv.")
    parser.add_argument("--min-barrier-kbt", type=float, default=MIN_BARRIER_KBT)
    parser.add_argument("--all-reports", action="store_true", help="Keep older reports too; default selects the latest per calculation.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--formats", nargs="+", default=list(FIGURE_FORMATS), choices=("png", "pdf", "svg"))
    args = parser.parse_args(argv)
    root = args.root if args.root else Path(__file__).resolve().parent
    settings_path = args.case_settings
    if settings_path is not None and not settings_path.is_absolute():
        settings_path = root / settings_path
    try:
        result = combine(root, args.md_folder, args.neb_folder, args.temperature, settings_path,
                         args.min_barrier_kbt, args.all_reports, not args.no_plots, args.formats)
        return 2 if result["errors"] or not result["comparisons"] else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
