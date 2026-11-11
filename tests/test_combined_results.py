from pathlib import Path
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("combined", REPO / "combine_neb_aimd_results.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def dump_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def dump_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def md_fixture(root, metal="Au", condition="6nn-au-100ps", env="B_vacancy_candidate",
               stamp="20261111_090000_000001", d=.2):
    case = {"Au": "au-ps", "Ag": "aag-ps", "Cu": "cu-ps"}[metal]
    processed = root / "folder-md" / condition / "AIMD_processed"
    report = processed / "reports" / stamp / case
    signature = "synthetic_" + metal
    composition = {"B": 2, "N": 3, metal: 1} if "B_vac" in env else {"B": 3, "N": 3, metal: 1}
    meta = {"case_id": case, "relative": case, "metal": metal, "environment": env,
            "case_folder": f"D:/original/research/{condition}/{case}", "composition": composition,
            "first_time_ps": .0005, "last_time_ps": 90.0}
    dump_json(processed / "raw_cache" / case / signature / "cache_manifest.json", meta)
    rows = []
    for window, start, alpha in (("whole", .0005, 1.0), ("equilibrated", 1., 1.0), ("post_capture", 8.5, .02)):
        rows.append({"case": case, "metal": metal, "environment": env, "window": window,
                     "first_saved_time_ps": start, "last_saved_time_ps": 90.,
                     "frames_in_window": 1000, "continuous_segments_used": 1,
                     "D_apparent_A2_per_ps": d, "D_apparent_m2_per_s": d * 1e-8,
                     "R2": .99 if alpha == 1 else .02, "alpha_fit": alpha,
                     "linear_screen_pass": alpha == 1,
                     "fit_status": "candidate_linear_regime_review_required" if alpha == 1 else "apparent_slope_only_not_diffusive_by_screen",
                     "geometric_capture_time_ps": 8., "observed_residence_ps": 82.,
                     "residence_right_censored": True,
                     "raw_cache_folder": f"D:\\original\\research\\{condition}\\AIMD_processed\\raw_cache\\{case}\\{signature}"})
        dump_csv(report / f"msd_{window}.csv", [{"lag_ps": i / 10, "msd_xy_A2": 4 * d * i / 10 if alpha == 1 else .1}
                                              for i in range(101)])
    dump_csv(report / "diffusion_summary.csv", rows)
    dump_json(report / "capture_result.json", {"definition": "synthetic persistent geometric entry",
              "radius_A": 1.5, "height_A": 3., "dwell_ps": .5,
              "capture": {"time_ps": 8., "index": 16000, "segment_id": 0,
                          "observed_residence_ps": 82., "residence_right_censored": True}})
    return report


def neb_fixture(root, metal="Au", name="M1-M2", env="B_vacancy", stamp="20261111_090000",
                barrier=.1, delta=0., interior=True):
    report = root / "folderm1m2m3m4" / metal / name / "NEB_extracted" / stamp
    composition = {"B": 2, "N": 3, metal: 1} if env == "B_vacancy" else {"B": 3, "N": 3, metal: 1}
    summary = {"metal": metal, "neb_folder": f"D:/original/research/NEB/{metal}/{name}",
               "composition": composition, "energy_type": "toten", "complete_energy_profile": True,
               "sampled_forward_energy_rise_eV": barrier, "sampled_reverse_energy_rise_eV": barrier - delta,
               "endpoint_energy_change_eV": delta, "maximum_is_interior": interior,
               "energy_tolerance_for_curvature_eV": .001, "warnings": []}
    dump_json(report / "neb_summary.json", summary)
    dump_csv(report / "neb_images.csv", [{"image": f"{i:02d}", "image_index": i,
              "selected_energy_eV": value, "relative_to_initial_eV": value,
              "metal_x_A": 1. + 2.5 * i / 4, "metal_y_A": 1., "metal_z_A": 3.}
              for i, value in enumerate([0., barrier / 2, barrier, barrier / 2, delta])])
    dump_csv(report / "neb_local_frequencies.csv", [{"endpoint": endpoint, "n_images": n,
              "frequency_THz": nu, "status": "exploratory_1d_only",
              "flags": "one_sided_fit; not_a_full_Vineyard_prefactor"}
              for endpoint, n, nu in [("initial", 3, 1.), ("initial", 4, 2.), ("final", 3, 1.)]])
    return report


class CombinedResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.report = md_fixture(self.root)
        self.neb = neb_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_combine(self, **kwargs):
        return c.combine(self.root, plots=False, **kwargs)

    def test_relocated_outputs_latest_report_and_no_raw_file_dependency(self):
        old = md_fixture(self.root, stamp="20261110_090000_000001", d=.9)
        # Copy time is misleading: timestamped report identity must win.
        os.utime(old / "diffusion_summary.csv", (2000000000, 2000000000))
        result = self.run_combine()
        self.assertEqual(len(result["md_cases"]), 1)
        self.assertEqual(len(result["comparisons"]), 3)
        self.assertEqual(result["comparisons"][0]["D_MD_apparent_A2_per_ps"], .2)
        self.assertEqual(result["comparisons"][0]["md_environment"], "B_vacancy")
        self.assertEqual(result["comparisons"][0]["residence_lower_bound_ps"], 82.)
        self.assertTrue((result["output"] / "comparison_summary.csv").is_file())
        self.assertTrue(any(not c.boolean(row["selected"]) for row in c.read_csv(result["output"] / "input_inventory.csv")))

    def test_rate_units_frequency_ranges_and_opt_in_diffusivity(self):
        result = self.run_combine()
        rate = next(row for row in result["rates"] if row["direction"] == "forward" and row["frequency_THz"] == 1.)
        expected = math.exp(-.1 / (c.KB_EV_K * 300))
        self.assertAlmostEqual(rate["formal_rate_ps_inv"], expected)
        self.assertAlmostEqual(rate["formal_single_channel_tau_ps"], 1 / expected)
        self.assertTrue(rate["activated_screen_pass"])
        self.assertTrue(all(row["D_hop_model_A2_per_ps"] is None for row in result["rates"]))
        settings = c.read_csv(result["output"] / "case_settings_template.csv")
        settings[0].update(process_kind="surface_hop", equivalent_sites=True, z=6)
        path = self.root / "chosen.csv"; dump_csv(path, settings)
        chosen = self.run_combine(settings_path=path)
        row = next(row for row in chosen["rates"] if row["direction"] == "forward" and row["frequency_THz"] == 1.)
        self.assertAlmostEqual(row["D_hop_model_A2_per_ps"], 6 * 2.5 ** 2 * expected / 4)
        self.assertAlmostEqual(row["D_hop_model_m2_per_s"], 6 * 2.5 ** 2 * expected / 4 * 1e-8)

    def test_paths_and_starting_conditions_are_not_collapsed(self):
        md_fixture(self.root, condition="7nn-au-100ps")
        neb_fixture(self.root, name="M2-M3")
        result = self.run_combine()
        self.assertEqual(len(result["md_cases"]), 2)
        self.assertEqual(len(result["neb_cases"]), 2)
        self.assertEqual(result["manifest"]["comparison_pairs"], 4)
        self.assertEqual(len(result["comparisons"]), 12)

    def test_known_pristine_and_vacancy_results_are_not_paired(self):
        neb_fixture(self.root, name="pristine-hop", env="pristine")
        result = self.run_combine()
        self.assertEqual(len(result["neb_cases"]), 2)
        self.assertEqual(result["manifest"]["comparison_pairs"], 1)

    def test_missing_capture_json_keeps_summary_but_does_not_assert_continuity(self):
        (self.report / "capture_result.json").unlink()
        result = self.run_combine()
        row = result["comparisons"][0]
        self.assertEqual(row["geometric_entry_absolute_ps"], 8.)
        self.assertIn("summary-only", row["geometric_entry_status"])
        self.assertIsNone(row["entry_delay_from_first_saved_frame_ps"])
        self.assertEqual(row["residence_lower_bound_ps"], 82.)

    def test_initially_inside_and_gapped_entries_are_not_observed_arrivals(self):
        capture_path = self.report / "capture_result.json"
        data = json.loads(capture_path.read_text())
        data["capture"].update(time_ps=.0005, index=0)
        dump_json(capture_path, data)
        result = self.run_combine()
        self.assertIn("left-censored", result["comparisons"][0]["geometric_entry_status"])
        self.assertIsNone(result["comparisons"][0]["entry_delay_from_first_saved_frame_ps"])
        data["capture"].update(time_ps=8., index=100, segment_id=1)
        dump_json(capture_path, data)
        result = self.run_combine()
        self.assertIn("continuity break", result["comparisons"][0]["geometric_entry_status"])

    def test_low_barrier_endpoint_maximum_and_bad_curvature_are_flagged(self):
        summary = json.loads((self.neb / "neb_summary.json").read_text())
        summary.update(sampled_forward_energy_rise_eV=.001, maximum_is_interior=False)
        dump_json(self.neb / "neb_summary.json", summary)
        fits = c.read_csv(self.neb / "neb_local_frequencies.csv")
        fits[0]["flags"] += "; nonparabolic_window"
        dump_csv(self.neb / "neb_local_frequencies.csv", fits)
        result = self.run_combine()
        row = next(row for row in result["rates"] if row["direction"] == "forward")
        self.assertFalse(row["activated_screen_pass"])
        self.assertIn("low_barrier", row["model_screen_flags"])
        self.assertIn("endpoint", row["model_screen_flags"])
        self.assertIn("nonparabolic", row["model_screen_flags"])
        self.assertIsNotNone(row["formal_rate_ps_inv"])
        self.assertIsNone(row["D_hop_model_A2_per_ps"])

    def test_legacy_per_metal_exports_and_duplicate_aggregates(self):
        shutil.rmtree(self.root / "folder-md")
        folder = self.root / "folder-md" / "AIMD_results_updated"
        row = {"label": "Au", "analysis_start_ps": 12., "analysis_end_ps": 50.,
               "D_candidate_A2_ps": "", "assessment": "not fitted"}
        dump_csv(folder / "Au_bvac_6nn_diffusion_summary.csv", [row])
        dump_csv(folder / "Au_Ag_Cu_bvac_diffusion_summary.csv", [row, dict(row, label="Ag")])
        dump_csv(folder / "Au_bvac_6nn_capture_and_residence.csv", [{"label": "Au", "metal": "Au", "condition": "VB",
                 "capture_status": "persistent entry observed", "capture_observed": True,
                 "first_persistent_entry_ps": 8., "first_residence_observed_ps": 82.,
                 "first_residence_right_censored": True}])
        result = self.run_combine()
        au_cases = [case for case in result["md_cases"] if case["metal"] == "Au"]
        self.assertEqual(len(au_cases), 1)
        self.assertEqual(result["comparisons"][0]["entry_delay_from_first_saved_frame_ps"], 8.)
        self.assertIsNone(result["comparisons"][0]["D_MD_apparent_A2_per_ps"])

    def test_multibasin_profile_is_not_treated_as_one_activated_hop(self):
        images = c.read_csv(self.neb / "neb_images.csv")
        for row, value in zip(images, [0., .1, .02, .09, 0.]):
            row.update(selected_energy_eV=value, relative_to_initial_eV=value)
        dump_csv(self.neb / "neb_images.csv", images)
        result = self.run_combine()
        self.assertTrue(all(not row["activated_screen_pass"] for row in result["rates"]))
        self.assertTrue(all("multiple_energy_basins" in row["model_screen_flags"] for row in result["rates"]))

    def test_inequivalent_endpoint_energies_block_hopping_diffusivity(self):
        neb_fixture(self.root, delta=-.05)
        first = self.run_combine()
        settings = c.read_csv(first["output"] / "case_settings_template.csv")
        settings[0].update(process_kind="surface_hop", equivalent_sites=True, z=6)
        path = self.root / "chosen.csv"; dump_csv(path, settings)
        result = self.run_combine(settings_path=path)
        forward = [row for row in result["rates"] if row["direction"] == "forward"]
        self.assertTrue(all(row["activated_screen_pass"] for row in forward))
        self.assertTrue(all(row["D_hop_model_A2_per_ps"] is None for row in forward))
        self.assertTrue(all("endpoint_energies_differ" in row["D_hop_status"] for row in forward))

    def test_cli_uses_script_parent_not_working_directory(self):
        shutil.copyfile(REPO / "combine_neb_aimd_results.py", self.root / "combine_neb_aimd_results.py")
        run = subprocess.run([sys.executable, str(self.root / "combine_neb_aimd_results.py"), "--no-plots"],
                             cwd=REPO, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertTrue(list((self.root / "Combined_analysis").glob("*/comparison_summary.csv")))


if __name__ == "__main__":
    unittest.main()
