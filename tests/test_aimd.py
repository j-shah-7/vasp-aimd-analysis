from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aimd_batch_analysis", ROOT / "aimd_batch_analysis.py")
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)


class MsdTests(unittest.TestCase):
    def test_fft_matches_independent_displacement_average(self):
        points = np.cumsum(np.random.default_rng(9).normal(size=(300, 2)), axis=0)
        sums, counts = md.fft_msd_sums(points, 70)
        for lag in (1, 2, 19, 70):
            expected = np.mean((points[lag:] - points[:-lag]) ** 2, axis=0)
            np.testing.assert_allclose(sums[lag] / counts[lag], expected, atol=1e-10)

    def test_units_and_insufficient_fit_coverage(self):
        t = np.arange(0, 8, .01)
        curve = {"lag_ps": t, "msd_xy_A2": 8 * t}
        fit = md.fit_msd(curve, [1, 5])
        self.assertAlmostEqual(fit["D_apparent_A2_per_ps"], 2)
        self.assertAlmostEqual(fit["D_apparent_m2_per_s"], 2e-8)
        self.assertTrue(fit["linear_screen_pass"])
        short = {key: value[:50] for key, value in curve.items()}
        self.assertIsNone(md.fit_msd(short, [1, 5])["D_apparent_A2_per_ps"])

    def test_90ps_random_walk_and_confinement(self):
        rng = np.random.default_rng(555)
        n, step_ps, expected_D = 180001, .0005, .35
        t = np.arange(n) * step_ps
        xy = np.cumsum(rng.normal(scale=np.sqrt(2 * expected_D * step_ps), size=(n, 2)), axis=0)
        data = {"time_ps": t, "positions_A": np.column_stack((xy, np.ones(n))),
                "segment_ids": np.zeros(n, int)}
        settings = dict(md.DEFAULTS, fit_lag_ps=[.05, .2])
        segments, dt, _ = md.window_segments(data, 0, None, settings)
        curve = md.pool_msd(segments, dt, settings)
        self.assertAlmostEqual(segments[0]["time_ps"][-1], 90)
        self.assertEqual(curve["time_origin_pairs"][1], 18000)
        fit = md.fit_msd(curve, settings["fit_lag_ps"])
        self.assertLess(abs(fit["D_apparent_A2_per_ps"] / expected_D - 1), .15)
        bound = rng.normal(scale=.1, size=(18001, 2))
        plateau = md.pool_msd([{"xy_A": bound, "time_ps": t[::10]}], .005, settings)
        self.assertFalse(md.fit_msd(plateau, [1, 5])["linear_screen_pass"])


class RawExtractionTests(unittest.TestCase):
    def test_saved_stride_periodic_unwrap_and_substrate_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "gold"
            parent.mkdir()

            def positions(step):
                drift = .001 * step
                return np.array([[1 + drift, 1, 8], [2.4 + drift, 1, 8],
                                 [(9.8 + .003 * step + drift) % 10, 3, 10]])

            def structure(step):
                body = "Synthetic MD\n1\n10 0 0\n0 10 0\n0 0 20\nB N Au\n1 1 1\nCartesian\n"
                body += "".join(" ".join(map(str, p)) + "\n" for p in positions(step))
                return body + "\nCartesian\n0 0 0\n0 0 0\n0 0 0\n"

            for run in (1, 2):
                folder = parent / str(run)
                folder.mkdir()
                offset = (run - 1) * 100
                (folder / "INCAR").write_text("POTIM = .5; NBLOCK = 2; IBRION = 0\n")
                (folder / "POSCAR").write_text(structure(offset))
                (folder / "CONTCAR").write_text(structure(offset + 100))
                (folder / "stdout").write_text("".join(
                    f" {i} T = 300.0 E= -1.000D+02 F= -1.010D+02 E0= -1.009D+02\n"
                    for i in range(1, 101)))
                xdat = "Synthetic MD\n1\n10 0 0\n0 10 0\n0 0 20\nB N Au\n1 1 1\n"
                for ordinal, step in enumerate(range(2, 101, 2), 1):
                    xdat += f"Direct configuration= {ordinal}\n"
                    xdat += "".join(" ".join(f"{v:.12f}" for v in p / [10, 10, 20]) + "\n"
                                    for p in positions(offset + step))
                (folder / "XDATCAR").write_text(xdat)
            settings = dict(md.DEFAULTS, make_plots=False, progress_every_frames=0)
            case = md.discover_cases(root)[0]
            result = md.prepare_case(case, root / "outputs", settings)
            data, meta = result["data"], result["metadata"]
            self.assertEqual(len(data["time_ps"]), 100)
            self.assertAlmostEqual(data["time_ps"][-1], .1)
            self.assertEqual(meta["number_of_continuous_segments"], 1)
            self.assertEqual(meta["metal"], "Au")
            self.assertAlmostEqual(data["positions_A"][-1, 0] - data["positions_A"][0, 0], .003 * 198)
            self.assertTrue((result["cache_dir"] / "movie-combined.xyz").is_file())

    def test_notebooks_are_valid_and_module_matches_current_figures(self):
        for filename in ("aimd_batch_analysis.ipynb", "hbn_aimd_local_analysis.ipynb"):
            nb = json.loads((ROOT / filename).read_text())
            for index, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"{filename}_cell_{index}", "exec")
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
        nb = json.loads((ROOT / "hbn_aimd_local_analysis.ipynb").read_text())
        self.assertEqual("".join(nb["cells"][3]["source"]).strip(),
                         (ROOT / "hbn_aimd_analysis.py").read_text().strip())
        nb = json.loads((ROOT / "aimd_batch_analysis.ipynb").read_text())
        self.assertEqual("".join(nb["cells"][4]["source"]).strip(),
                         (ROOT / "aimd_batch_analysis.py").read_text().split("\n\ndef main(argv=None):", 1)[0].strip())


if __name__ == "__main__":
    unittest.main()
