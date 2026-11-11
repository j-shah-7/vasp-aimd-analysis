from pathlib import Path
from html import escape
import importlib.util
import json
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


md = load("batch_lattice_test", "aimd_batch_analysis.py")
animation = load("hbn_lattice_test", "hbn_animation_with_lattice.py")


def geometry(metal="Au", rotated=False):
    cell = np.array([[2.5, 0., 0.], [-1.25, 2.165063509, 0.], [0., 0., 20.]])
    reference = np.array([[0., 0., 8.], [0., 1.443375673, 8.], [.6, .8, 10.]])
    if rotated:
        az, ax = .41, .27
        rz = np.array([[np.cos(az), -np.sin(az), 0.], [np.sin(az), np.cos(az), 0.], [0., 0., 1.]])
        rx = np.array([[1., 0., 0.], [0., np.cos(ax), -np.sin(ax)], [0., np.sin(ax), np.cos(ax)]])
        rotation = rx @ rz
        reference = reference @ rotation.T + [3., -2., 4.]
        cell = cell @ rotation.T
    e1 = cell[0] / np.linalg.norm(cell[0])
    normal = np.cross(cell[0], cell[1]); normal /= np.linalg.norm(normal)
    basis = np.array([e1, np.cross(normal, e1), normal])
    center = np.average(reference[:2], axis=0, weights=[10.81, 14.007])
    origin = (reference[2] - center) @ basis.T
    displacement = np.array([0., .1, .2, 4., 4.1, 4.2])
    p = origin + np.column_stack((displacement, np.zeros(6), np.zeros(6)))
    return {"symbols": np.array(["B", "N", metal]), "reference_positions_A": reference,
            "cell_A": cell, "positions_A": p, "time_ps": np.arange(6) * .1,
            "segment_ids": np.array([0, 0, 0, 1, 1, 1])}, basis


class LatticePlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_lattice_and_metal_share_origin_after_translation_and_rotation(self):
        expected = np.array([[-.6, -.8], [-.6, 1.443375673-.8], [0., 0.]])
        for rotated in (False, True):
            data, _ = geometry(rotated=rotated)
            _, sites, _ = md.reference_lattice_xy(data)
            np.testing.assert_allclose(sites, expected, atol=1e-9)

    def test_skew_lattice_image_is_present_and_cropped(self):
        data, _ = geometry()
        _, sites, lattice = md.reference_lattice_xy(data)
        xlim, ylim = (-1.9, -1.8), (1.3, 1.4)
        visible = md.tiled_sites_xy(sites[:1], lattice, xlim, ylim)
        np.testing.assert_allclose(visible, [[-1.85, 1.365063509]], atol=1e-9)

    def test_static_path_does_not_bridge_flagged_gaps(self):
        data, _ = geometry()
        result = {"extracted": {"data": data, "metadata": {"metal": "Au"}}}
        fig, ax = plt.subplots()
        md.draw_xy_trajectory(ax, result, md.DEFAULTS)
        lines = [line for artist in ax.collections if isinstance(artist, LineCollection)
                 for line in artist.get_segments()]
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(np.linalg.norm(line[1]-line[0]) < .101 for line in lines))
        self.assertEqual(set(ax.get_legend_handles_labels()[1]), {"B", "N", "Start", "End"})
        self.assertAlmostEqual(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()))

    def test_all_three_gifs_save_from_loaded_arrays(self):
        trajectories = []
        for metal in ("Au", "Ag", "Cu"):
            data, basis = geometry(metal)
            # Old h-BN notebook uses an absolute initial origin, unlike batch delta-x/y.
            projected = data["reference_positions_A"][2] @ basis.T + np.column_stack((np.linspace(0, 1, 6), np.zeros(6), np.zeros(6)))
            trajectories.append({"label": metal, "metal": metal, "time_ps": data["time_ps"],
                "projected_A": projected, "segment_id": data["segment_ids"],
                "snapshot_A": data["reference_positions_A"], "symbols": data["symbols"],
                "cell": data["cell_A"], "basis": basis})
        with tempfile.TemporaryDirectory() as folder:
            saved = animation.animate_hbn_with_lattice(trajectories, folder, label="all",
                prefix_function=lambda label: f"example_{label}", show_inline=False,
                frames=6, fps=10, dpi=40)
            self.assertEqual(set(saved), {"Au", "Ag", "Cu"})
            for metal, result in saved.items():
                with Image.open(result["gif_path"]) as gif:
                    self.assertEqual(gif.n_frames, 6)
                self.assertEqual(result["gif_path"].name, f"example_{metal}_trajectory_BN_lattice.gif")
                ax = result["animation"]._fig.axes[0]
                self.assertIn("B", ax.get_legend_handles_labels()[1])
                self.assertIn("N", ax.get_legend_handles_labels()[1])
                trail = result["animation"]._func(3)[0]
                self.assertEqual(len(trail.get_xdata()), 1)

    def test_notebook_preview_displays_all_successful_metals(self):
        nb = json.loads((ROOT / "aimd_batch_analysis.ipynb").read_text())
        displayed = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for metal in ("Au", "Ag", "Cu"):
                folder = root / metal; folder.mkdir()
                Image.new("RGB", (10, 10), "white").save(folder / "xy_trajectory.png")
                results.append({"extracted": {"metadata": {"metal": metal}},
                    "case": {"id": metal, "relative": metal}, "report_dir": folder})
            namespace = {"BATCH": {"results": results, "report_root": root},
                "DEFAULTS": md.DEFAULTS, "SETTINGS": {}, "COLORS": md.COLORS,
                "display": displayed.append, "HTML": lambda x: x,
                "Image": lambda **kwargs: kwargs["filename"], "escape": escape}
            exec(compile("".join(nb["cells"][13]["source"]), "preview_cell", "exec"), namespace)
        shown_paths = [value for value in displayed if value.endswith(".png")]
        self.assertEqual({Path(value).parent.name for value in shown_paths}, {"Au", "Ag", "Cu"})


if __name__ == "__main__":
    unittest.main()
