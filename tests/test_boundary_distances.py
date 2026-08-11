import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rawframe_analysis.boundary_distance_plots import BoundaryDistancePlotMixin
from rawframe_analysis.geometry import GeometryMixin
from rawframe_analysis.tracker import GasTracker


class PolygonBoundaryDistanceTests(unittest.TestCase):
    def test_separated_polygons_use_boundary_not_centroid_distance(self):
        first = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        second = np.array([[5, 0], [7, 0], [7, 2], [5, 2]], dtype=float)

        self.assertAlmostEqual(
            GeometryMixin.polygon_boundary_distance(first, second), 3.0
        )

    def test_touching_overlapping_and_contained_polygons_are_zero(self):
        outer = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        cases = (
            np.array([[4, 1], [6, 1], [6, 3], [4, 3]], dtype=float),
            np.array([[3, 1], [5, 1], [5, 3], [3, 3]], dtype=float),
            np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=float),
        )

        for other in cases:
            with self.subTest(other=other.tolist()):
                self.assertEqual(
                    GeometryMixin.polygon_boundary_distance(outer, other), 0.0
                )

    def test_diagonal_gap_uses_euclidean_distance(self):
        first = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        second = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], dtype=float)

        self.assertAlmostEqual(
            GeometryMixin.polygon_boundary_distance(first, second), np.sqrt(2.0)
        )


class BoundaryDistanceExportTests(unittest.TestCase):
    def test_pair_line_breaks_across_missing_frames(self):
        frames, distances = BoundaryDistancePlotMixin._line_with_frame_gaps(
            [(0, 2.0), (2, 8.0)]
        )

        np.testing.assert_allclose(frames[[0, 2]], [0.0, 2.0])
        np.testing.assert_allclose(distances[[0, 2]], [2.0, 8.0])
        self.assertTrue(np.isnan(frames[1]))
        self.assertTrue(np.isnan(distances[1]))

    def test_exports_all_pairs_and_per_particle_nearest_distances(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            raw_frame_dir = base / "raw_frames"
            output_dir = base / "output"
            json_dir.mkdir()
            raw_frame_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 0], [2, 0], [2, 2], [0, 2]],
                },
                {
                    "category": "nanocluster",
                    "segmentation": [[5, 0], [7, 0], [7, 2], [5, 2]],
                },
                {
                    "category": "nanodroplet",
                    "segmentation": [[1, 1], [3, 1], [3, 3], [1, 3]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": objects}), encoding="utf-8"
            )
            (json_dir / "frame_0002.json").write_text(
                json.dumps({"objects": [objects[1], objects[0], objects[2]]}),
                encoding="utf-8",
            )
            for stem in ("frame_0001", "frame_0002"):
                Image.new("RGB", (32, 32), "white").save(raw_frame_dir / f"{stem}.png")

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=2.0,
                    target_category="nanocluster",
                    compute_diameter_height_enabled=False,
                    compute_boundary_distances_enabled=True,
                    output_root=str(output_dir),
                )
                tracker.process_all_frames()
                tracker.export_results()
                particle_pairs, particle_droplet_pairs = (
                    tracker._tracked_boundary_distance_series(max_dist=10.0)
                )
                annotation_ids, prefixes = tracker._boundary_pair_annotation_ids(
                    max_dist=10.0
                )
                plot_paths = tracker.plot_boundary_distances_vs_frame(max_dist=10.0)
                tracker.annotate_boundary_pair_ids_on_rawframe(
                    raw_frame_dir=str(raw_frame_dir),
                    output_dir="annotated_pair_ids",
                    max_dist=10.0,
                    frame_step=1,
                )

            self.assertEqual(len(plot_paths), 2)
            self.assertTrue(all(Path(path).is_file() for path in plot_paths))
            self.assertEqual(len(particle_pairs), 1)
            self.assertEqual(len(next(iter(particle_pairs.values()))), 2)
            self.assertEqual(len(particle_droplet_pairs), 2)
            self.assertTrue(
                all(len(points) == 2 for points in particle_droplet_pairs.values())
            )
            self.assertEqual(prefixes, {"nanocluster": "P", "nanodroplet": "D"})
            self.assertEqual(annotation_ids["nanocluster"][0], [1, 2])
            self.assertEqual(annotation_ids["nanocluster"][1], [2, 1])
            self.assertEqual(annotation_ids["nanodroplet"][0], [1])
            self.assertEqual(annotation_ids["nanodroplet"][1], [1])
            annotated_dir = output_dir / "annotated_pair_ids"
            self.assertTrue((annotated_dir / "frame_0001.png").is_file())
            self.assertTrue((annotated_dir / "frame_0002.png").is_file())

            pair_rows = self._read_rows(
                output_dir / "nanocluster_to_nanocluster_boundary_distances.csv"
            )
            self.assertEqual(len(pair_rows), 2)
            self.assertEqual(
                [(row["particle_1_id"], row["particle_2_id"]) for row in pair_rows],
                [("P1", "P2"), ("P1", "P2")],
            )
            self.assertEqual(
                [float(row["boundary_distance_nm"]) for row in pair_rows],
                [6.0, 6.0],
            )

            droplet_rows = self._read_rows(
                output_dir / "nanocluster_to_nanodroplet_boundary_distances.csv"
            )
            self.assertEqual(len(droplet_rows), 4)
            self.assertEqual(
                [(row["particle_id"], row["droplet_id"]) for row in droplet_rows],
                [("P1", "D1"), ("P2", "D1"), ("P2", "D1"), ("P1", "D1")],
            )
            self.assertEqual(
                [float(row["boundary_distance_nm"]) for row in droplet_rows],
                [0.0, 4.0, 4.0, 0.0],
            )

            nearest_rows = self._read_rows(
                output_dir / "nanocluster_nearest_boundary_distances.csv"
            )
            self.assertEqual(len(nearest_rows), 4)
            self.assertEqual(
                [
                    (
                        row["particle_id"],
                        row["nearest_particle_id"],
                        row["nearest_droplet_id"],
                    )
                    for row in nearest_rows
                ],
                [
                    ("P1", "P2", "D1"),
                    ("P2", "P1", "D1"),
                    ("P2", "P1", "D1"),
                    ("P1", "P2", "D1"),
                ],
            )
            self.assertEqual(
                [
                    float(row["nearest_particle_boundary_distance_nm"])
                    for row in nearest_rows
                ],
                [6.0, 6.0, 6.0, 6.0],
            )
            self.assertEqual(
                [
                    float(row["nearest_droplet_boundary_distance_nm"])
                    for row in nearest_rows
                ],
                [0.0, 4.0, 4.0, 0.0],
            )

    @staticmethod
    def _read_rows(path):
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
