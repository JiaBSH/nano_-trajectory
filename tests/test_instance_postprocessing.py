import contextlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from rawframe_analysis.tracker import GasTracker
from rawframe_analysis.tracking import ObjectTrackingMixin


class InstancePostprocessingTests(unittest.TestCase):
    def test_nested_duplicates_and_particle_in_droplet_are_suppressed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
                {
                    "category": "nanocluster",
                    "segmentation": [[2, 2], [8, 2], [8, 8], [2, 8]],
                },
                {
                    "category": "nanocluster",
                    "segmentation": [[22, 2], [26, 2], [26, 6], [22, 6]],
                },
                {
                    "category": "nanodroplet",
                    "segmentation": [[20, 0], [30, 0], [30, 10], [20, 10]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": objects}), encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=1.0,
                    target_category="nanocluster",
                    compute_boundary_distances_enabled=True,
                    output_root=str(base / "output"),
                )
                tracker.process_all_frames()

            self.assertEqual(len(tracker.object_records), 1)
            self.assertEqual(len(tracker.boundary_particle_records), 1)
            self.assertEqual(len(tracker.boundary_droplet_records), 1)
            self.assertEqual(tracker.same_category_suppressed_count, 1)
            self.assertEqual(tracker.cross_category_suppressed_count, 1)
            self.assertEqual(
                {record["reason"] for record in tracker.instance_postprocess_records},
                {
                    "same_category_containment",
                    "particle_contained_in_droplet",
                },
            )

    def test_low_overlap_instances_remain_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
                {
                    "category": "nanocluster",
                    "segmentation": [[8, 0], [18, 0], [18, 10], [8, 10]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": objects}), encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=1.0,
                    target_category="nanocluster",
                    output_root=str(base / "output"),
                )
                tracker.process_all_frames()

            self.assertEqual(len(tracker.object_records), 2)
            self.assertEqual(tracker.same_category_suppressed_count, 0)

    def test_minor_particle_droplet_contact_remains_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 0], [10, 0], [10, 10], [0, 10]],
                },
                {
                    "category": "nanodroplet",
                    "segmentation": [[8, 0], [18, 0], [18, 10], [8, 10]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": objects}), encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=1.0,
                    target_category="nanocluster",
                    output_root=str(base / "output"),
                )
                tracker.process_all_frames()

            self.assertEqual(len(tracker.object_records), 1)
            self.assertEqual(tracker.cross_category_suppressed_count, 0)


class TrackingLifecycleTests(unittest.TestCase):
    def test_two_to_one_to_two_uses_exactly_five_lifecycle_ids(self):
        tracker = ObjectTrackingMixin()
        detections = defaultdict(list)
        detections[0] = [
            ("frame-0", 1.0, -2.0, 0.0, 5.0),
            ("frame-0", 1.0, 2.0, 0.0, 5.0),
        ]
        detections[1] = [("frame-1", 1.0, 0.0, 0.0, 10.0)]
        detections[2] = [
            ("frame-2", 1.0, -2.0, 0.0, 5.0),
            ("frame-2", 1.0, 2.0, 0.0, 5.0),
        ]

        series, assignments, events = tracker._build_event_id_series_with_assignments(
            detections, max_dist=10.0
        )

        self.assertEqual(assignments, {0: [1, 2], 1: [3], 2: [4, 5]})
        self.assertEqual(set(series), {1, 2, 3, 4, 5})
        self.assertEqual([event["type"] for event in events], ["merge", "split"])


if __name__ == "__main__":
    unittest.main()
