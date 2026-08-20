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
    def test_nested_same_category_masks_merge_but_cross_category_overlap_remains(self):
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

            self.assertEqual(len(tracker.object_records), 2)
            self.assertEqual(len(tracker.boundary_particle_records), 2)
            self.assertEqual(len(tracker.boundary_droplet_records), 1)
            self.assertEqual(tracker.same_category_suppressed_count, 1)
            self.assertEqual(tracker.cross_category_suppressed_count, 0)
            self.assertEqual(
                {record["reason"] for record in tracker.instance_postprocess_records},
                {"same_category_overlap_merge"},
            )

    def test_partial_same_category_overlap_uses_union_shape(self):
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
                    "segmentation": [[1, 0], [11, 0], [11, 10], [1, 10]],
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
            self.assertGreater(tracker.object_records[0][5], 100.0)

    def test_distant_same_category_instances_remain_separate(self):
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
                    "segmentation": [[17, 17], [27, 17], [27, 27], [17, 27]],
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

    def test_upper_and_lower_fragments_merge_by_boundary_contact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            objects = [
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 0], [20, 0], [20, 8], [0, 8]],
                },
                {
                    "category": "nanocluster",
                    "segmentation": [[0, 11], [20, 11], [20, 19], [0, 19]],
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
            self.assertEqual(tracker.same_category_suppressed_count, 1)
            self.assertEqual(
                tracker.instance_postprocess_records[0]["reason"],
                "same_category_contact_merge",
            )
            self.assertGreater(tracker.object_records[0][5], 320.0)

    def test_transient_split_fragments_merge_against_same_previous_instance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            previous = {
                "category": "nanodroplet",
                "segmentation": [[0, 0], [50, 0], [50, 20], [0, 20]],
            }
            fragments = [
                {
                    "category": "nanodroplet",
                    "segmentation": [[0, 0], [20, 0], [20, 20], [0, 20]],
                },
                {
                    "category": "nanodroplet",
                    "segmentation": [[32, 0], [50, 0], [50, 20], [32, 20]],
                },
            ]
            (json_dir / "frame_0001.json").write_text(
                json.dumps({"objects": [previous]}), encoding="utf-8"
            )
            (json_dir / "frame_0002.json").write_text(
                json.dumps({"objects": fragments}), encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=1.0,
                    target_category="nanodroplet",
                    output_root=str(base / "output"),
                )
                tracker.process_all_frames()

            frame_counts = defaultdict(int)
            for record in tracker.object_records:
                frame_counts[record[1]] += 1
            self.assertEqual(frame_counts, {"frame_0001": 1, "frame_0002": 1})
            self.assertEqual(tracker.same_category_suppressed_count, 1)
            self.assertEqual(
                tracker.instance_postprocess_records[0]["reason"],
                "same_category_temporal_contact_merge",
            )

    def test_nearby_persistent_instances_do_not_share_temporal_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            json_dir = base / "annotations"
            json_dir.mkdir()
            separate = [
                {
                    "category": "nanodroplet",
                    "segmentation": [[0, 0], [20, 0], [20, 20], [0, 20]],
                },
                {
                    "category": "nanodroplet",
                    "segmentation": [[32, 0], [50, 0], [50, 20], [32, 20]],
                },
            ]
            for frame in (1, 2):
                (json_dir / f"frame_{frame:04d}.json").write_text(
                    json.dumps({"objects": separate}), encoding="utf-8"
                )

            with contextlib.redirect_stdout(io.StringIO()):
                tracker = GasTracker(
                    json_dir=str(json_dir),
                    nm_per_px=1.0,
                    target_category="nanodroplet",
                    output_root=str(base / "output"),
                )
                tracker.process_all_frames()

            self.assertEqual(len(tracker.object_records), 4)
            self.assertEqual(tracker.same_category_suppressed_count, 0)

    def test_fully_overlapping_particle_and_droplet_remain_separate(self):
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
                    "segmentation": [[0, 0], [10, 0], [10, 10], [0, 10]],
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
