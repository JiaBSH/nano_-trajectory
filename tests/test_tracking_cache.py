import csv
import tempfile
import unittest
from pathlib import Path

from rawframe_analysis.exporting import CsvExportMixin
from rawframe_analysis.tracking import ObjectTrackingMixin


def object_record(frame_id, cx_nm):
    return [frame_id, f"frame-{frame_id}", 1.0, cx_nm, 0.0, 10.0]


class TrackerDouble(ObjectTrackingMixin, CsvExportMixin):
    pass


class TrackingCacheTests(unittest.TestCase):
    def setUp(self):
        self.tracker = TrackerDouble()
        self.tracker._object_detections_by_frame_cache = None
        self.tracker._object_detections_by_frame_cache_records = None
        self.tracker._object_detections_by_frame_cache_record_count = 0
        self.tracker._event_id_series_cache = {}

    def test_empty_frame_breaks_identity(self):
        self.tracker.json_files = ["frame-0.json", "frame-1.json", "frame-2.json"]
        self.tracker.object_records = [
            object_record(0, 0.0),
            object_record(2, 0.0),
        ]

        export_ids = self.tracker._build_export_instance_ids(
            max_dist=10.0,
            use_display_id=True,
        )

        self.assertEqual(export_ids, [1, 2])

    def test_speed_export_keeps_canonical_display_id_after_short_track_drops(self):
        self.tracker.json_files = ["frame-0.json", "frame-1.json", "frame-2.json"]
        self.tracker.target_category = "nanodroplet"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"
        self.tracker.object_records = [
            object_record(0, 0.0),
            object_record(0, 100.0),
            object_record(1, 101.0),
            object_record(2, 102.0),
        ]

        series, _assignments, _events = (
            self.tracker._event_id_series_for_object_records(max_dist=10.0)
        )
        display_id_of = self.tracker._display_id_mapping(series)
        speed_series = {
            instance_id: self.tracker._compute_speed_series_from_points(points)
            for instance_id, points in series.items()
            if len(points) >= 2
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "speed.csv"
            self.tracker.export_speed_series(
                speed_series,
                out_csv=str(output_path),
                display_id_of=display_id_of,
            )
            with output_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual({row["instance_id"] for row in rows}, {"D2"})

    def test_area_export_uses_the_same_prefixed_id_as_annotations(self):
        self.tracker.target_category = "nanocluster"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"
        series = {7: [(3, "frame-3", 1.0, 2.0, 4.0, 6.0)]}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "area.csv"
            self.tracker.export_id_series(series, out_csv=str(output_path))
            with output_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["instance_id"], "P7")

    def test_primary_droplet_exports_all_use_annotation_id(self):
        self.tracker.json_files = ["frame-0.json"]
        self.tracker.target_category = "nanodroplet"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"
        self.tracker.object_records = [object_record(0, 2.0)]
        self.tracker.area_records = [[0, "frame-0", 1.0, 10.0]]
        self.tracker.contour_records = [[0, "frame-0", "(0.000,0.000)"]]
        self.tracker.centroid_records = [[0, "frame-0", 1.0, 2.0, 0.0]]
        self.tracker.diameter_height_records = [
            [0, "frame-0", 1.0, 2.0, 0.0, 3.0, 4.0]
        ]
        self.tracker.pin_reference_enabled = False
        self.tracker.compute_boundary_distances_enabled = False

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tracker.output_root = temp_dir
            self.tracker.export_results(max_dist=10.0)
            filenames = [
                "nanodroplet_area_vs_frame.csv",
                "nanodroplet_contours_by_frame.csv",
                "nanodroplet_centroids.csv",
                "nanodroplet_diameter_height_vs_frame.csv",
            ]
            exported_ids = []
            for filename in filenames:
                with (Path(temp_dir) / filename).open(
                    "r", newline="", encoding="utf-8"
                ) as handle:
                    exported_ids.append(next(csv.DictReader(handle))["instance_id"])

        self.assertEqual(exported_ids, ["D1", "D1", "D1", "D1"])

    def test_csv_validator_checks_all_object_table_ids(self):
        self.tracker.target_category = "nanodroplet"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"
        full_header = "instance_id,frame_id,frame_name,value\n"
        full_row = "D1,0,frame-0,1\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tracker.output_root = temp_dir
            for suffix in (
                "area_vs_frame",
                "centroids",
                "contours_by_frame",
                "diameter_height_vs_frame",
                "instance_area_vs_frame",
            ):
                (Path(temp_dir) / f"nanodroplet_{suffix}.csv").write_text(
                    full_header + full_row, encoding="utf-8"
                )
            (Path(temp_dir) / "nanodroplet_instance_speed_mean_5frames.csv").write_text(
                "instance_id,frame_id,frame_name,speed_nm_per_s\n"
                "D1,0,frame-0,2\n",
                encoding="utf-8",
            )
            (Path(temp_dir) / "nanocluster_to_nanodroplet_boundary_distances.csv").write_text(
                "frame_id,particle_id,droplet_id\n0,P1,D1\n",
                encoding="utf-8",
            )

            report = self.tracker.validate_exported_csv_ids()

            self.assertEqual(report["files"], 7)
            self.assertEqual(report["complete_tables"], 5)
            self.assertGreaterEqual(report["id_columns"], 8)

            bad_path = Path(temp_dir) / "nanodroplet_centroids.csv"
            bad_path.write_text(full_header + "1,0,frame-0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "centroids.csv.instance_id"):
                self.tracker.validate_exported_csv_ids()

    def test_filtered_far_export_uses_boundary_annotation_id(self):
        self.tracker.json_files = ["frame-0.json"]
        self.tracker.target_category = "nanodroplet"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"
        self.tracker.object_records = [object_record(0, 2.0)]
        self.tracker.area_records = [[0, "frame-0", 1.0, 10.0]]
        self.tracker.contour_records = [[0, "frame-0", "(0.000,0.000)"]]
        self.tracker.centroid_records = [[0, "frame-0", 1.0, 2.0, 0.0]]
        self.tracker.diameter_height_records = [
            [0, "frame-0", 1.0, 2.0, 0.0, 3.0, 4.0]
        ]
        self.tracker.pin_reference_enabled = True
        self.tracker.max_particle_pin_distance_nm = 50.0
        self.tracker.pin_reference_records = []
        self.tracker.filtered_far_particle_records = [
            [0, "frame-0", 1.0, 100.0, 0.0, 100.0, 50.0, 8.0, 2]
        ]
        self.tracker.boundary_particle_records = []
        self.tracker.boundary_droplet_records = [
            [0, "frame-0", 1.0, 1, 0, 2.0, 0.0, 10.0],
            [0, "frame-0", 1.0, 2, 1, 100.0, 0.0, 8.0],
        ]
        self.tracker.compute_boundary_distances_enabled = False

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tracker.output_root = temp_dir
            self.tracker.export_results(max_dist=10.0)
            with (Path(temp_dir) / "nanodroplet_filtered_far_from_pin.csv").open(
                "r", newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["instance_id"], "D2")

    def test_csv_validator_rejects_object_table_without_instance_id(self):
        self.tracker.target_category = "nanodroplet"
        self.tracker.particle_category = "nanocluster"
        self.tracker.droplet_category = "nanodroplet"

        with tempfile.TemporaryDirectory() as temp_dir:
            self.tracker.output_root = temp_dir
            (Path(temp_dir) / "nanodroplet_filtered_far_from_pin.csv").write_text(
                "frame_id,frame_name,distance_to_pin_nm\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "has no instance_id column"):
                self.tracker.validate_exported_csv_ids()

    def test_event_ids_are_rebuilt_after_record_context_is_replaced(self):
        full_records = [
            object_record(0, 0.0),
            object_record(0, 100.0),
            object_record(1, 1.0),
            object_record(1, 101.0),
        ]
        left_records = [object_record(0, 0.0), object_record(1, 1.0)]

        self.tracker.object_records = full_records
        _full_series, full_assignments, _full_events = (
            self.tracker._event_id_series_for_object_records(max_dist=10.0)
        )

        self.tracker.object_records = left_records
        left_ids = self.tracker._build_export_instance_ids(
            max_dist=10.0,
            use_display_id=True,
        )

        self.assertEqual(sum(map(len, full_assignments.values())), 4)
        self.assertEqual(left_ids, [1, 1])

    def test_same_size_record_context_does_not_reuse_previous_frames(self):
        self.tracker.object_records = [
            object_record(0, 0.0),
            object_record(1, 1.0),
        ]
        self.tracker._event_id_series_for_object_records(max_dist=10.0)

        self.tracker.object_records = [
            object_record(10, 50.0),
            object_record(11, 51.0),
        ]
        series, assignments, _events = (
            self.tracker._event_id_series_for_object_records(max_dist=10.0)
        )

        self.assertEqual(set(assignments), {10, 11})
        self.assertEqual([point[0] for point in series[1]], [10, 11])

    def test_event_ids_are_rebuilt_after_records_are_appended(self):
        records = [object_record(0, 0.0)]
        self.tracker.object_records = records
        self.tracker._event_id_series_for_object_records(max_dist=10.0)

        records.append(object_record(1, 1.0))
        series, assignments, _events = (
            self.tracker._event_id_series_for_object_records(max_dist=10.0)
        )

        self.assertEqual(sum(map(len, assignments.values())), 2)
        self.assertEqual(len(series[1]), 2)


if __name__ == "__main__":
    unittest.main()
