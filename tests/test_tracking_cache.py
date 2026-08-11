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

        self.assertEqual({int(row["instance_id"]) for row in rows}, {2})

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
