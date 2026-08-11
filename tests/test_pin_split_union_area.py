import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "analyze-rawframe-pin-split180.py"
SPEC = importlib.util.spec_from_file_location("analyze_rawframe_pin_split180", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PinSplitGasTracker = MODULE.PinSplitGasTracker


class PinSplitUnionAreaTests(unittest.TestCase):
    @staticmethod
    def make_tracker():
        tracker = object.__new__(PinSplitGasTracker)
        tracker.W = None
        tracker.H = None
        tracker.union_area_records = []
        tracker.split_union_records = {}
        tracker.side_records = {
            "left": {"union_area_records": []},
            "right": {"union_area_records": []},
        }
        return tracker

    def test_overlapping_same_category_masks_count_shared_pixels_once(self):
        first = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=float)
        second = np.array([[3, 1], [7, 1], [7, 5], [3, 5]], dtype=float)

        first_pixels = np.count_nonzero(
            PinSplitGasTracker._rasterize_polygon_union([first], 10, 10)
        )
        second_pixels = np.count_nonzero(
            PinSplitGasTracker._rasterize_polygon_union([second], 10, 10)
        )
        union_pixels = np.count_nonzero(
            PinSplitGasTracker._rasterize_polygon_union([first, second], 10, 10)
        )

        self.assertEqual(first_pixels, 25)
        self.assertEqual(second_pixels, 25)
        self.assertEqual(union_pixels, 35)
        self.assertLess(union_pixels, first_pixels + second_pixels)

    def test_left_and_right_union_areas_add_up_to_global_union(self):
        first = np.array([[1, 1], [5, 1], [5, 5], [1, 5]], dtype=float)
        second = np.array([[3, 1], [7, 1], [7, 5], [3, 5]], dtype=float)
        tracker = self.make_tracker()

        tracker._record_frame_union_areas(
            data={"info": {"width": 10, "height": 10}},
            frame_id=0,
            frame_name="frame_0",
            nm_per_px=2.0,
            split_x_raw=4.0,
            all_polygons=[first, second],
            polygons_by_side={"left": [first], "right": [second]},
        )

        stats = tracker.split_union_records[0]
        self.assertEqual(stats["total_area_nm2"], 35 * 4)
        self.assertEqual(
            stats["left_area_nm2"] + stats["right_area_nm2"],
            stats["total_area_nm2"],
        )
        self.assertEqual(
            stats["left_clipped_area_nm2"] + stats["right_clipped_area_nm2"],
            stats["total_area_nm2"],
        )

    def test_summary_records_preserve_instance_count_but_use_union_total(self):
        tracker = self.make_tracker()
        tracker.area_records = [
            [0, "frame_0", 1.0, 25.0],
            [0, "frame_0", 1.0, 25.0],
        ]
        tracker.union_area_records = [[0, "frame_0", 1.0, 35.0]]

        adjusted = tracker._area_records_with_union_totals()

        self.assertEqual(len(adjusted), 2)
        self.assertEqual(sum(row[3] for row in adjusted), 35.0)


if __name__ == "__main__":
    unittest.main()
