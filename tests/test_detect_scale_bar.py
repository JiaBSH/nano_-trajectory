import tempfile
import unittest
from pathlib import Path

import numpy as np

import detect_scale_bar as detector
from rawframe_analysis.inputs import InputMixin


class DetectScaleBarDefaultsTests(unittest.TestCase):
    def test_frame_index_uses_trailing_counter_after_hash_prefix(self) -> None:
        frame_name = "11dd74426e8374ac110c4036c77c09ab_000000000338.jpg"

        self.assertEqual(detector.parse_frame_index(frame_name), 338)

    def test_defaults_detect_short_lower_right_scale_bar(self) -> None:
        args = detector.build_arg_parser().parse_args(["--input-dir", "."])
        image = np.zeros((1080, 1080), dtype=np.uint8)
        image[992:1000, 946:993] = 255

        detection, candidates, _ = detector.detect_one(
            gray=image,
            frame_name="frame_000.jpg",
            threshold=args.threshold,
            kernel_size=(args.kernel_width, args.kernel_height),
            roi_box=args.roi_box,
            roi_x_frac=args.roi_x_frac,
            roi_y_frac=args.roi_y_frac,
            polarity=args.polarity,
            min_width=args.min_width,
            max_height=args.max_height,
            min_ratio=args.min_ratio,
        )

        self.assertIsNotNone(detection)
        self.assertEqual(len(candidates), 1)
        assert detection is not None
        self.assertEqual((detection.x, detection.y), (946, 992))
        self.assertEqual((detection.width, detection.height), (47, 8))

    def test_csv_is_compatible_with_rawframe_scale_loader(self) -> None:
        frame = "11dd74426e8374ac110c4036c77c09ab_000000000000"
        detection = detector.Detection(
            frame=f"{frame}.jpg",
            x=946,
            y=992,
            width=47,
            height=8,
            x1=946,
            y1=995.5,
            x2=992,
            y2=995.5,
            length_px=47,
            candidate_count=1,
            threshold=220,
            kernel_width=35,
            kernel_height=5,
            roi_x0=594,
            roi_y0=648,
            roi_x1=1080,
            roi_y1=1080,
            frame_index=0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "scale_bar_detections.csv"
            detector.write_csv(csv_path, [detection])
            scale_map = InputMixin._load_nm_per_px_map(
                csv_path,
                default_scale_value_nm=20.0,
                allowed_stems={frame},
            )

        self.assertEqual(set(scale_map), {frame})
        self.assertAlmostEqual(scale_map[frame], 20.0 / 47.0)


if __name__ == "__main__":
    unittest.main()
