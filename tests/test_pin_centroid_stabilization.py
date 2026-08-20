import unittest

import numpy as np

from rawframe_analysis.processing import FrameProcessingMixin


class PinCentroidStabilizationTests(unittest.TestCase):
    @staticmethod
    def make_processor(alpha):
        processor = object.__new__(FrameProcessingMixin)
        processor.pin_centroid_smoothing_alpha = alpha
        processor.stabilized_pin_centroid = None
        return processor

    def test_first_observation_initializes_without_bias(self):
        processor = self.make_processor(0.25)

        result = processor._stabilize_pin_centroid(np.array([12.0, 8.0]))

        np.testing.assert_allclose(result, [12.0, 8.0])

    def test_ema_reduces_single_frame_centroid_jitter(self):
        processor = self.make_processor(0.25)
        processor._stabilize_pin_centroid(np.array([10.0, 10.0]))

        result = processor._stabilize_pin_centroid(np.array([14.0, 6.0]))

        np.testing.assert_allclose(result, [11.0, 9.0])

    def test_alpha_one_preserves_raw_centroid(self):
        processor = self.make_processor(1.0)
        processor._stabilize_pin_centroid(np.array([10.0, 10.0]))

        result = processor._stabilize_pin_centroid(np.array([14.0, 6.0]))

        np.testing.assert_allclose(result, [14.0, 6.0])


if __name__ == "__main__":
    unittest.main()
