import unittest

import numpy as np

from src.preprocessing.landmark_layout import LANDMARK_DIM, RIGHT_HAND_OFFSET
from src.preprocessing.missing_data import interpolate_missing_data
from src.preprocessing.normalization import (
    canonical_normalize_landmarks,
    normalize_landmarks,
)


class MissingDataTest(unittest.TestCase):
    def test_interpolates_middle_and_edges(self):
        landmarks = np.array(
            [[np.nan, 10.0], [1.0, np.nan], [3.0, 30.0], [np.nan, np.nan]],
            dtype=np.float32,
        )

        actual = interpolate_missing_data(landmarks)

        expected = np.array(
            [[1.0, 10.0], [1.0, 20.0], [3.0, 30.0], [3.0, 30.0]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected)


class NormalizationTest(unittest.TestCase):
    def setUp(self):
        self.landmarks = np.full((1, LANDMARK_DIM), np.nan, dtype=np.float32)
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[:, 0] = np.arange(21)
        hand[:, 1] = np.arange(21) ** 2
        hand[:, 2] = 1.0
        self.landmarks[0, RIGHT_HAND_OFFSET:] = hand.flatten()

    def test_wrist_scale_normalization(self):
        actual = normalize_landmarks(self.landmarks)
        hand = actual[0, RIGHT_HAND_OFFSET:].reshape(21, 3)

        np.testing.assert_allclose(hand[0], np.zeros(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(hand[9])), 1.0, places=6)

    def test_canonical_normalization(self):
        actual = canonical_normalize_landmarks(self.landmarks)
        hand = actual[0, RIGHT_HAND_OFFSET:].reshape(21, 3)

        np.testing.assert_allclose(hand[0], np.zeros(3), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(hand[17])), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
