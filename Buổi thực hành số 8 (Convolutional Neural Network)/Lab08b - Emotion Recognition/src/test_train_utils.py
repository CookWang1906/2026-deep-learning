import unittest

import numpy as np
import torch

from train_utils import balanced_sampler


class BalancedSamplerTests(unittest.TestCase):
    def test_minority_examples_receive_higher_sampling_weight(self):
        labels = np.repeat(np.arange(7), [6, 5, 4, 3, 2, 2, 1])

        sampler = balanced_sampler(labels)

        self.assertEqual(sampler.num_samples, len(labels))
        self.assertTrue(sampler.replacement)
        self.assertGreater(sampler.weights[-1].item(), sampler.weights[0].item())
        self.assertTrue(torch.isfinite(sampler.weights).all())

    def test_missing_emotion_class_is_reported(self):
        labels = np.repeat(np.arange(6), 2)

        with self.assertRaisesRegex(ValueError, "Neutral"):
            balanced_sampler(labels)


if __name__ == "__main__":
    unittest.main()
