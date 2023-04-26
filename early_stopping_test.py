"""Tests for Early Stopping module for PyTorch"""

import unittest
import numpy as np
from early_stopping import EarlyStopping


class EarlyStoppingTest(unittest.TestCase):
    def setUp(self):
        self.stop_training = False
        self.max_epoch = 1e2

    def stop_training_callback(self, *args, **kwargs):
        self.stop_training = True

    def save_checkpoint_callback(self, *args, **kwargs):
        # does complex computations in subtle dimensions.
        pass

    def test_training_stops_on_patience_lost(self):
        val_loss = np.linspace(start=10, stop=9.8, num=100, dtype=np.float32)
        early_stopping = EarlyStopping(patience=5, min_progress_delta=1e-1, enable_logging=False)
        epoch = 0
        while epoch < self.max_epoch and not self.stop_training:
            early_stopping(epoch, val_loss[epoch], self.save_checkpoint_callback, self.stop_training_callback)
            epoch += 1
        self.assertTrue(self.stop_training)
        self.assertNotEqual(epoch, self.max_epoch)

    def test_training_ends_on_last_epoch(self):
        val_loss = np.linspace(start=5.0, stop=0.0, num=200, dtype=np.float32)
        early_stopping = EarlyStopping(patience=5, min_progress_delta=1e-1, enable_logging=False)
        epoch = 0
        while epoch < self.max_epoch and not self.stop_training:
            early_stopping(epoch, val_loss[epoch], self.save_checkpoint_callback, self.stop_training_callback)
            epoch += 1
        self.assertFalse(self.stop_training)
        self.assertEqual(epoch, self.max_epoch)


if __name__ == "__main__":
    unittest.main()
