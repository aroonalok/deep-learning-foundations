"""Tests for Early Stopping module for PyTorch"""

import unittest
import os
import logging
import numpy as np
from torch import nn
from early_stopping import EarlyStopping

class TestModel(nn.Module):
    """Single Neuron "Deep" model for test."""

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

class EarlyStoppingTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "/tmp/test_model.ckpt"
        self.stop_training = False
        self.max_epoch = 1e2
        self.model = TestModel().to('cpu')

    def tearDown(self):
        os.remove(self.model_path)

    def stop_training_callback(self):
        self.stop_training = True

    def test_training_stops_on_patience_lost(self):
        val_loss = np.linspace(start=10, stop=9.8, num=100, dtype=np.float32)
        early_stopping = EarlyStopping(
            patience=5, min_progress_delta=1e-1, path=self.model_path, enable_logging=False)
        epoch = 0
        while epoch < self.max_epoch and not self.stop_training:
            early_stopping(val_loss[epoch], self.model,
                           self.stop_training_callback)
            epoch += 1
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(self.stop_training)
        self.assertNotEqual(epoch, self.max_epoch)

    def test_training_ends_on_last_epoch(self):
        val_loss = np.linspace(start=5.0, stop=0.0, num=200, dtype=np.float32)
        early_stopping = EarlyStopping(
            patience=5, min_progress_delta=1e-1, path=self.model_path, enable_logging=False)
        epoch = 0
        while epoch < self.max_epoch and not self.stop_training:
            early_stopping(val_loss[epoch], self.model,
                           self.stop_training_callback)
            epoch += 1
        self.assertTrue(os.path.exists(self.model_path))
        self.assertFalse(self.stop_training)
        self.assertEqual(epoch, self.max_epoch)
            
    def test_early_stopping_logs(self):
        logs_path = "/tmp/test.log"
        logging.basicConfig(filename=logs_path,
                            encoding='utf-8',
                            level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')

        val_loss = np.linspace(start=10, stop=9.8, num=100, dtype=np.float32)
        early_stopping = EarlyStopping(
            patience=5, min_progress_delta=1e-1, path=self.model_path, enable_logging=True)
        epoch = 0
        while epoch < self.max_epoch and not self.stop_training:
            early_stopping(val_loss[epoch], self.model,
                           self.stop_training_callback)
            epoch += 1
        self.assertTrue(os.path.exists(logs_path))

        os.remove(logs_path)

if __name__ == "__main__":
    unittest.main()
