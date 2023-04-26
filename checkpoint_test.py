"""Tests for Checkpoint Module"""

import unittest
import os
import torch
from checkpoint import Checkpoint


class TestModel(torch.nn.Module):
    """Single Neuron "Deep" model for test."""

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "/tmp/test_model.tar"
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def tearDown(self):
        os.remove(self.model_path)

    # helper method
    def _create_artifacts(self):
        model = TestModel().to('cpu')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_checkpoint(self):
        # Configuration for saving checkpoints
        saved_model, saved_optimizer = self._create_artifacts()
        saved_checkpoint = Checkpoint(path=self.model_path,
                                      model=saved_model,
                                      optimizer=saved_optimizer,
                                      enable_logging=False)

        epoch, max_epoch = 1, 100
        save_ckpt_every_n = 7
        loss_fn = torch.nn.MSELoss()
        checkpoint_loss = None

        # Training loop
        saved_model.train()
        while epoch < max_epoch:
            # Create data
            X = torch.rand(1)
            y = 7*X

            # Forward pass
            pred = saved_model(X)

            # Compute prediction error
            loss = loss_fn(pred, y)

            # Backpropagation
            saved_optimizer.zero_grad()
            loss.backward()
            saved_optimizer.step()

            if (epoch) % save_ckpt_every_n == 0:
                checkpoint_loss = loss
                saved_checkpoint.save(epoch, checkpoint_loss)

            epoch += 1

        self.assertTrue(os.path.exists(self.model_path))

        # Configuration for loading checkpoints
        loaded_model, loaded_optimizer = self._create_artifacts()
        loaded_checkpoint = Checkpoint(path=self.model_path,
                                       model=loaded_model,
                                       optimizer=loaded_optimizer,
                                       enable_logging=False)
        loaded_checkpoint.load()
        self.assertEqual(loaded_checkpoint.epoch(),
                         max_epoch - max_epoch % save_ckpt_every_n)
        self.assertAlmostEqual(loaded_checkpoint.loss(), checkpoint_loss)


if __name__ == "__main__":
    unittest.main()
