"""Module for saving and loading checkpoints in PyTorch"""

import logging
import torch


class Checkpoint:
    """
        Saves model checkpoints during training. Allows loading a model checkpoint.
        Training can resume from a saved checkpoint if previous training run was 
        abruptly stopped.
    """

    def __init__(self, path, model, optimizer, enable_logging=True):
        self.path = path
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        # Logging
        self._SAVE_CKPT_MSG = "Saving model checkpoint @ Epoch = {epoch}, Training Loss = {loss}"
        self._LOAD_CKPT_MSG = "Model Loaded from checkpoint @ Epoch = {epoch}, Training Loss = {loss}"
        self.logger = logging.getLogger(__name__)
        self.logger.disabled = not enable_logging

    def save(self, epoch, loss):
        self.checkpoint_dict['epoch'] = epoch
        self.checkpoint_dict['loss'] = loss
        self.logger.info(self._SAVE_CKPT_MSG.format(epoch=epoch, loss=loss))
        torch.save(self.checkpoint_dict, self.path)

    def load(self):
        self.checkpoint_dict = torch.load(self.path)
        self.model.load_state_dict(self.checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint_dict['optimizer_state_dict'])
        self.logger.info(self._LOAD_CKPT_MSG.format(epoch=self.epoch(), loss=self.loss()))

    def epoch(self):
        return self.checkpoint_dict['epoch']

    def loss(self):
        return self.checkpoint_dict['loss']
