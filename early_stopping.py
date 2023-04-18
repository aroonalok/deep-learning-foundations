"""Early Stopping module for PyTorch"""

import numpy as np
import torch

class EarlyStopping:
    """
        Implements Early Stopping.

        Reference : https://www.deeplearningbook.org/contents/regularization.html
    """

    def __init__(self, patience=10, min_progress_delta = 1e-1, path="model.ckpt", verbose=False):
        self.patience = patience
        self.min_progress_delta = min_progress_delta
        self.path = path
        self.verbose = verbose
        self.least_val_loss = np.inf
        self.no_progress_count = 0
        self._VAL_LOSS_IMPROVE_MSG = "Validation loss improved from {old_val_loss} -> {new_val_loss}, updating checkpoint."
        self._NO_PROGRESS_MSG = "Validation loss did not improve. Remaining patience = {remaining_patience}"
        self._IMPATIENCE_MSG = "No patience left. Stopping Training."

    def __call__(self, validation_loss, model, stop_training_callback):
        if validation_loss + self.min_progress_delta < self.least_val_loss:
            if self.verbose:
                print(self._VAL_LOSS_IMPROVE_MSG.format(old_val_loss=self.least_val_loss,new_val_loss=validation_loss))
            self.least_val_loss = validation_loss
            torch.save(model.state_dict(), self.path)
            self.no_progress_count = 0
        else:
            self.no_progress_count += 1
            if self.verbose:
                print(self._NO_PROGRESS_MSG.format(remaining_patience=self.patience - self.no_progress_count))
            if self.no_progress_count == self.patience:
              if self.verbose:
                print(self._IMPATIENCE_MSG)
              stop_training_callback() 