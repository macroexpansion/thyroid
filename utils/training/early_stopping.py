import torch
import numpy as np


class EarlyStopping(object):
    def __init__(self, mode="min", delta=0, patience=10, percentage=False):
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, percentage):
        if self.mode not in ["min", "max"]:
            raise ValueError("mode " + self.mode + " is unknown!")
        if not percentage:
            if self.mode == "min":
                self.is_better = lambda a, best: a < best - self.delta
            if self.mode == "max":
                self.is_better = lambda a, best: a > best + self.delta
        else:
            if self.mode == "min":
                self.is_better = lambda a, best: a < best - (best * self.delta / 100)
            if self.mode == "max":
                self.is_better = lambda a, best: a > best + (best * self.delta / 100)
