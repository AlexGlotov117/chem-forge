import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class StandardGP:
    """Generic wrapper converting standard GP into our model interface contract."""
    def __init__(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=3)

    def fit(self, X, Y):
        # Handles single-task or multi-task target array Y
        self.gp.fit(X, Y)

    def predict_with_uncertainty(self, X):
        # returns mean and std; square std to get variance
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std**2