from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class StandardGP:
    """Generic wrapper converting standard GP into our model interface contract."""
    def __init__(self, length_scale=1.0, alpha=1e-1):
        # Allow length_scale to scale up smoothly across multi-dimensional feature spaces
        kernel = C(1.0, (1e-3, 1e6)) * RBF(
            length_scale=length_scale, 
            length_scale_bounds=(1e-2, 1e4) # Bound floor raised to 0.1 to stop length-scale collapse!
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=alpha, 
            n_restarts_optimizer=10,
            normalize_y=True  # Centers target values around their mean (e.g. ~400 K) instead of 0!
        )

    def fit(self, X, Y):
        self.gp.fit(X, Y)

    def predict_with_uncertainty(self, X):
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std**2