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

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class MTGPR:
    """
    Gaussian Process architecture tailored for small datasets (N <= 50) of energetic precursors.
    Implements dimensionality reduction (PCA) and a Rational Quadratic kernel to prevent overfitting.
    """
    def __init__(self, variance_retained=0.95, max_components=5):
        self.variance_retained = variance_retained
        self.max_components = max_components
        
        # Preprocessing pipeline
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=self.variance_retained)
        
        # 
        self.model = None
        self.train_variances = {}
        self.target_names = []
        
        # Define target properties and whether they require log-transformation (non-negative bounds)
        self.targets = {
            'T_m': {'log_transform': True},
            'dH_fus': {'log_transform': True},
            'dH_f': {'log_transform': False}
        }

    def _transform_target(self, y, property_name):
        """Applies log transformation for strictly positive thermodynamic properties."""
        if self.targets[property_name]['log_transform']:
            # Add small epsilon to prevent log(0) if data is flawed
            return np.log(np.maximum(y, 1e-6))
        return y

    def _inverse_transform_target(self, mu, sigma, property_name):
        """
        Maps log-space predictions back to physical space.
        Applies log-normal distribution back-transformation: E[Y] = exp(mu + sigma^2 / 2)
        """
        if self.targets[property_name]['log_transform']:
            y_physical = np.exp(mu + 0.5 * sigma**2)
            # Approximate standard deviation in physical space
            sigma_physical = y_physical * np.sqrt(np.exp(sigma**2) - 1)
            return y_physical, sigma_physical
        return mu, sigma

    def fit(self, X, Y):
        """
        Fits the feature compression pipeline and the GP models.
        X: Feature matrix of descriptors (N x D)
        Y: Target matrix (N x 3) containing [T_m, dH_fus, dH_f]
        """
        N, D = X.shape
        print(f"Training on dataset size N={N}, Original Features D={D}")
        
        # 1. Feature Compression Pipeline (Section 2.2.2 Step 1)
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Enforce max components to maintain N / d_eff >= 10 heuristically
        if X_pca.shape[1] > self.max_components:
            print(f"Capping PCA components at {self.max_components} to prevent overfitting.")
            X_pca = X_pca[:, :self.max_components]
            self.pca.components_ = self.pca.components_[:self.max_components, :]
            
        print(f"Compressed feature space: d_eff = {X_pca.shape[1]}")

        # 2. Kernel Formulation (Section 2.2.2 Step 2)
        # Matern added for physical function modeling, WhiteKernel justified by 2401.17898v2.pdf for independent noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
                 Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + \
                 RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2)) + \
                 WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))

        # 3. Fit a single Multi-Task GP (MTGP) for all target properties simultaneously
        self.target_names = list(self.targets.keys())
        Y_transformed = np.zeros_like(Y)
        
        for i, prop in enumerate(self.target_names):
            Y_transformed[:, i] = self._transform_target(Y[:, i], prop)
            
        # Initialize GP with multiple restarts. Passing a 2D Y-matrix optimizes the 
        # joint log-marginal likelihood across all properties simultaneously.
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            normalize_y=True, # Standardizes target variables internally per property
            random_state=42
        )
        
        self.model.fit(X_pca, Y_transformed)
        
        # Store the training predictive variance for extrapolation flagging
        _, train_std = self.model.predict(X_pca, return_std=True)
        
        # sklearn's multi-output return_std is typically 1D (shared variance in normalized space) 
        # or 2D depending on the internal scaling. We handle both cleanly.
        for i, prop in enumerate(self.target_names):
            std_for_prop = train_std if train_std.ndim == 1 else train_std[:, i]
            self.train_variances[prop] = np.mean(std_for_prop**2)
            
        print(f"[Joint MTGP] Optimized Shared Kernel: {self.model.kernel_}")

    def predict(self, X_new):
        """
        Predicts properties for new molecules and flags high-risk extrapolations.
        Returns a dictionary containing predictions, standard deviations, and flags.
        """
        # Compress new features
        X_scaled = self.scaler.transform(X_new)
        X_pca = self.pca.transform(X_scaled)
        if X_pca.shape[1] > self.max_components:
            X_pca = X_pca[:, :self.max_components]

        results = {}
        
        # Predict all properties simultaneously in transformed space
        mu_z_all, std_z_all = self.model.predict(X_pca, return_std=True)
        
        for i, prop in enumerate(self.target_names):
            # Extract predictions for the specific task
            mu_z = mu_z_all[:, i]
            std_z = std_z_all if std_z_all.ndim == 1 else std_z_all[:, i]
            
            # Map back to physical space (Section 2.2.1)
            mu_y, std_y = self._inverse_transform_target(mu_z, std_z, prop)
            
            # Uncertainty-Gated Filtering (Section 2.2.3)
            # Flag if variance exceeds 2x the average training variance
            train_std_thresh = np.sqrt(self.train_variances[prop])
            flags = std_z > (2.0 * train_std_thresh)
            
            results[prop] = {
                'prediction': mu_y,
                'uncertainty': std_y,
                'high_risk_flag': flags
            }
            
        return results