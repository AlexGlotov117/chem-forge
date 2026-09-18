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

class MTGPR_sklearn:
    """
    Gaussian Process architecture tailored for small datasets (N <= 50) of energetic precursors.
    Implements dimensionality reduction (PCA) and a Rational Quadratic kernel to prevent overfitting.
    """
    def __init__(self):
        # self.variance_retained = variance_retained
        # self.max_components = max_components
        
        # # Preprocessing pipeline
        # self.scaler = MinMaxScaler()
        # self.pca = PCA(n_components=self.variance_retained)
        self.scaler = StandardScaler()
        
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
        
        # # 1. Feature Compression Pipeline (Section 2.2.2 Step 1)
        X_scaled = self.scaler.fit_transform(X)
        # X_pca = self.pca.fit_transform(X_scaled)
        
        # # Enforce max components to maintain N / d_eff >= 10 heuristically
        # if X_pca.shape[1] > self.max_components:
        #     print(f"Capping PCA components at {self.max_components} to prevent overfitting.")
        #     X_pca = X_pca[:, :self.max_components]
        #     self.pca.components_ = self.pca.components_[:self.max_components, :]
            
        # print(f"Compressed feature space: d_eff = {X_pca.shape[1]}")

        # 2. Kernel Formulation (Section 2.2.2 Step 2)
        # Matern added for physical function modeling, WhiteKernel justified by 2401.17898v2.pdf for independent noise
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
            + RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=(0.1, 10.0),
                alpha_bounds=(0.1, 10.0),
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-4, 1e-2))
        )

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
        
        self.model.fit(X_scaled, Y_transformed)
        
        # Store the training predictive variance for extrapolation flagging
        _, train_std = self.model.predict(X_scaled, return_std=True)
        
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
        # X_pca = self.pca.transform(X_scaled)
        # if X_pca.shape[1] > self.max_components:
        #     X_pca = X_pca[:, :self.max_components]

        results = {}
        
        # Predict all properties simultaneously in transformed space
        mu_z_all, std_z_all = self.model.predict(X_scaled, return_std=True)
        
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

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import gpytorch

class JobackMean(gpytorch.means.Mean):
    def __init__(self, train_priors_2d, test_priors_2d):
        super().__init__()
        self.train_priors = torch.tensor(train_priors_2d, dtype=torch.float32)
        self.test_priors = torch.tensor(test_priors_2d, dtype=torch.float32)
        
        # Buffer to hold the currently active 1D vector (sized 136 for train, N for test)
        self.register_buffer("active_priors", torch.zeros(0))

    def apply_train_mask(self, row_indices, task_indices):
        """Slices the 2D training matrix down to the exact 1D shape of valid targets."""
        self.active_priors = self.train_priors[row_indices, task_indices]

    def set_predict_task(self, task_idx):
        """Pulls the 1D test vector for the specific property being predicted."""
        self.active_priors = self.test_priors[:, task_idx]

    def forward(self, x):
        return self.active_priors

# =====================================================================
# 1. Low-Level GPyTorch Core Architecture
# =====================================================================
class _GPyTorchMTGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_i, train_y, likelihood, mean_module=None, num_tasks=3):
        super().__init__((train_x, train_i), train_y, likelihood)

        # self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[-1])
        # self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()

        # Spatial feature kernel (e.g., Matern 3/2 with ARD)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=1.5,
                ard_num_dims=train_x.shape[-1],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-2),
            )
        )

        # Task covariance kernel (learns task-to-task correlation matrix)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=2
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x)

        # Evaluate spatial covariance and task covariance together
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)

        # Combine feature distance and task correlation
        covar = covar_x * covar_i
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)


# =====================================================================
# 2. Generic High-Level MTGP Pipeline Class
# =====================================================================

class MTGPPipeline:
    """
    Generic Multi-Task Gaussian Process wrapper handling scaling, NaN masking,
    GPyTorch joint optimization, and physical target recovery (T_m reconstruction).
    """
    def __init__(self, mean_module=None, num_tasks=3, lr=0.01, num_epochs=1000):
        self.num_tasks = num_tasks
        self.lr = lr
        self.num_epochs = num_epochs

        self.x_scaler = StandardScaler()
        self.train_x = None
        self.train_i = None
        self.train_y = None
        
        self.y_mean_ = None
        self.y_std_ = None

        self.mean_module = mean_module
        self.model = None
        self.likelihood = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        # Standardize features X
        X_scaled = self.x_scaler.fit_transform(X)

        # Standardize targets Y independently per task (ignoring NaNs)
        self.y_mean_ = np.nanmean(Y, axis=0)
        self.y_std_ = np.nanstd(Y, axis=0)
        self.y_std_[self.y_std_ == 0] = 1.0

        Y_scaled = (Y - self.y_mean_) / self.y_std_

        # Unroll only valid (non-NaN) observations
        x_flat, i_flat, y_flat = [], [], []
        
        for row_idx in range(X_scaled.shape[0]):
            for task_idx in range(self.num_tasks):
                val = Y_scaled[row_idx, task_idx]
                if not np.isnan(val):
                    x_flat.append(X_scaled[row_idx])
                    i_flat.append(task_idx)
                    y_flat.append(val)
                    

        self.train_x = torch.tensor(np.array(x_flat), dtype=torch.float32)
        self.train_i = torch.tensor(np.array(i_flat), dtype=torch.long)
        self.train_y = torch.tensor(np.array(y_flat), dtype=torch.float32)

        task_noise_map = {0: 1e-3, 1: 1e-4, 2: 5e-2}  
        train_noise = torch.tensor([task_noise_map[i.item()] for i in self.train_i], dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=train_noise,
            learn_additional_noise=False
        )

        self.model = _GPyTorchMTGPModel(
            self.train_x,
            self.train_i,
            self.train_y,
            self.likelihood,
            self.mean_module,
            num_tasks=self.num_tasks,
        )

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        print(f"[MTGPPipeline] Optimization started ({self.num_epochs} epochs)...")

        with gpytorch.settings.cholesky_jitter(1e-3):
            for epoch in range(self.num_epochs):
                optimizer.zero_grad()
                output = self.model(self.train_x, self.train_i)
                loss = -mll(output, self.train_y)
                loss.backward()
                optimizer.step()

        print("[MTGPPipeline] Fit completed successfully.")

    def predict(self, X_test: np.ndarray):
        self.model.eval()
        self.likelihood.eval()

        X_test_scaled = self.x_scaler.transform(X_test)
        n_samples = X_test_scaled.shape[0]

        # Allocate arrays for full prediction matrix across all tasks
        means_scaled = np.zeros((n_samples, self.num_tasks))
        stds_scaled = np.zeros((n_samples, self.num_tasks))

        task_noise_map = {0: 1e-3, 1: 1e-4, 2: 5e-2}

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Query the model for each target task independently
            for task_idx in range(self.num_tasks):
                test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
                test_i = torch.full(
                    (n_samples,), task_idx, dtype=torch.long
                )

                test_noise = torch.full((n_samples,), task_noise_map[task_idx], dtype=torch.float32)
                pred_dist = self.likelihood(self.model(test_x, test_i), noise=test_noise)

                means_scaled[:, task_idx] = pred_dist.mean.numpy()
                stds_scaled[:, task_idx] = np.sqrt(pred_dist.variance.numpy())

        # Inverse scaling back to physical units
        mean_unscaled = (means_scaled * self.y_std_) + self.y_mean_
        std_unscaled = stds_scaled * self.y_std_

        return mean_unscaled, std_unscaled