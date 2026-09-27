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
from sklearn.preprocessing import StandardScaler
import torch
import gpytorch

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
import math

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# 1. Low-Level GPyTorch Core Architecture
# =====================================================================
class _GPyTorchMTGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_i, train_y, likelihood, mean_module=None, covar_module=None, num_tasks=3):
        super().__init__((train_x, train_i), train_y, likelihood)

        # self.mean_module = gpytorch.means.LinearMean(input_size=train_x.shape[-1])
        # self.mean_module = gpytorch.means.ZeroMean()
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()

        # Spatial feature kernel (e.g., Matern 3/2 with ARD)
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=1.5,
                ard_num_dims=train_x.shape[-1],
                lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-2),
            )
        )

        # Task covariance kernel (learns task-to-task correlation matrix)
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=1
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
    def __init__(
        self,
        mean_module=None,
        covar_module=None,
        num_tasks=3,
        lr=0.01,
        num_epochs=1000,
        task_names=None,
        task_noise_map={0: 1e-3, 1: 1e-4, 2: 5e-2},
        seed=42,
    ):
        self.num_tasks = num_tasks
        self.lr = lr
        self.num_epochs = num_epochs
        self.task_names = (
            task_names
            if task_names
            else [f"Task_{i}" for i in range(num_tasks)]
        )
        self.task_noise_map = task_noise_map

        self.x_scaler = StandardScaler()
        self.train_x = None
        self.train_i = None
        self.train_y = None

        self.y_mean_ = None
        self.y_std_ = None

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.model = None
        self.likelihood = None

        self.fig = None
        self.ax1 = None
        self.ax2 = None

        # Tracking metrics history
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_overall_rmse_scaled": [],
            "train_per_task_rmse_phys": {name: [] for name in self.task_names},
            "test_loss": [],
            "test_overall_rmse_scaled": [],
            "test_per_task_rmse_phys": {name: [] for name in self.task_names},
        }

        # # 1. Python & OS seeds
        # random.seed(seed)
        # os.environ["PYTHONHASHSEED"] = str(seed)
        # np.random.seed(seed)

        # # 2. PyTorch seeds
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        # # 3. CUDA Deterministic Operations
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        # # Force PyTorch to use deterministic algorithms
        # try:
        #     torch.use_deterministic_algorithms(True)
        # except AttributeError:
        #     pass  # Older PyTorch fallback

        # # 4. Disable GPyTorch randomized fast solvers during training
        # gpytorch.settings.fast_computations(
        #     covar_root_decomposition=False,
        #     log_prob=False,
        #     solves=False,
        # )

    def _prepare_tensors(self, X: np.ndarray, Y: np.ndarray, fit_scaler=False):
        """Helper to scale features/targets and unroll non-NaN entries into GP

        tensors.
        """
        if fit_scaler:
            X_scaled = self.x_scaler.fit_transform(X)
            self.y_mean_ = np.nanmean(Y, axis=0)
            self.y_std_ = np.nanstd(Y, axis=0)
            self.y_std_[self.y_std_ == 0] = 1.0
        else:
            X_scaled = self.x_scaler.transform(X)

        Y_scaled = (Y - self.y_mean_) / self.y_std_

        x_flat, i_flat, y_flat = [], [], []
        for row_idx in range(X_scaled.shape[0]):
            for task_idx in range(self.num_tasks):
                val = Y_scaled[row_idx, task_idx]
                if not np.isnan(val):
                    x_flat.append(X_scaled[row_idx])
                    i_flat.append(task_idx)
                    y_flat.append(val)

        x_tensor = torch.tensor(np.array(x_flat), dtype=torch.float32)
        i_tensor = torch.tensor(np.array(i_flat), dtype=torch.long)
        y_tensor = torch.tensor(np.array(y_flat), dtype=torch.float32)

        return x_tensor, i_tensor, y_tensor

    def _evaluate_dataset_metrics(self, X_raw: np.ndarray, Y_raw: np.ndarray):
        """Calculates both normalized (cross-task) and physical (per-task)

        metrics.
        """
        # Predict both physical and scaled units
        Y_pred_phys, _, Y_pred_scaled, _ = self.predict(X_raw)
        Y_true_scaled = (Y_raw - self.y_mean_) / self.y_std_

        per_task_rmse_phys = {}
        all_true_scaled, all_pred_scaled = [], []

        for task_idx, task_name in enumerate(self.task_names):
            valid_mask = ~np.isnan(Y_raw[:, task_idx])
            if np.sum(valid_mask) == 0:
                per_task_rmse_phys[task_name] = np.nan
                continue

            y_t_phys = Y_raw[valid_mask, task_idx]
            y_p_phys = Y_pred_phys[valid_mask, task_idx]

            # Physical units per task (e.g., °C, MPa)
            rmse_phys = np.sqrt(mean_squared_error(y_t_phys, y_p_phys))
            per_task_rmse_phys[task_name] = rmse_phys

            # Collect scaled z-scores across tasks for overall aggregate metric
            all_true_scaled.extend(Y_true_scaled[valid_mask, task_idx])
            all_pred_scaled.extend(Y_pred_scaled[valid_mask, task_idx])

        # Overall aggregate metric using dimensionless Z-scores
        overall_rmse_scaled = np.sqrt(
            mean_squared_error(all_true_scaled, all_pred_scaled)
        )

        return overall_rmse_scaled, per_task_rmse_phys

    def save_history_to_csv(self, filepath: str = "training_history.csv"):
        """Exports all tracked metrics across epochs to a single CSV file."""
        # Start with base metrics
        data = {
            "epoch": self.history["epoch"],
            "train_loss": self.history["train_loss"],
        }

        # Add optional test loss if present
        if "test_loss" in self.history and len(self.history["test_loss"]) > 0:
            data["test_loss"] = self.history["test_loss"]

        # Add per-task train RMSE metrics
        if "train_per_task_rmse_phys" in self.history:
            for task_name, values in self.history[
                "train_per_task_rmse_phys"
            ].items():
                data[f"train_rmse_{task_name}"] = values

        # Add per-task test RMSE metrics
        if "test_per_task_rmse_phys" in self.history:
            for task_name, values in self.history[
                "test_per_task_rmse_phys"
            ].items():
                if len(values) > 0:
                    data[f"test_rmse_{task_name}"] = values

        # Convert to pandas DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Training metrics successfully exported to {filepath}")

    def _init_live_plot(self):
        """Initializes a dynamic Grid layout for Loss + Per-Task subplots."""
        plt.ion()  # Non-blocking interactive mode

        # Total plots = 1 (Loss) + N (One per Task)
        total_plots = 1 + self.num_tasks

        # Calculate grid dimensions (up to 3 columns)
        self.n_cols = min(3, total_plots)
        self.n_rows = math.ceil(total_plots / self.n_cols)

        self.fig, self.axes = plt.subplots(
            self.n_rows,
            self.n_cols,
            figsize=(5 * self.n_cols, 4 * self.n_rows),
            squeeze=False,  # Keep 2D array structure
        )
        self.fig.suptitle("Live Multi-Task Training Metrics", fontsize=14)

        # Flatten axes array for simple 1D indexing
        self.ax_flat = self.axes.flatten()

        plt.show(block=False)

    def _update_live_plot(self, has_test: bool):
        """Dynamically updates each task's individual error plot in real-time."""
        if (
            self.fig is None
            or not hasattr(self, "ax_flat")
            or not plt.fignum_exists(self.fig.number)
        ):
            self._init_live_plot()

        try:
            # Clear all active subplot axes
            for ax in self.ax_flat:
                ax.clear()

            epochs = self.history["epoch"]

            # --- Graph 0: Overall Model Loss (NLML) ---
            ax_loss = self.ax_flat[0]
            ax_loss.plot(
                epochs,
                self.history["train_loss"],
                label="Train Loss",
                color="navy",
                linewidth=2,
            )
            if has_test and len(self.history["test_loss"]) > 0:
                ax_loss.plot(
                    epochs,
                    self.history["test_loss"],
                    label="Test Loss",
                    color="crimson",
                    linewidth=2,
                    linestyle="--",
                )

            ax_loss.set_xlabel("Epochs")
            ax_loss.set_ylabel("Negative Log Marginal Likelihood")
            ax_loss.set_title("Overall Model Loss (Residuals)")
            ax_loss.set_yscale("symlog")
            ax_loss.grid(True, linestyle=":", alpha=0.6)
            ax_loss.legend(loc="upper right")

            # --- Graphs 1 to N: Individual Plot per Task ---
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for idx, task_name in enumerate(self.task_names):
                ax_task = self.ax_flat[idx + 1]  # Task plots start at index 1
                color = colors[idx % len(colors)]

                # Train curve
                tr_phys = self.history["train_per_task_rmse_phys"][task_name]
                ax_task.plot(
                    epochs,
                    tr_phys,
                    label=f"Train {task_name}",
                    color=color,
                    linewidth=2,
                )

                # Test curve
                if (
                    has_test
                    and len(self.history["test_per_task_rmse_phys"][task_name]) > 0
                ):
                    ts_phys = self.history["test_per_task_rmse_phys"][task_name]
                    ax_task.plot(
                        epochs,
                        ts_phys,
                        label=f"Test {task_name}",
                        color=color,
                        linewidth=2,
                        linestyle="--",
                    )

                ax_task.set_xlabel("Epochs")
                ax_task.set_ylabel("RMSE")
                ax_task.set_title(f"Target Error: {task_name}")
                ax_task.set_yscale("log")
                ax_task.grid(True, linestyle=":", alpha=0.6)
                ax_task.legend(loc="upper right")

            # Hide any unused grid subplot slots
            for unused_idx in range(1 + self.num_tasks, len(self.ax_flat)):
                self.ax_flat[unused_idx].set_visible(False)

            plt.tight_layout()

            # Canvas redraw (Non-blocking GUI update)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)
                

        except Exception:
            # Prevent crashes if window is manually closed or running headless
            pass

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray = None,
        Y_test: np.ndarray = None,
        eval_freq: int = 100,
        live_plot: bool = True,
    ):
        """Fit model and log training/testing metrics every `eval_freq` epochs.

        Parameters
        ----------
        eval_freq : int
            Interval of epochs at which metrics (loss, physical RMSE per task)
            are evaluated and saved.
        """
        
        # 1. Prepare Training Tensors
        self.train_x, self.train_i, self.train_y = self._prepare_tensors(
            X_train, Y_train, fit_scaler=True
        )

        # 2. Prepare Testing Tensors (if available)
        has_test = X_test is not None and Y_test is not None
        if has_test:
            test_x, test_i, test_y = self._prepare_tensors(
                X_test, Y_test, fit_scaler=False
            )

        # 3. Setup Likelihood & Model
        train_noise = torch.tensor(
            [self.task_noise_map[i.item()] for i in self.train_i],
            dtype=torch.float32,
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
        )


        self.model = _GPyTorchMTGPModel(
            self.train_x,
            self.train_i,
            self.train_y,
            self.likelihood,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            num_tasks=self.num_tasks,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        print(
            f"[MTGPPipeline] Optimization started ({self.num_epochs} epochs)..."
        )

        with gpytorch.settings.cholesky_jitter(1e-7):
            for epoch in range(1, self.num_epochs + 1):
                # Standard Train Optimization Step
                self.model.train()
                self.likelihood.train()
                optimizer.zero_grad()

                train_output = self.model(self.train_x, self.train_i)
                train_loss = -mll(train_output, self.train_y)
                train_loss.backward()
                optimizer.step()

                # Evaluate and log metrics at requested interval
                if (
                    epoch == 1
                    or epoch % eval_freq == 0
                    or epoch == self.num_epochs
                ):
                    self.model.eval()
                    self.likelihood.eval()

                    # Compute losses
                    with torch.no_grad():
                        tr_loss_val = train_loss.item()

                        if has_test:
                            test_output = self.model(test_x, test_i)
                            test_noise = torch.tensor(
                                [self.task_noise_map[i.item()] for i in test_i],
                                dtype=torch.float32,
                            )
                            test_dist = self.likelihood(
                                test_output, noise=test_noise
                            )
                            ts_loss_val = (-mll(test_dist, test_y)).item()

                    # Compute physical (per-task) and scaled (overall) metrics
                    tr_rmse_scaled, tr_rmse_phys = (
                        self._evaluate_dataset_metrics(X_train, Y_train)
                    )

                    if has_test:
                        ts_rmse_scaled, ts_rmse_phys = (
                            self._evaluate_dataset_metrics(X_test, Y_test)
                        )

                    # Store into history
                    self.history["epoch"].append(epoch)
                    self.history["train_loss"].append(tr_loss_val)
                    self.history["train_overall_rmse_scaled"].append(
                        tr_rmse_scaled
                    )
                    for t_name in self.task_names:
                        self.history["train_per_task_rmse_phys"][
                            t_name
                        ].append(tr_rmse_phys[t_name])

                    if has_test:
                        self.history["test_loss"].append(ts_loss_val)
                        self.history["test_overall_rmse_scaled"].append(
                            ts_rmse_scaled
                        )
                        for t_name in self.task_names:
                            self.history["test_per_task_rmse_phys"][
                                t_name
                            ].append(ts_rmse_phys[t_name])

                    # Trigger Live Plot Refresh
                    if live_plot:
                        self._update_live_plot(has_test=has_test)

                    # Print Log Summary
                    log_msg = f"Epoch {epoch:4d}/{self.num_epochs} | Train Loss: {tr_loss_val:.4f} | Train RMSE (z-score): {tr_rmse_scaled:.4f}"
                    if has_test:
                        log_msg += f" || Test Loss: {ts_loss_val:.4f} | Test RMSE (z-score): {ts_rmse_scaled:.4f}"
                    print(log_msg)

        plt.ioff()
        self.save_history_to_csv(filepath='data/output/history.csv')
        if live_plot and self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.show()  # Keeps final figure open when script finishes
        
        print("[MTGPPipeline] Fit completed successfully.")

    def predict(self, X_test: np.ndarray):
        self.model.eval()
        self.likelihood.eval()

        X_test_scaled = self.x_scaler.transform(X_test)
        n_samples = X_test_scaled.shape[0]

        means_scaled = np.zeros((n_samples, self.num_tasks))
        stds_scaled = np.zeros((n_samples, self.num_tasks))

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for task_idx in range(self.num_tasks):
                test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
                test_i = torch.full((n_samples,), task_idx, dtype=torch.long)
                test_noise = torch.full(
                    (n_samples,), self.task_noise_map[task_idx], dtype=torch.float32
                )

                pred_dist = self.likelihood(self.model(test_x, test_i), noise=test_noise)
                means_scaled[:, task_idx] = pred_dist.mean.numpy()
                stds_scaled[:, task_idx] = np.sqrt(pred_dist.variance.numpy())

        mean_unscaled = (means_scaled * self.y_std_) + self.y_mean_
        std_unscaled = stds_scaled * self.y_std_

        return mean_unscaled, std_unscaled, means_scaled, stds_scaled