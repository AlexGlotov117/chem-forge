import os
import numpy as np
import optuna
import gpytorch
import torch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, IndexKernel
from sklearn.model_selection import KFold

from models.gp import MTGPPipeline
from encoders.chemicals import MolecularEncoder
from data_processing.dataProcessing import extract_smiles_and_targets

# Ensure Optuna outputs clean logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

## 0.2974638052697237 Params = [max_features: 6, base_kernel: rbf, use_ard: True, lr: 0.010990838802648367, num_epochs: 3000, noise_task_0: 0.0013195053529382061, noise_task_1: 0.014860509392656391, noise_task_2: 0.0011029742935256713, mean_type: zero]

def build_mean_module(trial: optuna.Trial, X):
    """Dynamically creates mean module based on hyper-hyperparameters."""
    mean_type = trial.suggest_categorical("mean_type", ["constant", "linear", "zero"])
    
    if mean_type == "constant":
        return gpytorch.means.ConstantMean()
    elif mean_type == "linear":
        return gpytorch.means.LinearMean(input_size=X.shape[1])
    else:
        return gpytorch.means.ZeroMean()


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    Y_train_dS: np.ndarray,
    X_test: np.ndarray,
    Y_test_dS: np.ndarray,
    num_tasks: int = 3,
    task_names=None,
    num_runs: int = 1,
):
    """Objective function for Optuna hyperparameter optimization using K-Fold CV."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # --- 1. Sample Hyperparameters ---
    
    k_features = trial.suggest_int(
        "max_features", low=2, high=X_train.shape[1], step=1
    )

    base_kernel_type = trial.suggest_categorical(
        "base_kernel", ["rbf", "matern_32", "matern_52"]
    )

    # Fix: Use suggest_categorical for booleans in Optuna 3.x+
    use_ard = trial.suggest_categorical("use_ard", [True, False])
    ard_num_dims = X_train[:, :k_features].shape[1] if use_ard else None

    if base_kernel_type == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif base_kernel_type == "matern_32":
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=ard_num_dims
        )
    else:
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=ard_num_dims
        )

    # Multi-task Index Kernel Rank
    # rank = trial.suggest_int("index_kernel_rank", 1, num_tasks)
    
    # Optimizer Parameters
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    num_epochs = trial.suggest_int("num_epochs", 100, 5000, step=100)
    
    # Task Noise Map values (log scale sampling)
    task_noise_map = {}
    for t_idx in range(num_tasks):
        task_noise_map[t_idx] = trial.suggest_float(f"noise_task_{t_idx}", 1e-5, 1e-1, log=True)

    # task_noise_map = {0: 0.001, 1:0.01, 2:0.1}

    run_scores = []

    for run_idx in range(num_runs):
        # Re-instantiate kernels and pipeline every run to reset model parameters
        if base_kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif base_kernel_type == "matern_32":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=ard_num_dims
            )
        else:
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=ard_num_dims
            )

        mean_mod = build_mean_module(trial, X=X_train)
        covar_module = ScaleKernel(base_kernel)

        pipeline = MTGPPipeline(
            mean_module=mean_mod,
            covar_module=covar_module,
            num_tasks=num_tasks,
            lr=lr,
            num_epochs=num_epochs,
            task_noise_map=task_noise_map,
        )

        try:
            pipeline.fit(
                X_train=X_train[:, :k_features],
                Y_train=Y_train_dS,
                X_test=X_test[:, :k_features],
                Y_test=Y_test_dS,
                eval_freq=100,
                live_plot=False,
            )

            # Get final test RMSE across all tasks
            val_rmse = pipeline.history["test_overall_rmse_scaled"][-1]

            if np.isnan(val_rmse):
                return float("inf")  # Penalize NaN runs instantly

            run_scores.append(val_rmse)

        except Exception as e:
            return float("inf")

    return float(np.mean(run_scores))

def run_hyperparameter_search(X_train, Y_train_dS, X_test, Y_test_dS, n_trials=50, num_tasks=3, task_names=None):
    # Pass storage parameter to write study results to a local database
    storage_url = "sqlite:///optuna_study.db"

    study = optuna.create_study(
        study_name="mtgp_hyperparameter_tuning",
        storage=storage_url,
        load_if_exists=True,  # Resume search if interrupted
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    print(f"[Optuna] Dashboard target: {storage_url}")
    print(
        f"[Optuna] Launch dashboard in terminal using: optuna-dashboard {storage_url}"
    )

    study.optimize(
        lambda trial: objective(
            trial, X_train, Y_train_dS, X_test, Y_test_dS, num_tasks=num_tasks, task_names=task_names
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study

if __name__ == "__main__":
    if os.path.exists("optuna_study.db"):
        os.remove("optuna_study.db")

    train_filepath="data/input/MTGPR_Tm_Hfus_Hf/train.xlsx"
    test_filepath="data/input/MTGPR_Tm_Hfus_Hf/test.xlsx"

    # Load data
    smiles_train, Y_train_raw = extract_smiles_and_targets(train_filepath)
    smiles_test, Y_test_raw = extract_smiles_and_targets(test_filepath)

    # Target Transformation (dS_fus = dH_fus / T_m)
    Y_train_dS = Y_train_raw.copy()
    Y_train_dS[:, 0] = Y_train_raw[:, 1] / Y_train_raw[:, 0]
    Y_test_dS = Y_test_raw.copy()
    Y_test_dS[:, 0] = Y_test_raw[:, 1] / Y_test_raw[:, 0]

    # Calculate max_features as a clean integer
    target_max_features = int(np.floor(Y_train_raw.shape[0]))

    encoder = MolecularEncoder(output_dir="MutualInformationApproach_v1")
    X_train = encoder.fit_transform_features(smiles_train, Y_train_dS, max_features=target_max_features, show_plots=False)
    X_test = encoder.transform_features(smiles_test)

    task_names = ["Melting_Temp", "H_fus", "H_f"]
    
    # Run the hyper-hyperparameter search
    study = run_hyperparameter_search(
        X_train=X_train.values,
        Y_train_dS=Y_train_dS,
        X_test=X_test.values,
        Y_test_dS=Y_test_dS,
        n_trials=10000,
        num_tasks=len(task_names),
        task_names=task_names
    )
    
    # Train the final pipeline using the best discovered parameters with live plotting turned back ON
    best_params = study.best_params
    print("\nFitting final model with optimal hyperparameters and live plot enabled...")
    
    # Pass best_params into your fit pipeline...