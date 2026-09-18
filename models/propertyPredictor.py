import os
import numpy as np
import torch
import gpytorch

from datetime import datetime
from data_processing.dataProcessing import extract_smiles_and_targets
from encoders.chemicals import MolecularEncoder
from models.gp import MTGPPipeline, _GPyTorchMTGPModel


class MTGPR_Tm_Hfus_Hf:
    def __init__(self, model_name=None, model_dir="models/model_weights", output_dir="data/output"):
        """
        Parameters:
        -----------
        model_name : str
            Identifier for the model (e.g., 'v1.0_CEA', 'run_2026_09_18').
        model_dir : str
            Shared directory storing all flat .pth weight files.
        output_dir : str
            Base directory containing model-specific subfolders for plots and artifacts.
        """
        # Generate timestamp if model_name is not explicitly passed
        if model_name is None:
            model_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.model_name = model_name
        self.model_dir = model_dir
        self.output_dir = os.path.join(output_dir, self.model_name)
        
        self.model_filename = f"MTGPR_Tm_Hfus_Hf_{self.model_name}.pth"
        self.model_path = os.path.join(self.model_dir, self.model_filename)
        
        # Ensure model-specific output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Pass dedicated output directory to encoder
        self.encoder = MolecularEncoder(output_dir=self.output_dir)
        self.pipeline = MTGPPipeline(num_tasks=3, num_epochs=1000)
        self.is_trained = False

    def train_and_save(self, train_filepath):
        print(f"--- Training new GPR Model: [{self.model_name}] ---")
        smiles_train, Y_train_raw = extract_smiles_and_targets(train_filepath)

        # Target Transformation (dS_fus = dH_fus / T_m)
        Y_train_dS = Y_train_raw.copy()
        Y_train_dS[:, 0] = Y_train_raw[:, 1] / Y_train_raw[:, 0]

        # Calculate max_features as a clean integer
        target_max_features = int(np.floor(Y_train_raw.shape[0] / 3.0))

        X_train = self.encoder.fit_transform_features(smiles_train, Y_train_dS, max_features=target_max_features)

        self.pipeline.fit(X_train.values, Y_train_dS)

        # Ensure model-specific folder exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save PyTorch state dict and encoder state
        torch.save({
            'model_state': self.pipeline.model.state_dict(),
            'likelihood_state': self.pipeline.likelihood.state_dict(),
            'encoder': self.encoder,
            'train_x': self.pipeline.train_x,
            'train_i': self.pipeline.train_i,
            'train_y': self.pipeline.train_y,
            'x_scaler': self.pipeline.x_scaler,
            'y_mean': self.pipeline.y_mean_,
            'y_std': self.pipeline.y_std_
        }, self.model_path)
        
        self.is_trained = True
        print(f"GPR Model successfully trained and saved to '{self.model_path}'.")

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No saved model found at '{self.model_path}'.")
        
        checkpoint = torch.load(self.model_path, weights_only=False)

        # 1. Restore Encoder
        self.encoder = checkpoint['encoder']

        # 2. Restore Scaler & Target Transformation State
        self.pipeline.x_scaler = checkpoint['x_scaler']
        self.pipeline.y_mean_ = checkpoint['y_mean']
        self.pipeline.y_std_ = checkpoint['y_std']

        # 3. Instantiate GPyTorch Model Shell with Stored Training Tensors
        self.pipeline.train_x = checkpoint['train_x']
        self.pipeline.train_i = checkpoint['train_i']
        self.pipeline.train_y = checkpoint['train_y']

        task_noise_map = {0: 1e-3, 1: 1e-4, 2: 5e-2}
        train_noise = torch.tensor([task_noise_map[i.item()] for i in self.pipeline.train_i], dtype=torch.float32)

        self.pipeline.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=train_noise,
            learn_additional_noise=False
        )

        self.pipeline.model = _GPyTorchMTGPModel(
            self.pipeline.train_x,
            self.pipeline.train_i,
            self.pipeline.train_y,
            self.pipeline.likelihood,
            self.pipeline.mean_module,
            num_tasks=self.pipeline.num_tasks
        )

        # 4. Load Saved Weights
        self.pipeline.model.load_state_dict(checkpoint['model_state'])
        self.pipeline.likelihood.load_state_dict(checkpoint['likelihood_state'])
        
        self.pipeline.model.eval()
        self.pipeline.likelihood.eval()

        self.is_trained = True
        print(f"Successfully loaded existing GPR model: [{self.model_name}] from '{self.model_path}'.")

    def predict_smiles(self, smiles_list):
        if not self.is_trained:
            self.load_model()

        X = self.encoder.transform_features(smiles_list)

        # Set evaluation mode and disable gradient calculation for accurate GP predictions
        self.pipeline.model.eval()
        self.pipeline.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            means, stds = self.pipeline.predict(X.values)

        dS_mean, dS_std = means[:, 0], stds[:, 0]
        dH_fus_mean, dH_fus_std = means[:, 1], stds[:, 1]
        dH_f_mean, dH_f_std = means[:, 2], stds[:, 2]

        T_m_mean = dH_fus_mean / dS_mean
        T_m_std = np.sqrt((dH_fus_std / dS_mean)**2 + ((dH_fus_mean * dS_std) / (dS_mean**2))**2)
        
        # Conversions to standard units (kJ/mol and K)
        return {
            "T_m": T_m_mean,
            "T_m_std": T_m_std,
            "dH_fus": dH_fus_mean,
            "dH_fus_std": dH_fus_std,
            "dH_f": dH_f_mean,
            "dH_f_std": dH_f_std
        }