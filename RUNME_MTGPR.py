# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"
# Use CEARun

import numpy as np
from encoders.chemicals import MolecularEncoder
from data_processing.dataProcessing import extract_smiles_and_targets
from models.gp import MTGPPipeline, JobackMean

from thermo.group_contribution.joback import Joback

if __name__ == "__main__":
    key_overrides = [

    ]
    # Load Raw Data
    smiles_train, Y_train_raw = extract_smiles_and_targets("data/input/MTGPR_Tm_Hfus_Hf/train.xlsx")
    smiles_test, Y_test_raw = extract_smiles_and_targets("data/input/MTGPR_Tm_Hfus_Hf/test.xlsx")

    # Target Transformation (dS_fus = dH_fus / T_m)
    Y_train_dS = Y_train_raw.copy()
    Y_train_dS[:, 0] = Y_train_raw[:, 1] / Y_train_raw[:, 0]

    # Encode Features & Compute Joback Priors
    encoder = MolecularEncoder("data/output/screening_results_v1",override_features=key_overrides)
    # raw_joback_train = encoder.compute_joback_priors(smiles_train)
    # raw_joback_test  = encoder.compute_joback_priors(smiles_test)

    X_train = encoder.fit_transform_features(smiles_train, Y_train_dS, max_features=18)
    X_test  = encoder.transform_features(smiles_test)

    # y_mean = np.nanmean(Y_train_dS, axis=0)
    # y_std  = np.nanstd(Y_train_dS, axis=0)

    # # Standardize & unroll Joback priors
    # joback_train_std = np.nan_to_num((raw_joback_train - y_mean) / y_std, nan=0.0).flatten()
    # joback_test_std  = np.nan_to_num((raw_joback_test - y_mean) / y_std, nan=0.0).flatten()

    # # # 4. Instantiate Custom Prior & GP Pipeline
    # joback_mean = JobackMean(joback_train_std, joback_test_std)
    # pipeline = MTGPPipeline(mean_module=joback_mean, num_tasks=3, num_epochs=1000)
    pipeline = MTGPPipeline(num_tasks=3, num_epochs=1000)

    # # 5. Fit and Predict
    pipeline.fit(X_train.values, Y_train_dS)
    means, stds = pipeline.predict(X_test.values)

    # 6. Reconstruct T_m physical prediction
    dS_pred, dS_std = means[:, 0], stds[:, 0]
    dH_fus_pred, dH_fus_std = means[:, 1], stds[:, 1]
    dH_f_pred, dH_f_std = means[:, 2], stds[:, 2]

    T_m_pred = dH_fus_pred / dS_pred
    T_m_std = np.sqrt((dH_fus_std / dS_pred)**2 + ((dH_fus_pred * dS_std) / (dS_pred**2))**2)

    # Display clean results
    for i in range(len(smiles_test)):
        print(f"Candidate {i+1}: {smiles_test[i]}")
        print(f"  T_m    : {T_m_pred[i]:.2f} ± {T_m_std[i]:.2f} K (Actual: {Y_test_raw[i,0]} K)")
        print(f"  dH_fus : {dH_fus_pred[i]:.2f} ± {stds[i,1]:.2f} kJ/mol (Actual: {Y_test_raw[i,1]} kJ/mol)")
        print(f"  dH_f : {dH_f_pred[i]:.2f} ± {stds[i,2]:.2f} kJ/mol (Actual: {Y_test_raw[i,2]} kJ/mol)\n")