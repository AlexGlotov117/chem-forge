import numpy as np
import pandas as pd


# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern
import sys
import os

# Force Python to look inside locally cloned folders first
sys.path.insert(0, os.path.abspath("/home/aglotov/chemicals"))
sys.path.insert(0, os.path.abspath("/home/aglotov/thermo"))

import chemicals
from thermo import Chemical

# Map generic strings to property functions
PROPERTY_MAP = {
    "T_fus": {
        "func": "Tm",
        "methods_func": ""
    },
    "H_fus": {
        "func": "Hfusm",
        "methods_func": "Hfus_method"
    },
    "H_formation": {
        "func": "Hf",
        "methods_func": ""
    }
}

class MultiTaskGaussianProcess:
    def __init__(self, property_list, feature_extractor):
        self.properties = property_list  # e.g., ["T_fus", "H_fus", "H_formation"]
        self.feature_extractor = feature_extractor
        
        # In-memory clean cache structure: self.cache[SMILES] = {prop: val}
        self.cache = {}
        
        # Underlying training matrices for the MT-GP
        self.X_train = pd.DataFrame()
        self.y_train = np.empty((0, len(self.properties)))
        self.trained_smiles = []
        
        self.model = None

    # def _query_databases(self, smiles_str, prop_name):
    #     """
    #     Dynamically queries all available thermodynamic data and calculation methods 
    #     from chemical librarys for ANY registered property.
    #     """
    #     if prop_name not in PROPERTY_MAP:
    #         print(f"Warning: Property '{prop_name}' is not configured in PROPERTY_MAP.")
    #         return None
            
    #     prop_config = PROPERTY_MAP[prop_name]
        
    #     try:
    #         # Resolve chemical ID and CAS Number from SMILES
    #         chemical_id = search_chemical(f"SMILES={smiles_str}")
    #         cas_rn = chemical_id.CASs
            
    #         # Dynamically import the target submodule on the fly
    #         submodule = importlib.import_module(prop_config["module"])
            
    #         # Grab the main function and the methods array function using string lookups
    #         calc_function = getattr(submodule, prop_config["func"])
    #         methods_function = getattr(submodule, prop_config["methods_func"])
            
    #         # Sweep through all valid methods for this molecule
    #         discovered_values = {}
            
    #         # Execution of methods_function(cas_rn) loops through available estimation approaches
    #         for method in methods_function(cas_rn):
    #             try:
    #                 val = calc_function(cas_rn, method=method)
    #                 if val is not None:
    #                     discovered_values[method] = val
    #             except Exception:
    #                 # Skip individual estimation failures if a specific equation fails to converge
    #                 continue
                    
    #         return discovered_values if discovered_values else None

    #     except Exception as e:
    #         print(f"Database query failed for {smiles_str} on property {prop_name}: {e}")
    #         return None

    # def seed_initial_dataset(self, smiles_list, true_target_matrix=None):
    #     """Seeds the orchestrator with your baseline laboratory Excel entries."""
    #     features = [self.feature_extractor(s) for s in smiles_list]
    #     self.X_train_df = pd.DataFrame(features)
    #     self.trained_smiles = list(smiles_list)
        
    #     if true_target_matrix is not None:
    #         self.y_train_matrix = np.array(true_target_matrix)
    #     else:
    #         # Fallback to auto-populating from Caleb Bell if no spreadsheet values provided
    #         self.y_train_matrix = np.zeros((len(smiles_list), len(self.properties)))
    #         for row_idx, s in enumerate(smiles_list):
    #             for col_idx, prop in enumerate(self.properties):
    #                 val = self._query_caleb_bell(s, prop)
    #                 self.y_train_matrix[row_idx, col_idx] = val if val is not None else np.nan
        
    #     # Clean up missing data limits for standard GP wrappers via mean imputation or native proxy
    #     nas = np.isnan(self.y_train_matrix)
    #     if np.any(nas):
    #         col_means = np.nanmean(self.y_train_matrix, axis=0)
    #         self.y_train_matrix[nas] = np.take(col_means, np.where(nas)[1])
            
    #     self._retrain_gp()

    # def _retrain_gp(self):
    #     """Re-fits the structural-task kernel matrix."""
    #     kernel = Matern(length_scale=[1.0] * self.X_train_df.shape[1], bounds_error=False)
    #     self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    #     self.model.fit(self.X_train_df, self.y_train_matrix)

    def queryProperties(self, smiles_str, requested_properties):
        """
        [THE GENERIC INTERFACE]
        Pass a molecule and ANY list of arbitrary registered properties.
        Returns: {prop: predicted_value}, {prop: uncertainty}
        """
        results_mean = {}
        results_unc = {}
        missing_props = []

        if smiles_str not in self.cache:
            self.cache[smiles_str] = {}
            
        # Instantiate the Chemical object *once* per molecule to save overhead
        chem_obj = None
        try:
            cas_rn = chemicals.CAS_from_any(f"SMILES={smiles_str}")
            chem_obj = Chemical(cas_rn)
        except Exception as e:
            # If the molecule completely fails database lookup, all requested properties are marked missing
            print(f"Database lookup failed for SMILES {smiles_str}: {e}")

        # Process each property cleanly
        for prop in requested_properties:
            # Check local active learning cache first
            if prop in self.cache[smiles_str]:
                results_mean[prop] = self.cache[smiles_str][prop]
                results_unc[prop] = 0.0
                continue
                
            # If cached value isn't found, pull directly from the initialized Chemical object
            if chem_obj is not None and prop in PROPERTY_MAP:
                thermo_attr = PROPERTY_MAP[prop]["func"]
                db_val = getattr(chem_obj, thermo_attr, None)
                
                if db_val is not None:
                    self.cache[smiles_str][prop] = db_val
                    results_mean[prop] = db_val
                    results_unc[prop] = 0.0
                    continue

            # If neither cache nor Chemical database has it, flag it for the MT-GP engine
            missing_props.append(prop)

        # # Fallback to Multi-Task GP for anything left over
        # if missing_props and hasattr(self, 'model') and self.model is not None:
        #     features = self.feature_extractor(smiles_str)
        #     pred_means, pred_stds = self.model.predict([features], return_std=True)
            
        #     for prop in missing_props:
        #         # Match the requested property to its index in the global MT-GP output vector
        #         if prop in self.properties:
        #             idx = self.properties.index(prop)
        #             results_mean[prop] = pred_means[0][idx]
        #             results_unc[prop] = pred_stds[0] if len(pred_stds.shape) == 1 else pred_stds[0][idx]
        #         else:
        #             results_mean[prop] = None
        #             results_unc[prop] = None
        #             print(f"Error: Property '{prop}' is unconfigured in the active MT-GP architecture.")

        return results_mean, results_unc

    # def active_learning_update(self, smiles_str, physical_solver_results):
    #     """
    #     Call this when your code finishes running a slow simulation (like CEA or SLE)
    #     or an experiment confirms a true property. It locks the point down.
    #     """
    #     features = self.feature_extractor(smiles_str)
        
    #     # Update cache
    #     if smiles_str not in self.cache:
    #         self.cache[smiles_str] = {}
    #     for prop, val in physical_solver_results.items():
    #         self.cache[smiles_str][prop] = val

    #     # Update training data array matrix
    #     new_row = np.array([physical_solver_results.get(p, self.cache[smiles_str].get(p, np.nan)) for p in self.properties])
        
    #     self.X_train_df = pd.concat([self.X_train_df, pd.DataFrame([features])], ignore_index=True)
    #     self.y_train_matrix = np.vstack([self.y_train_matrix, new_row])
    #     self.trained_smiles.append(smiles_str)
        
    #     # Clean missing values if any remain
    #     nas = np.isnan(self.y_train_matrix)
    #     if np.any(nas):
    #         col_means = np.nanmean(self.y_train_matrix, axis=0)
    #         self.y_train_matrix[nas] = np.take(col_means, np.where(nas)[1])

    #     self._retrain_gp()
    #     print(f"-> Active Learning: MT-GP completely retrained with new point for {smiles_str}")