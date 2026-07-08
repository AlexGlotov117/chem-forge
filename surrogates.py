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

    def _compute_distance_matrix(self, smiles_pool):
        """Computes a structural distance matrix across the candidate pool."""
        features = np.array([self.feature_extractor(s) for s in smiles_pool])
        # Compute pairwise Euclidean distance between molecular feature vectors
        dot_product = np.dot(features, features.T)
        square_norms = np.diag(dot_product)
        distances = np.sqrt(np.maximum(square_norms[:, None] + square_norms[None, :] - 2 * dot_product, 0.0))
        return distances, features

    def compileInitialSeed(self, smiles_pool, target_seed_size=10, user_target_smiles=None):
        """
        Builds a diverse initial seed dataset using ONLY molecules with 
        experimental/database data. Completely avoids simulation/DFT.
        """
        if target_seed_size > len(smiles_pool):
            raise ValueError(f"Target seed size ({target_seed_size}) cannot exceed your pool size ({len(smiles_pool)}).")
        
        distances, features = self._compute_distance_matrix(smiles_pool)
        
        selected_indices = []
        seed_X = []
        seed_y = []
        successful_smiles = []

        # Track our pool of candidates we haven't evaluated or rejected yet
        available_indices = list(range(len(smiles_pool)))

        # Determine initial starting point absolute index
        if user_target_smiles and user_target_smiles in smiles_pool:
            current_idx = smiles_pool.index(user_target_smiles)
        else:
            global_sum_dist = np.sum(distances, axis=1)
            current_idx = int(np.argmax(global_sum_dist))
            
        print(f"Initializing experimental seed collection (Target Size: {target_seed_size})...")

        while len(successful_smiles) < target_seed_size and len(available_indices) > 0:
            smiles = smiles_pool[current_idx]
            available_indices.remove(current_idx)  # Safely removes the absolute index
            
            # Query database without GP fallback
            means, _ = self.queryProperties(smiles, self.properties)
            
            row_values = []
            is_valid = True
            for prop in self.properties:
                val = means.get(prop, None)
                if val is None:
                    is_valid = False
                    break
                row_values.append(val)

            if is_valid:
                selected_indices.append(current_idx)
                seed_X.append(features[current_idx])
                seed_y.append(row_values)
                successful_smiles.append(smiles)
                print(f"  ✅ [Anchor #{len(successful_smiles)}] Secured: {smiles}")
            else:
                print(f"  ❌ Skipped (Incomplete database data): {smiles}")

            # Pick the next absolute index if we need more anchors
            if len(successful_smiles) < target_seed_size and len(available_indices) > 0:
                if not selected_indices:
                    # If everything checked so far failed, pick the next highest global variance
                    global_sum_dist = np.sum(distances[available_indices, :], axis=1)
                    relative_next_idx = np.argmax(global_sum_dist)
                    current_idx = available_indices[relative_next_idx]  # Map back to absolute
                else:
                    # Standard Max-Min selection against currently approved absolute anchors
                    sub_dist = distances[available_indices, :][:, selected_indices]
                    min_distances = np.min(sub_dist, axis=1)
                    relative_next_idx = np.argmax(min_distances)
                    current_idx = available_indices[relative_next_idx]

        return successful_smiles, np.array(seed_X), np.array(seed_y)

    def queryProperties(self, smiles_str, requested_properties, experimental_data_only=True):
        """
        [THE GENERIC INTERFACE]
        Pass a molecule and ANY list of arbitrary registered properties. Will first check if data is cached, then checks databases, then predicts using MT-GP
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

class TargetHarvestingEngine:
    """
    Modular Engine for:
    Step 1: Checking if a target entity has direct, complete ground-truth data in any of our databases.
    Step 2: Harvesting & ranking the top-k nearest complete neighbors from a pool.
    """

    def __init__(self, target_smiles, requested_properties, featurizer_fn, db_query_fn):
        """
        Parameters:
        -----------
        target_entity : object
            Arbitrary representation of your target (e.g., SMILES string, 
            graph, dict of formulation fractions, vector).
        requested_properties: 

        featurizer_fn : callable
            A function: entity -> 1D numpy array of features (X).
        db_query_fn : callable
            A function: entity -> dict of target values (Y) if complete, or None if missing.
        """
        self.target_smiles = target_smiles
        self.requested_properties = requested_properties
        self.featurizer_fn = featurizer_fn
        self.db_query_fn = db_query_fn

        # Extract features for target once
        self.x_target = np.array(self.featurizer_fn(self.target_smiles)).flatten()

    def _check_target(self):
        """
        Query the database interface for direct ground-truth data.
        Returns: (has_complete_data: bool, data: dict or None)
        """
        print("Checking database interface for target entity...")
        data = self.db_query_fn(self.target_smiles, self.requested_properties)

        if data is not None:
            print("Target entity has complete empirical data in the database!")
            return True, data
        else:
            print("Target entity lacks complete data in the database.")
            return False, None

    def _harvest_neighbors(self, mutator_fn, distance_metric_fn, k_neighbors=10, max_attempts=500):
        """
        Dynamically generates virtual molecules around the target, checks if they
        exist in the database, and harvests the top k complete neighbors on-the-fly.

        Parameters:
        -----------
        mutator_fn : callable
            A generator or function yielding virtual mutated candidates: mutator_fn(target_smiles) -> candidate.
        distance_metric_fn : callable
            A function: (1D array x_target, 1D array x_candidate) -> scalar distance.
        k_neighbors : int
            Target number of valid database neighbors to harvest.
        max_attempts : int
            Safety cutoff for the generation loop.
        """
        print(f"\nDynamic Generation Triggered: Harvesting {k_neighbors} database neighbors...")
        harvested_neighbors = []
        visited = {self.target_smiles}

        attempts = 0
        for candidate in mutator_fn(self.target_smiles):
            print(candidate)
            attempts += 1
            if attempts > max_attempts:
                print(f"Reached max generation attempts ({max_attempts}).")
                break

            # Avoid re-evaluating duplicate mutations or the target itself
            if candidate in visited:
                continue
            visited.add(candidate)

            # Query database for empirical hit on the virtual candidate
            cand_data = self.db_query_fn(candidate, self.requested_properties)
            if cand_data is None:
                continue  # Virtual molecule does not exist in the database or lacks data

            # Compute features & distance relative to target
            cand_features = np.array(self.featurizer_fn(candidate)).flatten()
            dist = distance_metric_fn(self.x_target, cand_features)

            harvested_neighbors.append({
                "entity": candidate,
                "features": cand_features,
                "properties": cand_data,
                "distance": dist
            })

            print(f"Found Valid Neighbor #{len(harvested_neighbors)}: {candidate} | Distance: {dist:.4f}")

            if len(harvested_neighbors) >= k_neighbors:
                break

        if not harvested_neighbors:
            print("Generator could not find any database-verified neighbors.")
            return []

        # Sort closest first
        harvested_neighbors.sort(key=lambda item: item["distance"])
        return harvested_neighbors
    
class SurrogateEvaluator:
    """
    Step 3: Converts harvested neighbor data into feature/target matrices,
    fits a surrogate model, and queries predictions + uncertainty at target_x.
    """
    def __init__(self, surrogate_model):
        """
        surrogate_model must expose:
            - .fit(X, Y)
            - .predict_with_uncertainty(X_query) -> (mean, variance)
        """
        self.model = surrogate_model

    def fit_and_evaluate(self, target_x, harvested_neighbors, property_keys):
        """
        Parameters:
        -----------
        target_x : 1D or 2D numpy array
            The feature vector of the uncharacterized target molecule.
        harvested_neighbors : list of dicts
            The output list from Step 2 containing 'features' and 'properties'.
        property_keys : list of str
            The ordered list of target property names (e.g., ['T_melt']).

        Returns:
        --------
        dict: Predicted property means
        dict: Predictive variance (uncertainty) per property
        """
        if not harvested_neighbors:
            raise ValueError("Harvested neighbor list is empty! Cannot fit surrogate.")

        # 1. Format X_train (N_samples x N_features)
        X_train = np.array([item["features"] for item in harvested_neighbors])

        # 2. Format Y_train (N_samples x N_properties)
        Y_train = np.array([
            [item["properties"][key] for key in property_keys]
            for item in harvested_neighbors
        ])

        # Ensure target_x is 2D matrix shape (1 x N_features)
        X_query = np.atleast_2d(target_x)

        # 3. Fit Model
        print(f"\n🧠 [Step 3] Fitting Surrogate Model on {len(X_train)} harvested neighbors...")
        self.model.fit(X_train, Y_train)

        # 4. Predict at Target
        pred_mean, pred_var = self.model.predict_with_uncertainty(X_query)

        # Map back to property names
        results_mean = dict(zip(property_keys, pred_mean.flatten()))
        results_var = dict(zip(property_keys, pred_var.flatten()))

        print("🎯 Prediction Complete at Target:")
        for key in property_keys:
            print(f"  • {key}: {results_mean[key]:.2f} ± {np.sqrt(results_var[key]):.2f} (Variance: {results_var[key]:.4f})")

        return results_mean, results_var
    
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class StandardGPWrapper:
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