import numpy as np

class HarvestedContextEvaluator:
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
        print(f"\nFitting Surrogate Model on {len(X_train)} harvested neighbors...")
        print(X_train)
        self.model.fit(X_train, Y_train)

        # 4. Predict at Target
        pred_mean, pred_var = self.model.predict_with_uncertainty(X_query)

        # Map back to property names
        results_mean = dict(zip(property_keys, pred_mean.flatten()))
        results_var = dict(zip(property_keys, pred_var.flatten()))

        print("Prediction Complete at Target:")
        for key in property_keys:
            print(f"  • {key}: {results_mean[key]:.2f} ± {np.sqrt(results_var[key]):.2f} (Variance: {results_var[key]:.4f})")

        return results_mean, results_var
