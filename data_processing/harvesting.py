import numpy as np

class TargetHarvestingEngine:
    """
    Modular Engine for:
    Checking if a target entity has direct, complete ground-truth data in any of our databases.
    Harvesting & ranking the top-k nearest complete neighbors from a pool of virtual entities.
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
        # print("Checking database interface for target entity...")
        data = self.db_query_fn(self.target_smiles, self.requested_properties)

        if data is not None:
            # print("Target entity has complete empirical data in the database!")
            return True, data
        else:
            # print("Target entity lacks complete data in the database.")
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