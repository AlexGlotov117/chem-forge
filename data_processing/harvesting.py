import numpy as np

class TargetHarvestingEngine:
    """
    Modular Engine for:
    Checking if a target entity has direct, complete ground-truth data in any of our databases.
    Harvesting & ranking the top-k nearest complete neighbors from a pool of virtual entities.
    """

    def __init__(self, target_smiles, requested_properties, featurizer_fn, adapter):
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
        self.adapter = adapter

        # Extract features for target once
        self.x_target = np.array(self.featurizer_fn(self.target_smiles)).flatten()

    # COME BACK TO THIS AND FIX
    def check_target(self):
        """
        Query the database interface for direct ground-truth data.
        Returns: (has_complete_data: bool, data: dict or None)
        """
        # print("Checking database interface for target entity...")
        data = self.adapter.query_target(self.target_smiles, self.requested_properties)
        
        return True, data

    def harvest_neighbors(self, distance_metric_fn, strategy='full', k_neighbors=1, max_attempts=1000):
        """
        Generically harvests neighbors and includes the target molecule itself 
        if it exists in the selected search space.
        
        Strategies:
        -----------
        'mutation'     : Uses adapter's mutation generator to pull targeted values on-the-fly.
        'full'         : Pulls the entire database from the adapter and evaluates all entries.
        'full_cropped' : Pulls the entire database but pre-filters/crops to candidate SMILES first.
        """
        import pandas as pd
        
        harvested_neighbors = []

        # ==========================================
        # STRATEGY 1: Dynamic Mutation Search
        # ==========================================
        if strategy == 'mutation':
            print(f"Harvesting via Dynamic Mutation (k={k_neighbors})...")
            
            visited = set()
            attempts = 0
            
            # Request mutation stream from the adapter
            mutation_generator = self.adapter.generate_mutated_candidates(self.target_smiles)
            if not mutation_generator:
                print("Adapter does not support mutations.")
                return []

            for candidate in mutation_generator:
                attempts += 1
                if attempts > max_attempts or len(harvested_neighbors) >= k_neighbors:
                    break

                if candidate in visited:
                    continue
                visited.add(candidate)

                # Direct point-query on the candidate
                cand_data = self.adapter.query_target(candidate, self.requested_properties)
                if cand_data is None:
                    continue

                cand_features = np.array(self.featurizer_fn(candidate)).flatten()
                dist = distance_metric_fn(self.x_target, cand_features)

                harvested_neighbors.append({
                    "entity": candidate,
                    "features": cand_features,
                    "properties": cand_data,
                    "distance": dist
                })

        # ==========================================
        # STRATEGY 2 & 3: Global Database / Cropped
        # ==========================================
        elif strategy in ('full', 'full_cropped'):
            print(f"Harvesting via Global Database Scan (Strategy: '{strategy}', k={k_neighbors})...")
            
            # Request the full database dump from the adapter
            all_records = self.adapter.get_full_database()
            
            # -----------------------------------------------------------------
            # FALLBACK: Adapter does not support full dumps (e.g. Caleb Bell)
            # -----------------------------------------------------------------
            if all_records is None:
                print("Adapter does not support global database dumps. Checking for target direct hit...")
                
                # Check if we can at least get the target itself
                target_has_data, target_data = self.check_target()
                if target_has_data:
                    print("Found direct target data! Returning single target record.")
                    return [{
                        "entity": self.target_smiles,
                        "features": self.x_target,
                        "properties": target_data,
                        "distance": 0.0
                    }]
                else:
                    # Return empty list cleanly instead of raising an error!
                    print("Target not found and global scan unsupported. Returning empty neighbor set.")
                    return []
            # -----------------------------------------------------------------

            candidates_pool = all_records.items()

            pool_with_distances = []
            for candidate, cand_data in candidates_pool:
                try:
                    cand_features = np.array(self.featurizer_fn(candidate)).flatten()
                    dist = distance_metric_fn(self.x_target, cand_features)
                    pool_with_distances.append((candidate, cand_data, dist, cand_features))
                except Exception:
                    continue

            pool_with_distances.sort(key=lambda x: x[2])

            harvested_neighbors = []
            for candidate, cand_data, dist, cand_features in pool_with_distances:
                # ONLY stop early at k if we are using the cropped strategy
                if strategy == 'full_cropped' and len(harvested_neighbors) >= k_neighbors:
                    break

                # Extract and pad properties
                filtered_properties = {
                    prop: cand_data.get(prop, np.nan) 
                    for prop in self.requested_properties
                }

                # Skip only if the candidate has absolutely none of the properties we want
                if all(pd.isna(val) for val in filtered_properties.values()):
                    continue

                harvested_neighbors.append({
                    "entity": candidate,
                    "features": cand_features,
                    "properties": filtered_properties,
                    "distance": dist
                })

            return harvested_neighbors

        else:
            raise ValueError(f"Unknown harvesting strategy: {strategy}")

        # Sort all discovered entries globally. 
        # If the target molecule was evaluated, its distance will be 0.0 and it will sort to index 0!
        harvested_neighbors.sort(key=lambda item: item["distance"])
        return harvested_neighbors[:k_neighbors]