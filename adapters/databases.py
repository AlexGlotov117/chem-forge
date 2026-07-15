import os
import sys

class BaseDatabaseAdapter:
    """
    Abstract Base Class defining the unified database interface.
    Implementations handle data retrieval, full dataset loading, and mutation.
    """

    def query_target(self, smiles: str, properties: list) -> dict:
        """
        Point Query: Retrieves property values for a single specific compound.
        
        Parameters:
        -----------
        smiles : str
            The target molecular SMILES string.
        properties : list
            List of requested keys, e.g. ['Melting Point', 'Enthalpy of Fusion'].

        Returns:
        --------
        dict or None:
            Returns dictionary {prop_name: value} if ALL properties exist.
            Otherwise, returns None.
        """
        raise NotImplementedError

    def get_full_database(self) -> dict or None:
        """
        Set Query: Dumps the entire database into memory for global scans.

        Returns:
        --------
        dict or None:
            A map of {smiles: {property_name: value}}.
            Returns None if the database does not support bulk dumps (e.g., massive APIs).
        """
        raise NotImplementedError

    def generate_mutated_candidates(self, smiles: str):
        """
        Stream Query: Creates a generator yielding structurally modified candidates.

        Parameters:
        -----------
        smiles : str
            The seed molecule from which mutated neighbors are derived.

        Yields:
        -------
        str : A virtual candidate SMILES string.
        """
        raise NotImplementedError

class BradleyDatabaseAdapter(BaseDatabaseAdapter):
    """
    Adapter interfacing with the Jean-Claude Bradley Open Melting Point Dataset.
    Supports point queries and fast in-memory global scans.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Raw Bradley keys are parsed, cleaned, and cached on initialization
        self._db = self._load_and_clean()

    def _load_and_clean(self) -> dict:
        import pandas as pd

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Bradley dataset file not found at: {self.file_path}")
        
        # We only need the smiles, temperature (Celsius), and the bad row flag
        df = pd.read_csv(self.file_path, usecols=['smiles', 'mpC', 'donotuse'], encoding='cp1252')
        
        # Clean bad or excluded rows
        df = df[df['donotuse'].isna() | (df['donotuse'].astype(str).str.strip() == '')]
        df = df.dropna(subset=['smiles', 'mpC'])
        
        # Convert Celsius to Kelvin
        df['Tm_K'] = df['mpC'] + 273.15
        
        # Build our standard dictionary map: {smiles: {property: value}}
        return {
            row.smiles: {"Melting Point": row.Tm_K}
            for row in df.itertuples(index=False)
        }

    def query_target(self, smiles: str, properties: list) -> dict:
        """Fast point-lookup in our cleaned local cache."""
        import numpy as np

        # Initialize all requested properties as np.nan by default
        extracted_properties = {prop: np.nan for prop in properties}
        
        # If the SMILES exists in our local dictionary, check for values
        if smiles in self._db:
            record = self._db[smiles]
            for prop in properties:
                db_val = record.get(prop, np.nan)
                
                # Overwrite only if a valid, non-null value is found
                if db_val is not None and not (isinstance(db_val, float) and np.isnan(db_val)):
                    extracted_properties[prop] = db_val
                    
        return extracted_properties

    def get_full_database(self) -> dict or None:
        """Exposes the internal cache for global neighbor harvesting."""
        return self._db

    def generate_mutated_candidates(self, smiles: str):
        """This is a static dataset; dynamic graph mutation is not supported."""
        return None

sys.path.insert(0, os.path.abspath("/home/aglotov/chemicals"))
sys.path.insert(0, os.path.abspath("/home/aglotov/thermo"))
import chemicals
from thermo import Chemical

class CalebBellDatabaseAdapter(BaseDatabaseAdapter):
    """
    Adapter interfacing with Caleb Bell's chemicals & thermo packages.
    Highly optimized for identifier searches, calculations, and dynamic mutations.
    """
    def __init__(self, smart_mutator):
        """
        Parameters:
        -----------
        smart_mutator : object
            Your initialized SmartActionMutator containing registered action rules.
        """
        self.smart_mutator = smart_mutator

    def query_target(self, smiles: str, properties: list) -> dict:
        """Resolves physical properties using Caleb Bell's on-demand solvers."""
        import numpy as np
        
        # Convert properties request to thermo API attributes
        prop_mapping = {
            "Melting Point": {"func": "Tm"},
            "Enthalpy of Fusion": {"func": "Hfusm"}
        }
    
        # Initialize all requested properties to np.nan by default
        extracted_properties = {prop: np.nan for prop in properties}
        
        # Attempt to instantiate the Chemical object
        chem_obj = None
        try:
            cas_rn = chemicals.CAS_from_any(f"SMILES={smiles}")
            chem_obj = Chemical(cas_rn)
        except Exception:
            # Chemical lookup failed entirely -> Return all properties as np.nan
            return extracted_properties
        
        # Process requested properties and overwrite np.nan where data exists
        for prop in properties:
            if prop in prop_mapping:
                thermo_attr = prop_mapping[prop]["func"]
                db_val = getattr(chem_obj, thermo_attr, None)

                # Overwrite only if a valid, non-null value is found
                if db_val is not None and not (isinstance(db_val, float) and np.isnan(db_val)):
                    extracted_properties[prop] = db_val
            
        return extracted_properties

    def get_full_database(self) -> dict or None:
        """Caleb Bell's underlying database is too heavy/lazy-loaded to dump globally."""
        return None

    def generate_mutated_candidates(self, smiles: str):
        """Yields virtual mutated compounds generated on-the-fly around the target."""
        # Request candidates dynamically from your smart mutator
        candidates = self.smart_mutator.mutate_until_k(smiles, k_neighbors=1000)
        for cand in candidates:
            yield cand