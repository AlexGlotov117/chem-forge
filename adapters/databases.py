import sys
import os

sys.path.insert(0, os.path.abspath("/home/aglotov/chemicals"))
sys.path.insert(0, os.path.abspath("/home/aglotov/thermo"))
import chemicals
from thermo import Chemical

def caleb_bell_db_adapter(target_smiles, requested_properties):
    """
    This is the actual function that executes when 
    self.db_query_fn(self.target_entity) is called inside _check_target().
    """
    PROPERTY_MAP = {
        "T_melt": {"func": "Tm"},
        "H_fus": {"func": "Hfusm"},
    }
    # 1. Instantiate Chemical object once per unique molecule
    chem_obj = None
    try:
        cas_rn = chemicals.CAS_from_any(f"SMILES={target_smiles}")
        chem_obj = Chemical(cas_rn)
    except Exception:
        # Chemical lookup failed entirely
        return None
    
    # Process requested properties
    extracted_properties = {}
    for prop in requested_properties:
        # Pull directly from Chemical object
        if chem_obj is not None and prop in PROPERTY_MAP:
            thermo_attr = PROPERTY_MAP[prop]["func"]
            db_val = getattr(chem_obj, thermo_attr, None)

            if db_val is not None:
                extracted_properties[prop] = db_val
            else:
                # Property is missing -> strictly incomplete
                return None
        else:
            return None

    return extracted_properties