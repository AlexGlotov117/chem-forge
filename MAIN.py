# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

import numpy as np
import pandas as pd
import os
# import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import ExactMolWt

from alexChemicals import Compound, Mixture
from solvers import generate_x_grid
from surrogates import MultiTaskGaussianProcess, TargetHarvestingEngine, StandardGPWrapper, SurrogateEvaluator
# output_dir = "screening_results"

# pureComponents = pd.read_excel("pureComponents.xlsx", sheet_name="Input")
# cominations = pd.read_excel("combinations.xlsx", sheet_name=None)

# num_points = 101

# def create_compound_from_smiles(input_smiles: str) -> Compound:
#     """
#     Validates the input SMILES, converts it to a standard canonical format,
#     looks up thermodynamic data from the database, and auto-generates 
#     molecular properties using RDKit.
#     """
#     mol = Chem.MolFromSmiles(str(input_smiles).strip())
#     if mol is None:
#         raise ValueError(f"Invalid SMILES string: {input_smiles}")
#     canonical_key = Chem.MolToSmiles(mol)
    
#     row = pureComponents[pureComponents["SMILES"] == canonical_key].iloc[0]
        
#     # Generate atomic formula dynamically for NASA-CEA
#     mol_with_hs = Chem.AddHs(mol)
#     formula_dict = {}
#     for atom in mol_with_hs.GetAtoms():
#         symbol = atom.GetSymbol()
#         formula_dict[symbol] = formula_dict.get(symbol, 0) + 1
    
#     # Force a fallback to a short name if Symbol is missing, and slice at 15 chars max
#     cea_name = str(row["Full Name"]).strip()[:15] # Hard limit to 15 characters for CEA safety

#     return Compound(
#         name=str(row["Full Name"]),  
#         mw=ExactMolWt(mol),
#         T_fus=float(row["Melting Temp\n[K]"]),
#         h_fus=float(row["Heat of Fusion\n[J/mol]"]),
#         h_f_298=float(row["Standard Heat of Formation\n[J/mol]"]),
#         formula=formula_dict
#     )

# # 3. Loop through each sheet (Binary, Ternary, Quaternary, etc.)
# for sheet_name, df_combos in cominations.items():
#     print(f"\n" + "="*50)
#     print(f"  STARTING SCREENING FOR SHEET: {sheet_name}")
#     print("="*50)
    
#     # Identify component columns dynamically for this specific sheet
#     component_cols = [col for col in df_combos.columns if col.startswith("Component ")]
    
#     # Loop through every row/combination in the current sheet
#     for idx, row in df_combos.iterrows():
#         row_smiles = [str(row[col]).strip() for col in component_cols if pd.notna(row[col])]
        
#         if not row_smiles:
#             continue
            
#         try:
#             # Generate the dynamic compound arrays using the SMILES strings directly
#             compounds_list = [create_compound_from_smiles(s) for s in row_smiles]
            
#             # Use the display symbols to format a clean filename string
#             system_label = "_".join([c.name for c in compounds_list])
#             print(f" -> Sweeping system [{idx+1}]: {system_label}")
            
#             # Build your uniform mixture engine
#             mixture = Mixture(compounds=compounds_list)

#             # Generate the composition matrix
#             x_grid_matrix = generate_x_grid(num_components=mixture.num_components, steps=num_points)

#             records = []

#             # 3. Sweep across the N-dimensional composition matrix
#             for x_vec in x_grid_matrix:
#                 # Pass the current composition row vector (e.g., [0.2, 0.5, 0.3]) to the state machine
#                 mixture.set_composition(x=x_vec)
                
#                 # 4. Safely extract properties on-the-fly with a fallback catch
#                 try:
#                     # If mixture.isp returns a list/array index accordingly, adjust to match your class output
#                     current_isp = mixture.isp[2]
#                     current_t_adi = mixture.T_adi[0]
#                     current_c_star = mixture.c_star[0]
#                 except Exception:
#                     current_isp = np.nan
#                     current_t_adi = np.nan
#                     current_c_star = np.nan

#                 # 5. Build a dynamic row record mapping compositions back to column names cleanly
#                 row_record = {}
#                 for i, name in enumerate(mixture.names):
#                     row_record[f"{name} Molar Composition \n[%]"] = x_vec[i]
#                     # If your Mixture class tracks liquidus temperatures per component:
#                     row_record[f"{name} Liquidus Temperature \n[K]"] = mixture.T_liq[i]
                    
#                 # Append the thermodynamic metrics
#                 row_record["Solid-Liquid Equilibrium Temperature \n[K]"] = mixture.T_fus
#                 row_record["Adiabatic Flame Temperature \n[K]"] = current_t_adi
#                 row_record["Characteristic Velocity \n[m/s]"] = current_c_star
#                 row_record["Specific Impulse \n[s]"] = current_isp
                
#                 records.append(row_record)

#             # 6. Convert to DataFrame and save to CSV
#             df_results = pd.DataFrame(records)
            
#             system_label = "_".join(mixture.names)
#             filename = f"{system_label}.csv"

#             os.makedirs(output_dir, exist_ok=True)
#             df_results.to_csv(os.path.join(output_dir, filename), index=False)
            
#         except Exception as e:
#             print(f"Skipping index row {idx} due to calculation error: {e}")

### CALEB BELL STUFF
def basic_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [Descriptors.ExactMolWt(mol), Descriptors.NumRotatableBonds(mol)]


def standard_molecular_features(target_smiles):
    """
    Converts a SMILES string into a high-dimensional structural fingerprint.
    Replaces simple metrics with a 2048-bit structural topology vector.
    """

    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        # Fallback to an empty vector if SMILES parsing fails
        return np.zeros(2048, dtype=float)
        
    # Generate a radius-2 Morgan Fingerprint (equivalent to ECFP4)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    # Convert the internal RDKit bit vector into a standard NumPy array for our distance matrix
    features = np.zeros((1,), dtype=float)
    Chem.DataStructs.ConvertToNumpyArray(fingerprint, features)
    
    return features

# # Initialize the tracking space with the full master property vector
# properties = ["T_fus", "H_fus"]
# engine = MultiTaskGaussianProcess(properties, standard_molecular_features)

# smile_pool = ["O", "C(C(CO)O)O", "CC(N)=O", "c1c[nH]cn1"]

# seed_smiles, X_seed, y_seed = engine.compileInitialSeed(smile_pool, 3)
# print(seed_smiles)
# print(X_seed)
# print(y_seed)

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

target_smiles = "[Li+].[BH4-]"
requested_properties = ["T_melt"]

engine = TargetHarvestingEngine(
    target_smiles=target_smiles,
    requested_properties=requested_properties,
    featurizer_fn=standard_molecular_features,
    db_query_fn=caleb_bell_db_adapter
)

def tanimoto_distance(x1, x2):
    """
    Computes Tanimoto Distance between two 1D binary feature vectors.
    Distance = 1.0 - (Intersection / Union)
    """
    intersection = np.logical_and(x1, x2).sum()
    union = np.logical_or(x1, x2).sum()
    
    if union == 0:
        return 1.0
        
    similarity = intersection / union
    return 1.0 - similarity

from rdkit import Chem
import random
# import selfies as sf
from rdkit.Chem import rdChemReactions

# Standard database-heavy anions commonly found in thermo/chemicals tables
COMMON_DATABASE_ANIONS = [
    "[BH4-]",          # Borohydride
    "[Cl-]",           # Chloride
    "[Br-]",           # Bromide
    "[I-]",            # Iodide
    "F[B-](F)(F)F",    # Tetrafluoroborate (BF4)
    "F[P-](F)(F)(F)(F)F", # Hexafluorophosphate (PF6)
    "[N-]([S](=O)(=O)C(F)(F)F)[S](=O)(=O)C(F)(F)F", # NTf2 (Bis(trifluoromethanesulfonyl)imide)
    "[O-]S(=O)(=O)C(F)(F)F", # Triflate (OTf)
    "[O-]C(=O)C",      # Acetate
    "[NO3-]",          # Nitrate
]

def mutator(target_smiles, num_mutations_per_target=100):
    # try:
    #     # 1. Convert SMILES to SELFIES
    #     target_selfies = sf.encoder(target_smiles)
    #     tokens = list(sf.split_selfies(target_selfies))
    # except Exception:
    #     return

    # # A pool of common organic & ionic tokens to insert/substitute
    # token_pool = [
    #     "[C]", "[C][C]", "[Branch1]", "[Ring1]", 
    #     "[N+1]", "[O]", "[F]", "[Cl]", "[Br]", "[Expl=B-1]"
    # ]

    # for _ in range(num_mutations_per_target):
    #     mutated_tokens = tokens.copy()
    #     mutation_type = random.choice(["substitute", "insert", "delete"])
    #     idx = random.randint(0, len(mutated_tokens) - 1)

    #     if mutation_type == "substitute" and len(mutated_tokens) > 0:
    #         mutated_tokens[idx] = random.choice(token_pool)
            
    #     elif mutation_type == "insert":
    #         mutated_tokens.insert(idx, random.choice(token_pool))
            
    #     elif mutation_type == "delete" and len(mutated_tokens) > 1:
    #         mutated_tokens.pop(idx)

    #     # Reconstruct SELFIES and decode back to SMILES
    #     try:
    #         mutated_selfies = "".join(mutated_tokens)
    #         candidate_smiles = sf.decoder(mutated_selfies)
            
    #         # Canonicalize and validate via RDKit
    #         mol = Chem.MolFromSmiles(candidate_smiles)
    #         if mol is None:
    #             continue
                
    #         canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
    #         # Skip returning exact target copy
    #         if canonical_smiles != Chem.MolToSmiles(Chem.MolFromSmiles(target_smiles), canonical=True):
    #             yield canonical_smiles
                
    #     except Exception:
    #         continue
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return

    # 1. Deconstruct Target into Cation and Anion
    frags = Chem.GetMolFrags(mol, asMols=True)
    cation_mols = []
    
    if len(frags) >= 2:
        for f in frags:
            f_smiles = Chem.MolToSmiles(f, canonical=True)
            # Separate organic/cation fragment from simple counterions
            if "+" in f_smiles or not ("B" in f_smiles or "-" in f_smiles):
                cation_mols.append(f)
    if not cation_mols:
        cation_mols = [mol]

    # 2. Comprehensive Cation Transformation SMARTS
    cation_reactions = [
        # Alkyl Extensions (C1 -> C2 -> C3 -> C4)
        '[C;H3:1] >> [C:1]C',                   # Add Methyl
        '[C;H3:1] >> [C:1]CC',                  # Add Ethyl
        '[C;H3:1] >> [C:1]CCC',                 # Add Propyl
        '[C;H3:1] >> [C:1]CCCC',                # Add Butyl
        
        # Branching & Isomerization
        '[C;H2:1] >> [CH:1](C)C',               # Isopropyl branching
        
        # Nitrogen / Carbon Functionalization (Alcohol / Hydroxyl ILs like Choline)
        '[C;H3:1] >> [C:1]CO',                  # Ethanolamine derivative
        
        # Symmetrical Alkyl Extension on Quaternary Nitrogen
        '[N+:1](C)(C)C >> [N+:1](CC)(CC)CC',    # TMA -> TEA
        '[N+:1](CC)(CC)CC >> [N+:1](CCC)(CCC)CCC', # TEA -> TPA
        '[N+:1](CCC)(CCC)CCC >> [N+:1](CCCC)(CCCC)CCCC', # TPA -> TBA
    ]

    compiled_rxns = [rdChemReactions.ReactionFromSmarts(s) for s in cation_reactions]

    # 3. Generate Mutated Cations
    generated_cations = set()
    
    for cat_mol in cation_mols:
        # Include original cation
        generated_cations.add(Chem.MolToSmiles(cat_mol, canonical=True))
        
        # Apply SMARTS Transformations
        for rxn in compiled_rxns:
            try:
                products = rxn.RunReactants((cat_mol,))
                for prod_tuple in products:
                    for prod in prod_tuple:
                        try:
                            Chem.SanitizeMol(prod)
                            generated_cations.add(Chem.MolToSmiles(prod, canonical=True))
                        except Exception:
                            continue
            except Exception:
                continue

    # 4. Combinatorial Cross-Product: (Mutated Cations) x (Common DB Anions)
    for cat_smiles in generated_cations:
        # A. Yield pure cation alone (if cataloged)
        yield cat_smiles
        
        # B. Yield cation paired with every common database anion
        for anion_smiles in COMMON_DATABASE_ANIONS:
            full_salt = f"{cat_smiles}.{anion_smiles}"
            validated_mol = Chem.MolFromSmiles(full_salt)
            if validated_mol is not None:
                yield Chem.MolToSmiles(validated_mol, canonical=True)

# Run Step 1
has_data, direct_data = engine._check_target()

# If Step 1 fails, run dynamic Step 2!
if not has_data:
    harvested_neighbors = engine._harvest_neighbors(
        mutator_fn=mutator,
        distance_metric_fn=tanimoto_distance,
        k_neighbors=5
    )

    # 3. Initialize Step 3 Evaluator
    evaluator = SurrogateEvaluator(surrogate_model=StandardGPWrapper())

    # 4. Fit & Predict
    predicted_means, predicted_variances = evaluator.fit_and_evaluate(
        target_x=standard_molecular_features(target_smiles),
        harvested_neighbors=harvested_neighbors,
        property_keys=requested_properties
    )
