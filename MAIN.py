# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

from data_processing.harvesting import TargetHarvestingEngine
from encoders.chemicals import hybrid_physicochemical_encoder, standard_molecular_features
from adapters.databases import CalebBellDatabaseAdapter, BradleyDatabaseAdapter
from mutators.combinatorial import mutator, create_smart_mutator, selfies_mutator
from metrics.chemicals_differences import tanimoto_distance
from models.gp import StandardGP
from evaluators.surrogates import HarvestedContextEvaluator

import numpy as np
from tabulate import tabulate

# TARGET_POOL = [
#     # ------------------------------------------------------------------
#     # 1. Simple Alcohols (Small & Linear Organics)
#     # ------------------------------------------------------------------
#     "CO",                             # Methanol (175.47 K)
#     "CCO",                            # Ethanol (159.05 K)
#     "CCCO",                           # 1-Propanol (146.95 K)
#     "CCCCO",                          # 1-Butanol (183.85 K)
#     "CC(O)C",                         # Isopropanol (185.25 K)
    
#     # ------------------------------------------------------------------
#     # 2. Carboxylic Acids & Carbonyls
#     # ------------------------------------------------------------------
#     "CC(=O)O",                        # Acetic Acid (289.65 K)
#     "CCC(=O)O",                       # Propionic Acid (252.45 K)
#     "CC(=O)C",                        # Acetone (178.45 K)
#     "CCC(=O)C",                       # Methyl Ethyl Ketone / MEK (186.45 K)

#     # ------------------------------------------------------------------
#     # 3. Simple Aromatics (Halogen & Alkyl Swaps)
#     # ------------------------------------------------------------------
#     "c1ccccc1",                       # Benzene (278.68 K)
#     "Cc1ccccc1",                      # Toluene (178.15 K)
#     "Fc1ccccc1",                      # Fluorobenzene (230.94 K)
#     "Clc1ccccc1",                     # Chlorobenzene (227.95 K)
#     "Brc1ccccc1",                     # Bromobenzene (242.43 K)
#     "Ic1ccccc1",                      # Iodobenzene (241.80 K)

#     # ------------------------------------------------------------------
#     # 4. Short-Chain Haloalkanes
#     # ------------------------------------------------------------------
#     "CCCl",                           # Chloroethane (134.45 K)
#     "CCBr",                           # Bromoethane (154.55 K)
#     "CCCCl",                          # 1-Chloropropane (150.35 K)

#     # ------------------------------------------------------------------
#     # 5. Ionic Salts & Quaternary Ammonium (Counterion & Chain Swaps)
#     # ------------------------------------------------------------------
#     "C[N+](C)(C)C.[Cl-]",             # Tetramethylammonium Chloride (TMAC)
#     "CC[N+](CC)(CC)CC.[Cl-]",         # Tetraethylammonium Chloride (TEAC)
#     "CC[N+](CC)(CC)CC.[Br-]",         # Tetraethylammonium Bromide (TEABr)
#     "CCCC[N+](CCCC)(CCCC)CCCC.[Cl-]", # Tetrabutylammonium Chloride (TBAC)
#     "CCCC[N+](CCCC)(CCCC)CCCC.[Br-]", # Tetrabutylammonium Bromide (TBABr)
#     "CCCC[N+](CCCC)(CCCC)CCCC.[I-]",  # Tetrabutylammonium Iodide (TBAI)
# TARGET_POOL = [
#     "[BH3-][NH3+]",
#     "[B-][NH2+]C",
#     "CNC.[B]",
#     "[B][N+](C)(C)C",
#     "CCN.[B]",
#     "[BH3-][NH2+]CCC",
#     "CCCCN.[B]",

#     "[BH4-].[Li+]",
#     "[BH4-].[Na+]",

#     "C[N+](C)(C)C.[BH4-]",
#     "CC[N+](CC)(CC)CC.[BH4-]",
#     "CCCC[N+](CCCC)(CCCC)CCCC.[BH4-]"
# ]

TARGET_POOL = ["O", "[BH4-].[Li+]", "[BH4-].[Na+]"]

requested_properties = ["Melting Point"]

ACTIVE_ADAPTER = BradleyDatabaseAdapter(file_path="data/input/BradleyMeltingPointDataset.csv")
# ACTIVE_ADAPTER = CalebBellDatabaseAdapter(smart_mutator=selfies_mutator)
HARVEST_STRATEGY = 'full_cropped'
REQUESTED_PROPERTIES = ['Melting Point', 'Enthalpy of Fusion']

summary_results = []

print("="*70)
print(" 🚀 STARTING BATCH HARVESTING & SURROGATE EVALUATION")
print("="*70)

for idx, target_smiles in enumerate(TARGET_POOL, 1):
    print("\n" + "="*80)
    print(f"[{idx}/{len(TARGET_POOL)}] Processing Target: {target_smiles}")
    print("="*80)
    
    # 1. Initialize the engine with our properties list
    harvester = TargetHarvestingEngine(
        target_smiles=target_smiles,
        requested_properties=REQUESTED_PROPERTIES,
        featurizer_fn=hybrid_physicochemical_encoder,
        adapter=ACTIVE_ADAPTER
    )
    
    neighbors = harvester.harvest_neighbors(
        distance_metric_fn=tanimoto_distance,
        strategy=HARVEST_STRATEGY,
        k_neighbors=100
    )

    evaluator = HarvestedContextEvaluator(surrogate_model=StandardGP())

    predicted_means, predicted_variances = evaluator.fit_and_evaluate(
        target_x=hybrid_physicochemical_encoder(target_smiles),
        harvested_neighbors=neighbors,
        property_keys=requested_properties
    )

    _, target_data = harvester.check_target()

    actual_val = target_data["Melting Point"]
    pred_mean = predicted_means['Melting Point']
    pred_std = np.sqrt(predicted_variances['Melting Point'])
    error = abs(pred_mean - actual_val)

    # Save to summary list for final tabulations
    summary_results.append({
        "smiles": target_smiles,
        "actual": actual_val,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "error": error
    })

# =====================================================================
# 3. POST-PROCESSING: Convert to Pandas and Print
# =====================================================================
import pandas as pd
df_results = pd.DataFrame(summary_results)

print("\n" + "#"*40)
print("FINAL TABULATION SUMMARY")
print("#"*40)
print(df_results.to_string(index=False))