# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

from data_processing.harvesting import TargetHarvestingEngine
from encoders.chemicals import hybrid_physicochemical_encoder, standard_molecular_features
from adapters.databases import caleb_bell_db_adapter
from mutators.combinatorial import mutator, create_smart_mutator, selfies_mutator
from metrics.chemicals_differences import tanimoto_distance
from models.gp import StandardGP
from evaluators.surrogates import HarvestedContextEvaluator

import numpy as np
from tabulate import tabulate

# target_smiles = "CCCC[N+](CCCC)(CCCC)CCCC.[BH4-]" # TBABH
# target_smiles = "CC[N+](CC)(CC)CC.[BH4-]" # TEABH
# target_smiles = "C[N+](C)(C)C.[BH4-]" # TMABH
# target_smiles = "CCO"
# requested_properties = ["Melting Point"]
# smart_mutator = create_smart_mutator()

# engine = TargetHarvestingEngine(
#     target_smiles=target_smiles,
#     requested_properties=requested_properties,
#     featurizer_fn=hybrid_physicochemical_encoder,
#     db_query_fn=caleb_bell_db_adapter
# )

# has_data, data = engine._check_target()

# harvested_neighbors = engine._harvest_neighbors(
#     mutator_fn=lambda smi: smart_mutator.mutate(smi),
#     distance_metric_fn=tanimoto_distance,
#     k_neighbors=10
# )

# smart_mutator.print_mutation_report()

# evaluator = HarvestedContextEvaluator(surrogate_model=StandardGP())

# predicted_means, predicted_variances = evaluator.fit_and_evaluate(
#     target_x=hybrid_physicochemical_encoder(target_smiles),
#     harvested_neighbors=harvested_neighbors,
#     property_keys=requested_properties
# )

# # =====================================================================
# # PRINT HARVESTED DATA VS. PREDICTIONS
# # =====================================================================

# print("\n" + "="*50)
# print(f" 🎯 TARGET RESULTS: {target_smiles}")
# print("="*50)
# print(f" Predicted Melting Point : {predicted_means['Melting Point']:.2f} K ± {np.sqrt(predicted_variances['Melting Point']):.2f} K")
# print(f" Actual Database Value   : {data['Melting Point']:.2f}")
# print("="*50 + "\n")

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
TARGET_POOL = [
    "[BH3-][NH3+]",
    "[B-][NH2+]C",
    "CNC.[B]",
    "[B][N+](C)(C)C",
    "CCN.[B]",
    "[BH3-][NH2+]CCC",
    "CCCCN.[B]",

    "[BH4-].[Li+]",
    "[BH4-].[Na+]",

    "C[N+](C)(C)C.[BH4-]",
    "CC[N+](CC)(CC)CC.[BH4-]",
    "CCCC[N+](CCCC)(CCCC)CCCC.[BH4-]"
]

requested_properties = ["Melting Point"]
smart_mutator = create_smart_mutator()
# List to accumulate final table results
summary_results = []

print("="*70)
print(" 🚀 STARTING BATCH HARVESTING & SURROGATE EVALUATION")
print("="*70)

for idx, target_smiles in enumerate(TARGET_POOL, 1):
    print(f"\n[{idx}/{len(TARGET_POOL)}] Processing Target: {target_smiles}")
    
    engine = TargetHarvestingEngine(
        target_smiles=target_smiles,
        requested_properties=requested_properties,
        featurizer_fn=hybrid_physicochemical_encoder,
        db_query_fn=caleb_bell_db_adapter
    )

    # 2. Check if ground truth data exists in the database
    has_data, target_data = engine._check_target()

    if not has_data:
        actual_val = np.inf
    else:
        # Extract actual value safely
        actual_val = target_data["Melting Point"]

    harvested_neighbors = engine._harvest_neighbors(
        mutator_fn=lambda smi: smart_mutator.mutate_until_k(smi),
        # mutator_fn=selfies_mutator,
        distance_metric_fn=tanimoto_distance,
        k_neighbors=10,
        max_attempts=2000
    )

    # smart_mutator.print_mutation_report()

    evaluator = HarvestedContextEvaluator(surrogate_model=StandardGP())

    predicted_means, predicted_variances = evaluator.fit_and_evaluate(
        target_x=hybrid_physicochemical_encoder(target_smiles),
        harvested_neighbors=harvested_neighbors,
        property_keys=requested_properties
    )

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
# 5. PRINT FINAL SUMMARY TABLE FOR ALL EVALUATED TARGETS
# =====================================================================
summary_results.sort(key=lambda item: item["error"])

print("\n\n" + "="*85)
print(" 📊 FINAL BATCH RESULTS: SORTED BY LOWEST TO HIGHEST ABSOLUTE ERROR")
print("="*85)
print(f"{'#':<3} | {'Target SMILES':<35} | {'Actual DB (K)':<13} | {'Predicted (K)':<18} | {'Abs Error (K)'}")
print("-" * 85)

for i, res in enumerate(summary_results, 1):
    actual_str = f"{res['actual']:.2f}"
    pred_str = f"{res['pred_mean']:.2f} ± {res['pred_std']:.2f}"
    err_str = f"{res['error']:.2f}"
    
    print(f"{i:<3} | {res['smiles']:<35} | {actual_str:<13} | {pred_str:<18} | {err_str}")

print("="*85 + "\n")