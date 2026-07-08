# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

from data_processing.harvesting import TargetHarvestingEngine
from encoders.chemicals import hybrid_physicochemical_encoder, standard_molecular_features
from adapters.databases import caleb_bell_db_adapter
from mutators.combinatorial import mutator
from metrics.chemicals_differences import tanimoto_distance
from models.gp import StandardGP
from evaluators.surrogates import HarvestedContextEvaluator

target_smiles = "CCCC[N+](CCCC)(CCCC)CCCC.[BH4-]" # TBABH
# target_smiles = "CC[N+](CC)(CC)CC.[BH4-]" # TEABH
# target_smiles = "C[N+](C)(C)C.[BH4-]" # TMABH
requested_properties = ["Melting Point"]

engine = TargetHarvestingEngine(
    target_smiles=target_smiles,
    requested_properties=requested_properties,
    featurizer_fn=hybrid_physicochemical_encoder,
    db_query_fn=caleb_bell_db_adapter
)

# has_data, data = engine._check_target()

harvested_neighbors = engine._harvest_neighbors(
    mutator_fn=mutator,
    distance_metric_fn=tanimoto_distance,
    k_neighbors=5
)

evaluator = HarvestedContextEvaluator(surrogate_model=StandardGP())

predicted_means, predicted_variances = evaluator.fit_and_evaluate(
    target_x=hybrid_physicochemical_encoder(target_smiles),
    harvested_neighbors=harvested_neighbors,
    property_keys=requested_properties
)
