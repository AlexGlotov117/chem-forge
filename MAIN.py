# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

from data_processing.harvesting import TargetHarvestingEngine
from encoders.chemicals import standard_molecular_features
from adapters.databases import caleb_bell_db_adapter
from mutators.combinatorial import mutator
from metrics.chemicals_differences import tanimoto_distance
from models.gp import StandardGP
from evaluators.surrogates import HarvestedContextEvaluator

target_smiles = "[BH4-].CCCC[N+](CCCC)(CCCC)CCCC"
requested_properties = ["Melting Point"]

engine = TargetHarvestingEngine(
    target_smiles=target_smiles,
    requested_properties=requested_properties,
    featurizer_fn=standard_molecular_features,
    db_query_fn=caleb_bell_db_adapter
)

has_data, data = engine._check_target()

if has_data:
    print(data)
else:
    harvested_neighbors = engine._harvest_neighbors(
        mutator_fn=mutator,
        distance_metric_fn=tanimoto_distance,
        k_neighbors=5
    )

    # 3. Initialize Step 3 Evaluator
    evaluator = HarvestedContextEvaluator(surrogate_model=StandardGP())

    # 4. Fit & Predict
    predicted_means, predicted_variances = evaluator.fit_and_evaluate(
        target_x=standard_molecular_features(target_smiles),
        harvested_neighbors=harvested_neighbors,
        property_keys=requested_properties
    )
