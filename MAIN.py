# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"

target_smiles = "[Li+].[BH4-]"
requested_properties = ["T_melt"]

engine = TargetHarvestingEngine(
    target_smiles=target_smiles,
    requested_properties=requested_properties,
    featurizer_fn=standard_molecular_features,
    db_query_fn=caleb_bell_db_adapter
)

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
