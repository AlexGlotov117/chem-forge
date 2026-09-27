# To setup conda use: eval "$(/home/aglotov/miniconda3/bin/conda shell.bash hook)"
# Use CEARun

from models.propertyPredictor import MTGPR_Tm_Hfus_Hf
from data_processing.dataProcessing import prepare_MTGPR_Tm_Hfus_Hf


if __name__ == "__main__":
    # USER PROVIDED INPUTS
    # model_name = "MutualInformationApproach_v1"
    model_name = "TestRSME000223"

    # ====================================================================================
    # Train/load model
    print(f"=== Model Training ===")
    prop_predictor = MTGPR_Tm_Hfus_Hf(model_name=model_name)

    saved_model_name = prop_predictor.model_name
    print(f"Using model name: {saved_model_name}")

    # prop_predictor.train_and_save(train_filepath="data/input/MTGPR_Tm_Hfus_Hf/train_TBABH.xlsx", test_filepath="data/input/MTGPR_Tm_Hfus_Hf/test_TBABH.xlsx")
    prop_predictor.load_model()

    print(f"=== Workflow Preparation ===")
    filled_path, combos_path = prepare_MTGPR_Tm_Hfus_Hf(
        excel_path="data/input/pureComponents_2026DecJANNAF_testing_predicted.xlsx",
        # excel_path="data/input/pureComponents.xlsx",
        predictor=prop_predictor,
        combination_arities=[2],
        output_dir="data/output/20260922"
    )