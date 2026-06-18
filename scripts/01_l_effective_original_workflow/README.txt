This folder contains the Python scripts used for the earlier L-effective framework-development phase.

The original data for this branch came from:

* sampling.csv for the bearing input parameters
* BearinX .l files for the simulation outputs

The scripts are grouped by purpose.

1. core_pipeline

This folder contains the main modelling workflow.

Main steps include:

* build_dataset.py
  Combines the prepared inputs and outputs into one machine-learning dataset.

* split_dataset.py
  Divides the dataset into training, validation and test sets.

* train_models.py
  Trains the available surrogate models.

* select_best_models.py
  Compares the models and selects the best model for each target.

* predict_test_set.py
  Uses the selected models to predict the unseen test data.

* evaluate_predictions.py
  Calculates evaluation metrics such as R², MAE and RMSE.

* uncertainty_coverage.py
  Checks the coverage of Gaussian Process uncertainty intervals.

* uncertainty_export.py
  Exports uncertainty-related prediction results.

2. diagnostics

This folder contains scripts used to investigate data quality, mapping and model behaviour.

Examples include:

* model_diagnostics.py
  Analyses model performance and weak prediction targets.

* vg2_mapping_check.py
  Checks the mapping between sampling rows and BearinX files.

* vg2_mapping_batch_check.py
  Performs mapping checks for several files.

* vg2_replacement_validation_initial.py
  Performs an initial validation of VG2 value replacement.

* vg2_replacement_validation_final.py
  Performs the final validation of VG2 value replacement.

These scripts were important because the earlier weak model results were mainly caused by incorrect input-output pairing.

3. optimization_experiments

This folder contains additional model-tuning and optimization experiments.

The experiments include:

* evolutionary-algorithm optimization
* SVR tuning
* Random Forest tuning
* XGBoost tuning
* ANN tuning
* boosting experiments
* feature-engineering experiments
* stacking and ensemble tests

File names usually contain:

* ea_svr for evolutionary optimization of SVR
* ea_rf for evolutionary optimization of Random Forest
* ea_xgb for evolutionary optimization of XGBoost
* ea_ann for evolutionary optimization of ANN

These scripts were used target by target to improve prediction quality.

4. adaptive_sampling

This folder contains the adaptive-sampling experiments.

Main scripts include:

* adaptive_enrichment_points.py
  Selects new points in weak or poorly covered regions.

* adaptive_full_input_proposals.py
  Creates new simulation proposals using the full input space.

* adaptive_vg2_file_export.py
  Attempts to export new BearinX VG2 files for the selected candidates.

The adaptive-sampling loop was not completed because the generated BearinX files produced syntax problems. This part therefore remained experimental and is documented as future work.

Project role:

This folder belongs to the earlier framework-development phase.

The final G28/G200 custom-versus-AutoGluon benchmark is stored separately under:

scripts/new_data_comparison