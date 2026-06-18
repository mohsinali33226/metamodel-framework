# Final Benchmark Data: Custom Workflow vs AutoGluon Benchmark

## Purpose

This folder documents the final comparison between the custom machine-learning workflow and AutoGluon as an established AutoML benchmark.

The comparison follows the supervisor's instruction: same dataset, same input features, same target variables, same train/test split, same metrics, and saved result tables/plots.

## Dataset setup

- Dataset: final benchmark bearing data
- Cases: G28 and G200
- Targets: Y/Z effective-value targets only
- Ignored targets: X targets, because supervisor stated that X vibration is almost zero and mostly noisy
- Features: input-only features
- Excluded columns: `res_*`, `id`, `group`
- Split: fixed 80/20 train/test split with `random_state=42`

## Scripts

1. `fixed_splits.py`  
   Creates fixed train/test CSV files for G28 and G200.

2. `run_custom_models_fixed_splits.py`  
   Runs the custom workflow using Ridge, RandomForest, ExtraTrees, SVR, and XGBoost.  
   Best final custom model was SVR for all final Y/Z targets.

3. `run_autogluon_fixed_splits.py`  
   Runs AutoGluon on the same fixed splits.  
   G28 uses `medium_quality`, G200 uses `best_quality`.

4. `custom_vs_autogluon_comparison.py`  
   Creates target-level and summary comparison tables.

5. `comparison_plots.py`  
   Creates predicted-vs-simulated plots, residual plots, and absolute-error distribution plots.

6. `error_analysis.py`  
   Creates error summaries, worst-case error tables, and per-case error files.

7. `workflow_comparison_table.py`  
   Creates a methodological workflow-level comparison table.

## Main result

AutoGluon outperformed the custom workflow on 15 out of 16 Y/Z targets.

Summary:

- G28 custom mean RÂ² â‰ˆ 0.8975
- G28 AutoGluon mean RÂ² â‰ˆ 0.9600
- G200 custom mean RÂ² â‰ˆ 0.8252
- G200 AutoGluon mean RÂ² â‰ˆ 0.8983

Only one target was better with the custom workflow:

- G200 `res_eff_low_y_my_s`

## Interpretation

AutoGluon gives better predictive performance because it uses automated model selection, bagging, stacking, and weighted ensembles.

The custom workflow is still useful because it is more transparent, easier to control, and better suited for integration into engineering workflows such as ProBeNo.

## Output folders

Main outputs are saved under:

```text
data/outputs/new_data/fixed_splits
data/outputs/new_data/custom_workflow
data/outputs/new_data/autogluon_fixed_split
data/outputs/new_data/comparison
data/outputs/new_data/comparison/plots
data/outputs/new_data/comparison/error_analysis