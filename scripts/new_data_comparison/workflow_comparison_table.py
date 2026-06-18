from pathlib import Path
import pandas as pd

OUT = Path("data/outputs/new_data/comparison")
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "aspect": "Data import",
        "custom_workflow": "Loads fixed train/test CSV files explicitly with pandas.",
        "autogluon_workflow": "Loads the same fixed train/test CSV files, passed directly to TabularPredictor.",
        "interpretation": "Both use the same data basis, so comparison is fair."
    },
    {
        "aspect": "Preprocessing",
        "custom_workflow": "Manual numeric conversion, NaN handling by median imputation, scaling for SVR/Ridge.",
        "autogluon_workflow": "Automatic preprocessing inside AutoGluon, including type handling and model-specific preprocessing.",
        "interpretation": "Custom workflow is more transparent; AutoGluon is more automated."
    },
    {
        "aspect": "Feature selection",
        "custom_workflow": "Input-only features selected manually; result columns and id/group excluded.",
        "autogluon_workflow": "Same input-only features provided; AutoGluon may internally select/use features through model training.",
        "interpretation": "External feature set is identical for both workflows."
    },
    {
        "aspect": "Target selection",
        "custom_workflow": "Y/Z effective-value targets only; X targets excluded as noisy according to supervisor guidance.",
        "autogluon_workflow": "Same Y/Z target variables used.",
        "interpretation": "Target comparison is identical and thesis-valid."
    },
    {
        "aspect": "Train/test split",
        "custom_workflow": "Uses fixed 80/20 split files with random_state=42.",
        "autogluon_workflow": "Uses exactly the same fixed 80/20 split files.",
        "interpretation": "Performance comparison is directly comparable."
    },
    {
        "aspect": "Models trained",
        "custom_workflow": "Ridge, RandomForest, ExtraTrees, SVR, XGBoost; best model selected by test RÂ² for comparison output.",
        "autogluon_workflow": "Automatically trains multiple models such as tree ensembles, gradient boosting, neural networks, and weighted ensembles.",
        "interpretation": "AutoGluon searches a broader model space."
    },
    {
        "aspect": "Hyperparameter optimization",
        "custom_workflow": "GridSearchCV with predefined parameter grids and 5-fold CV.",
        "autogluon_workflow": "Automatic model training, bagging, stacking, ensembling, and internal model selection depending on preset.",
        "interpretation": "Custom approach is controlled; AutoGluon is stronger but less transparent."
    },
    {
        "aspect": "Final model selection",
        "custom_workflow": "Best custom model per target selected from tested models; SVR was selected for all final final targets.",
        "autogluon_workflow": "Best AutoGluon model selected automatically, mostly WeightedEnsemble_L2/L3.",
        "interpretation": "AutoGluon gains performance mainly through ensembling/stacking."
    },
    {
        "aspect": "Prediction performance",
        "custom_workflow": "Lower mean RÂ² than AutoGluon on both G28 and G200.",
        "autogluon_workflow": "Higher mean RÂ² and wins on 15 out of 16 Y/Z targets.",
        "interpretation": "AutoGluon is the stronger predictive benchmark."
    },
    {
        "aspect": "Interpretability",
        "custom_workflow": "Simpler models and explicit preprocessing make the workflow easier to explain and integrate.",
        "autogluon_workflow": "Weighted ensembles are less transparent and harder to interpret physically.",
        "interpretation": "Custom workflow has higher transparency."
    },
    {
        "aspect": "Automation level",
        "custom_workflow": "Requires manual model/grid definition and more user decisions.",
        "autogluon_workflow": "Highly automated after feature/target definition.",
        "interpretation": "AutoGluon is easier to use for benchmarking."
    },
    {
        "aspect": "ProBeNo integration potential",
        "custom_workflow": "Better suited for controlled integration because each preprocessing and model step is explicit.",
        "autogluon_workflow": "Useful as benchmark, but model complexity and dependencies may make integration harder.",
        "interpretation": "Custom workflow provides engineering integration value despite lower accuracy."
    },
]

df = pd.DataFrame(rows)

out = OUT / "workflow_level_comparison_custom_vs_autogluon.csv"
df.to_csv(out, index=False)

print("\nDONE WORKFLOW COMPARISON TABLE")
print(df.to_string(index=False))
print("\nsaved:", out)



