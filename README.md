# Metamodel Framework for Rolling Bearing Optimization

## 1. Project Overview

This repository contains the Python work for the project thesis:

**Development of a Methodological Framework for Rapid Metamodel Generation in Rolling Bearing Optimization**

The project develops a script-based workflow for creating and comparing metamodels from rolling-bearing simulation data.

A metamodel is a fast prediction model that learns the relationship between simulation inputs and outputs.

The project contains two main stages:

1. an earlier framework-development stage using `sampling.csv` inputs and BearinX `.l` outputs,
2. a final G28/G200 benchmark comparing a custom Python workflow with AutoGluon.

The project operates through Python scripts in VS Code. It does not include a finished graphical user interface, automatic BearinX execution, or a complete closed-loop optimization system.

---

## 2. Project Objective

The workflow:

* loads prepared bearing simulation data,
* selects valid input features and targets,
* prevents data leakage,
* creates reproducible train-test splits,
* trains and compares several regression models,
* tunes model settings,
* calculates R², MAE, and RMSE,
* exports predictions, tables, and plots,
* compares a transparent custom workflow with AutoGluon.

---

## 3. Datasets Used

### Earlier L-effective dataset

The first dataset uses:

* `sampling.csv` for input parameters,
* BearinX `.l` files for simulation outputs.

This dataset supports framework development and testing.

The main work from this stage includes:

* correcting input-output pairing,
* training and tuning surrogate models,
* analysing weak targets,
* testing selected AutoGluon models,
* investigating adaptive sampling.

The results from this stage are stored under:

```text
data/outputs/l_effective
```

### Spectrum dataset

A separate exploratory branch processes BearinX spectrum files.

It:

* inspects spectrum-file structure,
* extracts numerical spectrum tables,
* calculates RMS values for frequency bands,
* builds spectrum-learning datasets.

The results are stored under:

```text
data/outputs/spectrum
```

### Final G28/G200 benchmark

The final benchmark uses two datasets:

* G28,
* G200.

Each dataset contains 500 simulation cases.

For each dataset:

* 400 cases are used for training,
* 100 cases are used for testing.

The final benchmark uses eight Y- and Z-direction vibration targets:

* broad Y and Z,
* low Y and Z,
* medium Y and Z,
* high Y and Z.

The X direction is excluded because its values are very small and irregular.

---

## 4. Main Results

The final comparison uses the same input features, targets, and fixed test cases for both workflows.

| Dataset | Custom mean R² | AutoGluon mean R² | Custom median R² | AutoGluon median R² |
| ------- | -------------: | ----------------: | ---------------: | ------------------: |
| G28     |         0.8975 |            0.9600 |           0.9045 |              0.9711 |
| G200    |         0.8252 |            0.8983 |           0.8245 |              0.9195 |

AutoGluon achieves the higher R² for 15 of the 16 evaluated targets.

The exception is the G200 low-frequency Y target, where the custom workflow performs better.

For this specific engineering context:

```text
R² > 0.90 is considered acceptable.
```

AutoGluon meets this criterion on average for G28 and remains slightly below it for G200.

---

## 5. Repository Structure

```text
metamodel_framework
├── .vscode
├── data
│   ├── raw
│   └── outputs
│       ├── l_effective
│       ├── new_data
│       └── spectrum
│
├── scripts
│   ├── 01_l_effective_original_workflow
│   └── new_data_comparison
│
├── src
│   ├── framework
│   └── spectrum_pipeline
│
├── outputs_paths.yaml
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Main folder meaning

* `data` contains input data and generated results.
* `scripts` contains Python files that perform complete workflow steps.
* `src` means source code and contains reusable Python functions and model classes.
* `outputs_paths.yaml` stores central output-path definitions.
* `requirements.txt` lists the required Python packages.

Detailed README files are included inside the main workflow and output folders.

---

## 6. Main Workflows

### Earlier L-effective workflow

This workflow is stored under:

```text
scripts/01_l_effective_original_workflow
```

The main stages are:

```text
sampling.csv + BearinX .l files
        ↓
input-output pairing
        ↓
dataset creation
        ↓
train-validation-test split
        ↓
model training and tuning
        ↓
test prediction and evaluation
```

The earlier reusable surrogate layer includes:

* Gaussian Process,
* RBF,
* SVR,
* ANN,
* ensemble models.

Additional experiments include tree models, boosting, feature engineering, evolutionary optimization, and adaptive sampling.

### Final G28/G200 benchmark

This workflow is stored under:

```text
scripts/new_data_comparison
```

The final custom workflow tests:

* Ridge Regression,
* Support Vector Regression,
* Random Forest,
* Extra Trees,
* XGBoost.

SVR is selected as the final custom model for all G28 and G200 targets.

AutoGluon automatically trains several model families and mainly selects weighted ensembles.

---

## 7. Important Scripts

The main final benchmark scripts are:

```text
fixed_splits.py
run_custom_models_fixed_splits.py
run_autogluon_fixed_splits.py
custom_vs_autogluon_comparison.py
comparison_plots.py
error_analysis.py
workflow_comparison_table.py
```

Recommended execution order:

```bash
python scripts/new_data_comparison/fixed_splits.py
python scripts/new_data_comparison/run_custom_models_fixed_splits.py
python scripts/new_data_comparison/run_autogluon_fixed_splits.py
python scripts/new_data_comparison/custom_vs_autogluon_comparison.py
python scripts/new_data_comparison/comparison_plots.py
python scripts/new_data_comparison/error_analysis.py
python scripts/new_data_comparison/workflow_comparison_table.py
```

---

## 8. Important Output Locations

### Earlier L-effective results

```text
data/outputs/l_effective
```

The `overview` subfolder groups the important earlier results into clear categories.

### Spectrum results

```text
data/outputs/spectrum
```

### Fixed train-test splits

```text
data/outputs/new_data/fixed_splits
```

### Custom-workflow results

```text
data/outputs/new_data/custom_workflow
```

### AutoGluon fixed-split results

```text
data/outputs/new_data/autogluon_fixed_split
```

### Final comparison results

```text
data/outputs/new_data/comparison
```

---

## 9. Running the Project

Install the required packages with:

```bash
pip install -r requirements.txt
```

The final benchmark uses the Conda environment:

```text
autogluon_env
```

Main environment information:

```text
Python 3.11.15
pandas 2.3.3
NumPy 2.1.3
scikit-learn 1.7.2
XGBoost 3.1.3
AutoGluon 1.5.0
matplotlib 3.10.9
```

To reproduce the final benchmark, run the scripts in the order listed in Section 7.

---

## 10. Evaluation and Limitations

The project uses:

* R²,
* MAE,
* RMSE,
* predicted-versus-simulated plots,
* residual plots,
* absolute-error distributions,
* case-level error comparisons.

These provide error-based validation.

The project does not include:

* automatic BearinX execution,
* completed adaptive sampling with valid new simulations,
* full probabilistic uncertainty quantification,
* a finished graphical user interface,
* complete closed-loop bearing optimization.

---

## 11. Final Submission Folder

The final thesis submission package is stored separately from the working VS Code project.

Its structure is:

```text
Thesis_Data
├── 01_Thesis
├── 02_Literature
├── 03_Methodology
├── 04_Python_Project
├── 05_Data_and_Results
├── 06_Graphics
└── 07_Documentation
```

Only the contents of `Thesis_Data` are copied to the USB stick.

---

## 12. Backup Note

During final organisation, duplicate files from:

```text
data/outputs/l_effective
```

that also exist inside:

```text
data/outputs/l_effective/overview
```

are moved to:

```text
C:\Thesis_Backup\l_effective_duplicate_files
```

They are not permanently deleted.

They can be restored if an earlier script expects one of the original file paths.

---

## 13. Future Work

Possible future improvements include:

* stable BearinX file generation,
* automatic BearinX execution,
* completed adaptive sampling,
* repeated-split validation,
* probabilistic uncertainty quantification,
* optional AutoGluon integration,
* graphical user-interface development,
* integration into a broader engineering environment such as ProBeNo.