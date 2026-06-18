This folder contains the reusable Python components of the metamodel framework.

Unlike the files under the scripts folder, these files are not normally executed directly. They provide reusable functions and model classes that can be imported by different scripts.

Folder structure:

1. data

Contains functions for loading, preparing and splitting data.

Main files:

* csv_loader.py
  Loads CSV datasets and performs basic data checks.

* data_split.py
  Creates training, validation and test splits.

* sampling.py
  Contains sampling-related helper functions.

2. optimization

Contains the framework location for optimization-related functions.

The folder currently mainly provides the package structure and can be extended with additional optimization methods in future work.

3. pipeline

Contains functions that connect separate framework steps.

Main file:

* runner.py
  Coordinates or supports the execution of the modelling pipeline.

4. surrogates

Contains reusable surrogate-model implementations.

Main files:

* base.py
  Defines the common structure used by surrogate models.

* gp.py
  Implements Gaussian Process regression.

* rbf.py
  Implements Radial Basis Function modelling.

* svr.py
  Implements Support Vector Regression.

* ann_tf.py
  Implements an Artificial Neural Network using TensorFlow.

* ensemble.py
  Combines predictions from several surrogate models.

These model classes were mainly developed for the earlier L-effective framework phase.

5. utils

Contains general helper functions shared by different scripts.

Main files:

* io.py
  Handles reading and writing files.

* logging.py
  Supports logging and recording execution information.

* paths.py
  Reads and manages project paths.

* run_paths.py
  Creates or manages paths for individual workflow runs.

6. validation

Contains reusable evaluation and validation functions.

Main files:

* metrics.py
  Calculates evaluation metrics such as R², MAE and RMSE.

* split.py
  Supports data-splitting and validation procedures.

* mop.py
  Contains model-selection or Metamodel of Optimal Prognosis-related functions.

Project role:

The src/framework folder provides the reusable technical foundation of the project.

The scripts folder contains the executable workflows that use these components.

No files were removed from this folder because they may be imported by earlier scripts and they document the modular framework architecture.
