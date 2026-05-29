# ML Foundations + Evaluation Toolkit from Scratch

## Motivation

This is a learning-oriented ML research engineering project for building foundations in Trustworthy ML, AI Systems, and Evaluation/Monitoring.

The focus is implementing core machine learning components from scratch while keeping the math visible and the project organized with configs, logging, tests, experiment scripts, and weekly notes. This is not intended to be a production-ready ML library.

## Current Status

Week 1 and Week 2 foundations are complete.

## Implemented Components

- synthetic data generation
- train/validation split
- feature standardization
- `set_seed`
- `get_logger`
- YAML config loading
- `LinearRegressionScratch`
- `LogisticRegressionScratch`
- `BatchGradientDescent`
- `mean_squared_error`
- `binary_cross_entropy`
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `confusion_matrix`
- loss curve plotting
- linear regression training experiment
- logistic regression training experiment
- unit tests and integration tests
- Week 1 and Week 2 reports

## Environment Setup

From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Tests

```powershell
pytest
```

## Run Experiments

```powershell
python experiments/run_linear_regression.py
python experiments/run_logistic_regression.py
```

Experiment logs are saved under `results/logs/`, and loss curves are saved under `results/figures/`. Generated logs and figures are ignored by Git.

## Design Principles

- separate model, optimizer, evaluation, data, and experiment responsibilities
- keep mathematical formulas visible in code
- use unit tests plus integration tests
- treat evaluation diagnostics as first-class
- avoid trusting scalar metrics without checking data distribution and confusion matrix

## Reports

- `reports/week1_setup_notes.md`
- `reports/week2_linear_logistic_regression.md`

## Next Steps

- implement SGD, Momentum, and Adam
- implement MLP from scratch
- compare optimizers
- add calibration and confidence analysis later
