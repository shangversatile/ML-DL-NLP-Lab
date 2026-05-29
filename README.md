# ML Foundations + Evaluation Toolkit from Scratch

## Motivation

This repository is a learning-oriented ML research engineering project focused on building foundational ability for Trustworthy ML, AI Systems, and Evaluation/Monitoring.

The goal is to implement core ML components from scratch while maintaining reproducibility, modularity, tests, logging, configs, and experiment notes. The project is intentionally educational rather than a polished production library.

## Current Status

Week 1 and major Week 2 foundations are complete.

Completed:

- project structure
- reproducibility utilities
- logging utilities
- YAML config loading
- synthetic data generation
- preprocessing
- linear regression from scratch
- logistic regression from scratch
- batch gradient descent optimizer
- regression metric: MSE
- classification metrics: accuracy, precision, recall, F1, confusion matrix
- loss curve plotting
- linear regression training experiment
- logistic regression training experiment
- unit tests and integration tests
- weekly reports

## Implemented Components

### Data

- synthetic linear regression data
- roughly balanced synthetic binary classification data
- train/validation split
- feature standardization using training statistics only

### Models

- `LinearRegressionScratch`
- `LogisticRegressionScratch`

### Optimization

- `BatchGradientDescent`

### Evaluation

- `mean_squared_error`
- `accuracy_score`
- `precision_score`
- `recall_score`
- `f1_score`
- `confusion_matrix`
- label distribution diagnostics in logistic regression experiment

### Utilities

- `set_seed`
- `get_logger`
- `load_config`
- `plot_loss_curve`

## Repository Structure

```text
configs/      YAML configuration files for experiments.
src/          Main Python package for data, models, optimization, evaluation, and utilities.
experiments/  Entry-point scripts for running training experiments.
tests/        Unit and integration tests.
reports/      Weekly learning and engineering notes.
results/      Generated logs and figures.
```

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

All tests should pass.

## Run Experiments

```powershell
python experiments/run_linear_regression.py
python experiments/run_logistic_regression.py
```

The linear regression experiment saves a loss curve to `results/figures/`. Logs are saved under `results/logs/`. Generated logs and figures are ignored by Git.

## Reports

- `reports/week1_setup_notes.md`
- `reports/week2_linear_logistic_regression.md`

## Current Design Principles

- separate model, optimizer, evaluation, data, and experiment responsibilities
- keep mathematical formulas visible in code
- prefer unit tests plus integration tests
- treat evaluation diagnostics as first-class components
- avoid trusting scalar metrics without inspecting data distribution and confusion matrix

## Next Steps

- add logistic regression loss curve plotting
- add cross entropy metric
- add threshold analysis
- add calibration interface
- implement SGD, Momentum, and Adam
- implement a small MLP from scratch
