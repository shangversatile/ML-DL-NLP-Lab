# ML Foundations + Evaluation Toolkit from Scratch

## Motivation

This is a learning-oriented ML research engineering project for building foundations in Trustworthy ML, AI Systems, and Evaluation/Monitoring.

The focus is implementing core machine learning components from scratch while keeping the math visible and the project organized with configs, logging, tests, experiment scripts, and weekly notes. This is not intended to be a production-ready ML library.

## Current Status

Week 1, Week 2, and Week 3 foundations are complete through the binary NumPy MLP. Week 4 is in progress with multiclass softmax and cross entropy utilities for the handwritten-digit recognition capstone.

| Week   | Theme                                                      | Status                                  |
| ------ | ---------------------------------------------------------- | --------------------------------------- |
| Week 1 | Reproducible Research Engineering Setup                    | complete                                |
| Week 2 | Linear / Logistic Regression and Probabilistic Foundations | complete                                |
| Week 3 | Optimization and Binary MLP Foundations                    | complete, tagged `week3-optimization-mlp` |
| Week 4 | Multiclass MLP and Handwritten-Digit Recognition Capstone  | in progress                             |
| Week 5 | Evaluation, Technical Debt, and Trustworthy ML Diagnostics | planned                                 |

Current notes are tracked in [Week 3 Optimization and MLP Notes](reports/week3_optimization_and_mlp.md) and [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](reports/week4_multiclass_digits_capstone.md).

## Implemented Components

- reproducible experiment utilities
- synthetic-data pipelines
- linear regression from scratch
- logistic regression from scratch
- binary MLP forward and backpropagation
- analytical and numerical gradient verification
- SGD, Momentum, and Adam
- generic parameter-dictionary optimizers
- nonlinear XOR-style training
- controlled optimizer comparison
- controlled overfitting analysis
- multiclass one-hot, stable softmax, cross entropy, and output-gradient utilities

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
- `reports/week3_optimization_and_mlp.md`
- `reports/week4_multiclass_digits_capstone.md`
- `reports/week5_evaluation_technical_debt.md`

## Next Steps

- Multiclass MLP from scratch
- handwritten-digit training
- error analysis
- local prediction application
- trustworthy-ML evaluation extensions
