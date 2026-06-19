# ML Foundations + Evaluation Toolkit from Scratch

## Motivation

This is a learning-oriented ML research engineering project for building foundations in Trustworthy ML, AI Systems, and Evaluation/Monitoring.

The focus is implementing core machine learning components from scratch while keeping the math visible and the project organized with configs, logging, tests, experiment scripts, and weekly notes. This is not intended to be a production-ready ML library.

## Current Status

Week 1, Week 2, and Week 3 foundations are complete through the binary NumPy MLP. Week 4 has a baseline recognizer, local app, shift diagnostics, an augmented robustness comparison, real canvas sample debugging, and real canvas validation diagnostics. The augmented model strongly improves configured synthetic shift robustness, but the first real canvas validation set shows weak Top-1 accuracy with stronger Top-3 accuracy. This remains an educational/research prototype, not a production recognizer.

| Week   | Theme                                                      | Status                                  |
| ------ | ---------------------------------------------------------- | --------------------------------------- |
| Week 1 | Reproducible Research Engineering Setup                    | complete                                |
| Week 2 | Linear / Logistic Regression and Probabilistic Foundations | complete                                |
| Week 3 | Optimization and Binary MLP Foundations                    | complete, tagged `week3-optimization-mlp` |
| Week 4 | Multiclass MLP and Handwritten-Digit Recognition Capstone  | ready for final review                  |
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
- multiclass MLP forward pass, prediction, loss, analytical backpropagation, and gradient checking
- real handwritten-digit data pipeline with stratified train/validation/test split
- scratch multiclass MLP baseline training on the digits dataset
- confusion matrix and per-class diagnostics for multiclass predictions
- top-k accuracy, confidence summaries, high-confidence errors, and top-loss example inspection
- checkpoint saving/loading for the scratch multiclass MLP
- reusable handwritten-digit inference helpers with top-k outputs
- local Tkinter handwritten-digit drawing app using the checkpoint-loaded scratch MLP
- canvas preprocessing from local drawings to 64-feature digit inputs
- synthetic local-input distribution-shift probes
- confidence-bin and ECE-style diagnostics
- augmented training robustness-comparison experiment
- real canvas model-input debugging and user-sample evaluation
- real canvas validation diagnostics with per-class summaries, confusion counts, high-confidence errors, and Top-k miss analysis
- Week 4 baseline diagnosis, robustness-loop interpretation, and MNIST extension roadmap

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
python experiments/run_digits_mlp.py
python experiments/analyze_digits_errors.py
python experiments/train_save_load_digits_mlp.py
python experiments/analyze_digits_shift_diagnostics.py
python experiments/compare_digits_augmented_training.py
python experiments/evaluate_canvas_samples.py
```

Run the local digit drawing app:

```powershell
python experiments/compare_digits_augmented_training.py
python apps/digit_draw_app.py
python experiments/evaluate_canvas_samples.py
```

The app prefers `results/checkpoints/digits_mlp_augmented.npz`, displays the `8 x 8` model input after preprocessing, and can save labeled canvas samples under `data/user_digits/samples/`.

Experiment logs are saved under `results/logs/`, and loss curves are saved under `results/figures/`. Generated logs and figures are ignored by Git.

Real canvas diagnostic figures from `experiments/evaluate_canvas_samples.py` are saved under `results/canvas_debug/`, which is also ignored by Git.

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

- Final Week 4 review with real canvas diagnostics included
- Then: Week 5 evaluation, calibration, and technical debt cleanup
- Next research steps: per-class failure analysis, more real canvas samples, a held-out canvas split after enough data exists, and MNIST/CNN reasoning after Week 5 hardening
