# ML Foundations + Evaluation Toolkit from Scratch

## Motivation

This project is a learning-oriented ML research engineering project focused on building foundational ability for Trustworthy ML, AI Systems, and Evaluation/Monitoring.

The goal is to implement core ML components from scratch over time while keeping the project structure simple, reproducible, and easy to inspect.

## May Roadmap

- Week 1: Project setup, structure, configuration files, and report templates.
- Week 2: Linear regression and logistic regression from scratch.
- Week 3: Optimization methods and a basic MLP.
- Week 4: Evaluation, calibration, error analysis, and technical debt review.

## Current Status

Week 1 setup.

No model logic, optimizers, or metrics have been implemented yet.

## Project Structure

```text
configs/      Minimal YAML configuration files for experiments.
src/          Main Python package for data, models, optimization, evaluation, and utilities.
experiments/  Entry-point scripts for running future experiments.
tests/        Placeholder test files for future validation.
reports/      Weekly learning and engineering notes.
results/      Output folders for logs and generated figures.
```

## Environment Setup

From Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Running Tests

Placeholder command:

```powershell
pytest
```

## Run a Week 1 smoke test

```powershell
python experiments/run_linear_regression.py
```

Specific placeholder test examples:

```powershell
pytest tests/test_metrics.py
pytest tests/test_linear_regression.py
pytest tests/test_logistic_regression.py
```
