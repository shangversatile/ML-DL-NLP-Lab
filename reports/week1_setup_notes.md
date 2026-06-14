# Week 1 Setup Notes

## 1. What I built this week

I initialized the ML-DL-NLP-Lab repository for the May project: "ML Foundations + Evaluation Toolkit from Scratch." The repository now has a modular project structure for configs, source code, experiments, tests, reports, and results. I also added a README, `requirements.txt`, `.gitignore`, config files, placeholder modules, tests, and report notes.

I implemented the first concrete utilities for the project:

- `set_seed` in `src/utils/seed.py`
- `get_logger` in `src/utils/logging_utils.py`
- `load_config` in `src/utils/config.py`
- synthetic data generation in `src/data/datasets.py`
- `train_val_split` and `standardize_features` in `src/data/preprocessing.py`
- a Week 1 smoke test in `experiments/run_linear_regression.py`

## 2. Why project structure matters in ML systems

We build the project structure before implementing models because ML research code needs to be reproducible, extensible, and testable from the beginning. A clear separation between configs, source code, experiments, tests, reports, and results makes it easier to track what was run, how it was configured, and whether future changes break existing behavior.

This structure also reduces technical debt. Instead of mixing model logic, metrics, plots, data handling, and experiment scripts together, each component has a stable place and interface. For trustworthy ML, evaluation, error analysis, and calibration should be treated as first-class modules rather than afterthoughts.

## 3. Reproducibility decisions

I separated experiment configuration files into `configs/` so future training scripts can load parameters from YAML instead of hard-coding them. I created `requirements.txt` to make the Python environment easier to reproduce. The `results/` folder is reserved for generated logs, figures, and experiment outputs, while `.gitignore` keeps large or frequently changing artifacts out of Git history.

### Why random seeds matter

Randomness affects initialization, data shuffling, train/validation splitting, synthetic data generation, and eventually training curves or model performance. Without seed control, it is hard to know whether a result changed because of a real modeling decision or because of random variation.

### Why logging matters

Logs preserve experiment information beyond temporary terminal output. They help with debugging, comparing runs, and tracing the sequence of pipeline steps. This is especially useful when experiments become longer and produce multiple outputs.

### Why preprocessing must be fitted only on training data

`standardize_features()` must compute mean and standard deviation only from `X_train` because validation data should simulate unseen data. Using `X_val` statistics would introduce data leakage and make evaluation overly optimistic. The correct procedure is to fit preprocessing on training data and apply the same transformation to validation or test data.

### Why configuration files matter

Experiment parameters should be stored in `configs/*.yaml` instead of being hard-coded in experiment scripts or model files because configuration and code serve different purposes. The source code should define reusable logic, while configuration files should describe the specific settings of one experiment, such as seed, learning rate, number of epochs, dataset size, and noise level.

This separation improves reproducibility, experiment comparison, and the separation of code and configuration. It also supports future scaling to many experiments, where changing a seed, dataset size, or model setting should not require editing reusable source files.

### Why smoke-test experiments matter

A smoke-test experiment verifies the pipeline end to end before model training. In this project, the smoke test connects config loading, seed setting, logging, synthetic data generation, splitting, and preprocessing. It helps detect path, config, shape, and logging issues early, before they become harder to debug inside model training code.

## 4. Connection to Hidden Technical Debt in ML Systems

An initial connection I see to "Hidden Technical Debt in Machine Learning Systems" is that ML systems are not only models. They also include data, configuration, dependencies, evaluation, monitoring, and infrastructure. Even a simple learning project can become hard to maintain if these parts are treated informally.

Technical debt can appear when data processing, model logic, metrics, and experiment scripts are mixed together. That kind of mixing may be fast at the beginning, but it makes later changes harder to test and reason about. The Week 1 structure tries to reduce this debt by separating concerns and creating tests early.

This project is still small, so the structure may feel heavier than a single notebook or script. However, the same design principles matter more as the system grows. Starting with clear boundaries should make it easier to add models, evaluation tools, calibration methods, and error analysis without turning the codebase into a collection of disconnected experiments.

## 5. Connection to later weeks

Week 1 is not merely repository setup. It establishes the experimental infrastructure required by later weeks: reproducible seeds, YAML configuration loading, logging, deterministic synthetic data, preprocessing utilities, unit testing, experiment scripts, results directories, and report writing discipline.

| Later stage                         | Dependency on Week 1                                                      |
| ----------------------------------- | ------------------------------------------------------------------------- |
| Week 2 Linear / Logistic Regression | uses configs, reproducible data generation, preprocessing, logging, tests |
| Week 3 MLP and Optimizers           | relies on the same experiment and testing structure                       |
| Week 4 Digits Capstone              | extends the same reproducible pipeline to real data                       |
| Week 5 Evaluation                   | depends on logs, saved results, figures, and structured diagnostics       |

Related notes:

- [Week 2 Linear / Logistic Regression](week2_linear_logistic_regression.md)
- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
- [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](week4_multiclass_digits_capstone.md)
- [Week 5 Evaluation, Technical Debt, and Trustworthy ML Diagnostics](week5_evaluation_technical_debt.md)

## 6. Open questions

- How much structure is enough before it becomes over-engineering?
- How should experiment logs evolve when there are many runs?
- When should I introduce tools like MLflow, TensorBoard, or Weights & Biases?
- How can I design evaluation modules so that calibration and error analysis are not afterthoughts?
- How should tests be designed for ML code where exact numerical results may vary?
