# Week 1 Setup Notes

## 1. What I built this week

I initialized the ML-DL-NLP-Lab repository for the May project: ML Foundations + Evaluation Toolkit from Scratch. The repository now contains a modular folder structure for configurations, source code, experiments, tests, reports, and results. I also added a README, requirements.txt, .gitignore, placeholder Python modules, YAML config files, and weekly report templates.

## 2. Why project structure matters in ML systems

We build the project structure before implementing models because ML research code needs to be reproducible, extensible, and testable from the beginning. A clear separation between configs, source code, experiments, tests, reports, and results makes it easier to track what was run, how it was configured, and whether future changes break existing behavior.

This structure also reduces technical debt: instead of mixing model logic, metrics, plots, and experimental scripts together, each component has a stable place and interface. For trustworthy ML, this is especially important because evaluation, error analysis, and calibration should be treated as 
first-class modules rather than afterthoughts.

## 3. Reproducibility decisions

I separated experiment configuration files into the configs/ folder so future training scripts can load hyperparameters from YAML instead of hard-coding them. I also created requirements.txt to make the Python environment easier to reproduce. The results/ folder is reserved for generated logs and figures, while .gitignore prevents large or frequently changing output files from polluting the Git history.

Reproducibility is important because randomness can affect initialization, data shuffling, train/validation splits, mini-batch order, and eventually the observed training curves or model performance. Without controlling random seeds, it becomes difficult to know whether a result changed because of a real code or modeling decision, or simply because of random variation.

Logging is equally important because it records what happened during an experiment instead of relying only on temporary terminal output. Logs make it easier to debug failures, compare runs, and understand the sequence of training or evaluation steps.

For trustworthy ML, reproducibility is a scientific requirement. If an experiment cannot be repeated or inspected by others, then the result becomes hard to verify or falsify. A model that cannot be reproduced cannot be reliably trusted.

### Data splitting and preprocessing

`standardize_features()` must compute mean and standard deviation only from `X_train` because the validation set should simulate unseen data. If we use `X_val` to compute preprocessing statistics, information from the validation distribution leaks into the training pipeline, making evaluation overly optimistic. In real deployment, we cannot compute preprocessing parameters using future unseen samples in advance. Therefore, the correct procedure is to fit preprocessing on training data and apply the same transformation to validation or test data.

### Why configuration files matter

Experiment parameters should be stored in `configs/*.yaml` instead of being hard-coded in experiment scripts or model files because configuration and code serve different purposes. The source code should define reusable logic, while configuration files should describe the specific settings of one experiment, such as seed, learning rate, number of epochs, dataset size, and noise level.

This separation improves reproducibility because each experiment can be traced back to a concrete configuration file. It also makes experiment comparison easier: changing a learning rate or seed does not require modifying source code, and different runs can be represented by different config files. From a technical debt perspective, config-driven experiments prevent training scripts from becoming messy collections of hard-coded values, which will become especially important when the project scales to more models, datasets, metrics, and trustworthy ML analyses.

## 4. Connection to Hidden Technical Debt in ML Systems

TODO: Add notes after reading the paper.

## 5. Open questions

TODO: Add 2-3 questions about reproducibility, configuration, and ML system maintenance.