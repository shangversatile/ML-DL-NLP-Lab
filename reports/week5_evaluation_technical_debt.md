# Week 5: Evaluation, Technical Debt, and Trustworthy ML Diagnostics

## Scope

Week 5 comes after the Week 4 handwritten-digit recognition capstone. It focuses on evaluating trained models beyond aggregate accuracy, diagnosing errors and confidence failures, analyzing calibration, identifying distribution shift between training data and application inputs, cleaning technical debt accumulated during scratch implementation, and improving experiment reliability and reproducibility.

## Why this comes after Week 4

Evaluation becomes more meaningful after a real-data task exists. XOR-style synthetic data are useful for controlled learning and debugging, but they are too simple for realistic evaluation. They do not expose many class-specific failures, ambiguous samples, confidence errors, or input-distribution mismatches.

Handwritten-digit recognition introduces real multiclass errors, ambiguous samples, and user-input distribution shift. Therefore, evaluation and technical debt cleanup are better placed after the capstone baseline, when there is a trained multiclass system to inspect and improve.

## Planned modules

| Module                                  | Purpose                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| Confusion matrix and per-class accuracy | Identify class-specific failures                           |
| Error sample inspection                 | Understand concrete failure modes                          |
| Confidence analysis                     | Detect high-confidence incorrect predictions               |
| Calibration and reliability diagrams    | Compare predicted confidence with empirical correctness    |
| Distribution-shift probes               | Compare dataset images with local handwritten inputs       |
| Technical debt cleanup                  | Simplify APIs, remove duplication, improve maintainability |
| Experiment registry                     | Track configs, metrics, figures, and model checkpoints     |

## Relationship to Week 4

Week 4 builds the real-data multiclass training and inference system. Week 5 evaluates how reliable that system is.

Week 5 should not replace Week 4; it depends on Week 4.

## Starting point inherited from Week 4

Week 5 starts from a system with:

- strong clean benchmark performance
- improved configured synthetic robustness
- weak real canvas Top-1 performance
- high-confidence real canvas errors
- a dataset protocol preventing leakage

Therefore Week 5 should focus on calibration, reliability, evaluation discipline, artifact cleanup, experiment registry design, and final tag readiness.

## Dependency on Week 4 robustness loop

Week 5 should begin after Week 4 has completed the augmentation-based robustness improvement loop. Calibration and reliability diagrams should not be used to hide a weak base model.

Task 6H showed strong improvement on the configured synthetic shift probes, especially the thick-stroke condition. Week 5 should treat that as useful evidence, not as a proof of real-world robustness.

If later held-out transformations or real canvas samples show that augmentation is insufficient, Week 5 should explicitly record that limitation and use it to motivate MNIST-scale and convolutional extensions.

## Input-robustness evidence inherited from Week 4

Week 4 now includes both failure diagnosis and augmentation-based improvement. The original clean-data baseline failed badly under thickened inputs while remaining highly confident. The augmented model strongly improved configured shift robustness under the same scratch MLP architecture and a fixed update budget.

Week 5 should not repeat the same shift probes blindly. It should harden the evaluation system through calibration, experiment registry design, multiple seeds, held-out transforms, real canvas validation where possible, and artifact management.

## Real canvas dataset protocol inherited from Week 4

Week 5 should preserve `Canvas-Diagnostic-v1` as diagnostic-only. The first 56 labeled real canvas samples exposed failure modes, but they should not be used for training, model selection, calibration threshold selection, or final reported real-canvas performance.

Future `Canvas-Train-v1`, `Canvas-Val-v1`, and `Canvas-Test-v1` splits should be explicit. Training or augmentation may use only the train split. Preprocessing sweeps, calibration, abstention, and top-k decision policies should be selected on validation data. Final claims should use a held-out canvas test split.

Calibration and abstention should be evaluated on validation and test splits, not on training data. Otherwise, reliability improvements can become another form of leakage.

## Relationship to MNIST extension

Week 5 hardens the current digits baseline before scaling it. Calibration, experiment registry design, artifact management, and technical debt cleanup should happen before MNIST-scale expansion.

MNIST should inherit the same discipline established in Week 4: explicit split design, checkpointing, error analysis, calibration diagnostics, and shift diagnostics. It should be a scale-up of the evaluation system, not only a larger training run.

## Current status

Planned

## Links

- [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](week4_multiclass_digits_capstone.md)
- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
