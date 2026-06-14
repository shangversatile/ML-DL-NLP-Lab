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

## Current status

Planned

## Links

- [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](week4_multiclass_digits_capstone.md)
- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
