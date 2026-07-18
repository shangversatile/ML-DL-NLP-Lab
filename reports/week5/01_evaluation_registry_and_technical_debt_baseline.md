[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)

# Evaluation Registry and Technical Debt Baseline

## 1. Why Week 5 starts with evaluation infrastructure

Week 4 showed that clean benchmark performance, synthetic-shift robustness, and real canvas robustness can disagree. The scratch multiclass MLP reached strong `load_digits` performance, but the same model behaved very differently under configured synthetic shifts and real canvas inputs. Week 5 should therefore make evaluation records explicit and auditable before adding calibration or abstention logic.

## 2. What an experiment registry solves

An experiment registry helps prevent:

- scattered metrics
- unclear dataset splits
- accidental comparison across different checkpoints
- hidden data leakage
- missing notes about diagnostic versus train/validation/test roles

## 3. Registry schema

The registry stores one JSON object per line with these fields:

- `name`: experiment or evaluation name
- `split`: dataset role used for the record
- `metrics`: validated scalar metric dictionary
- `model`: optional model identifier
- `dataset`: optional dataset identifier
- `checkpoint`: optional checkpoint path or identifier
- `notes`: optional context about the run
- `tags`: optional labels for filtering related records
- `created_at_unix`: Unix timestamp when the record was created

The schema is intentionally lightweight and JSONL-based so it can be used without a database. This keeps early Week 5 evaluation discipline simple while still making later calibration, reliability, and abstention comparisons easier to audit.

## 4. Inherited Week 4 evaluation lessons

Week 5 inherits several evaluation lessons from the Week 4 capstone:

- clean `load_digits` accuracy was high
- synthetic thickening collapsed the baseline model
- augmentation improved configured synthetic robustness
- real canvas validation remained weak
- Canvas-Diagnostic-v1 must not be used as training data

## 5. Technical debt baseline

Current categories of technical debt to audit in Week 5:

- calibration metrics are currently diagnostic rather than corrective
- figures and local artifacts are generated outside version control
- checkpoints are local artifacts and should remain ignored
- real canvas samples are local diagnostic data and should remain ignored
- evaluation outputs need consistent naming
- reports need link consistency
- experiment scripts should avoid hidden assumptions
- future dataset splits must be explicit

## 6. What this task intentionally does not do

This task intentionally avoids:

- no retraining
- no MNIST
- no CNN
- no calibration correction yet
- no changes to model behavior
- no app changes
- no use of Canvas-Diagnostic-v1 for optimization

## 7. Transition to the next Week 5 task

After the registry foundation, the next task should implement or audit calibration metrics and reliability diagrams with clear split discipline. Registry records should make it explicit which split produced each metric, which checkpoint was evaluated, and whether a result is diagnostic or eligible for model-selection decisions.

[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)
