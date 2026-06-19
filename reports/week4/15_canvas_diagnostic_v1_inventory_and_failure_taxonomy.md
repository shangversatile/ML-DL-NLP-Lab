# Canvas-Diagnostic-v1 Inventory and Failure Taxonomy

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Dataset role

`Canvas-Diagnostic-v1` contains the first 56 labeled real canvas samples.

Purpose:

- failure discovery
- per-class diagnosis
- high-confidence error inspection
- Top-3 miss analysis
- future experiment planning

Not allowed:

- training
- model selection
- preprocessing tuning with reported final claims
- final benchmark reporting

## 2. Overall inventory

| Metric                                   |    Value |
| ---------------------------------------- | -------: |
| Samples                                  |       56 |
| Errors                                   |       23 |
| Accuracy                                 | 0.589286 |
| Top-3 accuracy                           | 0.839286 |
| Cross entropy                            | 3.766522 |
| Mean confidence                          | 0.886175 |
| Mean confidence on correct predictions   | 0.935936 |
| Mean confidence on incorrect predictions | 0.814779 |
| Overconfidence gap                       | 0.296889 |
| Top-3 misses among errors                |   9 / 23 |

The Top-3 miss rate is computed among errors, not among all samples. The observed rate is `0.391304`, computed as 9 Top-3 misses out of 23 total errors.

## 3. Per-class taxonomy

| Digit | Status                | Main observed failure                     |
| ----: | --------------------- | ----------------------------------------- |
|     0 | stable                | no observed errors                        |
|     1 | moderate              | confused with 8                           |
|     2 | mostly stable         | one high-confidence 2→0 error             |
|     3 | moderate              | confused with 9                           |
|     4 | mostly stable         | one 4→7 error                             |
|     5 | mostly stable         | one 5→3 error                             |
|     6 | severe                | often predicted as 5 or 4                 |
|     7 | moderate              | confused with 2 or 3                      |
|     8 | severe                | often predicted as 0, 3, or 9             |
|     9 | severe boundary issue | often predicted as 4 with high confidence |

## 4. Failure categories

### Category 1: Stable class

The class is currently stable in `Canvas-Diagnostic-v1`.

Example:

- digit 0

### Category 2: Ranking uncertainty

The correct label often appears in Top-3 even when Top-1 is wrong.

Examples:

- 1→8
- 3→9
- 9→4

### Category 3: Representation/data-coverage failure

The correct label is often missing from Top-3.

Examples:

- many 8 errors
- some 6→5 errors

### Category 4: High-confidence boundary error

The model makes a wrong prediction with confidence above `0.90`.

Examples:

- 9→4 with confidence near 1.0
- 6→5 with confidence near 1.0
- 8→3 with high confidence

## 5. Condition A versus Condition B

Condition A: the `8 x 8` model input does not resemble the intended digit. This is likely a preprocessing issue.

Condition B: the `8 x 8` model input resembles the intended digit, but the prediction is wrong. This is likely a model/data coverage, representation, or calibration issue.

The user reports that many failures are Condition B. Therefore, the immediate research issue is not only preprocessing; it is also model/data coverage and representation.

## 6. Training policy

`Canvas-Diagnostic-v1` should not be used for training.

Reason: the samples have already influenced analysis and decision-making. Training on them and evaluating on them would cause leakage.

Allowed:

- inspect errors
- build taxonomy
- design future collection protocol

Not allowed:

- use them as `Canvas-Train-v1`
- tune model based on them and report improvement on them
- use them as final benchmark

## 7. Future use

Future training should use separately collected `Canvas-Train-v1`. Future model selection should use `Canvas-Val-v1`. Final real-canvas claims require `Canvas-Test-v1`.

## 8. Conceptual summary

`Canvas-Diagnostic-v1` transforms anecdotal app failure into structured evidence. Its value is not to train the current model, but to reveal which kinds of real-input failures the current MLP pipeline cannot handle.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
