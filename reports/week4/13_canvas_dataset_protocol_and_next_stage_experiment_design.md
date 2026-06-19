# Canvas Dataset Protocol and Next-Stage Experiment Design

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why a dataset protocol is needed

Task 6K revealed that real canvas inputs still fail even after the augmented checkpoint improved configured synthetic-shift probes. The current 56 samples are valuable diagnostic evidence because they show concrete real-input failure modes, especially Condition-B failures where the `8 x 8` model input is recognizable but the prediction is wrong.

These samples should not be used directly for training and then evaluated again as evidence of improvement. That would create data leakage: the model or preprocessing choices would have seen the evaluation examples, so the measured result would no longer estimate performance on unseen canvas inputs.

A proper research protocol must separate diagnosis, training, validation, and final testing. The goal is not to make the current 56 samples look good. The goal is to measure whether future changes generalize to unseen real canvas inputs.

## 2. Current evidence: Canvas-Diagnostic-v1

`Canvas-Diagnostic-v1` is the first user-specific real canvas diagnostic set.

- Size: 56 labeled real canvas samples.
- Collection method: saved through the Task 6J app.
- Evaluation: scored in Task 6K using the augmented checkpoint.
- Purpose: failure diagnosis only.
- Restriction: should not be used for model training or model selection.

| Metric                    |    Value |
| ------------------------- | -------: |
| Samples                   |       56 |
| Errors                    |       23 |
| Accuracy                  | 0.589286 |
| Top-3 accuracy            | 0.839286 |
| Cross entropy             | 3.766522 |
| Mean confidence           | 0.886175 |
| Overconfidence gap        | 0.296889 |
| Top-3 misses among errors |   9 / 23 |

The `top_k_miss_error_rate = 0.391304` is computed among errors, not among all samples. It means 9 of the 23 wrong Top-1 predictions also missed the true label in Top-3.

See [Canvas-Diagnostic-v1 Inventory and Failure Taxonomy](15_canvas_diagnostic_v1_inventory_and_failure_taxonomy.md) for the diagnostic inventory, per-class taxonomy, and training policy.

## 3. Dataset split policy

Future real canvas work should use explicit dataset roles.

| Dataset              | Purpose                                             | Can train on it? | Can tune on it? | Can report final result on it? |
| -------------------- | --------------------------------------------------- | ---------------- | --------------- | ------------------------------ |
| Canvas-Diagnostic-v1 | Failure discovery                                   | No               | No              | No                             |
| Canvas-Train-v1      | Real-input augmentation / fine-tuning               | Yes              | No              | No                             |
| Canvas-Val-v1        | Model selection / preprocessing sweep / calibration | No               | Yes             | No                             |
| Canvas-Test-v1       | Final held-out real canvas evaluation               | No               | No              | Yes                            |

`Canvas-Diagnostic-v1` exists to understand failure structure. It can motivate hypotheses, but it should not decide final model settings.

`Canvas-Train-v1` is the only real canvas split that future training or augmentation may use. It can support fine-tuning or app-specific augmentation, but it cannot provide an unbiased estimate of performance.

`Canvas-Val-v1` is for decisions: preprocessing sweeps, model selection, confidence thresholds, abstention policies, and calibration choices. It should not be used for gradient updates.

`Canvas-Test-v1` is the final held-out evidence. It should be opened only after the experiment design and selected settings are fixed.

## 4. Recommended sample collection protocol

For a first prototype, collect at least 30-50 samples per split. Ideally, each split should be balanced across digits 0-9. At minimum, collect 3-5 samples per digit per split.

Stronger evidence requires multiple collection sessions and varied writing styles. The collection process should include routine examples, unusual styles, and ambiguous inputs. It should not only save failures, because a failure-only set cannot estimate real operating accuracy.

Save correct and incorrect examples. Always save true labels. Samples should remain local under `data/user_digits/` and should not be committed to Git.

## 5. Experimental paths after the protocol

### Path A: Preprocessing sweep

Use only `Canvas-Val-v1` for model-selection decisions. Possible variables include crop padding, foreground threshold, blank threshold, resize rule, centering, and stroke normalization.

Use `Canvas-Test-v1` only once after choosing settings.

### Path B: Canvas-derived augmentation

Use `Canvas-Train-v1` to generate app-specific augmentation. Never train on `Canvas-Diagnostic-v1`. Never tune on `Canvas-Test-v1`.

Compare against the current augmented checkpoint. Report clean, synthetic-shift, and real-canvas metrics so that app-input gains do not hide clean-data or configured-shift regressions.

### Path C: Calibration and abstention

Use `Canvas-Val-v1` for reliability diagrams, confidence thresholds, abstention rules, and Top-k assisted decision policies.

Do not use `Canvas-Test-v1` until final evaluation.

### Path D: MNIST scale-up

MNIST gives higher-resolution `28 x 28` images. It tests scale and representation, but it does not replace real canvas validation. MNIST should inherit the same evaluation discipline: explicit splits, held-out test evaluation, calibration diagnostics, and real-input validation.

### Path E: CNN inductive-bias extension

Condition-B failures suggest that the current MLP may have representation limits. CNNs introduce local spatial bias and parameter sharing, which are natural hypotheses for image-like inputs.

A CNN should be introduced as a hypothesis about image structure, not as a black-box upgrade.

## 6. Evaluation metrics for future experiments

Future experiments should report:

- clean `load_digits` accuracy
- synthetic shift accuracy
- real canvas Top-1 accuracy
- real canvas Top-3 accuracy
- cross entropy
- mean confidence
- high-confidence error count
- top-k miss count
- per-class accuracy
- confusion matrix
- calibration diagnostics

A future model is not better just because one metric improves. It should be evaluated on clean data, synthetic shifts, and real canvas samples.

## 7. Decision criteria for moving forward

A future model or preprocessing change is promising if:

- `Canvas-Val-v1` Top-1 accuracy improves meaningfully
- `Canvas-Val-v1` Top-3 accuracy remains high or improves
- high-confidence errors decrease
- digit 6 and digit 8 improve
- clean `load_digits` accuracy does not collapse
- synthetic-shift robustness does not collapse

A change is not acceptable if:

- it improves the diagnostic set only
- it hurts clean performance severely
- it increases high-confidence errors
- it was selected using `Canvas-Test-v1`

## 8. Research interpretation

Task 6K changes the project from "Can the model classify clean digits?" to "Can the full application system generalize from curated digit data to real user input?" This is a harder and more meaningful question.

Real input validation is necessary. Synthetic probes are useful but incomplete. A strong protocol prevents self-deception. The next stage should test hypotheses, not chase demo accuracy.

## 9. Conceptual summary

The current 56 samples are diagnostic evidence. Future training requires separate data. Evaluation must distinguish train, validation, and test roles.

MNIST and CNNs are natural next steps, but they must not replace real canvas validation.

Task 6M selects MNIST Option A for Week 4 closure. Week 4 will not implement MNIST. MNIST and CNN work is deferred to the later deep-learning stage, and `Canvas-Diagnostic-v1` remains diagnostic only.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
