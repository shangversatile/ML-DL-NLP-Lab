[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)

# Confidence Thresholding and Abstention Policy

## 1. Why abstention follows calibration

Calibration diagnostics measure whether confidence is reliable. Abstention asks whether the system can avoid answering when confidence is low. If confidence is poorly aligned with correctness, then an abstention rule based only on confidence must be interpreted cautiously.

## 2. Selective prediction

A selective classifier answers only when confidence exceeds a threshold; otherwise it abstains. In this project, confidence is the largest predicted class probability.

For threshold `t`, the classifier answers when `confidence >= t` and abstains when `confidence < t`.

## 3. Metrics implemented

Task 7C implements:

- coverage: fraction of samples answered
- abstention rate: fraction of samples abstained
- selective accuracy: Top-1 accuracy among answered samples
- selective error rate: Top-1 error rate among answered samples
- total errors: Top-1 errors before abstention
- answered errors: Top-1 errors that remain answered
- abstained errors: Top-1 errors moved into the abstained set
- error abstention rate: fraction of original errors that are abstained
- mean answered confidence: average confidence among answered samples
- mean abstained confidence: average confidence among abstained samples
- Top-k fallback hit rate: fraction of abstained samples whose true label appears in the Top-k candidates

Empty analytical groups preserve `NaN` instead of being coerced to zero.

## 4. Why high-confidence errors matter

If an error has very high confidence, confidence thresholding will not catch it unless the threshold is nearly `1.0`, which may destroy coverage. This is expected to matter for real canvas samples because Week 4 found high-confidence errors on `Canvas-Diagnostic-v1`.

## 5. Top-k fallback

When the model abstains from Top-1, showing Top-k candidates may still be useful if the true label is often in Top-k. This is diagnostic, not a complete user-interface solution. It does not decide how a user should confirm, correct, or reject a prediction.

## 6. Split discipline

Thresholds should be selected on validation data, not on diagnostic or test data. `Canvas-Diagnostic-v1` may be analyzed, but thresholds selected from it should not be treated as final policy choices.

## 7. Expected interpretation

Likely outcomes:

- clean data may keep high coverage and high selective accuracy
- synthetic shifts may benefit from thresholding
- real canvas may still contain high-confidence errors that thresholding cannot catch
- a high Top-3 hit rate can support future top-k or human-confirmation workflows

## 8. What this task does not solve

This task intentionally does not:

- retrain the model
- correct calibration
- remove high-confidence errors
- prove robustness
- implement deployment policy

## 9. Next step

The next task should either implement temperature scaling on a proper validation split or create a technical-debt and artifact audit before any calibration correction.

[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)
