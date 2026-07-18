[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)

# Calibration Metrics and Reliability Diagrams

## 1. Why calibration matters after Week 4

Week 4 found high-confidence errors on synthetic shifts and real canvas samples. That means accuracy alone is not enough: a model can be correct often on clean benchmark data while still assigning excessive confidence to wrong predictions under distribution shift. Week 5 must distinguish predictive accuracy from reliability.

## 2. What calibration measures

Calibration asks whether confidence values match empirical correctness frequencies. A model that predicts 90% confidence should be correct about 90% of the time on comparable examples. Calibration is therefore split-dependent: a confidence value can be reliable on clean test data and unreliable on shifted or real canvas inputs.

## 3. Metrics implemented

Task 7B implements:

- confidence: the maximum predicted class probability
- accuracy: the fraction of correct top-label predictions
- ECE: weighted average absolute calibration gap across confidence bins
- MCE: maximum absolute calibration gap across non-empty confidence bins
- Brier score: mean squared distance between predicted probabilities and one-hot labels
- NLL: negative log likelihood of the true class probability
- overconfidence gap: mean confidence minus accuracy

These metrics measure reliability. They do not change the model or correct its predictions.

## 4. Reliability diagrams

Reliability diagrams compare average confidence with empirical accuracy in confidence bins. Points near the diagonal indicate that confidence and observed correctness are aligned. Points below the diagonal indicate overconfidence. Points above the diagonal indicate underconfidence.

## 5. Split discipline

Calibration results are only meaningful relative to explicit splits:

- clean test
- synthetic shift probes
- real canvas diagnostic samples
- future Canvas-Val and Canvas-Test splits

Canvas-Diagnostic-v1 remains diagnostic only and should not be used for calibration tuning.

## 6. Experiment registry integration

Each calibration evaluation is recorded as a JSONL experiment record with dataset, split, checkpoint, metrics, notes, and tags. The registry makes it clear whether a result came from clean test data, a configured synthetic shift, or local diagnostic canvas samples.

## 7. What this task does not solve

This task intentionally does not:

- recalibrate the model
- improve accuracy
- implement temperature scaling
- define abstention yet
- validate production readiness

## 8. Next step

The next task should introduce calibration-aware decision policies, such as confidence thresholds, abstention, and top-k fallback, while keeping validation/test separation. Thresholds should be selected on validation data and evaluated on held-out test data.

[← Back to Week 5 Index](../week5_evaluation_technical_debt.md)
