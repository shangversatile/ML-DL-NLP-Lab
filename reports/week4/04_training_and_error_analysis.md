# Training and Error Analysis

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why aggregate accuracy is not enough

Accuracy compresses the entire error structure into one scalar. That makes it useful as a first check, but it hides which classes fail, which predictions are overconfident, and which samples dominate cross entropy.

For the handwritten-digits baseline, a high test accuracy does not tell us whether the model confuses visually similar digits, whether one class is over-predicted, or whether the largest losses come from ambiguous images. Task 6D adds diagnostics that make those questions visible.

## 2. Confusion matrix

```math
C_{a,b}
=
\#\{
i:
y_i=a,
\hat{y}_i=b
\}
```

Rows are true labels. Columns are predicted labels. Diagonal cells are correct predictions, and off-diagonal cells reveal concrete confusions.

A confusion matrix keeps the class structure intact. Instead of saying only that the model is wrong on some fraction of examples, it shows which true digits are mapped to which predicted digits.

## 3. Per-class recall

```math
recall_k
=
\frac{
C_{k,k}
}{
\sum_j C_{k,j}
}
```

Recall answers: among true class-k examples, how many did the model recover? Low recall for a class means the model often fails to recognize that digit when it appears.

## 4. Per-class precision

```math
precision_k
=
\frac{
C_{k,k}
}{
\sum_i C_{i,k}
}
```

Precision answers: among examples predicted as class k, how many were truly class k? Low precision for a class means the model over-predicts that digit or attracts examples from other classes.

## 5. Macro versus aggregate metrics

Aggregate accuracy weights classes by their frequency. Macro averaging gives each class equal weight by averaging class-level values.

The digits dataset is roughly balanced, so aggregate and macro metrics should often agree. Macro metrics are still useful diagnostic habits because they make class-level failures harder to hide.

## 6. Top-k accuracy

```math
TopKAcc
=
\frac{1}{n}
\sum_{i=1}^{n}
1[
y_i
\in
TopK(P_i)
]
```

Top-1 accuracy measures the final predicted class. Top-k accuracy measures whether the true class remains highly ranked by the model.

This is useful for applications that can display multiple candidates or ask for confirmation. A model with weak top-1 but strong top-3 performance may still contain useful ranking information.

## 7. Confidence and high-confidence errors

```math
c_i
=
\max_k
P_{i,k}
```

```math
\hat{y}_i
=
\arg\max_k
P_{i,k}
```

High-confidence error condition:

```math
\hat{y}_i
\ne
y_i
\quad
and
\quad
c_i
\ge
\tau
```

High confidence is not the same as correctness. High-confidence errors are more dangerous than low-confidence errors because they are harder for a user or downstream system to distrust.

This motivates calibration analysis later. Task 6D only identifies high-confidence errors; it does not prove whether the probability estimates are calibrated.

## 8. Per-sample negative log-likelihood

```math
\ell_i
=
-\log
P_{i,y_i}
```

Cross entropy is the mean of per-sample negative log-likelihood. High-loss examples are samples where the model assigns low probability to the true class.

These samples often reveal ambiguous data, mislabeled data, unusual handwriting, or model blind spots. They are especially useful because they connect the training objective to concrete examples.

## 9. Why visualize error examples

Numbers show where to look. Images explain what the model is actually seeing.

Visual error inspection connects probability outputs to data geometry. It helps distinguish between an obvious model failure, an ambiguous digit, and a sample that may be hard even for a human to classify from an `8 x 8` image.

## 10. Interpretation boundaries

The confusion matrix is descriptive, not causal proof. Per-class metrics depend on test-set composition. High-confidence errors do not automatically imply poor calibration, but they motivate calibration checks.

Error analysis should not be used to repeatedly tune against the test set. Later development should use validation diagnostics for iteration and reserve test results for final reporting.

## 11. Relationship to Week 5 trustworthy diagnostics

Task 6D gives the first structured error analysis for the digits capstone. Week 5 will extend this into calibration, reliability diagrams, distribution-shift probes, and technical debt cleanup.

The current diagnostics provide the bridge: they show what the model gets wrong, how confident it is, and which concrete samples deserve inspection.

## 12. Conceptual summary

Aggregate accuracy is only the first layer. The confusion matrix reveals error structure. Per-class recall and precision separate two kinds of class failure. Top-k accuracy analyzes ranking. Confidence analysis detects risky errors. Per-sample negative log-likelihood connects cross entropy to concrete examples.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
