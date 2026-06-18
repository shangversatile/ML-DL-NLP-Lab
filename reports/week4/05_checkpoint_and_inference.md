# Checkpoint and Inference

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why checkpointing matters

Training creates a temporary Python object in memory. Once the process exits, that object disappears unless its learned state is saved.

Applications need reusable model state. Checkpointing separates training from inference by turning a trained model into an artifact that can be loaded later without retraining.

## 2. What a scratch-model checkpoint must contain

A scratch MLP checkpoint must contain the learned parameter arrays: `W1`, `b1`, `W2`, and `b2`. These arrays define the trained function only when paired with the correct architecture.

The checkpoint also needs architecture metadata: `n_features`, `hidden_dim`, and `num_classes`. It should include class names, input scaling assumptions, version information, and optional training metadata such as the seed, optimizer settings, and final metrics.

## 3. Model function preservation

Define the trained model:

```math
f_\theta(x)
```

Define the loaded model:

```math
\tilde{f}_\theta(x)
```

Correct checkpoint round trip should satisfy:

```math
f_\theta(x)
=
\tilde{f}_\theta(x)
```

for the same input batch up to floating-point tolerance.

This is stronger than checking that a file exists. The saved artifact must preserve probabilities and predictions after loading.

## 4. Why metadata is part of correctness

Parameters alone are not enough. Array shapes depend on architecture. Input interpretation depends on preprocessing. Class index interpretation depends on class names.

Metadata prevents silent misuse. A checkpoint that says its inputs are scaled digits and its classes are `"0"` through `"9"` carries semantics that raw arrays do not.

## 5. Reusable inference pipeline

Inference should normalize input shape before calling the model. The same helper should support a single `8 x 8` image, a single flattened vector with 64 features, a batch of images, and a batch of flattened vectors.

The reusable inference output should include probabilities, predicted labels, confidences, and top-k candidates. Future GUI and application code should call the same inference utility instead of duplicating prediction logic.

## 6. Top-k output as uncertainty interface

Top-1 prediction hides alternatives. Top-k output shows competing hypotheses and their probabilities.

This matters for handwritten input because ambiguity is common. A digit that looks between a `3` and an `8` may be more honestly represented by a ranked list than by only one label.

## 7. Interpretation boundaries

Checkpointing does not prove model reliability. It does not solve calibration. It does not handle local handwritten-input distribution shift. GUI input preprocessing will require additional validation before user-drawn digits can be treated like the scikit-learn digits data.

Task 6E only makes the trained scratch model reusable and verifies loaded-model equivalence.

## 8. Conceptual summary

Checkpointing converts a trained model into a reusable artifact. Metadata is part of model semantics. Inference is a pipeline, not merely a function call. Loaded-model equivalence must be tested numerically.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
