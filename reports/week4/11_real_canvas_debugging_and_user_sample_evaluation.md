# Real Canvas Debugging and User-Sample Evaluation

Related: [Real Canvas Validation Findings](12_real_canvas_validation_findings.md)

Task 6K analyzes the first 56 labeled real canvas samples and shows that real canvas robustness remains weak despite synthetic-shift improvement.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why synthetic robustness was not enough

Task 6H improved configured synthetic shift probes. Real canvas input can still fail. Therefore, synthetic robustness and real app robustness must be separated.

The controlled probes test transformations chosen by the project. User drawings come from a different process: a mouse or trackpad, a Tkinter canvas, preprocessing choices, and individual handwriting. A model can improve under configured probe distributions while still failing on the true app-input distribution.

## 2. The true application pipeline

```math
g(x)
=
f_{\theta}(T(x))
```

Here `x` is raw canvas input, `T` is preprocessing, and `f_theta` is the checkpoint-loaded MLP. Failure can occur in preprocessing, in the model, or in their interaction.

The user sees a high-resolution drawing. The model receives only the flattened `8 x 8` output of `T(x)`.

## 3. Condition A versus Condition B

Condition A:

- the `8 x 8` model input does not resemble the intended digit
- likely preprocessing failure

Condition B:

- the `8 x 8` model input resembles the intended digit
- prediction is wrong
- likely model/data coverage, representation bottleneck, or calibration failure

The user currently observes many Condition-B failures. This means the next research focus should be model/data coverage, real canvas sample evaluation, and eventually MNIST/CNN reasoning, not only preprocessing.

## 4. Why the 8x8 model input must be displayed

The model does not see the Tkinter canvas. It sees a small `8 x 8` image after grayscale normalization, polarity correction, cropping, resizing, clipping, and flattening.

Debugging must inspect the model input. A wrong prediction with a distorted `8 x 8` input is mainly a preprocessing problem. A wrong prediction with a clear `8 x 8` input is more likely a model/data coverage problem.

## 5. Why saving real canvas samples matters

Anecdotal testing is not a dataset. Saved labeled samples make real failures measurable. Labels allow accuracy, confidence, top-k, and error analysis on actual user-drawn inputs.

Small sample size must be reported honestly. A few examples are useful for debugging and hypothesis generation, not for broad robustness claims.

## 6. Expected failure modes

Expected failure modes include overly thick strokes, off-center digits, crops that are too tight, resize distortion, unusual personal handwriting, blank or near-blank input, correct-looking `8 x 8` input but wrong prediction, and high-confidence wrong prediction.

The app debug panel is intended to separate these causes. It shows whether the final model input resembles the intended digit before deciding whether to focus on preprocessing or model/data coverage.

## 7. Relationship to Task 6H

Task 6H showed that configured synthetic robustness improves under app-like augmentation. Task 6J tests whether that improvement transfers to real user-drawn inputs.

Condition-B failures suggest the synthetic augmentation did not cover the real handwriting manifold sufficiently. That does not invalidate Task 6H; it narrows the scope of what Task 6H demonstrated.

## 8. Interpretation boundaries

A few saved user samples are not a benchmark. The goal is debugging and hypothesis generation. The project should not tune repeatedly against a tiny personal sample set and then claim robustness.

Robust claims require larger labeled real-canvas data and held-out evaluation. The sample store is a step toward that evidence, not the evidence itself.

## 9. Concrete next research steps

- Collect at least 30-50 labeled canvas samples.
- Evaluate the augmented checkpoint on saved samples.
- Inspect the `8 x 8` input for every error.
- Categorize errors into Condition A and Condition B.
- Run preprocessing sweeps only for Condition A.
- Run canvas-derived augmentation only after separating a validation set.
- Compare with an MNIST-scale MLP.
- Eventually introduce CNNs for spatial inductive bias.

## 10. Conceptual summary

Synthetic robustness and real canvas robustness are different. Preprocessing is part of the model system. Condition-B failures reveal model/data coverage limitations. Saved user samples make app failures measurable.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
