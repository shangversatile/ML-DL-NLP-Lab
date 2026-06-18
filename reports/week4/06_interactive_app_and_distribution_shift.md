# Interactive App and Distribution Shift

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why an interactive app changes the problem

Offline test data come from `load_digits`. The app input comes from a human drawing on a local canvas. That changes the input distribution.

The model is no longer evaluated only on fixed dataset arrays. Inference now includes preprocessing from raw canvas pixels into the 64-feature convention expected by the trained scratch MLP.

## 2. The full application function

```math
g(x)
=
f_\theta(T(x))
```

Here `x` is raw canvas input, `T` is preprocessing, and `f_theta` is the trained MLP loaded from a checkpoint. Errors can come from either the model or preprocessing.

## 3. Canvas-to-feature preprocessing

The local app uses a grayscale canvas buffer. Preprocessing normalizes intensities, corrects polarity so digit strokes are bright on a dark background, crops around foreground pixels, resizes to `8 x 8`, and flattens to 64 features.

The app then passes those features into the checkpoint-loaded scratch MLP through the reusable inference helper.

## 4. Why preprocessing is part of the model system

Training data use a specific pixel convention. App data must be transformed into a compatible convention before prediction.

Hidden preprocessing assumptions can cause silent failure. Checkpoint metadata records input scaling, but GUI preprocessing must still be validated because user drawings are not automatically equivalent to `load_digits` samples.

## 5. Top-k display

Top-1 prediction hides ambiguity. Top-3 candidates expose competing hypotheses and their probabilities.

This is useful for uncertain handwriting. If a drawing looks like both a `3` and an `8`, the ranked candidates are more informative than a single label.

## 6. Expected failure modes

Expected failure modes include digits that are too small, off-center digits, thick strokes, thin strokes, unusual writing style, background polarity mismatch, cropping that removes important context, resizing that loses information, and blank canvases that still produce a prediction.

These are system-level failures, not only model failures.

## 7. Why this leads to distribution-shift diagnostics

Offline accuracy does not guarantee interactive robustness. User drawings may differ systematically from `load_digits` in stroke width, centering, scale, contrast, and shape.

Later analysis should compare activation, confidence, and error behavior between dataset samples and canvas samples.

## 8. Interpretation boundaries

This app is a local educational demo. It is not a production system. It lacks calibration, rejection or abstention, and robust preprocessing.

It should be used to surface failure modes, not hide them.

## 9. Conceptual summary

Application reliability depends on both preprocessing and model behavior. Top-k output is a minimal uncertainty interface. Interactive inputs expose distribution shift. This motivates Week 5 trustworthy diagnostics.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
