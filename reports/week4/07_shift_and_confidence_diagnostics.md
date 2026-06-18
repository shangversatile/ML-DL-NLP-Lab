# Shift and Confidence Diagnostics

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Why distribution shift matters after the app

The offline test set comes from `load_digits`. User canvas inputs come from a different process: a person draws on a local canvas, then preprocessing converts that drawing into an `8 x 8` digit-like input.

The model now sees `T(x)`, not only original dataset images. App success requires robustness to preprocessing choices and input variation.

## 2. Risk under different input distributions

Clean test risk:

```math
R_{\mathrm{test}}(\theta)
=
E_{(X,Y)\sim P_{\mathrm{test}}}
[
\ell(f_\theta(X),Y)
]
```

App-like risk:

```math
R_{\mathrm{app}}(\theta)
=
E_{(X,Y)\sim P_{\mathrm{app}}}
[
\ell(f_\theta(T(X)),Y)
]
```

Controlled probes approximate possible changes between these distributions. They are diagnostics, not complete guarantees.

## 3. Shift probes

Translation shifts stress centering assumptions. Intensity scaling stresses contrast and stroke brightness. Pixel noise stresses local image corruption. Thresholding stresses binary canvas-like conversion. Thickening stresses broad strokes. Thinning stresses faint or narrow strokes.

Each probe is deliberately simple so the failure mode is easy to inspect.

## 4. Shift sensitivity metrics

```math
\Delta Acc
=
Acc_{\mathrm{shift}}
-
Acc_{\mathrm{clean}}
```

```math
\Delta CE
=
CE_{\mathrm{shift}}
-
CE_{\mathrm{clean}}
```

Accuracy degradation shows prediction failure. Cross-entropy increase shows probability-quality degradation. Confidence remaining high while accuracy falls is especially concerning.

## 5. Confidence bins

Confidence:

```math
c_i
=
\max_k P_{i,k}
```

Bin:

```math
B_m
=
\{
i:
c_i
\in
(a_m,b_m]
\}
```

Bin accuracy:

```math
acc(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m}
1[
\hat{y}_i
=
y_i
]
```

Bin mean confidence:

```math
conf(B_m)
=
\frac{1}{|B_m|}
\sum_{i\in B_m}
c_i
```

If confidence is much higher than accuracy, the model is overconfident. This motivates calibration analysis later.

## 6. ECE-style diagnostic

```math
ECE
=
\sum_m
\frac{|B_m|}{n}
|
acc(B_m)
-
conf(B_m)
|
```

This is a diagnostic summary. This task measures calibration error but does not correct it. Temperature scaling and calibration curves belong to Week 5.

## 7. Interpretation boundaries

Synthetic probes are not the same as real canvas data. Passing these probes does not prove robustness. Failing these probes identifies useful weaknesses.

Diagnostics should guide future validation, not be overfit directly.

## 8. Relationship to Week 5

Week 4 identifies shift and confidence issues. Week 5 will clean technical debt and extend this into calibration, reliability diagrams, abstention ideas, and structured evaluation.

## 9. Conceptual summary

Real applications change the input distribution. Confidence can fail under shift. Shift probes make hidden assumptions visible. Calibration is a measurement problem before it is a correction problem.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
