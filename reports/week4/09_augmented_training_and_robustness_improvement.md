[<- Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

# Augmented Training and Robustness Improvement

## 1. Why improvement is required

The clean benchmark result is strong, but the shift diagnostics exposed a severe failure under thickened inputs. This matters because local canvas strokes can be thicker than the original `load_digits` examples. A model that remains highly confident while accuracy collapses cannot be treated as a final recognizer.

| Condition           | Accuracy | Cross entropy | Mean confidence | ECE-style diagnostic |
| ------------------- | -------: | ------------: | --------------: | -------------------: |
| Clean test baseline | 0.977695 |      0.082859 |        0.994004 |             0.019867 |
| Thickened baseline  | 0.182156 |     22.020438 |        0.967042 |             0.784886 |

The failure is not just low accuracy. The baseline is often confidently wrong under the thick-stroke probe.

## 2. Augmentation as distributional coverage

Clean empirical risk measures loss on the original training distribution:

```math
R_{\mathrm{clean}}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell
(
f_\theta(x_i),
y_i
)
```

Augmented empirical risk measures loss after applying a set of transformations:

```math
R_{\mathrm{aug}}(\theta)
=
\frac{1}{n|S|}
\sum_{i=1}^{n}
\sum_{s\in S}
\ell
(
f_\theta(T_s(x_i)),
y_i
)
```

The transformations `T_s` are app-like probes such as small translations, intensity changes, noise, thresholding, thickening, and thinning. Augmentation expands the training support, but it is still a hypothesis about likely deployment variation. It is not a robustness proof.

## 3. Fair comparison: fixed update budget

The augmented dataset is larger than the clean-only dataset. Equal epochs would give the augmented model more parameter updates, so equal epochs would not isolate the effect of augmentation. A fixed update budget gives both models the same number of optimizer steps.

The comparison should also use the same initialization and optimizer configuration. This makes the contrast about the training data distribution rather than a random initialization advantage.

## 4. Evaluation protocol

Both models should be evaluated on:

- clean test data
- translation shifts
- intensity changes
- noise
- thresholding
- thickening
- thinning

The primary metrics are:

- accuracy
- cross entropy
- mean confidence
- ECE-style diagnostic
- top-k accuracy

This keeps the experiment focused on both prediction correctness and probability behavior.

## 5. Tradeoffs

Augmentation may improve shifted performance, but it can also reduce clean accuracy or alter confidence behavior. Improvement must be measured rather than assumed. The important comparison is not only whether thickened accuracy increases, but also whether clean performance degrades and whether confidence remains meaningful under the evaluated shifts.

## 6. Interpretation boundaries

If augmentation improves thickened accuracy, it does not prove real app robustness. The app still produces inputs through a different process than the synthetic probes. If augmentation fails, that suggests limits of the current `8 x 8` MLP and simple transformations.

Repeated tuning on the same fixed shift probes can overfit the probes. MNIST and convolutional reasoning remain natural future extensions, but they should follow the current robustness loop rather than replace it.

## 7. Conceptual summary

The baseline found a real weakness. Augmentation is the first controlled improvement loop. A fair comparison requires equal update budgets, shared initialization, and evaluation beyond clean accuracy. Robustness has to be measured across likely input changes, not inferred from a clean test score.

[<- Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
