# Real Canvas Validation Findings

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. Validation setup

The first real canvas validation pass used 56 labeled real canvas samples. The augmented checkpoint was used, and the samples were saved through the Task 6J app.

This is a small user-specific validation set, not a full benchmark.

| Metric            |    Value |
| ----------------- | -------: |
| Number of samples |       56 |
| Number of errors  |       23 |
| Accuracy          | 0.589286 |
| Top-3 accuracy    | 0.839286 |
| Cross entropy     | 3.766522 |
| Mean confidence   | 0.886175 |

## 2. Why this changes the interpretation

Task 6H improved configured synthetic shifts, especially the thick-stroke probe. Real canvas accuracy is still low. Therefore, configured synthetic robustness does not imply real canvas robustness.

This result does not erase the Task 6H improvement. It sharpens the research problem: the system has become stronger on the probes it was trained and evaluated against, but the actual app-input distribution remains a harder and less controlled target.

## 3. Error structure

The errors are structured rather than random.

- Digit 0 is comparatively stable.
- Digit 6 is often confused with 5 or 4.
- Digit 8 is the most severe failure mode and is often predicted as 0, 3, or 9.
- Digit 9 is often confused with 4, but the true label often remains in Top-3.
- Some classes show ranking uncertainty while others show representation failure.

This distinction matters. A wrong Top-1 prediction with the true label still ranked nearby is different from a failure where the true label is absent from the candidate set.

## 4. Top-1 versus Top-3

Top-3 accuracy is much higher than Top-1 accuracy. That means the model often assigns useful probability mass to the correct digit even when the final decision is wrong.

For some classes, the correct answer is in the candidate set. For digit 8, the true label is often missing from Top-3, which is more severe. The former points toward ranking, thresholding, or abstention design. The latter points toward data coverage, representation, or model-capacity limits.

This distinction matters for future UI design. A system that exposes candidates or abstains when uncertain needs reliable candidate sets. A Top-3 miss gives the UI no correct option to recover.

## 5. Confidence failure

Mean confidence is high relative to accuracy. Some wrong predictions have very high confidence, which is especially concerning.

Confidence cannot currently be used as reliability without calibration and validation. At this stage, confidence is a diagnostic signal, not a trustworthy guarantee.

## 6. Condition-B failures

Condition-B failures occur when the `8 x 8` input looks recognizable but the prediction is wrong. In those cases, the issue is not mainly preprocessing.

The likely causes are training distribution mismatch, representation limitations, or model capacity limitations. The current `8 x 8` MLP may be too brittle for personal handwriting variation, especially when the real canvas style differs from the `load_digits` training distribution and the configured synthetic augmentations.

The full app system should be evaluated as:

```math
g(x) = f_\theta(T(x))
```

Here `T(x)` is the canvas preprocessing path and `f_theta` is the checkpointed MLP.

## 7. Research implications

The next stage should collect more real canvas samples, but it should avoid training and testing on the same user samples. A real canvas train/validation split should only be created after enough samples exist.

Canvas-derived augmentation should be run carefully. If the same tiny sample set drives both preprocessing decisions and evaluation, the project will overfit the diagnostic set.

Useful next comparisons include a larger MNIST-scale MLP and CNNs with structured image inductive bias. These are future research steps, not changes made in Task 6K.

## 8. Interpretation boundaries

The 56 samples are informative but not conclusive. The sample set is user-specific. It should not be treated as a broad handwriting benchmark.

Do not overfit the model or preprocessing to this exact set. This is a diagnostic validation set, not a final benchmark.

## 9. Conceptual summary

Real canvas validation exposes failures hidden by synthetic probes. The app system should be evaluated as `g(x)=f_theta(T(x))`, not only as the model on clean `load_digits` inputs.

High Top-3 accuracy but low Top-1 accuracy suggests ranking and decision issues. Severe Top-k misses for digit 8 suggest representation or data coverage failure.

The next stage should move from anecdotal app testing to controlled real-input evaluation, with clear separation between data used for diagnosis, data used for model changes, and data used for final validation.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
