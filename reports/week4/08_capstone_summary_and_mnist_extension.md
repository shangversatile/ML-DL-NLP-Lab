# Week 4 Capstone Summary and MNIST Extension Roadmap

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. What Week 4 accomplished

Week 4 extended the binary MLP foundation into a real multiclass handwritten-digit system. The work moved from probability theory and gradient validation to training, evaluation, persistence, local inference, and diagnostic stress tests.

The capstone pipeline is:

```text
softmax theory
-> multiclass MLP
-> real digits training
-> error analysis
-> checkpointed inference
-> interactive app
-> shift and confidence diagnostics
```

The system can train a scratch NumPy MLP on real handwritten-digit data from `load_digits`. It can save and reload model state. It can run local app inference from user-drawn digits. It can diagnose failure modes under synthetic app-like shifts.

The local app is an educational inference interface, not evidence of robust handwritten-input performance.

## 2. Why this is more than a toy classifier

The project includes mathematical derivation, manual implementation, numerical gradient verification, train/validation/test evaluation, diagnostic metrics beyond aggregate accuracy, checkpoint equivalence checks, inference interface design, app-input preprocessing, distribution-shift stress tests, and confidence diagnostics.

A toy classifier often stops at a single accuracy score. This capstone studies the full model system: how probabilities are computed, how gradients are verified, how training behaves, where errors occur, whether saved models preserve predictions, how inputs are transformed for inference, and how confidence behaves when the input distribution changes.

## 3. Theory-to-code-to-diagnostics chain

| Layer             | Mathematical idea                       | Code artifact                              | Diagnostic question                                 |
| ----------------- | --------------------------------------- | ------------------------------------------ | --------------------------------------------------- |
| Softmax           | categorical probability distribution    | `src/utils/multiclass.py`                  | do probabilities sum to one and remain stable?      |
| Cross entropy     | negative log-likelihood                 | `multiclass_cross_entropy()`               | how much probability is assigned to the true class? |
| Backprop          | `P - Y` residual                        | `MulticlassMLPScratch.compute_gradients()` | do analytical gradients match finite differences?   |
| Optimization      | parameter update dynamics               | `train_multiclass_mlp()`                   | does training reduce empirical risk?                |
| Evaluation        | confusion matrix / per-class metrics    | `src/evaluation/multiclass_metrics.py`     | where does the model fail?                          |
| Error analysis    | per-sample NLL / high-confidence errors | `src/evaluation/error_analysis.py`         | which errors dominate loss and risk?                |
| Checkpointing     | function preservation                   | `src/models/checkpoint.py`                 | does loaded model match original model outputs?     |
| Inference         | input-to-probability pipeline           | `src/inference/digits_inference.py`        | are predictions reusable outside training scripts?  |
| App preprocessing | `g(x)=f_theta(T(x))`                    | `digit_canvas_preprocessing.py`            | does user input match training assumptions?         |
| Shift diagnostics | risk changes under transformed inputs   | `shift_diagnostics.py`                     | does confidence remain meaningful under shift?      |

## 4. Key empirical findings

On the held-out `load_digits` test split, the trained scratch MLP reached:

- clean test accuracy: `0.977695`
- clean test cross entropy: `0.082859`
- macro recall: `0.977635`
- macro precision: `0.978411`
- top-3 accuracy: `1.000000`
- clean ECE-style diagnostic: `0.019867`

The worst synthetic shift condition was `thicken`:

- thickened accuracy: `0.182156`
- thickened cross entropy: `22.020438`
- thickened mean confidence: `0.967042`
- thickened ECE-style diagnostic: `0.784886`

The offline model is strong on held-out `load_digits` data. The system is highly sensitive to some app-like shifts. Confidence can remain high even when accuracy collapses. This justifies Week 5 calibration and reliability diagnostics.

## 5. Current limitations

- Input resolution is only `8 x 8`, which limits visual detail.
- The model is an MLP without spatial inductive bias.
- App preprocessing is intentionally simple.
- Synthetic shift probes are not real user-distribution measurement.
- No calibration correction has been applied.
- No abstention or rejection policy exists.
- No CNN or convolutional inductive bias has been implemented.
- No MNIST-scale experiment has been run yet.

These limitations do not invalidate the capstone. They define the boundary between the current educational system and future scale-up work.

## 6. Why MNIST is the natural next extension

`load_digits` is small and low-resolution. MNIST introduces higher-resolution `28 x 28` images and a larger benchmark setting. Flattened MNIST increases input dimension from `64` to `784`, which tests whether the NumPy MLP pipeline scales beyond the small teaching dataset.

MNIST also makes spatial structure more important. The move from `8 x 8` to `28 x 28` should therefore be treated as both a scale extension and an inductive-bias extension.

MNIST should not replace the current capstone. It should inherit its discipline: explicit preprocessing, split design, checkpointing, error analysis, confidence diagnostics, and shift tests.

## 7. MNIST extension roadmap

| Stage   | Goal                                            | Key changes                                                               |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| MNIST-1 | Load and train a scratch MLP on MNIST           | new data loader, input dimension 784, memory-aware mini-batching          |
| MNIST-2 | Compare `load_digits` and MNIST error structure | confusion matrix, per-class recall/precision, confidence diagnostics      |
| MNIST-3 | Compare app preprocessing targets               | 8x8 app pipeline vs 28x28 MNIST-style pipeline                            |
| MNIST-4 | Study why MLP is limited for images             | lack of translation equivariance and local spatial bias                   |
| MNIST-5 | Introduce convolutional reasoning               | explain CNN as structured parameter sharing rather than black-box upgrade |

MNIST should be introduced as a scale and inductive-bias extension, not as a replacement for the current capstone.

## 8. Bridge to Week 5

Week 5 should focus on calibration and reliability diagrams, technical debt cleanup, experiment registry design, checkpoint and artifact management, validation-vs-test discipline, potential abstention rules, and preparation for MNIST-scale extension.

Week 5 should not erase Week 4. It should evaluate and harden it.

## 9. Conceptual conclusion

Week 4 demonstrates that building an AI model is not just training a predictor. It requires probability modeling, gradient correctness, optimization, evaluation, persistence, inference design, app preprocessing, and failure-mode diagnostics. The MNIST extension should preserve this full-stack discipline while increasing data scale and visual complexity.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
