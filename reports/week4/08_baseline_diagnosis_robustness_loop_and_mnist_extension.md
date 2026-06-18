# Baseline Diagnosis, Robustness Loop, and MNIST Extension Roadmap

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

```text
Update after real canvas testing:
Although Task 6H substantially improved configured synthetic shift robustness, real hand-drawn canvas inputs can still fail. In many observed cases, the 8x8 model input appears recognizable but the model still predicts incorrectly. This suggests a model/data-coverage gap rather than only a preprocessing failure. Task 6J adds model-input visualization and user-sample evaluation to measure this gap.
```

```text
Current status:
The Week 4 system has progressed from a clean-data baseline to an augmented robustness-improved recognizer on configured synthetic probes.

It can:
- train a scratch NumPy multiclass MLP
- evaluate clean and shifted digit inputs
- save and reload model checkpoints
- run local drawing-app inference
- compare baseline and augmented training under a fixed update budget

It still cannot claim:
- real-world canvas robustness
- calibrated probabilities under all shifts
- production readiness
- MNIST-scale generalization
- convolutional image understanding
```

## 1. What Week 4 accomplished

Week 4 extended the binary MLP foundation into a real multiclass handwritten-digit system. It moved from stable multiclass probability modeling to a trained scratch NumPy MLP, then to checkpointed inference, local app input, synthetic shift diagnostics, and an augmentation-based robustness loop.

The implemented pipeline is:

```text
softmax theory
-> multiclass MLP
-> real digits training
-> error analysis
-> checkpointed inference
-> interactive app
-> shift and confidence diagnostics
-> augmented robustness comparison
```

The system can train a scratch NumPy MLP on real handwritten-digit data from `load_digits`, save and reload model state, run local app inference from user-drawn digits, and diagnose failure modes under configured synthetic app-like shifts.

The initial baseline was not sufficient. Task 6G exposed a severe confidence failure under thickened inputs: accuracy collapsed while mean confidence stayed high. Task 6H substantially improved robustness on the configured shift probes, but that improvement is not a proof of real app robustness.

## 2. Why this is more than a toy classifier

The project includes mathematical derivation, manual implementation, numerical gradient verification, train/validation/test evaluation, diagnostic metrics beyond aggregate accuracy, checkpoint equivalence checks, inference interface design, app-input preprocessing, distribution-shift stress tests, confidence diagnostics, and a controlled augmentation intervention.

A toy classifier often stops at a single clean accuracy score. This capstone studies the full model system: how probabilities are computed, how gradients are verified, how training behaves, where errors occur, whether saved models preserve predictions, how inputs are transformed for inference, and how confidence behaves when the input distribution changes.

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
| Augmentation      | empirical distribution expansion        | `src/data/digit_augmentation.py`           | can configured shift failures be reduced?           |

## 4. What Task 6G revealed

On the held-out `load_digits` test split, the clean baseline was strong:

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

The baseline failure was not merely an accuracy drop. It was a reliability failure: the model was often confidently wrong under a plausible app-like perturbation.

## 5. What Task 6H changed

Task 6H compared a clean-only baseline model against an augmented-training model under a fixed update budget and shared initialization. The architecture remained a single-hidden-layer scratch MLP. The intervention was the training distribution.

| Condition | Metric        |  Baseline | Augmented |     Change |
| --------- | ------------- | --------: | --------: | ---------: |
| Clean     | Accuracy      |  0.977695 |  0.985130 |  +0.007435 |
| Clean     | Cross entropy |  0.082859 |  0.032913 |  -0.049946 |
| Clean     | ECE-style     |  0.019867 |  0.011971 |  -0.007896 |
| Thicken   | Accuracy      |  0.182156 |  0.914498 |  +0.732342 |
| Thicken   | Cross entropy | 22.020438 |  0.375293 | -21.645145 |
| Thicken   | ECE-style     |  0.784886 |  0.040985 |  -0.743901 |

The baseline failure was not random noise. It was strongly related to missing support for app-like transformations. Augmentation expanded the effective training distribution, and the same MLP architecture became much less brittle under the configured shifts.

This is meaningful evidence that distributional coverage matters. It is not a proof of real-world canvas robustness, because the training and evaluation probes share transformation families and real drawings may differ in ways not represented here.

## 6. Augmentation as empirical distribution expansion

Clean empirical risk optimizes the model over the original training examples:

```math
R_{\text{clean}}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell
(
f_\theta(x_i),
y_i
)
```

Augmented empirical risk optimizes the model over transformed versions of each example:

```math
R_{\text{aug}}(\theta)
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

The transformations `T_s` include shifts, thickening, thinning, noise, thresholding, and intensity changes. Augmentation asks the model to assign the same label to a neighborhood of transformed inputs. This encourages local invariance in data space.

For the current MLP, augmentation is a data-level substitute for missing architectural image bias. It can reduce brittleness under configured transformations, but it does not provide formal worst-case robustness.

## 7. Why fixed update budget matters

The augmented dataset is larger than the clean dataset. Equal epochs would give the augmented model more parameter updates, which would confound data coverage with optimization opportunity.

Task 6H used a fixed update budget so both models received the same number of optimizer steps. Shared initialization controlled random-start confounding. Fresh optimizer instances controlled momentum and Adam-state leakage. This makes the comparison more scientifically meaningful than an equal-epoch comparison.

## 8. Threats to validity

1. Probe overlap:

Training augmentation and evaluation shifts share transformation families. The experiment therefore tests configured shift robustness, not all real shifts.

2. Single dataset:

`load_digits` is low-resolution and small. Results may not transfer to MNIST or real handwritten inputs.

3. Single architecture:

The MLP lacks convolutional spatial bias. Augmentation compensates for this under selected transformations, but it does not replace structured image modeling.

4. Single split and seed:

The current results are deterministic, but they should eventually be repeated under multiple seeds and splits.

5. No real canvas dataset:

Synthetic probes approximate app-like variation. Real user drawings should be collected for validation later.

6. Calibration not corrected:

ECE-style diagnostics are measured, not directly optimized or calibrated.

## 9. Concrete next research steps

### 1. Augmentation ablation

Run thicken-only training, shift-only training, noise-only training, all-but-thicken training, and full augmentation. The purpose is to identify which transformations actually drive robustness improvement.

### 2. Held-out transformation test

Train on one set of transforms and evaluate on unseen transform strengths, such as shift by 2 pixels, different noise levels, different thickening/thinning variants, and different thresholds. The purpose is to measure whether the model learns local invariance or merely memorizes configured probes.

### 3. Real canvas sample collection

Save user-drawn canvas samples and evaluate them separately. The purpose is to replace synthetic-only shift diagnostics with real app-input validation.

### 4. Preprocessing sweep

Compare different crop padding, foreground thresholds, resize rules, and optional centering. The purpose is to determine whether failures come from preprocessing or model limitations.

### 5. Calibration after augmentation

Use validation data for reliability diagrams, ECE, and temperature scaling. The purpose is to make confidence more meaningful after robustness improvement.

### 6. MNIST scale-up

Move from `8 x 8` `load_digits` to `28 x 28` MNIST. The purpose is to test whether the same scratch MLP pipeline scales to higher-dimensional images.

### 7. Convolutional reasoning

Introduce CNNs after MNIST as structured parameter sharing and local spatial bias. The purpose is to understand why image models need architectural inductive bias rather than only data augmentation.

## 10. Current limitations

- Input resolution is only `8 x 8`, which limits visual detail.
- The model is an MLP without spatial inductive bias.
- App preprocessing is intentionally simple.
- Synthetic shift probes are not real user-distribution measurement.
- Augmentation and evaluation probes overlap.
- No calibration correction has been applied.
- No abstention or rejection policy exists.
- No CNN or convolutional inductive bias has been implemented.
- No MNIST-scale experiment has been run yet.

These limitations define the boundary between the current educational capstone and stronger evidence for deployment-style robustness.

## 11. Why MNIST is the natural next extension

`load_digits` is small and low-resolution. MNIST introduces higher-resolution `28 x 28` images and a larger benchmark setting. Flattened MNIST increases input dimension from `64` to `784`, which tests whether the NumPy MLP pipeline scales beyond the small teaching dataset.

MNIST also makes spatial structure more important. The move from `8 x 8` to `28 x 28` should therefore be treated as both a scale extension and an inductive-bias extension.

MNIST should be introduced after the current robustness loop, not as a way to avoid the thick-stroke failure diagnosis. The current `load_digits` setup remains valuable because it is small enough for bottom-up implementation, controlled experiments, and fast robustness comparisons.

MNIST should inherit the same discipline: scratch implementation, explicit split design, checkpointing, error analysis, shift diagnostics, confidence diagnostics, and cautious interpretation.

## 12. MNIST extension roadmap

| Stage   | Goal                                            | Key changes                                                               |
| ------- | ----------------------------------------------- | ------------------------------------------------------------------------- |
| MNIST-1 | Load and train a scratch MLP on MNIST           | new data loader, input dimension 784, memory-aware mini-batching          |
| MNIST-2 | Compare `load_digits` and MNIST error structure | confusion matrix, per-class recall/precision, confidence diagnostics      |
| MNIST-3 | Compare app preprocessing targets               | 8x8 app pipeline vs 28x28 MNIST-style pipeline                            |
| MNIST-4 | Study why MLP is limited for images             | lack of translation equivariance and local spatial bias                   |
| MNIST-5 | Introduce convolutional reasoning               | explain CNN as structured parameter sharing rather than black-box upgrade |

MNIST extension should be treated as scale-up and inductive-bias study, not as a replacement for the current baseline or the current robustness analysis.

## 13. Bridge to Week 5

Week 5 should focus on calibration and reliability diagrams, technical debt cleanup, experiment registry design, checkpoint and artifact management, validation-vs-test discipline, held-out transformation tests, multiple seeds, and preparation for MNIST-scale extension.

Week 5 should not hide remaining weaknesses behind calibration plots. It should evaluate and harden the Week 4 system after the augmentation-based robustness loop.

## 14. Conceptual conclusion

Week 4 demonstrates that building an AI model is not just training a predictor. It requires probability modeling, gradient correctness, optimization, evaluation, persistence, inference design, app preprocessing, failure-mode diagnostics, and measured improvement loops.

Task 6H changes the Week 4 story: the baseline was brittle, but augmentation repaired a large part of the configured shift failure. The next research question is the scope of that repair.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
