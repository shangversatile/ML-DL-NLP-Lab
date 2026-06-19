# Week 4 Final Synthesis and Transition to Week 5

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. What Week 4 actually accomplished

Week 4 extended the binary MLP foundations into a full handwritten-digit application system.

```text
softmax theory
→ multiclass MLP
→ load_digits training
→ error analysis
→ checkpoint inference
→ local drawing app
→ synthetic shift diagnostics
→ augmented robustness comparison
→ real canvas validation
→ dataset protocol
```

This was not just a classifier exercise. Week 4 built a model system with training, inference, diagnostics, app input, real-input evaluation, and a validation protocol. The system now exposes the difference between a clean benchmark result, a configured synthetic robustness result, and a true app-input result.

## 2. The theory-to-system chain

| Layer              | Core idea                        | Implementation             | What it tested                            |
| ------------------ | -------------------------------- | -------------------------- | ----------------------------------------- |
| Probability        | softmax categorical distribution | `src/utils/multiclass.py`  | stable probabilities                      |
| Learning objective | cross entropy / NLL              | multiclass loss utilities  | probability assigned to true class        |
| Optimization       | gradient descent / Adam          | optimizers and train loops | empirical risk reduction                  |
| Representation     | hidden-layer MLP                 | `MulticlassMLPScratch`     | non-linear decision boundaries            |
| Verification       | finite-difference gradient check | model gradient tests       | correctness of backprop                   |
| Evaluation         | confusion / per-class / top-k    | evaluation modules         | error structure beyond accuracy           |
| Persistence        | checkpointing                    | checkpoint utilities       | reproducible inference                    |
| Application        | canvas app                       | app + preprocessing        | raw user input to model input             |
| Robustness         | synthetic shift probes           | shift diagnostics          | sensitivity to controlled transformations |
| Real validation    | saved canvas samples             | canvas diagnostics         | true app-input behavior                   |

## 3. Main empirical arc

### Clean benchmark

The baseline model reached strong held-out `load_digits` performance: accuracy `0.977695` and cross entropy `0.082859`. This showed that the scratch multiclass MLP, loss, optimizer, and data pipeline were working on the curated benchmark.

### Synthetic shift failure

Thickened synthetic inputs caused a major baseline collapse: accuracy `0.182156`, cross entropy `22.020438`, mean confidence `0.967042`, and ECE-style diagnostic `0.784886`. The failure was not only wrong predictions. It was wrong predictions with high confidence.

### Augmented training intervention

Augmented training dramatically improved configured thickened-shift performance while preserving clean performance in this run. Clean accuracy reached `0.985130`, clean cross entropy reached `0.032913`, and clean ECE-style diagnostic reached `0.011971`. Thickened accuracy reached `0.914498`, thickened cross entropy reached `0.375293`, and thickened ECE-style diagnostic reached `0.040985`.

### Real canvas validation

Real user-drawn inputs remained weak. On 56 labeled real canvas samples, the augmented checkpoint had 23 errors, accuracy `0.589286`, Top-3 accuracy `0.839286`, cross entropy `3.766522`, mean confidence `0.886175`, overconfidence gap `0.296889`, and 9 Top-3 misses among 23 errors.

| Evaluation stage          | Key result                 | Interpretation                                              |
| ------------------------- | -------------------------- | ----------------------------------------------------------- |
| Clean `load_digits`       | 0.977695 baseline accuracy | clean benchmark solved reasonably well                      |
| Thickened synthetic shift | 0.182156 baseline accuracy | app-like shift exposed failure                              |
| Augmented thickened shift | 0.914498 accuracy          | configured shift robustness improved                        |
| Real canvas validation    | 0.589286 accuracy          | synthetic robustness did not fully transfer                 |
| Real canvas Top-3         | 0.839286                   | ranking contains useful information but Top-1 is unreliable |

## 4. What the real canvas result teaches

Real inputs are not the same as synthetic transformed `load_digits` inputs. The configured augmentation helped, but it did not cover the full app-input distribution.

Condition-B failures are especially important. When the `8 x 8` model input looks recognizable but the prediction is wrong, the main issue is not just preprocessing. It points toward model/data coverage limits, an `8 x 8` representation bottleneck, and the MLP's lack of spatial inductive bias.

Digit 6 and digit 8 reveal representation failure. Digit 6 often becomes 5 or 4. Digit 8 is the most severe failure mode and often becomes 0, 3, or 9. Digit 9 often becomes 4 with high confidence, which suggests a high-confidence boundary error. Confidence remains unreliable under real app inputs.

## 5. Why this is still a successful Week 4

Success does not mean the recognizer is robust. Success means the project found the boundary of the method.

A weaker project would stop at clean accuracy. This project continued until the clean benchmark result, synthetic robustness result, and real canvas result disagreed. That disagreement is the scientific finding.

Week 4 is an end-to-end MLP application and failure-analysis capstone. It demonstrates how a model can be correct in the benchmark setting, improved under configured probes, and still unreliable under real application input.

## 6. What should not be done next

- Do not train on `Canvas-Diagnostic-v1` and evaluate on it.
- Do not tune preprocessing on the final sample set and claim improvement.
- Do not claim production readiness.
- Do not use synthetic shift success as real robustness proof.
- Do not chase MNIST SOTA inside this MLP capstone.

## 7. MNIST decision

MNIST is a natural scale-up, but it is not necessary to prove the Week 4 point. Week 4 already demonstrates the central MLP application lesson: clean benchmark success is not application reliability.

Option A: do not implement MNIST in Week 4. Keep MNIST and CNN work for the later deep-learning stage.

Option B: add a minimal MNIST-MLP appendix before leaving Week 4:

- flattened `28 x 28` input
- same scratch MLP philosophy
- no CNN
- no SOTA chasing
- purpose: demonstrate scale-up and prepare the deep-learning transition

Recommended decision: unless a minimal MNIST appendix is required for portfolio completeness, Week 4 can close without MNIST because the central MLP application lesson is already complete.

## 8. Transition to Week 5

Week 5 should focus on:

- calibration and reliability diagrams
- ECE implementation review
- confidence thresholding
- abstention policies
- experiment registry
- artifact management
- validation/test discipline
- technical debt cleanup
- README/report link audit
- final Week 4 tag preparation

Week 5 should harden the evaluation system rather than add new model architectures. The immediate goal is not a larger recognizer. The immediate goal is a cleaner, more reliable evaluation and artifact system for the models already built.

## 9. Conceptual conclusion

Week 4 demonstrates that building an AI model is not merely training a predictor. It requires understanding probability, optimization, implementation correctness, evaluation design, deployment input transformations, and real-input validation.

The final lesson is that benchmark success is not the same as application reliability.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
