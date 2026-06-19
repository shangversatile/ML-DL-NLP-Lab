# Week 4 Final Synthesis and Transition to Week 5

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. What Week 4 actually accomplished

Week 4 extended binary MLP foundations into a full handwritten-digit application and diagnostic system.

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

This is not merely a classifier. It is a model system with training, inference, diagnostics, app input, real-input validation, and a leakage-aware validation protocol.

## 2. Theory-to-system chain

| Layer                | Core idea                        | Implementation             | What it tested                            |
| -------------------- | -------------------------------- | -------------------------- | ----------------------------------------- |
| Probability          | softmax categorical distribution | `src/utils/multiclass.py`  | stable probabilities                      |
| Objective            | cross entropy / NLL              | multiclass loss utilities  | probability assigned to true class        |
| Optimization         | gradient descent / Adam          | optimizers and train loops | empirical risk reduction                  |
| Representation       | hidden-layer MLP                 | `MulticlassMLPScratch`     | nonlinear decision boundaries             |
| Verification         | finite-difference gradient check | model gradient tests       | correctness of backprop                   |
| Evaluation           | confusion / per-class / top-k    | evaluation modules         | error structure beyond accuracy           |
| Persistence          | checkpointing                    | checkpoint utilities       | reproducible inference                    |
| Application          | canvas app                       | app + preprocessing        | raw user input to model input             |
| Synthetic robustness | shift probes                     | shift diagnostics          | sensitivity to controlled transformations |
| Real validation      | saved canvas samples             | canvas diagnostics         | true app-input behavior                   |

## 3. Main empirical arc

| Evaluation stage                              | Key result        | Interpretation                                        |
| --------------------------------------------- | ----------------- | ----------------------------------------------------- |
| Clean `load_digits` baseline                  | 0.977695 accuracy | clean benchmark was strong                            |
| Synthetic thickened shift before augmentation | 0.182156 accuracy | app-like synthetic shift exposed failure              |
| Augmented thickened shift                     | 0.914498 accuracy | configured synthetic shift robustness improved        |
| Real canvas validation                        | 0.589286 accuracy | synthetic robustness did not fully transfer           |
| Real canvas Top-3                             | 0.839286          | ranking contains information, but Top-1 is unreliable |

The disagreement among clean, synthetic-shift, and real-canvas evaluations is the central Week 4 scientific finding.

## 4. What the real canvas result teaches

Real inputs are not the same as synthetic transformed `load_digits` inputs. The configured augmentation helped on configured probes, but it did not cover the full app-input distribution.

Condition-B failures show model/data coverage limits. When the `8 x 8` model input is recognizable but the prediction is wrong, the issue is not only preprocessing. The `8 x 8` representation can be too lossy, and the MLP lacks spatial inductive bias.

Confidence remains unreliable under real app inputs. Digit 6 and digit 8 reveal representation and data-coverage failures. Digit 9 reveals a high-confidence decision-boundary error, often becoming 4 with confidence near 1.0.

## 5. Why Week 4 is still successful

Success does not mean a robust recognizer. Success means the project found the boundary of the method.

A weaker project would stop at clean accuracy. This project continued until clean benchmark performance, synthetic robustness, and real canvas validation disagreed. That disagreement is the scientific result.

Week 4 closes as an MLP application and failure-analysis capstone, not as a production recognizer.

## 6. What should not be done next

- Do not train on `Canvas-Diagnostic-v1` and evaluate on it.
- Do not tune preprocessing on the final sample set and claim improvement.
- Do not claim production readiness.
- Do not use synthetic shift success as real robustness proof.
- Do not chase MNIST SOTA inside this MLP capstone.

## 7. MNIST decision: Option A

Week 4 will not implement MNIST.

Decision: Option A is selected. Do not implement MNIST in Week 4. Keep MNIST and CNN work for the later deep-learning stage.

Reasoning:

- Week 4 has already achieved its MLP application and diagnostic purpose.
- MNIST would shift the project from MLP foundations into image deep learning.
- CNN reasoning belongs naturally to the later deep-learning stage.
- MNIST should not distract from the real finding: clean benchmark, synthetic robustness, and real app robustness are different.
- Future MNIST/CNN work should retain the Week 4 evaluation discipline.

The Week 4 capstone closes without MNIST implementation.

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
- final tag preparation

Week 5 hardens the evaluation system rather than adding new model architectures.

## 9. Conceptual conclusion

Week 4 demonstrates that building an AI model is not merely training a predictor. It requires probability modeling, optimization, implementation correctness, evaluation design, deployment input transformations, real-input validation, and leakage-aware data protocol.

The final lesson is that benchmark success is not the same as application reliability.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
