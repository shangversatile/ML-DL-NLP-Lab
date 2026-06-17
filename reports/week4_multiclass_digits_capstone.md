# Week 4: Multiclass MLP and Handwritten-Digit Recognition Capstone

## Scope

Week 4 extends the binary MLP foundation into multiclass probability modeling, stable softmax, multiclass cross entropy, explicit multiclass MLP backpropagation, real handwritten-digit data, checkpointed inference, error analysis, and confidence and distribution-shift diagnostics.

Week 4 is the active stage of the project.

The current implementation scope covers the probability and loss foundation plus the scratch multiclass MLP forward and backpropagation path. Handwritten-digit training, checkpoint inference, and application work are intentionally deferred to later modules.

## Learning objectives

- Represent mutually exclusive class predictions as categorical probability distributions.
- Implement stable row-wise softmax without exponentiating raw large logits.
- Compute multiclass cross entropy from integer labels and predicted probabilities.
- Derive the softmax-cross-entropy gradient used by the output layer.
- Implement a single-hidden-layer `MulticlassMLPScratch` model with analytical backpropagation.
- Verify model gradients with generic central finite-difference checks.
- Connect binary sigmoid BCE intuition to multiclass softmax CE.
- Prepare the mathematical interface needed for future handwritten-digit recognition.

## Module map

| Module | File                                                 | Purpose                                                    |
| ------ | ---------------------------------------------------- | ---------------------------------------------------------- |
| 1      | `week4/01_softmax_and_multiclass_cross_entropy.md`   | Softmax, multiclass cross entropy, and gradient derivation |
| 2      | `week4/02_multiclass_mlp_backprop.md`                | Future multiclass MLP architecture and backprop            |
| 3      | `week4/03_digits_data_pipeline.md`                   | Future real handwritten-digit data pipeline                |
| 4      | `week4/04_training_and_error_analysis.md`            | Future training diagnostics and error analysis             |
| 5      | `week4/05_checkpoint_and_inference.md`               | Future checkpoint and prediction pipeline                  |
| 6      | `week4/06_interactive_app_and_distribution_shift.md` | Future local application and shift analysis                |

## Current status

Task 6A is complete:

- `src/utils/multiclass.py` implements one-hot encoding, stable softmax, multiclass cross entropy, cross entropy from logits, and the softmax-cross-entropy gradient.
- `tests/test_multiclass_utils.py` covers valid behavior, numerical stability, input validation, clipping behavior, and gradient structure.
- `week4/01_softmax_and_multiclass_cross_entropy.md` records the theory foundation.

Task 6B is complete:

- `src/models/multiclass_mlp.py` implements `MulticlassMLPScratch` forward prediction, multiclass CE loss, analytical gradients, and safe parameter access.
- `src/utils/model_gradient_check.py` implements generic central finite-difference gradient checking for scratch models with `get_parameters()` and `set_parameters()`.
- `tests/test_multiclass_mlp.py` and `tests/test_multiclass_mlp_gradient_check.py` cover validation, forward values, predictions, loss, backpropagation shapes, parameter safety, and numerical gradient agreement.
- [Multiclass MLP Backpropagation](week4/02_multiclass_mlp_backprop.md) records the architecture and gradient derivation.

No handwritten-digit training, data pipeline, GUI code, or checkpoint pipeline is included through Task 6B.

## Next steps

- Task 6C: add a real handwritten-digit data pipeline and baseline training.
- Train and evaluate the multiclass MLP on digit data.
- Add error analysis for common digit confusions.
- Add checkpointed prediction and confidence diagnostics.
- Add local inference interaction and distribution-shift analysis.
- Continue to Week 5 evaluation and technical debt work after the digits capstone baseline exists.

## Links

- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
- [Softmax and Multiclass Cross Entropy](week4/01_softmax_and_multiclass_cross_entropy.md)
- [Multiclass MLP Backpropagation](week4/02_multiclass_mlp_backprop.md)
- [Week 5 Evaluation, Technical Debt, and Trustworthy ML Diagnostics](week5_evaluation_technical_debt.md)
