# Week 4: Multiclass MLP and Handwritten-Digit Recognition Capstone

## Scope

Week 4 extends the binary MLP foundation into multiclass probability modeling, stable softmax, multiclass cross entropy, real handwritten-digit data, checkpointed inference, error analysis, and confidence and distribution-shift diagnostics.

The initial implementation scope is the probability and loss foundation required by a future `MulticlassMLPScratch` model. Full multiclass MLP training, handwritten-digit training, checkpoint inference, and application work are intentionally deferred to later modules.

## Learning objectives

- Represent mutually exclusive class predictions as categorical probability distributions.
- Implement stable row-wise softmax without exponentiating raw large logits.
- Compute multiclass cross entropy from integer labels and predicted probabilities.
- Derive the softmax-cross-entropy gradient used by the output layer.
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

No handwritten-digit training, full multiclass MLP model, GUI code, or checkpoint pipeline is included in Task 6A.

## Next steps

- Design the future `MulticlassMLPScratch` output layer and backpropagation path.
- Add a real handwritten-digit data pipeline.
- Train and evaluate the multiclass MLP on digit data.
- Add error analysis for common digit confusions.
- Add checkpointed prediction and confidence diagnostics.
- Add local inference interaction and distribution-shift analysis.

## Links to Week 3

- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
