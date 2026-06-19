# Week 4: Multiclass MLP and Handwritten-Digit Recognition Capstone

## Scope

Week 4 extends the binary MLP foundation into multiclass probability modeling, stable softmax, multiclass cross entropy, explicit multiclass MLP backpropagation, real handwritten-digit data, checkpointed inference, error analysis, confidence and distribution-shift diagnostics, augmentation-based robustness improvement, real canvas debugging, real canvas validation diagnostics, and a protocol for future real-input experiments.

Week 4 includes a strong baseline-plus-improvement system, but it must not claim real-world canvas robustness. Tasks 6A through 6H built the baseline, local app, diagnostics, and an augmented robustness-improvement loop. Task 6J adds a toolchain to collect and evaluate real canvas samples because user-drawn inputs can still fail. Task 6K turns the first 56 labeled canvas samples into a diagnostic validation report. Task 6L defines the dataset protocol needed before further real-canvas optimization.

The current implementation scope covers the probability and loss foundation, the scratch multiclass MLP forward and backpropagation path, a baseline real handwritten-digit training pipeline, structured error analysis, checkpointed inference, a local handwritten-digit drawing app, local-input shift/confidence diagnostics, fixed-update augmented training comparison, research interpretation, real canvas sample evaluation, real canvas validation diagnostics, and a dataset protocol for the next stage.

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
| 3      | `week4/03_digits_data_pipeline.md`                   | Real handwritten-digit data pipeline and baseline training |
| 4      | `week4/04_training_and_error_analysis.md`            | Confusion matrix, per-class metrics, and error analysis    |
| 5      | `week4/05_checkpoint_and_inference.md`               | Checkpoint saving/loading and reusable inference pipeline  |
| 6      | `week4/06_interactive_app_and_distribution_shift.md` | Local drawing app and distribution-shift boundary          |
| 7      | `week4/07_shift_and_confidence_diagnostics.md`       | Synthetic shift probes and confidence diagnostics          |
| 8      | `week4/08_baseline_diagnosis_robustness_loop_and_mnist_extension.md` | Baseline diagnosis, robustness loop, and MNIST roadmap |
| 9      | `week4/09_augmented_training_and_robustness_improvement.md` | Augmented training and robustness comparison        |
| 10     | `week4/10_research_interpretation_and_next_steps.md` | Research interpretation and next steps                      |
| 11     | `week4/11_real_canvas_debugging_and_user_sample_evaluation.md` | Real canvas debugging and user-sample evaluation       |
| 12     | `week4/12_real_canvas_validation_findings.md`        | Real canvas validation findings                             |
| 13     | `week4/13_canvas_dataset_protocol_and_next_stage_experiment_design.md` | Canvas dataset protocol and next-stage experiment design |

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

Task 6C is complete:

- `src/data/digits.py` loads the real handwritten digits dataset with optional 0-1 pixel scaling.
- `src/data/preprocessing.py` provides a reproducible stratified train/validation/test split without scikit-learn split helpers.
- `src/train.py` includes multiclass MLP evaluation and epoch-level training utilities.
- `experiments/run_digits_mlp.py` trains the scratch multiclass MLP baseline and saves cross-entropy and accuracy curves.
- `tests/test_digits_data.py` and `tests/test_multiclass_training.py` cover the data pipeline and multiclass training loop.
- [Digits Data Pipeline](week4/03_digits_data_pipeline.md) records the real-data baseline design and interpretation limits.

Task 6D is complete:

- `src/evaluation/multiclass_metrics.py` implements confusion matrices, confusion-derived accuracy, per-class recall and precision, macro averaging, top-k accuracy, prediction confidence, and per-sample negative log-likelihood.
- `src/evaluation/error_analysis.py` summarizes error and confidence behavior, selects high-loss examples, and selects high-confidence errors.
- `src/utils/plotting.py` can save confusion-matrix figures and digit-example grids.
- `experiments/analyze_digits_errors.py` trains the baseline and produces structured test-set diagnostics and figures.
- `tests/test_multiclass_metrics.py`, `tests/test_error_analysis.py`, and `tests/test_plotting.py` cover the new diagnostics.
- [Training and Error Analysis](week4/04_training_and_error_analysis.md) records the evaluation concepts and interpretation limits.

Task 6E is complete:

- `src/models/checkpoint.py` saves and loads scratch multiclass MLP checkpoints with JSON metadata and NumPy parameter arrays.
- `src/inference/digits_inference.py` normalizes digit input shapes and returns probabilities, predictions, confidences, and top-k candidates.
- `experiments/train_save_load_digits_mlp.py` trains the baseline, saves a checkpoint, loads it, and verifies probability and prediction equivalence.
- `tests/test_model_checkpoint.py` and `tests/test_digits_inference.py` cover checkpoint round trips, metadata validation, and reusable inference behavior.
- [Checkpoint and Inference](week4/05_checkpoint_and_inference.md) records the checkpoint semantics and inference boundary.

Task 6F is complete:

- `src/inference/digit_canvas_preprocessing.py` converts canvas grayscale arrays into 64-feature digit vectors with normalization, polarity correction, cropping, and 8 x 8 resizing.
- `apps/digit_draw_app.py` loads the saved scratch MLP checkpoint once at startup and predicts user-drawn digits with top-3 candidates.
- `tests/test_digit_canvas_preprocessing.py` covers the preprocessing bridge without launching Tkinter.
- [Interactive App and Distribution Shift](week4/06_interactive_app_and_distribution_shift.md) records why local canvas inference changes the problem.

Task 6G is complete:

- `src/evaluation/shift_diagnostics.py` implements controlled `8 x 8` shift probes for translation, intensity, noise, thresholding, thickening, and thinning.
- `src/evaluation/confidence_diagnostics.py` implements confidence-bin summaries and ECE-style diagnostics.
- `experiments/analyze_digits_shift_diagnostics.py` loads the saved checkpoint without retraining and compares clean test performance to synthetic shift conditions.
- `tests/test_shift_diagnostics.py`, `tests/test_confidence_diagnostics.py`, and `tests/test_plotting.py` cover the new diagnostics and figures.
- [Shift and Confidence Diagnostics](week4/07_shift_and_confidence_diagnostics.md) records the diagnostic framing and calibration boundary.

No calibration correction, temperature scaling, abstention, or full real-canvas distribution-shift analysis is included through Task 6G.

Task 6H is complete:

- `src/data/digit_augmentation.py` applies explicit app-like training augmentations to `8 x 8` digit batches.
- `src/train.py` includes fixed-update multiclass training so clean-only and augmented models can be compared under the same optimizer-step budget.
- `experiments/compare_digits_augmented_training.py` compares clean-only and augmented training under the same synthetic shift probes.
- `tests/test_digit_augmentation.py`, `tests/test_multiclass_fixed_update_training.py`, and `tests/test_plotting.py` cover the new augmentation, training, and plotting utilities.
- [Augmented Training and Robustness Improvement](week4/09_augmented_training_and_robustness_improvement.md) records the improvement protocol and interpretation boundaries.

Task 6I is complete:

- [Baseline Diagnosis, Robustness Loop, and MNIST Extension Roadmap](week4/08_baseline_diagnosis_robustness_loop_and_mnist_extension.md) incorporates the Task 6H results into the Week 4 synthesis without overclaiming real-world robustness.
- [Research Interpretation and Next Steps After Augmented Robustness](week4/10_research_interpretation_and_next_steps.md) records the scientific interpretation, threats to validity, and next research questions.

Task 6J is complete:

- `src/inference/digit_canvas_preprocessing.py` exposes stage-aware preprocessing so the final `8 x 8` model input can be inspected.
- `apps/digit_draw_app.py` prefers the augmented checkpoint, displays the preprocessed model input, and saves labeled or unlabeled canvas samples.
- `src/inference/canvas_sample_store.py` saves and loads real user-drawn samples without pickle.
- `experiments/evaluate_canvas_samples.py` evaluates saved labeled canvas samples without retraining.
- `tests/test_digit_canvas_preprocessing.py` and `tests/test_canvas_sample_store.py` cover the non-GUI utilities.
- [Real Canvas Debugging and User-Sample Evaluation](week4/11_real_canvas_debugging_and_user_sample_evaluation.md) records the debugging method and interpretation boundaries.

Task 6K is complete:

- `src/evaluation/canvas_diagnostics.py` summarizes real canvas validation performance, per-class behavior, confusion counts, high-confidence errors, and Top-k misses without retraining.
- `experiments/evaluate_canvas_samples.py` now prints richer real canvas diagnostics and saves ignored diagnostic figures under `results/canvas_debug/`.
- `tests/test_canvas_diagnostics.py` and `tests/test_plotting.py` cover the non-GUI diagnostics and plotting helpers.
- [Real Canvas Validation Findings](week4/12_real_canvas_validation_findings.md) records the first 56-sample validation result and interpretation boundaries.

Task 6L is complete:

- [Canvas Dataset Protocol and Next-Stage Experiment Design](week4/13_canvas_dataset_protocol_and_next_stage_experiment_design.md) defines `Canvas-Diagnostic-v1`, `Canvas-Train-v1`, `Canvas-Val-v1`, and `Canvas-Test-v1`.
- The protocol keeps the first 56 real canvas samples diagnostic-only and prevents leakage before future preprocessing, augmentation, calibration, MNIST, or CNN work.

Week 4 can be considered ready for final review after Task 6L. It should still not claim real-world canvas robustness; it now includes a toolchain to collect, evaluate, diagnose, and responsibly split real canvas evidence.

Week 4 now includes baseline modeling, synthetic-shift diagnosis, augmentation improvement, real canvas validation, and a protocol for future real-input experiments. It distinguishes configured synthetic robustness, real canvas robustness, preprocessing failure, and model/data coverage failure.

## Next steps

- Final Week 4 review with real canvas sample evidence and dataset protocol clearly separated from synthetic shift evidence.
- Then: Week 5 Evaluation, Technical Debt, and Trustworthy ML Diagnostics.

## Links

- [Week 3 Optimization and MLP Notes](week3_optimization_and_mlp.md)
- [Softmax and Multiclass Cross Entropy](week4/01_softmax_and_multiclass_cross_entropy.md)
- [Multiclass MLP Backpropagation](week4/02_multiclass_mlp_backprop.md)
- [Digits Data Pipeline](week4/03_digits_data_pipeline.md)
- [Training and Error Analysis](week4/04_training_and_error_analysis.md)
- [Checkpoint and Inference](week4/05_checkpoint_and_inference.md)
- [Interactive App and Distribution Shift](week4/06_interactive_app_and_distribution_shift.md)
- [Shift and Confidence Diagnostics](week4/07_shift_and_confidence_diagnostics.md)
- [Baseline Diagnosis, Robustness Loop, and MNIST Extension Roadmap](week4/08_baseline_diagnosis_robustness_loop_and_mnist_extension.md)
- [Augmented Training and Robustness Improvement](week4/09_augmented_training_and_robustness_improvement.md)
- [Research Interpretation and Next Steps After Augmented Robustness](week4/10_research_interpretation_and_next_steps.md)
- [Real Canvas Debugging and User-Sample Evaluation](week4/11_real_canvas_debugging_and_user_sample_evaluation.md)
- [Real Canvas Validation Findings](week4/12_real_canvas_validation_findings.md)
- [Canvas Dataset Protocol and Next-Stage Experiment Design](week4/13_canvas_dataset_protocol_and_next_stage_experiment_design.md)
- [Week 5 Evaluation, Technical Debt, and Trustworthy ML Diagnostics](week5_evaluation_technical_debt.md)
