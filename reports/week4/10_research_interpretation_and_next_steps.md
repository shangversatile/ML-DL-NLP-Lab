# Research Interpretation and Next Steps After Augmented Robustness

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

## 1. The scientific question

The key question is not whether the model can achieve high clean accuracy. The key question is whether the model's probability outputs remain meaningful when the input distribution changes.

Clean accuracy answers only one part of the problem. A local handwritten-digit app also depends on preprocessing, input distribution, confidence behavior, and robustness to plausible user-input variation.

## 2. Baseline failure mode

The baseline model was strong on clean held-out `load_digits` data. Under thickened inputs, however, accuracy collapsed while confidence remained high. This is a failure of reliability, not merely a failure of accuracy.

A low-confidence error and a high-confidence error have different practical meanings. The baseline thick-stroke result showed that the model could be wrong while still producing probability outputs that looked decisive.

## 3. Intervention

Task 6H changed the training distribution through app-like augmentation. The architecture stayed the same. The optimizer family stayed the same. The comparison used a fixed update budget, shared initialization, and fresh optimizer state for each model.

Therefore, the improvement is plausibly tied to distributional coverage rather than a larger model, different optimizer, or longer training run. This does not prove that augmentation is the only cause, but it makes the causal story more credible than an uncontrolled comparison.

## 4. Result

| Condition | Baseline acc |  Aug acc | Baseline CE |   Aug CE | Baseline ECE |  Aug ECE |
| --------- | -----------: | -------: | ----------: | -------: | -----------: | -------: |
| Clean     |     0.977695 | 0.985130 |    0.082859 | 0.032913 |     0.019867 | 0.011971 |
| Thicken   |     0.182156 | 0.914498 |   22.020438 | 0.375293 |     0.784886 | 0.040985 |

Augmentation strongly improves configured shift robustness. Confidence diagnostics improve together with accuracy. Clean performance does not degrade in this run.

The result is important because the same MLP architecture became far less brittle once the training distribution included app-like transformations. The result should still be interpreted as configured-probe evidence, not as broad robustness evidence.

## 5. Why this is still not final robustness

The synthetic probes are limited. The augmentation and evaluation probes overlap. Real canvas inputs may differ in stroke width, centering, pressure, smoothness, digit style, and preprocessing artifacts. The current result is deterministic for one split and seed, but it is not yet a multiple-seed study.

No calibration correction has been applied. The ECE-style diagnostic improved, but it remains a measurement rather than a calibration solution. No MNIST-scale validation has been run. No convolutional comparison has tested whether structured image bias changes the robustness story.

## 6. What would count as stronger evidence

Stronger evidence would include held-out transformation strengths, a real canvas validation set, multiple random seeds, augmentation ablation studies, validation-based calibration, MNIST comparison, and eventual CNN comparison.

These studies would separate configured-probe improvement from broader invariance. They would also clarify whether the observed gains come from thickening alone, from combined transformations, or from a more general regularizing effect.

## 7. Research attitude

A strong result should not be treated as the end of the investigation. It should sharpen the next question.

Task 6H changes the hypothesis from "the model is generally unreliable under app-like shifts" to "distributional coverage through augmentation can repair a large part of the configured shift failure, but the scope of that repair must be measured."

This is the right research posture for Week 4: acknowledge the improvement, preserve the limitation, and define the next test.

## 8. Conclusion

Week 4 should now be treated as a strong baseline-plus-improvement capstone, but the final tag should wait until this interpretation is integrated and links are consistent.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
