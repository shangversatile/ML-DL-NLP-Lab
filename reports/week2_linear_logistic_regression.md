# Week 2 Linear and Logistic Regression Notes

## 1. Current progress

I implemented the mathematical core of linear regression from scratch in `src/models/linear_regression.py`. The current model supports parameter initialization, prediction, MSE loss computation, and analytical gradient computation. I also added unit tests in `tests/test_linear_regression.py` to verify initialization, prediction shape, prediction values, loss computation, gradient computation, and invalid input handling.

## 2. Linear regression mathematical core

The linear regression prediction rule is:

y_hat = Xw + b

where `X` has shape `(n_samples, n_features)`, `w` has shape `(n_features,)`, `b` is a scalar, and `y_hat` has shape `(n_samples,)`.

The MSE loss is:

L = mean((y_hat - y)^2)

The analytical gradients are:

dw = (2 / n) * X.T @ (y_hat - y)

db = (2 / n) * sum(y_hat - y)

These formulas are implemented directly in NumPy, which makes the relationship between the math and the code explicit.

## 3. Why implement predict, loss, and gradients before fit?

We implement `predict()`, `compute_loss()`, and `compute_gradients()` before writing a full `fit()` method because these are the mathematical core of linear regression. Separating them makes the relationship between the formula, the NumPy implementation, and the test cases much clearer. If we put everything directly into `fit()`, it would be harder to debug whether an error comes from prediction, loss computation, gradient derivation, or parameter updates.

This separation also supports unit testing. We can verify prediction values, MSE loss, and analytical gradients independently before building a training loop. From an engineering perspective, the model should be responsible for forward prediction and gradients, while optimizers should be responsible for updating parameters. This separation will make it easier to later plug in Batch Gradient Descent, SGD, Momentum, and Adam without rewriting the model.

## 4. Shape reasoning

For `X.shape = (n, d)` and `weights.shape = (d,)`, the prediction `X @ weights + bias` has shape `(n,)`.

For gradients, `error = predictions - y` has shape `(n,)`. Since `X.T` has shape `(d, n)`, `X.T @ error` has shape `(d,)`, which matches the shape of `weights`. The bias gradient is a scalar because it sums the prediction error over all samples.

## 5. Testing decisions

The tests are designed to check both behavior and mathematical correctness. Prediction is tested with manually assigned weights and bias. Loss is tested with a small example where the MSE can be computed by hand. Gradients are tested using a simple one-dimensional dataset where `dw = -56/3` and `db = -8` can be manually derived.

## 6. Batch Gradient Descent optimizer

I implemented a minimal `BatchGradientDescent` optimizer in `src/optimization/gradient_descent.py`. The optimizer stores a positive learning rate and exposes a `step()` method that updates weights and bias using the rule:

w_new = w_old - learning_rate * dw

b_new = b_old - learning_rate * db

This separates the model's mathematical responsibilities from the optimizer's update responsibilities. The linear regression model computes predictions, loss, and gradients, while the optimizer decides how parameters move based on those gradients.

## 7. Why optimizer step returns new parameters

The optimizer should return new weights and bias instead of modifying the original arrays in place because this reduces hidden side effects. If `step()` directly mutates the input weights, it becomes harder to debug whether a parameter changed because of the optimizer, the model, or another part of the training loop.

Returning new parameters also makes the optimizer easier to test: given fixed inputs `(weights, bias, gradients, learning_rate)`, we can check the exact returned outputs without worrying about unexpected mutation. This design also supports cleaner comparisons between Batch Gradient Descent, SGD, Momentum, and Adam, because different optimizers can operate on the same initial parameters safely.

## 8. Linear regression training loop

I extended `experiments/run_linear_regression.py` from a Week 1 smoke test into a minimal batch gradient descent training loop. The script now loads configuration, sets the random seed, generates synthetic linear regression data, splits train/validation sets, standardizes features using training statistics, initializes `LinearRegressionScratch`, initializes `BatchGradientDescent`, and trains for multiple epochs.

The training loop explicitly performs:

1. compute train loss
2. compute analytical gradients
3. call `optimizer.step()`
4. assign the returned parameters back to the model
5. record loss history

This keeps the training process transparent and makes the connection between the model, optimizer, and experiment script easy to inspect.

## 9. Training result

The initial train loss was 1.617364 and the final train loss was 0.010139. The final validation loss was 0.012910. Since the synthetic dataset used Gaussian noise with standard deviation 0.1, the noise variance is approximately 0.01, so a final MSE around 0.01 is reasonable. This suggests that the training loop is working and that the model is fitting the learnable linear signal rather than trying to reduce the loss unrealistically to zero.

## 10. Parameter interpretation under feature standardization

The model is trained on standardized features, so the learned weights and bias live in standardized feature space. The data generator's `true_weights` and `true_bias` live in the original feature space. Therefore, the learned parameters should not be directly compared with the true parameters before converting them back.

For standardized features:

X_scaled = (X_original - mean) / std

The original-space parameters can be recovered as:

recovered_weights = learned_weights / std

recovered_bias = learned_bias - mean @ recovered_weights

After recovery, the learned parameters were close to the true parameters:

- Recovered weights: [0.32944361, 1.3954707]
- True weights: [0.33757455, 1.40748186]
- Recovered bias: 0.085046
- True bias: 0.090585

This confirms that the model learned the underlying linear relationship, while also showing why preprocessing changes parameter interpretation.

## 11. Loss curve as a training diagnostic

I added a plotting utility in `src/utils/plotting.py` and used it in `experiments/run_linear_regression.py` to save the linear regression training loss curve. The loss curve is more informative than only reporting final train loss because it shows the training dynamics across epochs.

A final loss value is only a snapshot at the end of training. In contrast, a loss curve can show whether optimization is stable, whether the loss decreases smoothly, whether it oscillates, whether it diverges, or whether it plateaus early. This makes it useful for diagnosing learning rate problems, gradient issues, numerical instability, and convergence behavior.

For this linear regression experiment, the curve should show a rapid drop in loss during early epochs and then a plateau near the noise level. Since the synthetic dataset uses Gaussian noise with standard deviation 0.1, an MSE near 0.01 is expected and suggests that the model is fitting the learnable signal rather than over-interpreting noise.

## 12. Separating training loss from evaluation metrics

I implemented `mean_squared_error()` in `src/evaluation/metrics.py` and added unit tests in `tests/test_metrics.py`. The function validates that `y_true` and `y_pred` are NumPy arrays, that both are one-dimensional, and that they have the same shape. It then computes `mean((y_true - y_pred) ** 2)` and returns a Python float.

Although linear regression currently uses MSE both as the training loss and as the evaluation metric, these two uses have different responsibilities. The training loss is part of the optimization process: it is used to compute gradients and update model parameters. The evaluation metric is part of reporting and diagnosis: it tells us how well the trained model performs after or during training.

This separation keeps the project structure cleaner. The model is responsible for prediction, loss, and gradients; the optimizer is responsible for parameter updates; and the evaluation module is responsible for metrics used in reporting and analysis. This will become more important for logistic regression and trustworthy ML, where the training loss may be cross entropy but evaluation may involve accuracy, precision, recall, F1, confusion matrix, calibration error, and error analysis.

## 13. Linear regression training integration test

I added an integration test for linear regression training in `tests/test_linear_regression_training.py`. Unlike unit tests that check individual functions, this test verifies whether multiple components work together as a small training pipeline: synthetic data generation, train/validation split, feature standardization, `LinearRegressionScratch`, and `BatchGradientDescent`.

The test checks that training reduces the loss after multiple gradient descent steps and that validation loss remains reasonably low. This is important because individual components can pass their unit tests while still failing when connected together. In ML research code, many bugs appear at the boundaries between modules, such as shape mismatches, incorrect preprocessing assumptions, or parameter updates not being assigned back to the model.

This integration test gives a behavioral guarantee: the system should not only compute formulas correctly in isolation, but should also learn from data when the components are composed.

## 14. Logistic regression mathematical core

I implemented the core binary logistic regression model in `src/models/logistic_regression.py` and added unit tests in `tests/test_logistic_regression.py`. The model supports parameter initialization, sigmoid probability transformation, probability prediction, threshold-based class prediction, binary cross entropy loss, and analytical gradients.

The model first computes logits:

z = Xw + b

Then it converts logits into probabilities with the sigmoid function:

p = sigmoid(z) = 1 / (1 + exp(-z))

The probability `p` represents the model's estimated probability that a sample belongs to class 1. A hard class prediction is then obtained by comparing this probability with a threshold, usually 0.5.

The binary cross entropy loss is:

L = -mean(y log(p) + (1 - y) log(1 - p))

The gradients are:

dw = (1 / n) * X.T @ (p - y)

db = (1 / n) * sum(p - y)

This implementation is a direct NumPy translation of the mathematical definition, similar to the linear regression implementation but adapted for binary classification.

## 15. Why predict_proba matters

`predict_proba()` is important because logistic regression is not only a hard classifier but also a probabilistic model. The probability output can be interpreted as the model's confidence for the positive class, which gives more information than a direct 0/1 label.

This also allows threshold adjustment. In different applications, predicting class 1 at probability 0.5 may not be appropriate; a medical or risk-sensitive system may require a higher threshold, while a screening system may use a lower threshold to improve recall.

For trustworthy ML, probability outputs are essential for calibration, confidence analysis, error analysis, and monitoring. If we only keep the final hard labels, we lose information about uncertainty and cannot evaluate whether the model is confidently wrong or appropriately uncertain.

## 16. Binary classification metrics

I implemented binary classification metrics in `src/evaluation/metrics.py` and added tests in `tests/test_metrics.py`. The metrics include confusion matrix, accuracy, precision, recall, and F1 score.

The confusion matrix uses the format:

[[TN, FP],
 [FN, TP]]

where TN is true negative, FP is false positive, FN is false negative, and TP is true positive.

Accuracy measures the overall fraction of correct predictions:

accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision measures how many predicted positives are truly positive:

precision = TP / (TP + FP)

Recall measures how many actual positives are recovered:

recall = TP / (TP + FN)

F1 score summarizes the trade-off between precision and recall:

F1 = 2 * precision * recall / (precision + recall)

Accuracy alone is not enough for binary classification, especially under class imbalance. A model can achieve high accuracy by predicting the majority class while failing to identify the minority class. Precision and recall expose different error types, and the confusion matrix makes false positives and false negatives explicit. This is important for trustworthy ML because different errors can have very different real-world costs.

## 17. Logistic regression training experiment

I implemented a minimal logistic regression training experiment in `experiments/run_logistic_regression.py`. The script loads `configs/logistic_regression.yaml`, sets the random seed, generates synthetic binary classification data, performs train/validation split, standardizes features using training statistics, initializes `LogisticRegressionScratch`, initializes `BatchGradientDescent`, and trains the model with batch gradient descent.

The training loop follows the same explicit structure as linear regression:

1. compute binary cross entropy loss
2. compute analytical gradients
3. call optimizer.step()
4. assign the returned parameters back to the model
5. record loss history

The initial train loss was approximately 0.693147, which is expected for a binary classifier initialized near probability 0.5. After training, the final train loss decreased to approximately 0.035329, showing that the logistic regression training loop is connected correctly.

## 18. Evaluation caveat: label distribution matters

The validation metrics were accuracy = 1.0, precision = 1.0, recall = 1.0, and F1 = 1.0. However, the validation confusion matrix was:

[[0, 0],
 [0, 40]]

This means that the validation split contained only positive examples and no negative examples. Therefore, the perfect validation metrics should not be interpreted as strong evidence of generalization across both classes. They mainly show that the current model predicts the positive validation samples correctly.

This is an important trustworthy ML lesson: metrics cannot be interpreted without inspecting class distribution and the confusion matrix. A single scalar metric can look perfect while hiding a weak or incomplete evaluation setup. The next improvement should make the synthetic classification dataset or splitting strategy more balanced so that both positive and negative classes appear in train and validation sets.

## 19. Fixing synthetic classification data balance

After adding label distribution diagnostics, I found that the original synthetic binary classification generator produced an all-positive dataset under the current seed. Both the training and validation splits contained only class 1 examples, which made the validation metrics look perfect but not meaningful.

To fix this, I updated `make_binary_classification_data()` so that it generates a roughly balanced linearly separable binary classification dataset. Instead of relying on a randomly sampled bias that can shift all logits to one side, the updated generator computes logits from `X @ true_weights` and thresholds them around the median logit. This makes the dataset contain both class 0 and class 1 examples more reliably.

This change improves the validity of the logistic regression experiment. After the fix, classification metrics such as accuracy, precision, recall, F1, and the confusion matrix are more informative because both classes are represented. The key lesson is that evaluation quality depends not only on the metric formula, but also on whether the evaluation data actually contains the cases the metric is supposed to measure.

From a trustworthy ML perspective, this is an example of why we should inspect data distributions before trusting scalar metrics. A model can appear perfect under a flawed validation split, but distribution diagnostics and confusion matrices can reveal that the evaluation setup is incomplete.

## 20. Logistic regression loss curve

I added loss curve plotting for the logistic regression experiment by reusing `plot_loss_curve()` from `src/utils/plotting.py`. The experiment now saves a Binary Cross Entropy training curve to `results/figures/logistic_regression_loss_curve.png`.

The logistic regression loss curve should use Binary Cross Entropy Loss rather than accuracy because BCE is the actual continuous optimization objective used to update the model parameters. Accuracy is a thresholded, discrete metric: it only checks whether predicted labels are correct after converting probabilities into 0/1 decisions.

During training, the model's predicted probabilities may improve substantially even when accuracy stays unchanged. For example, increasing the probability of the correct class from 0.55 to 0.95 does not change accuracy, but it significantly lowers BCE. BCE also penalizes confidently wrong predictions more strongly, which makes it more informative for optimization diagnostics.

Therefore, BCE is more useful for observing training dynamics, learning rate stability, convergence, oscillation, or divergence. Accuracy, precision, recall, F1, and confusion matrix are better treated as evaluation metrics after the model probabilities are converted into class predictions.

## 21. Threshold analysis and decision policy

Threshold analysis is part of trustworthy ML evaluation because it studies how probabilistic model outputs are converted into real decisions. A logistic regression model outputs probabilities, but downstream systems must choose a decision threshold to produce class labels. Different thresholds create different false positive and false negative trade-offs, and these errors may have very different real-world costs.

Therefore, threshold analysis is not merely tuning a number; it evaluates the decision policy built on top of the model. By comparing accuracy, precision, recall, F1, and confusion matrices across thresholds such as 0.3, 0.5, and 0.7, we can understand whether the model remains useful under different risk preferences. This is closely related to ROC/PR-style thinking: instead of trusting one fixed operating point, we examine how the model behaves across multiple decision boundaries.

## 22. Updated open questions

- Should the model class eventually include a `fit()` method, or should training remain fully controlled by external experiment scripts?
- Should we add numerical gradient checking to compare analytical gradients against finite-difference approximations?
- How should this implementation change when moving from linear regression to logistic regression?
- How should the training loop connect model gradients and optimizer updates while keeping responsibilities separated?
- Should the optimizer update model parameters directly, or should the experiment script assign returned parameters back to the model?
- How can we test that loss actually decreases over multiple gradient descent steps?
- What learning rates are stable for synthetic linear regression data?
- How will this optimizer design change for stateful optimizers like Momentum and Adam?
