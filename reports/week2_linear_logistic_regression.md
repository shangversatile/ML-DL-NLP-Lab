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

## 11. Updated open questions

- Should the model class eventually include a `fit()` method, or should training remain fully controlled by external experiment scripts?
- Should we add numerical gradient checking to compare analytical gradients against finite-difference approximations?
- How should this implementation change when moving from linear regression to logistic regression?
- How should the training loop connect model gradients and optimizer updates while keeping responsibilities separated?
- Should the optimizer update model parameters directly, or should the experiment script assign returned parameters back to the model?
- How can we test that loss actually decreases over multiple gradient descent steps?
- What learning rates are stable for synthetic linear regression data?
- How will this optimizer design change for stateful optimizers like Momentum and Adam?
