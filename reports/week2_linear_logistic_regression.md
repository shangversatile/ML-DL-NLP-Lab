# Week 2 Linear and Logistic Regression Notes

## 1. Current progress

I implemented the mathematical core of linear regression from scratch in `src/models/linear_regression.py`. The current model supports parameter initialization, prediction, MSE loss computation, and analytical gradient computation. I also added unit tests in `tests/test_linear_regression.py` to verify initialization, prediction shape, prediction values, loss computation, gradient computation, and invalid input handling.

## 2. Linear regression mathematical core

The linear regression prediction rule is:

ŷ = Xw + b

where `X` has shape `(n_samples, n_features)`, `w` has shape `(n_features,)`, `b` is a scalar, and `ŷ` has shape `(n_samples,)`.

The MSE loss is:

L = mean((ŷ - y)^2)

The analytical gradients are:

dw = (2 / n) * X.T @ (ŷ - y)

db = (2 / n) * sum(ŷ - y)

These formulas are implemented directly in NumPy, which makes the relationship between the math and the code explicit.

## 3. Why implement predict, loss, and gradients before fit?

We implement `predict()`, `compute_loss()`, and `compute_gradients()` before writing a full `fit()` method because these are the mathematical core of linear regression. Separating them makes the relationship between the formula, the NumPy implementation, and the test cases much clearer. If we put everything directly into `fit()`, it would be harder to debug whether an error comes from prediction, loss computation, gradient derivation, or parameter updates.

This separation also supports unit testing. We can verify prediction values, MSE loss, and analytical gradients independently before building a training loop. From an engineering perspective, the model should be responsible for forward prediction and gradients, while optimizers should be responsible for updating parameters. This separation will make it easier to later plug in Batch Gradient Descent, SGD, Momentum, and Adam without rewriting the model.

## 4. Shape reasoning

For `X.shape = (n, d)` and `weights.shape = (d,)`, the prediction `X @ weights + bias` has shape `(n,)`.

For gradients, `error = predictions - y` has shape `(n,)`. Since `X.T` has shape `(d, n)`, `X.T @ error` has shape `(d,)`, which matches the shape of `weights`. The bias gradient is a scalar because it sums the prediction error over all samples.

## 5. Testing decisions

The tests are designed to check both behavior and mathematical correctness. Prediction is tested with manually assigned weights and bias. Loss is tested with a small example where the MSE can be computed by hand. Gradients are tested using a simple one-dimensional dataset where `dw = -56/3` and `db = -8` can be manually derived.

## 6. Open questions

- Should the model class eventually include a `fit()` method, or should training remain fully controlled by external experiment scripts?
- How should optimizers update model parameters while keeping the model and optimizer responsibilities separate?
- Should we add numerical gradient checking to compare analytical gradients against finite-difference approximations?
- How should this implementation change when moving from linear regression to logistic regression?
