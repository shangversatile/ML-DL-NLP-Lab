[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# Optimization Algorithms

## 1. SGD optimizer

I implemented `SGD` in `src/optimization/sgd.py` and added unit tests in `tests/test_sgd.py`. The optimizer applies the update rule:

```math
w_{\mathrm{new}} = w_{\mathrm{old}} - \alpha \, dw
```

```math
b_{\mathrm{new}} = b_{\mathrm{old}} - \alpha \, db
```

The update formula looks the same as Batch Gradient Descent. The key difference is not the `step()` formula itself, but how gradients are computed. Batch Gradient Descent uses the full training set, classic SGD uses one sample at a time, and mini-batch SGD uses a subset of samples.
## 2. Why mini-batch sampling belongs to the training loop

The SGD optimizer should only update parameters from the gradients it receives, while the training loop should decide how samples are shuffled and grouped into mini-batches. This separation keeps the optimizer reusable: the same update rule can operate on gradients computed from one sample, a mini-batch, or the full dataset.

If mini-batch sampling were embedded inside the optimizer, data iteration logic and optimization logic would become tightly coupled. Keeping them separate makes the system easier to test, debug, and extend to Momentum and Adam.

The iterator implementation details live in [Engineering Validation](05_engineering_validation.md#1-mini-batch-iterator-implementation).
## 3. Momentum optimizer

I implemented `Momentum` in `src/optimization/momentum.py` and added unit tests in `tests/test_momentum.py`. Unlike SGD, Momentum is a stateful optimizer. It stores `velocity_weights` and `velocity_bias`, which summarize recent gradient directions.

The update rule is:

```math
v_t = \beta v_{t-1} + (1 - \beta)g_t
```

```math
\theta_t = \theta_{t-1} - \alpha v_t
```

where $`g_t`$ is the current gradient, $`v_t`$ is the exponentially weighted moving average of recent gradients, $`\alpha`$ is the learning rate, and $`\beta`$ controls how strongly historical gradients are retained.
## 4. Why Momentum reduces oscillation

Momentum helps reduce oscillation because gradients that repeatedly change sign partially cancel each other in the moving average, while gradients that point in a stable direction accumulate. This is useful in narrow optimization valleys, where gradients may oscillate strongly across the steep direction while remaining consistent along the direction toward the minimum.

If the gradient direction remains stable, velocity gradually increases toward that direction. If gradients alternate between positive and negative values, the moving average becomes smaller than the raw gradients. This creates smoother and more stable parameter updates.
## 5. Interpreting beta

The coefficient $`\beta`$ controls the effective memory length of Momentum. A useful approximation is:

```math
\mathrm{effective\ memory\ length} \approx \frac{1}{1 - \beta}
```

For example:

- $`\beta = 0.0`$ behaves like ordinary SGD
- $`\beta = 0.9`$ remembers roughly 10 recent steps
- $`\beta = 0.99`$ remembers roughly 100 recent steps

A larger $`\beta`$ creates smoother but slower-reacting updates. A smaller $`\beta`$ responds more strongly to current gradients but performs less smoothing. The common default $`\beta = 0.9`$ is a practical compromise rather than a universal mathematical optimum.
## 6. Adam optimizer overview

Adam combines the main idea of Momentum with an adaptive parameter-wise scaling term. It keeps two exponentially weighted moving averages for each parameter:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

Here, $`g_t`$ is the current gradient, $`m_t`$ is the first-moment estimate, and $`v_t`$ is the second-moment estimate. The square $`g_t^2`$ is applied elementwise for weights.

Adam then applies bias correction:

```math
\hat{m}_t
=
\frac{m_t}
{1-\beta_1^t}
```

```math
\hat{v}_t
=
\frac{v_t}
{1-\beta_2^t}
```

The final update is:

```math
\theta_t =
\theta_{t-1}
- \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

where $`\alpha`$ is the learning rate and $`\epsilon`$ is a small positive constant for numerical stability.

[See the detailed Adam derivation appendix](appendix_adam_derivation.md).
## 7. Why Adam tracks the first moment

The first moment is an exponentially weighted moving average of recent gradients. Under a stationary-gradient approximation, its bias-corrected form estimates the recent mean gradient. It should not be treated as automatically equal to the true expected gradient throughout non-stationary neural-network training. In Adam, $`m_t`$ plays a role similar to Momentum's velocity:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

If the gradient direction is consistent across steps, the moving average points strongly in that direction. If the gradient direction alternates, positive and negative gradient components partially cancel.

This makes $`m_t`$ a smoothed estimate of descent direction. It reduces the impact of noisy mini-batch gradients while still allowing the optimizer to accumulate evidence about directions that are repeatedly useful.

For the expectation derivation, see [the detailed Adam derivation appendix](appendix_adam_derivation.md#6-deriving-the-expected-first-moment).
## 8. Why Adam tracks the second moment

The second moment tracks the average squared gradient:

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

Unlike the first moment, the second moment does not preserve sign. Squaring makes both positive and negative gradients contribute positively. This means $`v_t`$ estimates the recent magnitude of gradients for each parameter.

If one parameter often has large gradients and another often has small gradients, their $`v_t`$ values will differ. Adam uses this difference to scale updates parameter by parameter. Parameters with consistently large gradients receive smaller effective steps, while parameters with consistently small gradients receive relatively larger effective steps.

For the expectation derivation, see [the detailed Adam derivation appendix](appendix_adam_derivation.md#7-deriving-the-expected-second-moment).
## 9. Why Adam divides by the square root of the second moment

Adam divides the first-moment direction by the root mean square scale of recent gradients:

```math
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

This division has a units interpretation. If the gradient $`g_t`$ has units of loss per parameter, then $`g_t^2`$ has squared gradient units. Taking $`\sqrt{v_t}`$ returns the scale back to gradient units. Therefore, $`\hat{m}_t / \sqrt{\hat{v}_t}`$ is dimensionless up to the small $`\epsilon`$ term, and the learning rate controls the parameter step size.

It also has an RMS normalization interpretation. Since $`v_t`$ averages squared gradients, $`\sqrt{v_t}`$ is similar to a running root mean square gradient magnitude. Dividing by this term normalizes the update by recent gradient scale.

This creates parameter-wise adaptive learning rates. The update can be rewritten as:

```math
\theta_t =
\theta_{t-1}
- \left(\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\right)\hat{m}_t
```

For parameter coordinate $`i`$, the effective learning rate is:

```math
\alpha_{\mathrm{effective},t,i}
=
\frac{\alpha}
{\sqrt{\hat{v}_{t,i}}+\epsilon}
```

Here $`i`$ denotes the parameter coordinate. Large recent squared gradients in coordinate $`i`$ make $`\sqrt{\hat{v}_{t,i}}`$ larger, which reduces that coordinate's effective learning rate. Small recent squared gradients make the denominator smaller, which allows a larger effective learning rate for that coordinate.
## 10. Comparing SGD, Momentum, and Adam

SGD uses the current gradient directly:

```math
\theta_t = \theta_{t-1} - \alpha g_t
```

It is simple and has no optimizer state beyond the learning rate. Its weakness is that each update can be noisy, especially with small mini-batches.

Momentum smooths the gradient direction:

```math
v_t = \beta v_{t-1} + (1 - \beta)g_t
```

```math
\theta_t = \theta_{t-1} - \alpha v_t
```

It is stateful and helps reduce oscillation by averaging recent gradients. It still uses one global learning rate for all parameters.

Adam tracks both direction and scale:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

```math
\theta_t =
\theta_{t-1}
- \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Adam is stateful like Momentum, but it also adapts the effective learning rate for each parameter. This is useful when different parameters have gradients with very different magnitudes.
## 11. Teaching-oriented optimizer comparison

I compared Batch Gradient Descent, mini-batch SGD, Momentum, and Adam on the same synthetic binary classification dataset using `LogisticRegressionScratch`.

All methods trained for 100 epochs on the same standardized training data. However, this is not a strict optimizer benchmark. Batch Gradient Descent performs one parameter update per epoch, while the mini-batch methods use a batch size of 32 and perform ten parameter updates per epoch.

The comparison results were:

| Optimizer | Parameter updates | Validation BCE | Validation accuracy |
|---|---:|---:|---:|
| Batch Gradient Descent | 100 | 0.259780 | 0.9750 |
| Mini-batch SGD | 1000 | 0.138172 | 0.9875 |
| Momentum | 1000 | 0.137438 | 0.9875 |
| Adam | 1000 | 0.109719 | 0.9875 |

The experiment controls the number of dataset passes, or epoch budget. Each method sees approximately the same total number of training samples. However, it does not control the number of parameter updates, gradient-estimation granularity, learning rate, wall-clock time, or hyperparameter-search budget.

Mini-batch SGD lowers BCE faster than full-batch gradient descent because it updates parameters more frequently during each pass through the dataset. Momentum performs only slightly better than SGD in this simple convex problem, where strong oscillation is limited. Adam achieves the lowest BCE under the current configuration. This result is consistent with adaptive parameter-wise scaling being helpful on this task, but the experiment does not isolate a single causal mechanism.

The validation accuracies of SGD, Momentum, and Adam are identical even though their BCE values differ. This illustrates why BCE provides more information than accuracy: accuracy only reflects thresholded class decisions, while BCE also reflects confidence quality.

These results should not be generalized into a universal optimizer ranking. The experiment uses a simple linearly separable synthetic dataset, one random seed, and different optimizer hyperparameters. Its purpose is to build optimization intuition before moving to MLP training.
## 12. Optimizer comparison implementation logic

The optimizer comparison in `experiments/compare_optimizers.py` uses two training-loop patterns. `train_full_batch()` computes gradients using the entire training set and performs one parameter update per epoch. This matches Batch Gradient Descent: each epoch evaluates the current model on all training examples, computes one averaged gradient, and then calls the optimizer once.

`train_minibatch()` instead iterates over batches produced by `iterate_minibatches()`. Each mini-batch gives the loop an `X_batch` and `y_batch`. The model computes gradients on that mini-batch, producing a gradient estimate at the current parameter values, and the optimizer immediately performs one parameter update. As a result, mini-batch SGD, Momentum, and Adam perform multiple updates inside one epoch when the training set contains multiple mini-batches.

At the end of every epoch, both training loops compute binary cross-entropy using the entire training set. The plotted curves therefore use a common reporting metric: epoch-level full-training-set BCE. Every plotted loss is the BCE of the full training set under the model parameters at the end of that epoch. Raw mini-batch losses are useful for debugging training dynamics, but they are noisier and harder to compare directly because each value is computed on a different subset of examples.

Each optimizer comparison run gets a fresh `LogisticRegressionScratch` model initialized with zero weights and zero bias. This prevents one optimizer from benefiting from parameters learned by a previous optimizer. Momentum and Adam also receive fresh optimizer objects so that historical state does not leak between experiments. Momentum gets a new velocity state, and Adam gets new first-moment, second-moment, and time-step state.

The mini-batch iterator uses `seed + epoch` to create different but reproducible shuffling each epoch. The shuffle order changes from epoch to epoch, but rerunning the experiment with the same base seed produces the same sequence of epoch-level shuffles.

The current experiment boundary is important. All optimizers train for the same number of epochs, and each optimizer sees approximately the same total number of training examples. However, Batch Gradient Descent performs one update per epoch, while the mini-batch methods perform multiple parameter updates per epoch. This is a teaching-oriented same-epoch comparison, not a strict optimizer benchmark. It does not control update count, wall-clock time, hardware efficiency, or hyperparameter-search budget.
## 13. Optimizer comparison interpretation boundaries

The optimizer comparison is designed for teaching, not for a universal optimizer ranking. It uses a simple synthetic binary-classification dataset, one seed, fixed hyperparameters, and a same-epoch budget. Because mini-batch methods perform more updates per epoch than full-batch gradient descent, the experiment mixes optimizer choice with gradient-estimation granularity and update count.

A stricter benchmark would separately control same-update budget, same sample-processing budget, same wall-clock budget, and hyperparameter-search budget. The current result is still useful because it shows how SGD, Momentum, and Adam behave under one transparent training-loop configuration.
## 14. From scalar-pair optimizers to parameter dictionaries

The earlier optimizers update one weight array and one scalar bias:

```python
new_weights, new_bias = optimizer.step(
    weights,
    bias,
    weight_gradients,
    bias_gradient,
)
```

That API is sufficient for linear regression and logistic regression because those models expose exactly one weight vector and one bias scalar. A one-hidden-layer MLP has multiple trainable tensors:

```text
W1
b1
W2
b2
```

For that case, the optimizer needs a generic parameter-dictionary interface:

```python
updated_parameters = optimizer.step(
    parameters,
    gradients,
)
```

The `parameters` and `gradients` inputs are dictionaries. Matching keys define the update contract, and each dictionary value is a NumPy parameter tensor. With this boundary, `ParameterSGD`, `ParameterMomentum`, and `ParameterAdam` do not depend on layer names, hidden-layer semantics, or MLP-specific structure.
## 15. Why `ParameterSGD` covers both full-batch and mini-batch updates

The SGD-style parameter update remains:

```math
\theta_{t+1}
=
\theta_t
-
\eta g_t
```

The update rule is identical for full-batch gradient descent and mini-batch SGD. The difference lies in the source of $`g_t`$. Full-batch training passes the empirical-risk gradient computed from the full training set. Mini-batch training passes a stochastic gradient estimate computed from a subset of examples.

Data sampling belongs to the training loop, not the optimizer. `ParameterSGD` only receives parameter tensors and gradient tensors and applies the update it was given.
## 16. Per-parameter state in Momentum and Adam

Each trainable tensor needs an independent optimizer-state tensor with the same shape. For Momentum, the state is the velocity:

```math
v_t
=
\beta v_{t-1}
+
(1-\beta)g_t
```

For Adam, the state includes first and second moments:

```math
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
```

```math
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
```

`W1`, `b1`, `W2`, and `b2` have different shapes, and their gradients have different coordinate-wise histories. A single global state array would mix unrelated parameter coordinates. State dictionaries preserve one state array per parameter tensor, so each parameter keeps history that matches its own shape and gradient stream.
## 17. Generic optimizer API boundary

The generic optimizer does not infer that `dW1` corresponds to `W1`, or that `db1` corresponds to `b1`. The training layer performs explicit gradient-key mapping before calling the optimizer.

The optimizer receives matching parameter and gradient keys, validates that contract, and then applies the update tensor by tensor. This avoids hidden naming conventions and keeps the optimizer reusable for future models whose parameter names may not look like MLP layer names.
## 18. Open questions

- How should learning-rate schedules interact with SGD, Momentum, and Adam?
- How do optimizer conclusions change when moving from convex logistic regression to a non-convex MLP objective?
- When does lower BCE reflect better confidence quality, and when is explicit calibration analysis still needed?
- What changes when optimizer comparisons control update count, processed-sample count, wall-clock time, or hyperparameter-search budget separately?

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Next: Gradient, Risk, and Sampling →](02_gradient_risk_and_sampling.md)
