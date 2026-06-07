# Week 3 Optimization and MLP Notes

Current Week 3 scope: SGD, Momentum, Adam, mini-batch iteration, optimizer comparison, and the MLP forward pass are complete.

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

## 3. Mini-batch iterator implementation

`iterate_minibatches()` belongs to the data/preprocessing layer rather than the optimizer layer. It creates sample indices with `np.arange(n_samples)`, optionally shuffles those indices, and slices the index array into batches instead of modifying `X` and `y` in place. Indexing both arrays with the same `batch_indices` keeps features and labels aligned.

The final batch may contain fewer than `batch_size` examples. In the optimizer comparison, the training loop passes `seed + epoch`, so each epoch has a different but reproducible order. The same iterator can be reused by SGD, Momentum, Adam, and future MLP training because it only controls data access, not parameter-update logic.

## 4. Momentum optimizer

I implemented `Momentum` in `src/optimization/momentum.py` and added unit tests in `tests/test_momentum.py`. Unlike SGD, Momentum is a stateful optimizer. It stores `velocity_weights` and `velocity_bias`, which summarize recent gradient directions.

The update rule is:

```math
v_t = \beta v_{t-1} + (1 - \beta)g_t
```

```math
\theta_t = \theta_{t-1} - \alpha v_t
```

where $`g_t`$ is the current gradient, $`v_t`$ is the exponentially weighted moving average of recent gradients, $`\alpha`$ is the learning rate, and $`\beta`$ controls how strongly historical gradients are retained.

## 5. Why Momentum reduces oscillation

Momentum helps reduce oscillation because gradients that repeatedly change sign partially cancel each other in the moving average, while gradients that point in a stable direction accumulate. This is useful in narrow optimization valleys, where gradients may oscillate strongly across the steep direction while remaining consistent along the direction toward the minimum.

If the gradient direction remains stable, velocity gradually increases toward that direction. If gradients alternate between positive and negative values, the moving average becomes smaller than the raw gradients. This creates smoother and more stable parameter updates.

## 6. Interpreting beta

The coefficient $`\beta`$ controls the effective memory length of Momentum. A useful approximation is:

```math
\mathrm{effective\ memory\ length} \approx \frac{1}{1 - \beta}
```

For example:

- $`\beta = 0.0`$ behaves like ordinary SGD
- $`\beta = 0.9`$ remembers roughly 10 recent steps
- $`\beta = 0.99`$ remembers roughly 100 recent steps

A larger $`\beta`$ creates smoother but slower-reacting updates. A smaller $`\beta`$ responds more strongly to current gradients but performs less smoothing. The common default $`\beta = 0.9`$ is a practical compromise rather than a universal mathematical optimum.

## 7. Adam optimizer overview

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

## 8. Why Adam tracks the first moment

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

## 9. Why Adam tracks the second moment

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

## 10. Why Adam divides by $`\sqrt{v_t}`$

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

## 11. Why Adam needs bias correction

Adam initializes $`m_0 = 0`$ and $`v_0 = 0`$. Early moving averages are therefore biased toward zero because they have not yet accumulated enough history.

For example, on the first step:

```math
m_1 = \beta_1 m_0 + (1 - \beta_1)g_1
= (1 - \beta_1)g_1
```

With $`\beta_1 = 0.9`$, this gives only $`10\%`$ of the first gradient. Without correction, the early first-moment estimate would be too small. The same issue is stronger for the second moment when $`\beta_2 = 0.999`$, because the first $`v_t`$ value contains only $`0.1\%`$ of $`g_t^2`$.

Bias correction divides out the missing weight:

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

This makes the early moment estimates comparable to the gradient statistics they are trying to estimate.

## 12. Deriving the expected first and second moments

Assume gradients are sampled from a stationary distribution. Let:

```math
\mu = \mathbb{E}[g_t]
```

and:

```math
\nu = \mathbb{E}[g_t^2]
```

For the first moment, start with:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

Expand recursively from $`m_0 = 0`$:

```math
m_t =
(1 - \beta_1)g_t
+ \beta_1(1 - \beta_1)g_{t-1}
+ \beta_1^2(1 - \beta_1)g_{t-2}
+ \cdots
+ \beta_1^{t-1}(1 - \beta_1)g_1
```

Taking expectation:

```math
\mathbb{E}[m_t]
=
(1 - \beta_1)\mathbb{E}[g_t]
+ \beta_1(1 - \beta_1)\mathbb{E}[g_{t-1}]
+ \cdots
+ \beta_1^{t-1}(1 - \beta_1)\mathbb{E}[g_1]
```

Using $`\mathbb{E}[g_i] = \mu`$:

```math
\mathbb{E}[m_t]
=
(1 - \beta_1)\mu
\left(1 + \beta_1 + \beta_1^2 + \cdots + \beta_1^{t-1}\right)
```

The finite geometric sum is:

```math
1 + \beta_1 + \beta_1^2 + \cdots + \beta_1^{t-1}
=
\frac{1 - \beta_1^t}{1 - \beta_1}
```

Therefore:

```math
\mathbb{E}[m_t]
=
(1 - \beta_1)\mu
\left(\frac{1 - \beta_1^t}{1 - \beta_1}\right)
=
(1 - \beta_1^t)\mu
```

For the second moment:

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

With $`v_0 = 0`$, recursive expansion gives:

```math
v_t =
(1 - \beta_2)g_t^2
+ \beta_2(1 - \beta_2)g_{t-1}^2
+ \cdots
+ \beta_2^{t-1}(1 - \beta_2)g_1^2
```

Taking expectation and using $`\mathbb{E}[g_i^2] = \nu`$:

```math
\mathbb{E}[v_t]
=
(1 - \beta_2)\nu
\left(1 + \beta_2 + \beta_2^2 + \cdots + \beta_2^{t-1}\right)
```

Using the geometric sum:

```math
\mathbb{E}[v_t]
=
(1 - \beta_2)\nu
\left(\frac{1 - \beta_2^t}{1 - \beta_2}\right)
=
(1 - \beta_2^t)\nu
```

So the uncorrected estimates are biased low by factors $`1 - \beta_1^t`$ and $`1 - \beta_2^t`$.

## 13. Interpreting the bias-corrected estimates

The corrected first moment is:

```math
\hat{m}_t
=
\frac{m_t}
{1-\beta_1^t}
```

Since $`\mathbb{E}[m_t] = (1 - \beta_1^t)\mu`$, the corrected estimate has expectation:

```math
\mathbb{E}[\hat{m}_t]
=
\mathbb{E}\left[\frac{m_t}{1 - \beta_1^t}\right]
=
\frac{\mathbb{E}[m_t]}{1 - \beta_1^t}
=
\mu
```

Similarly:

```math
\hat{v}_t
=
\frac{v_t}
{1-\beta_2^t}
```

and since $`\mathbb{E}[v_t] = (1 - \beta_2^t)\nu`$:

```math
\mathbb{E}[\hat{v}_t]
=
\nu
```

The correction does not remove all noise from stochastic gradients. It specifically corrects the initialization bias caused by starting the moving averages at zero.

In neural-network training, gradients change as parameters change, and data batches introduce stochastic variation. Therefore, the stationary-moment assumption is an analytical simplification. Bias correction fixes the early zero-initialization shrinkage, but it does not make Adam universally unbiased throughout a non-stationary optimization process.

## 14. Physical intuition and mathematical boundaries

The squared gradient $`g_t^2`$ can be interpreted as a gradient energy scale or signal power scale because it measures magnitude without regard to sign. In signal processing, squaring a signal is a standard way to measure power-like magnitude. In optimization, $`g_t^2`$ similarly measures how large the local gradient signal is for each parameter.

This interpretation is a mathematically supported metaphor, not literal physical energy. Adam's second moment does not store mechanical energy. It stores an exponential moving average of squared gradients:

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
```

The useful point is that $`v_t`$ captures a recent squared-gradient scale. It can be described as energy-like or power-like because it is quadratic in the gradient, but the rigorous object is still a statistical moment estimate.

Momentum has a closer connection to physical dynamics. Polyak's heavy-ball method is motivated by damped second-order motion. In continuous time, a common idealized form is:

```math
\frac{d^2\theta}{dt^2}
+
\gamma
\frac{d\theta}{dt}
+
\nabla L(\theta)
=
0
```

Here $`\theta`$ is the position in parameter space, $`\frac{d\theta}{dt}`$ is velocity, $`\frac{d^2\theta}{dt^2}`$ is acceleration, and $`\gamma`$ is a damping coefficient. The loss $`L(\theta)`$ can be interpreted as a potential surface. The negative gradient $`-\nabla L(\theta)`$ acts like the downhill force induced by that potential surface, while $`\nabla L(\theta)`$ points toward local steepest increase.

This physical picture is useful, but it has boundaries. Different momentum methods, discretizations, learning-rate schedules, and stochastic gradient methods do not all correspond to exactly the same ordinary differential equation. The continuous-time equation is an idealized model that helps explain acceleration and damping, not a unique exact description of every optimizer implementation.

For Adam, the most rigorous description is adaptive diagonal preconditioning. Define:

```math
D_t
=
\mathrm{diag}
\left(
\frac{1}
{\sqrt{\hat{v}_t}+\epsilon}
\right)
```

Then the Adam update can be written as:

```math
\theta_t =
\theta_{t-1}
-
\eta D_t \hat{m}_t
```

This says that Adam rescales each coordinate of the bias-corrected first moment by a diagonal matrix built from the bias-corrected second moment. Physical analogies such as variable friction or variable effective mass can be useful for intuition, because parameters with larger recent squared gradients receive smaller effective steps. But those analogies are not the unique rigorous interpretation, and Adam should not be treated as strictly equivalent to one specific variable-mass mechanical system.

## 15. First-step numerical example

Use a single scalar gradient:

```math
g_1 = 0.1
```

with:

```math
\beta_1 = 0.9,\quad
\beta_2 = 0.999,\quad
\alpha = 0.1
```

Assume $`m_0 = 0`$, $`v_0 = 0`$, and $`\epsilon = 10^{-8}`$.

First moment:

```math
m_1 = 0.9(0) + (1 - 0.9)(0.1) = 0.01
```

Second moment:

```math
v_1 = 0.999(0) + (1 - 0.999)(0.1^2)
= 0.001(0.01)
= 0.00001
```

Bias correction:

```math
\hat{m}_1 =
\frac{0.01}{1 - 0.9^1}
=
\frac{0.01}{0.1}
=
0.1
```

```math
\hat{v}_1 =
\frac{0.00001}{1 - 0.999^1}
=
\frac{0.00001}{0.001}
=
0.01
```

The update amount is:

```math
\alpha\frac{\hat{m}_1}{\sqrt{\hat{v}_1} + \epsilon}
=
0.1 \frac{0.1}{\sqrt{0.01} + 10^{-8}}
\approx
0.1
```

So for a scalar parameter $`\theta_0`$:

```math
\theta_1 \approx \theta_0 - 0.1
```

The first Adam step behaves like a normalized sign step for this scalar example because $`\hat{m}_1 = g_1`$ and $`\sqrt{\hat{v}_1} = |g_1|`$.

## 16. Why epsilon is included

The $`\epsilon`$ term prevents division by zero or division by an extremely small number:

```math
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

If a parameter has very small recent gradients, $`\sqrt{\hat{v}_t}`$ can be close to zero. Without $`\epsilon`$, the effective learning rate could become numerically unstable. With $`\epsilon`$, the denominator is always positive.

In most ordinary updates, $`\epsilon`$ is much smaller than $`\sqrt{\hat{v}_t}`$, so it has little effect. It mainly protects edge cases and floating-point stability.

## 17. Adam implementation mapping

I implemented `Adam` in `src/optimization/adam.py` and added unit tests in `tests/test_adam.py`. The implementation state mirrors the Adam equations:

- `first_moment_weights` stores $`m_t`$ for every weight parameter.
- `second_moment_weights` stores $`v_t`$ for every weight parameter.
- `first_moment_bias` stores the scalar first moment for the bias gradient.
- `second_moment_bias` stores the scalar second moment for the bias gradient.
- `time_step` stores $`t`$, which is required for the bias-correction factors $`1 - \beta_1^t`$ and $`1 - \beta_2^t`$.

The implementation performs the following sequence during every call to `step()`:

1. initialize moment state arrays on the first update using arrays shaped like `weights`
2. increment the time step
3. update first-moment estimates
4. update second-moment estimates
5. compute bias-corrected moments
6. divide the corrected first moment by the square root of the corrected second moment plus epsilon
7. return updated parameters without mutating the original arrays in place

The first moment estimates the smoothed gradient direction. The second moment estimates the typical squared-gradient scale for each parameter. Dividing by the square root of the second moment gives parameter-wise adaptive scaling, while epsilon prevents numerical instability.

On the first call to `step()`, the moment arrays are initialized with zeros matching the shape of `weights`. Each later call reuses and updates the stored moments. This is why Adam is a stateful optimizer: repeated calls with the same current gradient can still produce different updates depending on accumulated moment history and `time_step`.

The implementation computes the weight and bias paths separately but with the same equations. The weight path uses NumPy elementwise operations, while the bias path uses scalar arithmetic. The returned `new_weights` should be a new array, and `new_bias` should be converted to a Python `float`.

## 18. Testing stateful optimizers

One-step tests are necessary but insufficient for stateful optimizers. A faulty optimizer may produce the correct first update while incorrectly resetting or misusing internal state later.

Adam tests should verify the transition from $`state_0`$ to $`state_1`$ to $`state_2`$. The strengthened second-step test checks `first_moment_weights`, `second_moment_weights`, `first_moment_bias`, `second_moment_bias`, `time_step`, updated weights, and updated bias. This protects against errors such as resetting moments each step, mixing `beta1` and `beta2`, omitting gradient squaring, or applying bias correction incorrectly.

## 19. Comparing SGD, Momentum, and Adam

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

## 20. Key intuition summary

Adam can be read as Momentum plus adaptive scaling. The first moment estimates the useful direction of movement. The second moment estimates the recent gradient scale. Bias correction fixes the early underestimation caused by zero initialization. Dividing by $`\sqrt{\hat{v}_t}`$ makes each parameter's update relative to its own recent gradient magnitude, while $`\epsilon`$ keeps the denominator numerically safe.

## 21. Teaching-oriented optimizer comparison

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

## 22. Optimizer comparison implementation logic

The optimizer comparison in `experiments/compare_optimizers.py` uses two training-loop patterns. `train_full_batch()` computes gradients using the entire training set and performs one parameter update per epoch. This matches Batch Gradient Descent: each epoch evaluates the current model on all training examples, computes one averaged gradient, and then calls the optimizer once.

`train_minibatch()` instead iterates over batches produced by `iterate_minibatches()`. Each mini-batch gives the loop an `X_batch` and `y_batch`. The model computes gradients on that mini-batch, producing a gradient estimate at the current parameter values, and the optimizer immediately performs one parameter update. As a result, mini-batch SGD, Momentum, and Adam perform multiple updates inside one epoch when the training set contains multiple mini-batches.

At the end of every epoch, both training loops compute binary cross-entropy using the entire training set. The plotted curves therefore use a common reporting metric: epoch-level full-training-set BCE. Every plotted loss is the BCE of the full training set under the model parameters at the end of that epoch. Raw mini-batch losses are useful for debugging training dynamics, but they are noisier and harder to compare directly because each value is computed on a different subset of examples.

Each optimizer comparison run gets a fresh `LogisticRegressionScratch` model initialized with zero weights and zero bias. This prevents one optimizer from benefiting from parameters learned by a previous optimizer. Momentum and Adam also receive fresh optimizer objects so that historical state does not leak between experiments. Momentum gets a new velocity state, and Adam gets new first-moment, second-moment, and time-step state.

The mini-batch iterator uses `seed + epoch` to create different but reproducible shuffling each epoch. The shuffle order changes from epoch to epoch, but rerunning the experiment with the same base seed produces the same sequence of epoch-level shuffles.

The current experiment boundary is important. All optimizers train for the same number of epochs, and each optimizer sees approximately the same total number of training examples. However, Batch Gradient Descent performs one update per epoch, while the mini-batch methods perform multiple parameter updates per epoch. This is a teaching-oriented same-epoch comparison, not a strict optimizer benchmark. It does not control update count, wall-clock time, hardware efficiency, or hyperparameter-search budget.

## 23. Loss, cost, empirical risk, and objective function

Terminology varies across textbooks, research papers, and software libraries. Some sources distinguish loss, cost, risk, and objective very strictly, while engineering code often uses `loss` for both per-example values and averaged training metrics. Therefore, it is better to define the quantities explicitly instead of assuming universal naming conventions.

A per-example loss measures model error for one observation. For one training example, it is commonly written as:

```math
\ell_i(\theta)
=
\ell
\left(
f(x_i;\theta),
y_i
\right)
```

Here $`\theta`$ represents the model parameters, $`x_i`$ is the feature vector for sample $`i`$, $`y_i`$ is the target label, and $`f(x_i;\theta)`$ is the model prediction. The scalar $`\ell_i(\theta)`$ measures prediction error for one sample.

The empirical risk is the average loss over the finite training set:

```math
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell_i(\theta)
```

Empirical risk is the average loss across the finite training dataset. Some textbooks call this averaged training quantity a cost function. Many engineering codebases also call this averaged quantity `loss`, especially when reporting metrics during training. Naming conventions vary, so the mathematical level matters more than the vocabulary.

With regularization, the optimized objective can be written as:

```math
F(\theta)
=
R_{\mathrm{emp}}(\theta)
+
\lambda
\Omega(\theta)
```

In this expression, $`\Omega(\theta)`$ is a regularization term and $`\lambda`$ controls regularization strength. The objective function $`F(\theta)`$ is the scalar quantity the optimization algorithm minimizes.

## 24. Why the negative gradient is a descent direction

For a small perturbation $`\Delta\theta`$, the first-order Taylor approximation around the current parameter point gives:

```math
F(\theta+\Delta\theta)
\approx
F(\theta)
+
\nabla F(\theta)^{\mathsf{T}}
\Delta\theta
```

The inner product $`\nabla F(\theta)^{\mathsf{T}}\Delta\theta`$ describes the local linear change in the objective. Under a fixed Euclidean step length, this quantity is largest when $`\Delta\theta`$ points in the same direction as $`\nabla F(\theta)`$. Therefore, the gradient points in the direction of steepest local increase under the Euclidean norm.

The negative gradient points in the opposite direction, so it gives the steepest local decrease under the same fixed step-length constraint. Gradient descent uses this direction and scales it by a learning rate:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\nabla F(\theta_t)
```

Here $`\nabla F(\theta_t)`$ determines direction and $`\eta`$ determines how far to move. In code, `learning_rate` is the implementation parameter corresponding to $`\eta`$. If $`\eta`$ is too large, the local linear approximation may become inaccurate, which can cause overshooting, oscillation, or divergence. If $`\eta`$ is too small, optimization may be stable but slow.

Gradient descent is not an arbitrary engineering heuristic. It follows from a local first-order approximation to the objective, although its practical behavior depends strongly on step-size selection and on how well the local approximation describes the objective over the chosen step.

## 25. From per-example gradients to full-batch gradients

Start from the empirical risk:

```math
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell_i(\theta)
```

Because differentiation is linear, the gradient of the empirical risk is the average of the per-example gradients:

```math
\nabla
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\nabla
\ell_i(\theta)
```

The full-batch gradient is the average of all per-example gradients. In Batch Gradient Descent, every per-example gradient is evaluated at the same parameter point $`\theta_t`$. The optimizer computes this exact empirical-risk gradient and then updates once:

```math
\theta_{t+1}
=
\theta_t
-
\eta
\nabla
R_{\mathrm{emp}}(\theta_t)
```

For classic SGD, one sampled example produces:

```math
G_t
=
\nabla
\ell_{I_t}(\theta_t)
```

and the model updates immediately:

```math
\theta_{t+1}
=
\theta_t
-
\eta
G_t
```

Sequential single-sample updates are not mathematically identical to one averaged full-batch update. In full-batch gradient descent, every per-example gradient is evaluated at the same parameter vector. In classic SGD, the next sample gradient is evaluated at a new parameter point after the previous update has already changed the parameters. The two methods therefore follow different optimization trajectories.

Conceptually:

- Batch Gradient Descent computes all sample gradients at one shared parameter point, averages them, and updates once.
- Classic SGD computes one sampled gradient, updates immediately, and then evaluates the next gradient at a changed parameter point.
- Mini-batch SGD averages gradients inside the current batch and updates once per batch.

## 26. Statistical meaning of stochastic gradients

SGD is not merely an engineering shortcut. It is a stochastic approximation method with a statistical interpretation. For a fixed training dataset:

```math
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell_i(\theta)
```

The full-batch gradient is:

```math
\nabla
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\nabla
\ell_i(\theta)
```

This is the exact gradient of the empirical risk for the fixed training dataset.

Now assume sample index $`I_t`$ is drawn uniformly at random. One sampled example produces:

```math
G_t
=
\nabla
\ell_{I_t}(\theta_t)
```

Then:

```math
\mathbb{E}
\left[
G_t
\mid
\theta_t
\right]
=
\nabla
R_{\mathrm{emp}}(\theta_t)
```

The expectation is conditional on the current parameter value $`\theta_t`$. This matters because $`\theta_t`$ is itself influenced by previous random updates. Conditioned on the current parameter point, the random sample gradient is an unbiased estimator of the empirical-risk gradient under the uniform-sampling assumption.

For a mini-batch $`\mathcal{B}_t`$, the mini-batch gradient is:

```math
G_t^{(\mathcal{B})}
=
\frac{1}
{\lvert
\mathcal{B}_t
\rvert}
\sum_{i\in\mathcal{B}_t}
\nabla
\ell_i(\theta_t)
```

Under suitable random-sampling assumptions:

```math
\mathbb{E}
\left[
G_t^{(\mathcal{B})}
\mid
\theta_t
\right]
=
\nabla
R_{\mathrm{emp}}(\theta_t)
```

This means mini-batch gradients estimate the empirical-risk gradient. They introduce sampling variance:

```math
\mathrm{Var}
\left(
G_t^{(\mathcal{B})}
\mid
\theta_t
\right)
>
0
```

Smaller batches produce noisier estimates, but they allow more frequent updates and can make each update cheaper. Larger batches produce more stable gradient estimates, but each update is more expensive because it processes more examples before moving the parameters.

In a simplified independent-sampling setting, variance typically decreases as batch size grows:

```math
\mathrm{Var}
\left(
G_t^{(\mathcal{B})}
\right)
\propto
\frac{1}
{\lvert
\mathcal{B}_t
\rvert}
```

This proportionality is a simplified intuition, not a universal exact identity under every sampling strategy.

## 27. Two layers of statistical approximation

The true population-level goal is expected risk:

```math
R_{\mathrm{exp}}(\theta)
=
\mathbb{E}_{(X,Y)\sim P}
\left[
\ell
\left(
f(X;\theta),
Y
\right)
\right]
```

Here $`P(X,Y)`$ is the unknown real-world data distribution. Expected risk measures average model error over that unknown distribution. In practice, expected risk cannot usually be computed exactly, so training uses finite data.

The finite training set produces empirical risk:

```math
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell
\left(
f(x_i;\theta),
y_i
\right)
```

Empirical risk estimates expected risk from finite data. The quality of this approximation depends on the dataset and assumptions such as sampling representativeness.

There is a second approximation layer during mini-batch training:

```math
G_t^{(\mathcal{B})}
=
\frac{1}
{\lvert
\mathcal{B}_t
\rvert}
\sum_{i\in\mathcal{B}_t}
\nabla
\ell_i(\theta_t)
```

The mini-batch gradient estimates the empirical-risk gradient. This approximation reduces per-update computation cost, but it also introduces gradient noise.

The hierarchy is:

```text
unknown real-world distribution
-> expected risk
-> finite training dataset
-> empirical risk
-> random mini-batch
-> stochastic gradient estimate
-> optimizer update rule
-> new parameter state
```

Mini-batch gradients approximate empirical-risk gradients, and empirical risk approximates expected risk. The goal of training is not merely to reduce one mini-batch loss value. The broader goal is to find parameters that reduce empirical risk in a way that also generalizes to low expected risk on future data from the same underlying distribution.

## 28. Sampling with replacement versus random reshuffling

The clean introductory unbiased-gradient derivation often assumes that sample indices are drawn independently, uniformly, and with replacement. Under that simplified model, the next sampled index does not depend on which indices were sampled earlier.

The current repository uses `iterate_minibatches()` with `shuffle=True` and `seed=seed + epoch`. This means the training dataset is shuffled once per epoch, mini-batches are consumed without replacement inside that epoch, and every sample is covered once per epoch. The sample order changes across epochs, but the sequence remains reproducible because the seed is deterministic.

Random reshuffling without replacement is a common practical training strategy, but it is not mathematically identical to independent sampling with replacement. Later mini-batches in an epoch depend on which samples have already been consumed, so one should not blindly claim that every successive mini-batch is conditionally independent.

The simplified unbiased-gradient derivation remains a useful theoretical entry point because it explains why stochastic gradients can point in the correct direction on average under clear assumptions. The repository implementation should be described more precisely as reproducible random reshuffling without replacement.

## 29. Final conceptual summary

- Batch Gradient Descent uses the exact empirical-risk gradient.
- Classic SGD uses a stochastic estimate based on one sampled observation.
- Mini-batch SGD uses a stochastic estimate based on a subset of observations.
- Mini-batching is both statistically motivated and computationally useful.
- The gradient gives a local descent direction.
- The learning rate determines the step size.
- Empirical risk approximates expected risk.
- Mini-batch gradients approximate empirical-risk gradients.
- The repository uses reproducible random reshuffling without replacement.

## 30. One-hidden-layer MLP forward pass

`BinaryMLPScratch` implements a one-hidden-layer binary classifier. The forward pass computes:

```math
Z_1 = XW_1 + b_1
```

```math
A_1 = \mathrm{ReLU}(Z_1)
```

```math
Z_2 = A_1W_2 + b_2
```

```math
P = \sigma(Z_2)
```

| Variable | Shape | Meaning |
|---|---:|---|
| `X` | `(n_samples, n_features)` | input features |
| `W1` | `(n_features, hidden_dim)` | input-to-hidden weights |
| `b1` | `(hidden_dim,)` | hidden bias |
| `Z1` | `(n_samples, hidden_dim)` | hidden pre-activation |
| `A1` | `(n_samples, hidden_dim)` | hidden activation |
| `W2` | `(hidden_dim, 1)` | hidden-to-output weights |
| `b2` | `(1,)` | output bias |
| `Z2` | `(n_samples, 1)` | output logits |
| `probabilities` | `(n_samples,)` | positive-class probabilities |

NumPy broadcasts `b1` across all sample rows in the hidden layer and broadcasts `b2` across the output rows.

## 31. Why the hidden layer needs a nonlinear activation

Without ReLU, the two affine layers collapse into one affine transformation:

```math
Z_2
=
(XW_1+b_1)W_2+b_2
```

which can be rewritten as:

```math
Z_2
=
X(W_1W_2)
+
(b_1W_2+b_2)
```

With biases, the precise term is affine transformation rather than linear transformation. Stacking affine transformations without a nonlinear activation still produces only one equivalent affine transformation. ReLU breaks this collapsibility, which allows the MLP to represent nonlinear decision boundaries.

## 32. Stable sigmoid computation

The direct sigmoid formula is:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

For a very negative input, computing `exp(-z)` can overflow because $`-z`$ becomes very large and positive. The implementation uses an equivalent form for negative inputs:

```math
\sigma(z)=\frac{e^z}{1+e^z}
```

This avoids computing an enormous exponential value and improves numerical stability. The test with `0.0`, `1000.0`, and `-1000.0` verifies the behavior at ordinary, very positive, and very negative inputs.

## 33. Why forward pass returns a cache

The forward pass returns a cache containing `X`, `Z1`, `A1`, and `Z2`. These intermediate values are needed later during backpropagation: `A1` is needed to compute `dW2`, `Z1` is needed to compute the ReLU derivative, and `X` is needed to compute `dW1`.

Caching avoids recomputing the full forward pass during backpropagation. The trade-off is memory versus computation: storing intermediate arrays uses more memory, but it avoids redundant matrix multiplications and activation computations.

## 34. Why probabilities are reshaped

`Z2` has shape `(n_samples, 1)`, while labels usually have shape `(n_samples,)`. Calling `.reshape(-1)` converts output probabilities to shape `(n_samples,)`, aligning predictions with labels.

This also prevents accidental NumPy broadcasting from producing a `(n_samples, n_samples)` array during loss or gradient computation. Keeping probabilities and labels as matching one-dimensional arrays makes the binary-classification interface less error-prone.

## 35. Forward-pass testing strategy

The forward-pass tests cover parameter shapes, reproducible initialization, different initialization under different seeds, ReLU values, numerically stable sigmoid behavior, forward-pass shapes, manually verifiable forward-pass values, invalid dimensions, and invalid input shape.

Shape tests catch interface errors. Manual-value tests validate the complete mathematical path from input features through hidden activations to output probabilities. Stable-sigmoid tests catch numerical errors that would not necessarily appear in purely symbolic formula checks.

## 36. Further reading

- Adam: A Method for Stochastic Optimization
- Without-Replacement Sampling for Stochastic Gradient Methods: Convergence Results and Application to Distributed Optimization
- Why Random Reshuffling Beats Stochastic Gradient Descent

## 37. Binary MLP backpropagation roadmap

The backward pass for the binary MLP computes gradients in reverse order from the scalar batch loss back to the trainable parameters. For the one-hidden-layer network, the required intermediate and parameter gradients are:

- `dZ2`
- `dW2`
- `db2`
- `dA1`
- `dZ1`
- `dW1`
- `db1`

Each gradient is obtained by applying the chain rule from the binary cross-entropy loss backward through the sigmoid output activation, the output affine layer, the ReLU hidden activation, and the input affine layer. This reverse order mirrors the forward computation: first compute the output-layer error, then propagate it into the hidden layer, then compute gradients for the input-layer parameters.

## 38. Why sigmoid and BCE simplify to P minus y

For one training example, the binary cross-entropy loss is:

```math
\ell_i
=
-
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right]
```

The predicted probability is the sigmoid of the output logit:

```math
p_i
=
\sigma(z_{2,i})
```

Differentiate the loss with respect to the probability:

```math
\frac{\partial \ell_i}
{\partial p_i}
=
-
\left[
\frac{y_i}{p_i}
-
\frac{1-y_i}{1-p_i}
\right]
```

Putting the terms over a common denominator gives:

```math
\frac{\partial \ell_i}
{\partial p_i}
=
\frac{p_i-y_i}
{p_i(1-p_i)}
```

The sigmoid derivative is:

```math
\frac{\partial p_i}
{\partial z_{2,i}}
=
p_i(1-p_i)
```

By the chain rule:

```math
\frac{\partial \ell_i}
{\partial z_{2,i}}
=
\frac{\partial \ell_i}
{\partial p_i}
\frac{\partial p_i}
{\partial z_{2,i}}
```

Substituting the two factors cancels `p_i(1-p_i)`:

```math
\frac{\partial \ell_i}
{\partial z_{2,i}}
=
\frac{p_i-y_i}
{p_i(1-p_i)}
p_i(1-p_i)
```

Therefore:

```math
\frac{\partial \ell_i}
{\partial z_{2,i}}
=
p_i-y_i
```

For batch-average binary cross-entropy, the factor `1 / n` enters at the output logit gradient:

```math
dZ_2
=
\frac{P-y}{n}
```

The intuition is direct: if the predicted probability is too high, `P - y` is positive and gradient descent pushes the logit downward; if the predicted probability is too low, `P - y` is negative and gradient descent pushes the logit upward; if the prediction is close to the target, the gradient is small.

## 39. Output-layer gradients

The output affine layer is:

```math
Z_2
=
A_1W_2+b_2
```

Since each output logit is a weighted sum of hidden activations, the weight gradient is the hidden activation matrix transposed times the output-logit gradient:

```math
dW_2
=
A_1^{\mathsf{T}}dZ_2
```

The bias is shared across all samples in the batch, so every sample contributes to the same bias parameter. Therefore, the bias gradient sums the output-logit gradients across the sample dimension:

```math
db_2
=
\sum_{i=1}^{n}
dZ_{2,i}
```

| Gradient |             Shape |
| -------- | ----------------: |
| `dZ2`    |  `(n_samples, 1)` |
| `dW2`    | `(hidden_dim, 1)` |
| `db2`    |            `(1,)` |

## 40. Backpropagating through ReLU

The output-layer affine operation also sends gradient back into the hidden activations:

```math
dA_1
=
dZ_2W_2^{\mathsf{T}}
```

The ReLU derivative is `1` when the input is greater than zero. In this implementation, the derivative is `0` when the input is less than or equal to zero.

The hidden pre-activation gradient is:

```math
dZ_1
=
dA_1
\odot
\mathbf{1}[Z_1>0]
```

Here, `\odot` means elementwise multiplication. ReLU blocks gradient flow where the forward pre-activation is non-positive. The cached `Z1` value from the forward pass is needed to construct this mask, because the mask depends on the hidden pre-activation values seen during that same forward pass.

## 41. Input-layer gradients

The input affine layer is:

```math
Z_1
=
XW_1+b_1
```

By the same affine-layer gradient rule used for the output layer, the input-layer weight gradient is:

```math
dW_1
=
X^{\mathsf{T}}dZ_1
```

The hidden bias is shared across all samples, so its gradient sums the hidden pre-activation gradients across the sample dimension:

```math
db_1
=
\sum_{i=1}^{n}
dZ_{1,i}
```

| Gradient |                      Shape |
| -------- | -------------------------: |
| `dA1`    |  `(n_samples, hidden_dim)` |
| `dZ1`    |  `(n_samples, hidden_dim)` |
| `dW1`    | `(n_features, hidden_dim)` |
| `db1`    |            `(hidden_dim,)` |

## 42. Why divide by batch size only once

The training objective is batch-average binary cross-entropy, so the average over samples contributes one factor of `1 / n`. That factor enters through the output-logit gradient:

```math
dZ_2
=
\frac{P-y}{n}
```

All later gradients inherit this factor through the chain rule. For example, `dW2`, `db2`, `dA1`, `dZ1`, `dW1`, and `db1` are all computed from values that already include the batch-average scaling. Dividing again at each layer would incorrectly shrink the gradients multiple times and would no longer match the derivative of the stated batch-average objective.

## 43. Cache-to-gradient mapping

| Cached variable | Backward use                                  |
| --------------- | --------------------------------------------- |
| `X`             | compute `dW1 = X.T @ dZ1`                     |
| `Z1`            | construct the ReLU mask                       |
| `A1`            | compute `dW2 = A1.T @ dZ2`                    |
| `Z2`            | inspect logits or recompute sigmoid if needed |

Caching reduces repeated computation at the cost of additional memory. The cached values preserve the exact forward-pass intermediates needed by the backward pass, avoiding redundant matrix multiplications and activation evaluations.

## 44. Backpropagation shape invariants

| Parameter |            Parameter shape | Gradient |             Gradient shape |
| --------- | -------------------------: | -------- | -------------------------: |
| `W1`      | `(n_features, hidden_dim)` | `dW1`    | `(n_features, hidden_dim)` |
| `b1`      |            `(hidden_dim,)` | `db1`    |            `(hidden_dim,)` |
| `W2`      |          `(hidden_dim, 1)` | `dW2`    |          `(hidden_dim, 1)` |
| `b2`      |                     `(1,)` | `db2`    |                     `(1,)` |

Each parameter gradient must have exactly the same shape as its corresponding parameter. This invariant is what allows the optimizer to update parameters element by element without broadcasting mistakes or shape-dependent special cases.

## 45. Why sigmoid and BCE produce a logit-space residual

The simplification:

```math
\frac{\partial \ell_i}
{\partial z_i}
=
p_i-y_i
```

is not merely an accidental algebraic cancellation. It reflects a clean match between the Bernoulli likelihood model, the binary cross-entropy loss, and the sigmoid link from logits to probabilities.

The output layer first produces a real-valued logit, then sigmoid maps that logit into probability space:

```text
logit space
z in the real numbers
-> sigmoid
probability space
p in the interval (0, 1)
```

The logit corresponding to a probability is:

```math
z
=
\log
\frac{p}
{1-p}
```

and the sigmoid maps back from logit space to probability space:

```math
p
=
\sigma(z)
=
\frac{1}
{1+e^{-z}}
```

Binary cross-entropy is the negative log-likelihood of a Bernoulli model. For a binary target, the Bernoulli probability assigned to the observed outcome is high when the model assigns high probability to the observed class and low otherwise. Taking the negative logarithm turns high assigned probability into low loss and low assigned probability into high loss.

After substituting the sigmoid expression into binary cross-entropy, the per-example loss can be written directly in logit space:

```math
\ell(z,y)
=
\log(1+e^z)
-
yz
```

Differentiating this logit-space loss gives:

```math
\frac{\partial \ell}
{\partial z}
=
\frac{e^z}{1+e^z}
-
y
```

Since:

```math
\frac{e^z}{1+e^z}
=
\frac{1}
{1+e^{-z}}
=
\sigma(z)
```

the derivative is:

```math
\frac{\partial \ell}
{\partial z}
=
\sigma(z)-y
=
p-y
```

The deeper intuition is that `p` is the model-implied expected positive-class value, while `y` is the observed binary target. Their difference, `p - y`, is the residual in logit space. Binary cross-entropy has strong sensitivity near the edges of probability space, while sigmoid compresses logits into the interval `(0, 1)`. When the gradient is pulled back into logit space, these two effects cancel, leaving a stable and interpretable residual signal.

For one sample, `p_i - y_i` is a per-example residual. For a batch-average objective, the output-logit gradient is:

```math
dZ_2
=
\frac{P-y}{n}
```

The final parameter gradient then aggregates these residual contributions across samples.

## 46. Why output-layer gradients use `A1.T @ dZ2`

The output affine layer is:

```math
Z_2
=
A_1W_2+b_2
```

For one sample:

```math
z_{2,i}
=
a_{1,i}^{\mathsf{T}}W_2+b_2
```

Each output-layer weight scales one hidden activation. Therefore, for output-layer weight coordinate `j`:

```math
\frac{\partial z_{2,i}}
{\partial W_{2,j}}
=
a_{1,i,j}
```

Using the chain rule across the batch:

```math
\frac{\partial L}
{\partial W_{2,j}}
=
\sum_{i=1}^{n}
dZ_{2,i}
a_{1,i,j}
```

Stacking these coordinate-wise derivatives gives the matrix form:

```math
dW_2
=
A_1^{\mathsf{T}}dZ_2
```

There are three complementary ways to read this formula.

First, the shapes line up with the parameter. `A1.T` has shape `(hidden_dim, n_samples)`, `dZ2` has shape `(n_samples, 1)`, and the result has shape `(hidden_dim, 1)`, which matches `W2`.

Second, each sample contributes its hidden activation vector multiplied by its output residual. The batch gradient sums these per-example contributions. If a sample has a larger output residual, its hidden activation pattern contributes more strongly to the update direction.

Third, the transpose has a linear-map meaning. In the forward pass, parameter values are mapped through `A1 @ W2` into output-logit space. In the backward pass, the transpose map pulls output-space error back into parameter space. The transpose is therefore not merely a shape trick; it is the adjoint direction required by the chain rule for this linear map.

Using the opposite multiplication order gives the transposed orientation:

```math
dZ_2^{\mathsf{T}}A_1
=
dW_2^{\mathsf{T}}
```

That expression contains the same scalar products, but it produces `(1, hidden_dim)` rather than the `(hidden_dim, 1)` shape of `W2`.

## 47. Why cache `Z1` for ReLU backward

The hidden activation is:

```math
A_1
=
\mathrm{ReLU}(Z_1)
```

The ReLU backward step is:

```math
dZ_1
=
dA_1
\odot
\mathbf{1}[Z_1>0]
```

`Z1` is the hidden-layer pre-activation. It determines which forward paths were active: non-positive pre-activations block gradient flow, while positive pre-activations allow gradient flow.

There is an important nuance for basic ReLU. If the derivative at exactly `Z1 = 0` is defined as zero, then the masks `Z1 > 0` and `A1 > 0` are equivalent for this specific derivative calculation. In that narrow case, `A1` could technically be sufficient.

Caching `Z1` is still preferable because it preserves more information than `A1`. It supports debugging, activation-distribution analysis, dead-neuron diagnosis, and future replacement with other activation functions whose backward pass may depend directly on pre-activation values. Caching pre-activations also mirrors the chain-rule structure directly: the backward pass through an activation uses the derivative of the activation with respect to its own input.

## 48. Open questions

- How does batch size quantitatively affect gradient variance and optimization speed?
- How should learning-rate schedules interact with SGD, Momentum, and Adam?
- How should the optimizer API be generalized from one weight vector and one scalar bias to multiple MLP parameter tensors?
- How can numerical gradient checking validate an MLP backpropagation implementation?
- How do optimizer conclusions change when moving from convex logistic regression to a non-convex MLP objective?
- When does lower BCE reflect better confidence quality, and when is explicit calibration analysis still needed?
