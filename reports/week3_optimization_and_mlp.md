# Week 3 Optimization and MLP Notes

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

## 7. Why Adam tracks the first moment

The first moment is the expected gradient direction. In Adam, $`m_t`$ plays a role similar to Momentum's velocity:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
```

This is an exponentially weighted moving average of recent gradients. If the gradient direction is consistent across steps, the average points strongly in that direction. If the gradient direction alternates, positive and negative gradient components partially cancel.

This makes $`m_t`$ a smoothed estimate of descent direction. It reduces the impact of noisy mini-batch gradients while still allowing the optimizer to accumulate evidence about directions that are repeatedly useful.

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

## 9. Why Adam divides by $`\sqrt{v_t}`$

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

The effective learning rate for each parameter is:

```math
\alpha_{\mathrm{effective}, t}
=
\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}
```

Large recent squared gradients make $`\sqrt{\hat{v}_t}`$ larger, which reduces the effective learning rate. Small recent squared gradients make the denominator smaller, which allows a larger effective learning rate.

## 10. Why Adam needs bias correction

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

## 11. Deriving the expected first and second moments

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

## 12. Interpreting the bias-corrected estimates

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

## 13. Physical intuition and mathematical boundaries

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

Here $`\theta`$ is the position in parameter space, $`\frac{d\theta}{dt}`$ is velocity, $`\frac{d^2\theta}{dt^2}`$ is acceleration, $`\gamma`$ is a damping coefficient, and $`\nabla L(\theta)`$ acts like a force pushing downhill on the loss. The loss $`L(\theta)`$ can be interpreted as a potential surface because the negative gradient $`-\nabla L(\theta)`$ points in the direction of steepest decrease, analogous to force derived from a potential.

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

## 14. What bias correction actually corrects

An estimator is unbiased for a target quantity when its expectation equals that target:

```math
\mathbb{E}[\mathrm{estimate}] = \mathrm{target}
```

Unbiased estimation is a general statistical idea, not something invented specifically for Adam. Adam uses this idea because its moving-average estimates start from zero. Zero initialization is natural: before training begins, the optimizer has no gradient history from which to initialize $`m_0`$ or $`v_0`$. However, this also means the early exponential moving averages are systematically too small.

Assume, as an analytical simplification, that gradients are sampled from a stationary process with:

```math
\mathbb{E}[g_t]
=
\mu
```

and:

```math
\mathbb{E}[g_t^2]
=
\nu
```

Starting from $`m_0 = 0`$, the first moment is:

```math
m_t =
(1 - \beta_1)
\sum_{i=1}^{t}
\beta_1^{t-i} g_i
```

Taking expectation:

```math
\mathbb{E}[m_t]
=
(1 - \beta_1)
\sum_{i=1}^{t}
\beta_1^{t-i} \mathbb{E}[g_i]
```

Using:

```math
\mathbb{E}[g_i]
=
\mu
```

This gives:

```math
\mathbb{E}[m_t]
=
(1-\beta_1^t)\mu
```

Similarly, starting from $`v_0 = 0`$:

```math
v_t =
(1 - \beta_2)
\sum_{i=1}^{t}
\beta_2^{t-i} g_i^2
```

Taking expectation and using $`\mathbb{E}[g_i^2] = \nu`$:

```math
\mathbb{E}[v_t]
=
(1-\beta_2^t)\nu
```

The factors $`1 - \beta_1^t`$ and $`1 - \beta_2^t`$ are less than one during the early steps, so the raw estimates are biased toward zero. Adam corrects this initialization bias with:

```math
\hat{m}_t =
\frac{m_t}
{1-\beta_1^t}
```

and:

```math
\hat{v}_t =
\frac{v_t}
{1-\beta_2^t}
```

Under the stationary-moment assumption, these corrected estimates have expectations $`\mu`$ and $`\nu`$. This does not mean Adam removes every possible source of statistical bias. It corrects the specific bias introduced by initializing the exponential moving averages at zero.

In neural-network training, gradients change as parameters change, and the data batches introduce stochastic variation. Therefore, the stationary-moment assumption is an analytical simplification. Bias correction is still useful because it fixes a real early-step underestimation problem, but it does not make Adam universally unbiased throughout a non-stationary optimization process.

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

I implemented `Adam` in `src/optimization/adam.py` and added unit tests in `tests/test_adam.py`. The optimizer stores:

- `first_moment_weights`
- `second_moment_weights`
- `first_moment_bias`
- `second_moment_bias`
- `time_step`

The implementation performs the following sequence during every call to `step()`:

1. initialize moment state arrays on the first update
2. increment the time step
3. update first-moment estimates
4. update second-moment estimates
5. compute bias-corrected moments
6. divide the corrected first moment by the square root of the corrected second moment plus epsilon
7. return updated parameters without mutating the original arrays in place

The first moment estimates the smoothed gradient direction. The second moment estimates the typical squared-gradient scale for each parameter. Dividing by the square root of the second moment gives parameter-wise adaptive scaling, while epsilon prevents numerical instability.

## 18. Why Adam needs bias correction

Adam needs bias correction because its first-moment and second-moment estimates are initialized at zero. During the first few optimization steps, the exponential moving averages are systematically biased toward zero, especially when `beta1` and `beta2` are close to one.

For a roughly stationary expected gradient:

```math
\mathbb{E}[m_t]
=
(1 - \beta_1^t)\mu
```

and for the expected squared gradient:

```math
\mathbb{E}[v_t]
=
(1 - \beta_2^t)\nu
```

Therefore, Adam corrects the estimates using:

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

Without this correction, the effective update scale during the early training stage would be distorted by zero initialization.

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

## 20. Mapping the math to implementation state

The implementation state mirrors the Adam equations:

- `first_moment_weights` stores $`m_t`$ for every weight parameter.
- `second_moment_weights` stores $`v_t`$ for every weight parameter.
- `first_moment_bias` stores the scalar first moment for the bias gradient.
- `second_moment_bias` stores the scalar second moment for the bias gradient.
- `time_step` stores $`t`$, which is required for the bias-correction factors $`1 - \beta_1^t`$ and $`1 - \beta_2^t`$.

On the first call to `step()`, the moment arrays are initialized with zeros matching the shape of `weights`. Each later call reuses and updates the stored moments. This is why Adam is a stateful optimizer: two calls with the same current gradient can produce different updates depending on the accumulated moment history and `time_step`.

The implementation should compute the weight and bias paths separately but with the same equations. The weight path uses NumPy elementwise operations, while the bias path uses scalar arithmetic. The returned `new_weights` should be a new array, and `new_bias` should be converted to a Python `float`.

## 21. Key intuition summary

Adam can be read as Momentum plus adaptive scaling. The first moment estimates the useful direction of movement. The second moment estimates the recent gradient scale. Bias correction fixes the early underestimation caused by zero initialization. Dividing by $`\sqrt{\hat{v}_t}`$ makes each parameter's update relative to its own recent gradient magnitude, while $`\epsilon`$ keeps the denominator numerically safe.

## 22. Teaching-oriented optimizer comparison

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

Mini-batch SGD lowers BCE faster than full-batch gradient descent because it updates parameters more frequently during each pass through the dataset. Momentum performs only slightly better than SGD in this simple convex problem, where strong oscillation is limited. Adam achieves the lowest BCE under the current configuration because its adaptive parameter-wise scaling improves probability estimates.

The validation accuracies of SGD, Momentum, and Adam are identical even though their BCE values differ. This illustrates why BCE provides more information than accuracy: accuracy only reflects thresholded class decisions, while BCE also reflects confidence quality.

These results should not be generalized into a universal optimizer ranking. The experiment uses a simple linearly separable synthetic dataset, one random seed, and different optimizer hyperparameters. Its purpose is to build optimization intuition before moving to MLP training.

## 23. Optimizer comparison implementation logic

The optimizer comparison in `experiments/compare_optimizers.py` uses two training-loop patterns. `train_full_batch()` computes gradients using the entire training set and performs one parameter update per epoch. This matches Batch Gradient Descent: each epoch evaluates the current model on all training examples, computes one averaged gradient, and then calls the optimizer once.

`train_minibatch()` instead iterates over batches produced by `iterate_minibatches()`. Each mini-batch gives the loop an `X_batch` and `y_batch`. The model computes gradients on that mini-batch, producing a gradient estimate at the current parameter values, and the optimizer immediately performs one parameter update. As a result, mini-batch SGD, Momentum, and Adam perform multiple updates inside one epoch when the training set contains multiple mini-batches.

At the end of every epoch, both training loops compute binary cross-entropy using the entire training set. The plotted curves therefore use a common reporting metric: epoch-level full-training-set BCE. Every plotted loss is the BCE of the full training set under the model parameters at the end of that epoch. Raw mini-batch losses are useful for debugging training dynamics, but they are noisier and harder to compare directly because each value is computed on a different subset of examples.

Each optimizer comparison run gets a fresh `LogisticRegressionScratch` model initialized with zero weights and zero bias. This prevents one optimizer from benefiting from parameters learned by a previous optimizer. Momentum and Adam also receive fresh optimizer objects so that historical state does not leak between experiments. Momentum gets a new velocity state, and Adam gets new first-moment, second-moment, and time-step state.

The mini-batch iterator uses `seed + epoch` to create different but reproducible shuffling each epoch. The shuffle order changes from epoch to epoch, but rerunning the experiment with the same base seed produces the same sequence of epoch-level shuffles.

The current experiment boundary is important. All optimizers train for the same number of epochs, and each optimizer sees approximately the same total number of training examples. However, Batch Gradient Descent performs one update per epoch, while the mini-batch methods perform multiple parameter updates per epoch. This is a teaching-oriented same-epoch comparison, not a strict optimizer benchmark. It does not control update count, wall-clock time, hardware efficiency, or hyperparameter-search budget.

## 24. Loss, cost, empirical risk, and objective function

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

## 25. Why the negative gradient is a descent direction

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

## 26. From per-example gradients to full-batch gradients

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

## 27. Statistical meaning of stochastic gradients

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

## 28. Two layers of statistical approximation

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

## 29. Sampling with replacement versus random reshuffling

The clean introductory unbiased-gradient derivation often assumes that sample indices are drawn independently, uniformly, and with replacement. Under that simplified model, the next sampled index does not depend on which indices were sampled earlier.

The current repository uses `iterate_minibatches()` with `shuffle=True` and `seed=seed + epoch`. This means the training dataset is shuffled once per epoch, mini-batches are consumed without replacement inside that epoch, and every sample is covered once per epoch. The sample order changes across epochs, but the sequence remains reproducible because the seed is deterministic.

Random reshuffling without replacement is a common practical training strategy, but it is not mathematically identical to independent sampling with replacement. Later mini-batches in an epoch depend on which samples have already been consumed, so one should not blindly claim that every successive mini-batch is conditionally independent.

The simplified unbiased-gradient derivation remains a useful theoretical entry point because it explains why stochastic gradients can point in the correct direction on average under clear assumptions. The repository implementation should be described more precisely as reproducible random reshuffling without replacement.

## 30. Final conceptual summary

- Batch Gradient Descent uses the exact empirical-risk gradient.
- Classic SGD uses a stochastic estimate based on one sampled observation.
- Mini-batch SGD uses a stochastic estimate based on a subset of observations.
- Mini-batching is both statistically motivated and computationally useful.
- The gradient gives a local descent direction.
- The learning rate determines the step size.
- Empirical risk approximates expected risk.
- Mini-batch gradients approximate empirical-risk gradients.
- The repository uses reproducible random reshuffling without replacement.

## 31. Open questions

- How does batch size affect gradient noise and convergence?
- Why can noisy SGD updates sometimes help optimization?
- How should Momentum store and update optimizer state?
- How does Adam combine first-moment and second-moment estimates?
