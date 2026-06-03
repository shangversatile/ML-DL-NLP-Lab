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

## 23. Open questions

- How does batch size affect gradient noise and convergence?
- Why can noisy SGD updates sometimes help optimization?
- How should Momentum store and update optimizer state?
- How does Adam combine first-moment and second-moment estimates?
