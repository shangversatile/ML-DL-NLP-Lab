[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# Appendix: Adam Derivation

## 1. Adam update equations

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
## 2. Why the first moment is useful

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
## 3. Why the second moment is useful

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
## 4. Why divide by the square root of the second moment

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
## 5. Why bias correction is needed

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
## 6. Deriving the expected first moment

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

## 7. Deriving the expected second moment

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
## 8. Interpreting the corrected estimates

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
## 9. Initialization-bias boundary

Bias correction addresses the initialization bias caused by starting $`m_0`$ and $`v_0`$ at zero. Under the stationary-gradient approximation, dividing by $`1-\beta_1^t`$ and $`1-\beta_2^t`$ removes the early shrinkage in the expected moment estimates.

This is not a claim that Adam is universally unbiased during neural-network training. Gradients change as parameters change, mini-batches introduce stochastic variation, and the optimization process is non-stationary. The correction is specifically about zero-initialized exponential moving averages.
## 10. First-step numerical example

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
## 11. Why epsilon is included

The $`\epsilon`$ term prevents division by zero or division by an extremely small number:

```math
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

If a parameter has very small recent gradients, $`\sqrt{\hat{v}_t}`$ can be close to zero. Without $`\epsilon`$, the effective learning rate could become numerically unstable. With $`\epsilon`$, the denominator is always positive.

In most ordinary updates, $`\epsilon`$ is much smaller than $`\sqrt{\hat{v}_t}`$, so it has little effect. It mainly protects edge cases and floating-point stability.
## 12. Adam implementation mapping

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
## 13. Testing stateful Adam transitions

One-step tests are necessary but insufficient for stateful optimizers. A faulty optimizer may produce the correct first update while incorrectly resetting or misusing internal state later.

Adam tests should verify the transition from $`state_0`$ to $`state_1`$ to $`state_2`$. The strengthened second-step test checks `first_moment_weights`, `second_moment_weights`, `first_moment_bias`, `second_moment_bias`, `time_step`, updated weights, and updated bias. This protects against errors such as resetting moments each step, mixing `beta1` and `beta2`, omitting gradient squaring, or applying bias correction incorrectly.
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
## 15. Key intuition summary

Adam can be read as Momentum plus adaptive scaling. The first moment estimates the useful direction of movement. The second moment estimates the recent gradient scale. Bias correction fixes the early underestimation caused by zero initialization. Dividing by $`\sqrt{\hat{v}_t}`$ makes each parameter's update relative to its own recent gradient magnitude, while $`\epsilon`$ keeps the denominator numerically safe.

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)
