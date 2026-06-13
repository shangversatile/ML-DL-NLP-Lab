[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# Gradient, Risk, and Sampling

## 1. Loss, cost, empirical risk, and objective function

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
## 2. Why the negative gradient is a descent direction

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
## 3. From per-example gradients to full-batch gradients

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
## 4. Statistical meaning of stochastic gradients

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
## 5. Two layers of statistical approximation

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
## 6. Sampling with replacement versus random reshuffling

The clean introductory unbiased-gradient derivation often assumes that sample indices are drawn independently, uniformly, and with replacement. Under that simplified model, the next sampled index does not depend on which indices were sampled earlier.

The current repository uses `iterate_minibatches()` with `shuffle=True` and `seed=seed + epoch`. This means the training dataset is shuffled once per epoch, mini-batches are consumed without replacement inside that epoch, and every sample is covered once per epoch. The sample order changes across epochs, but the sequence remains reproducible because the seed is deterministic.

Random reshuffling without replacement is a common practical training strategy, but it is not mathematically identical to independent sampling with replacement. Later mini-batches in an epoch depend on which samples have already been consumed, so one should not blindly claim that every successive mini-batch is conditionally independent.

The simplified unbiased-gradient derivation remains a useful theoretical entry point because it explains why stochastic gradients can point in the correct direction on average under clear assumptions. The repository implementation should be described more precisely as reproducible random reshuffling without replacement.
## 7. Optimization budget

The optimizer comparison currently uses a same-epoch budget: every method trains for the same number of dataset passes. This means each method sees approximately the same total number of training examples, but it does not mean each method performs the same number of parameter updates.

Different budget definitions answer different questions:

- Same epoch budget controls dataset passes.
- Same update budget controls the number of optimizer steps.
- Same sample-processing budget controls the number of examples processed.
- Same wall-clock budget controls elapsed training time under a particular hardware and implementation setup.

The Week 3 experiment is intentionally a teaching-oriented same-epoch comparison. It is useful for seeing training-loop behavior, but it should not be read as a strict optimizer benchmark.
## 8. Conceptual summary

- Batch Gradient Descent uses the exact empirical-risk gradient.
- Classic SGD uses a stochastic estimate based on one sampled observation.
- Mini-batch SGD uses a stochastic estimate based on a subset of observations.
- Mini-batching is both statistically motivated and computationally useful.
- The gradient gives a local descent direction.
- The learning rate determines the step size.
- Empirical risk approximates expected risk.
- Mini-batch gradients approximate empirical-risk gradients.
- The repository uses reproducible random reshuffling without replacement.
## 9. Empirical-risk minimization and overfitting

Training minimizes empirical risk on the observed samples:

```math
R_{\mathrm{emp}}(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell(f_{\theta}(x_i), y_i)
```

The real objective is low expected risk on unseen samples:

```math
R_{\mathrm{exp}}(\theta)
=
\mathbb{E}_{(X,Y)}
\left[
\ell(f_{\theta}(X), Y)
\right]
```

A model can continue reducing empirical risk while its expected-risk estimate worsens. This divergence is a core overfitting signal: optimization on the training objective is still succeeding, but generalization is degrading.

Validation loss is an imperfect but useful estimate of unseen-data performance. Training loss and validation loss answer different questions: training loss measures fit to the optimization target, while validation loss estimates behavior on held-out examples.
## 10. Why label corruption is a controlled probe

Label corruption modifies the training objective without corrupting the validation target. Keeping validation labels clean preserves a stable evaluation reference while the training labels become deliberately less trustworthy.

Task 5F-C isolates the label-condition difference by keeping the initial parameters, data split, preprocessing, optimizer settings, mini-batch order, and update budget shared across conditions. Label noise is a useful controlled probe because it creates a known gap between the training objective and the clean evaluation target, but it is not the only source of overfitting.
## 11. Why clean labels can still overfit

Clean labels do not guarantee perfect generalization. Finite datasets contain sampling variation, feature noise creates ambiguous examples, large models can fit local irregularities, and long training can increase confidence on unstable decision boundaries.

The clean-label result in Task 5F-C is therefore meaningful rather than contradictory. A high-capacity MLP trained for many epochs on a small noisy training set can overfit even when no labels are intentionally corrupted.
## 12. Validation-set reuse and meta-overfitting

Repeated hyperparameter decisions based on the same validation set leak information from the validation set into model development. Over time, the validation set gradually becomes part of the optimization process, even if gradients are never computed from it directly.

Final reporting should use an untouched test set. Complex tuning workflows may require nested validation or nested cross-validation so that model selection and final performance estimation remain separated.
## 13. Open questions

- How does batch size quantitatively affect gradient variance and optimization speed?
- How should the repository compare optimizers under same-update, same-sample, and same-wall-clock budgets?
- How does random reshuffling without replacement change the clean conditional-unbiasedness derivation?
- Which assumptions are needed before empirical risk is a useful approximation to expected risk?

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Next: MLP Forward Pass and Backpropagation →](03_mlp_forward_and_backprop.md)
