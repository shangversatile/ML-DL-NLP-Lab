[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# MLP Forward Pass and Backpropagation

## 1. One-hidden-layer binary MLP architecture

`BinaryMLPScratch` implements a one-hidden-layer binary classifier. The architecture is:

```text
X -> affine hidden layer -> ReLU -> affine output layer -> sigmoid -> probabilities
```

The model has input-to-hidden parameters `W1` and `b1`, hidden-to-output parameters `W2` and `b2`, and returns positive-class probabilities for binary classification.
## 2. Forward-pass equations

The forward pass computes:

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

NumPy broadcasts `b1` across all sample rows in the hidden layer and broadcasts `b2` across the output rows.
## 3. Shape table

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
## 4. Why nonlinear activation is required

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

For a deeper geometric treatment, see [Activation, Initialization, and Normalization](04_activation_initialization_normalization.md).
## 5. Stable sigmoid computation

The direct sigmoid formula is:

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

For a very negative input, computing `exp(-z)` can overflow because $`-z`$ becomes very large and positive. The implementation uses an equivalent form for negative inputs:

```math
\sigma(z)=\frac{e^z}{1+e^z}
```

This avoids computing an enormous exponential value and improves numerical stability. The test with `0.0`, `1000.0`, and `-1000.0` verifies the behavior at ordinary, very positive, and very negative inputs.
## 6. Why forward pass returns a cache

The forward pass returns a cache containing `X`, `Z1`, `A1`, and `Z2`. These intermediate values are needed later during backpropagation: `A1` is needed to compute `dW2`, `Z1` is needed to compute the ReLU derivative, and `X` is needed to compute `dW1`.

Caching avoids recomputing the full forward pass during backpropagation. The trade-off is memory versus computation: storing intermediate arrays uses more memory, but it avoids redundant matrix multiplications and activation computations.
## 7. Why probabilities are reshaped

`Z2` has shape `(n_samples, 1)`, while labels usually have shape `(n_samples,)`. Calling `.reshape(-1)` converts output probabilities to shape `(n_samples,)`, aligning predictions with labels.

This also prevents accidental NumPy broadcasting from producing a `(n_samples, n_samples)` array during loss or gradient computation. Keeping probabilities and labels as matching one-dimensional arrays makes the binary-classification interface less error-prone.
## 8. Forward-pass testing strategy

The forward-pass tests cover parameter shapes, reproducible initialization, different initialization under different seeds, ReLU values, numerically stable sigmoid behavior, forward-pass shapes, manually verifiable forward-pass values, invalid dimensions, and invalid input shape.

Shape tests catch interface errors. Manual-value tests validate the complete mathematical path from input features through hidden activations to output probabilities. Stable-sigmoid tests catch numerical errors that would not necessarily appear in purely symbolic formula checks.
## 9. Binary MLP backpropagation roadmap

The backward pass for the binary MLP computes gradients in reverse order from the scalar batch loss back to the trainable parameters. For the one-hidden-layer network, the required intermediate and parameter gradients are:

- `dZ2`
- `dW2`
- `db2`
- `dA1`
- `dZ1`
- `dW1`
- `db1`

Each gradient is obtained by applying the chain rule from the binary cross-entropy loss backward through the sigmoid output activation, the output affine layer, the ReLU hidden activation, and the input affine layer. This reverse order mirrors the forward computation: first compute the output-layer error, then propagate it into the hidden layer, then compute gradients for the input-layer parameters.
## 10. Why sigmoid and BCE simplify to `P - y`

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
## 11. Why sigmoid and BCE produce a logit-space residual

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
## 12. Output-layer gradients

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
## 13. Why `dW2 = A1.T @ dZ2`

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
## 14. Backpropagating through ReLU

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
## 15. Why cache `Z1` for ReLU backward

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
## 16. Input-layer gradients

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
## 17. Why divide by batch size only once

The training objective is batch-average binary cross-entropy, so the average over samples contributes one factor of `1 / n`. That factor enters through the output-logit gradient:

```math
dZ_2
=
\frac{P-y}{n}
```

All later gradients inherit this factor through the chain rule. For example, `dW2`, `db2`, `dA1`, `dZ1`, `dW1`, and `db1` are all computed from values that already include the batch-average scaling. Dividing again at each layer would incorrectly shrink the gradients multiple times and would no longer match the derivative of the stated batch-average objective.
## 18. Cache-to-gradient mapping

| Cached variable | Backward use                                  |
| --------------- | --------------------------------------------- |
| `X`             | compute `dW1 = X.T @ dZ1`                     |
| `Z1`            | construct the ReLU mask                       |
| `A1`            | compute `dW2 = A1.T @ dZ2`                    |
| `Z2`            | inspect logits or recompute sigmoid if needed |

Caching reduces repeated computation at the cost of additional memory. The cached values preserve the exact forward-pass intermediates needed by the backward pass, avoiding redundant matrix multiplications and activation evaluations.
## 19. Backpropagation shape invariants

| Parameter |            Parameter shape | Gradient |             Gradient shape |
| --------- | -------------------------: | -------- | -------------------------: |
| `W1`      | `(n_features, hidden_dim)` | `dW1`    | `(n_features, hidden_dim)` |
| `b1`      |            `(hidden_dim,)` | `db1`    |            `(hidden_dim,)` |
| `W2`      |          `(hidden_dim, 1)` | `dW2`    |          `(hidden_dim, 1)` |
| `b2`      |                     `(1,)` | `db2`    |                     `(1,)` |

Each parameter gradient must have exactly the same shape as its corresponding parameter. This invariant is what allows the optimizer to update parameters element by element without broadcasting mistakes or shape-dependent special cases.
## 20. Open questions

- How should the optimizer API be generalized from one weight vector and one scalar bias to multiple MLP parameter tensors?
- How can numerical gradient checking validate an MLP backpropagation implementation?
- How should MLP training diagnostics separate optimization failure from implementation bugs?
- Which activation and initialization choices should be compared once MLP training is implemented?

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Next: Activation, Initialization, and Normalization →](04_activation_initialization_normalization.md)
