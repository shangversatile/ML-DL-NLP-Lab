[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# Activation, Initialization, and Normalization

## 1. Geometric meaning of affine transformations

A fully connected layer with bias is an affine transformation:

```math
Z = XW+b
```

Affine maps preserve the affine structure of the space. Geometrically, they can stretch, rotate, shear, and translate space. That intuition should be kept separate from the exact mathematical claim: affine maps do not create curved or piecewise-folded decision geometry by themselves.
## 2. Why deep affine networks collapse into one affine map

For multiple affine layers without nonlinear activations, the composition collapses into one affine transformation. For two layers:

```math
Z_2
=
(XW_1+b_1)W_2+b_2
```

```math
Z_2
=
X(W_1W_2)
+
(b_1W_2+b_2)
```

For many layers:

```math
Y
=
XW_{\mathrm{effective}}
+
b_{\mathrm{effective}}
```

The reason is that matrix multiplication is associative. Without nonlinear activations, depth alone cannot represent nonlinear decision boundaries. The final classification boundary remains affine, such as a hyperplane in the relevant representation space.
## 3. ReLU and sigmoid as different geometric transformations

ReLU is defined coordinate-wise by:

```math
\mathrm{ReLU}(z)
=
\max(0,z)
```

ReLU is piecewise linear and introduces a hinge at zero. A ReLU network is therefore piecewise affine: for a fixed pattern of active and inactive ReLU units, the network reduces locally to an affine map. Different activation masks correspond to different local affine regions.

The origami or folding-space picture is a useful geometric metaphor, as long as it is kept separate from the exact claim. A finite ReLU network has finitely many linear regions. Increasing width and depth can increase representational capacity and can create more complex piecewise-linear geometry.

The sigmoid function is:

```math
\sigma(z)
=
\frac{1}
{1+e^{-z}}
```

Sigmoid maps the real line into the interval `(0, 1)`. Applied coordinate-wise, it maps a vector into a bounded hypercube. The central region is more responsive, while the tails saturate. Saturation compresses large-magnitude logits and can reduce gradient magnitudes.

ReLU and sigmoid introduce nonlinearity in different ways. ReLU creates piecewise-affine regions; sigmoid smoothly compresses unbounded coordinates into bounded probabilities or activations.
## 4. Distribution of hidden pre-activations

For one hidden unit:

```math
Z
=
\sum_{i=1}^{n_{\mathrm{in}}}
w_ix_i
+
b
```

First analyze the common initialization setting:

```math
b=0
```

Assume $`x_i`$ are approximately independent across coordinates, $`\mathbb{E}[x_i]=0`$, $`\mathrm{Var}(x_i)=\sigma_x^2`$, $`w_i`$ are independent across coordinates, $`\mathbb{E}[w_i]=0`$, $`\mathrm{Var}(w_i)=\sigma_w^2`$, weights and inputs are independent, and second moments are finite.

Define:

```math
U_i = w_ix_i
```

Then:

```math
\mathbb{E}[U_i]
=
\mathbb{E}[w_i]
\mathbb{E}[x_i]
=
0
```

Also:

```math
\mathrm{Var}(U_i)
=
\mathbb{E}[w_i^2x_i^2]
-
\mathbb{E}[w_ix_i]^2
```

Using independence and zero means:

```math
\mathrm{Var}(U_i)
=
\mathbb{E}[w_i^2]
\mathbb{E}[x_i^2]
=
\sigma_w^2\sigma_x^2
```

Therefore:

```math
\mathrm{Var}(Z)
=
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
```
## 5. Central Limit Theorem analysis of `Z = XW + b`

Start with one hidden unit:

```math
Z
=
\sum_{i=1}^{n_{\mathrm{in}}}
w_i x_i
+
b
```

Here `Z` is one hidden unit's pre-activation, `x_i` is an input coordinate, `w_i` is the weight connecting input coordinate `i` to this hidden unit, and `b` is the bias. The Central Limit Theorem applies to the sum of product terms, not to multiplication alone.

### 5.1 Viewpoint A: fixed inputs and Gaussian weights

Treat input coordinates as fixed constants:

```math
x_1,
x_2,
\dots,
x_{n_{\mathrm{in}}}
```

Assume independent Gaussian weights:

```math
w_i
\sim
\mathrm{N}
\left(
0,
\sigma_w^2
\right)
```

A linear combination of independent Gaussian variables is exactly Gaussian. This conclusion does not require the Central Limit Theorem. Conditional on the input vector:

```math
Z
\mid
x
\sim
\mathrm{N}
\left(
b,
\sigma_w^2
\sum_{i=1}^{n_{\mathrm{in}}}
x_i^2
\right)
```

Equivalently:

```math
Z
\mid
x
\sim
\mathrm{N}
\left(
b,
\sigma_w^2
\lVert x\rVert_2^2
\right)
```

This is the conditional distribution of the hidden pre-activation given the input vector. If `b = 0`, the conditional mean is zero.

If input coordinates have an average squared scale approximately equal to $`\sigma_x^2`$:

```math
\frac{1}
{n_{\mathrm{in}}}
\sum_{i=1}^{n_{\mathrm{in}}}
x_i^2
\approx
\sigma_x^2
```

then:

```math
\sum_{i=1}^{n_{\mathrm{in}}}
x_i^2
\approx
n_{\mathrm{in}}
\sigma_x^2
```

and therefore:

```math
Z
\mid
x
\approx
\mathrm{N}
\left(
b,
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
\right)
```

### 5.2 Viewpoint B: random inputs and random weights

Define:

```math
U_i
=
w_i x_i
```

Assume $`\mathbb{E}[w_i] = 0`$, $`\mathrm{Var}(w_i) = \sigma_w^2`$, $`\mathbb{E}[x_i] = 0`$, $`\mathrm{Var}(x_i) = \sigma_x^2`$, weights and inputs are independent, product terms are independent or sufficiently weakly dependent across coordinates, and second moments are finite.

Derive the mean:

```math
\mathbb{E}[U_i]
=
\mathbb{E}[w_i x_i]
```

```math
\mathbb{E}[U_i]
=
\mathbb{E}[w_i]
\mathbb{E}[x_i]
```

```math
\mathbb{E}[U_i]
=
0
```

Then derive the variance:

```math
\mathrm{Var}(U_i)
=
\mathbb{E}[U_i^2]
-
\mathbb{E}[U_i]^2
```

```math
\mathrm{Var}(U_i)
=
\mathbb{E}[w_i^2 x_i^2]
```

```math
\mathrm{Var}(U_i)
=
\mathbb{E}[w_i^2]
\mathbb{E}[x_i^2]
```

```math
\mathrm{Var}(U_i)
=
\sigma_w^2
\sigma_x^2
```

Now write:

```math
Z-b
=
\sum_{i=1}^{n_{\mathrm{in}}}
U_i
```

Under the independence assumptions:

```math
\mathbb{E}[Z-b]
=
0
```

```math
\mathrm{Var}(Z-b)
=
\sum_{i=1}^{n_{\mathrm{in}}}
\mathrm{Var}(U_i)
```

```math
\mathrm{Var}(Z-b)
=
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
```

If `b = 0`:

```math
\mathbb{E}[Z]
=
0
```

```math
\mathrm{Var}(Z)
=
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
```

Then the CLT approximation is:

```math
Z \approx \mathcal{N}\left(0, n_{\mathrm{in}}\sigma_w^2\sigma_x^2\right)
```

The summed random variables are $`U_i = w_i x_i`$. The CLT applies to their sum. Sufficiently large fan-in helps the approximation, but finite variance and the absence of a dominating coordinate are also important. This is an approximation, not a universal exact identity.

### 5.3 Effect of the bias term

If:

```math
b=0
```

then:

```math
\mathbb{E}[Z]
=
0
```

If `b` is a fixed constant:

```math
Z
\approx
\mathrm{N}
\left(
b,
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
\right)
```

If `b` is an independent random variable with:

```math
\mathbb{E}[b]
=
\mu_b
```

```math
\mathrm{Var}(b)
=
\sigma_b^2
```

then:

```math
\mathbb{E}[Z]
=
\mu_b
```

```math
\mathrm{Var}(Z)
=
n_{\mathrm{in}}
\sigma_w^2
\sigma_x^2
+
\sigma_b^2
```

The current educational MLP initializes biases to zero. This keeps the initialization analysis simple, and zero bias helps maintain a centered pre-activation distribution initially.

### 5.4 Why the Gaussian approximation has limits

Real neural-network activations are not guaranteed to be independent. Input coordinates can be correlated, hidden units can become dependent during training, weight updates alter the initialization-time assumptions, and nonlinear activations change later-layer distributions.

The derivation is an idealized initialization analysis. It is still useful because it explains the design logic behind Xavier and He initialization: the initial weight scale is chosen to keep activation and gradient magnitudes from growing or shrinking too aggressively across layers.

### 5.5 Connection to ReLU half-zero activations

If the pre-activation distribution is approximately symmetric around zero, then the probability of a positive or negative pre-activation is approximately one half:

```math
\mathbb{P}(Z>0)
\approx
\frac{1}{2}
```

```math
\mathbb{P}(Z<0)
\approx
\frac{1}{2}
```

This motivates the following ReLU sparsity analysis. It does not mean that half of all neurons are permanently dead.

## 6. Why ReLU produces approximately half-zero activations at initialization

Assume:

```math
Z
\approx
\mathrm{N}(0,q)
```

or more generally that the distribution of $`Z`$ is symmetric around zero. Then:

```math
\mathbb{P}(Z>0)
=
\mathbb{P}(Z<0)
=
\frac{1}{2}
```

For:

```math
A
=
\mathrm{ReLU}(Z)
```

negative pre-activations become zero, and positive pre-activations remain positive. Under the symmetry assumption, approximately half of activation entries are zero at initialization.

This is an initialization-time statistical expectation. It does not mean that half of neurons are permanently dead. Different inputs activate different units, training changes the weight distribution and activation pattern, and a dead ReLU refers to a stronger failure mode where a unit remains inactive across relevant inputs.

## 7. Second moment versus variance after ReLU

For symmetric Gaussian $`Z`$:

```math
\mathbb{E}
\left[
\mathrm{ReLU}(Z)^2
\right]
=
\frac{1}{2}
\mathbb{E}[Z^2]
=
\frac{q}{2}
```

He initialization tracks the second moment or activation-energy scale. This is not identical to claiming that ReLU output variance is exactly one-half of input variance. More precisely, for Gaussian $`Z`$:

```math
\mathbb{E}
\left[
\mathrm{ReLU}(Z)
\right]
=
\sqrt{
\frac{q}
{2\pi}
}
```

```math
\mathrm{Var}
\left(
\mathrm{ReLU}(Z)
\right)
=
q
\left(
\frac{1}{2}
-
\frac{1}
{2\pi}
\right)
```

The second moment is $`\mathbb{E}[A^2]`$. The variance is $`\mathbb{E}[A^2] - \mathbb{E}[A]^2`$. Because ReLU outputs have positive mean under a symmetric nondegenerate input distribution, these two quantities are not the same.
## 8. Sparse activation intuition and its limits

ReLU activation masks create input-dependent active paths. Different inputs can activate different subsets of hidden units. This supports a useful conditional-subnetwork intuition, and zero activations can reduce unnecessary activity and create sparse intermediate representations.

The boundary of the claim matters. Sparse ReLU activations do not by themselves prove feature disentanglement, and they do not prove that semantic concepts automatically separate into different neurons. Biological-neuron analogies may be motivational, but they are not a mathematical justification for ReLU.
## 9. Variance propagation through an affine layer

Start with:

```math
Y
=
\sum_{i=1}^{n_{\mathrm{in}}}
w_ix_i
```

Under the same idealized assumptions:

```math
\mathrm{Var}(Y)
=
n_{\mathrm{in}}
\mathrm{Var}(w)
\mathrm{Var}(x)
```

If weight variance is too large, activation scale can grow rapidly across layers. If weight variance is too small, activation scale can shrink toward zero. Initialization aims to preserve a stable signal scale across depth. This is a mean-field-style idealization, not an exact description of all trained neural networks.
## 10. Xavier-style initialization

For approximately linear activations near the operating region, a forward-scale preservation condition is:

```math
\mathrm{Var}(Y)
\approx
\mathrm{Var}(x)
```

Using:

```math
\mathrm{Var}(Y)
=
n_{\mathrm{in}}
\mathrm{Var}(w)
\mathrm{Var}(x)
```

gives:

```math
\mathrm{Var}(w)
\approx
\frac{1}
{n_{\mathrm{in}}}
```

An analogous backward-flow argument gives:

```math
\mathrm{Var}(w)
\approx
\frac{1}
{n_{\mathrm{out}}}
```

The common compromise is:

```math
\mathrm{Var}(W)
=
\frac{2}
{n_{\mathrm{in}}+n_{\mathrm{out}}}
```

This compromise balances forward and backward signal scales. Xavier-style initialization is especially natural for approximately symmetric activations such as `tanh`, but it is not a universal optimum for every architecture.
## 11. He-style initialization for ReLU

ReLU approximately preserves half of the pre-activation second moment:

```math
\mathbb{E}
\left[
\mathrm{ReLU}(Z)^2
\right]
\approx
\frac{1}{2}
\mathbb{E}[Z^2]
```

To preserve activation-energy scale across a ReLU layer, require:

```math
\frac{1}{2}
n_{\mathrm{in}}
\mathrm{Var}(W)
\approx
1
```

Then:

```math
\mathrm{Var}(W)
\approx
\frac{2}
{n_{\mathrm{in}}}
```

This is the fan-in form. A backward-flow analysis leads to a corresponding fan-out form. Choosing fan-in prioritizes forward activation-scale preservation, while choosing fan-out prioritizes backward gradient-scale preservation. These conditions improve initialization stability but do not guarantee convergence for every architecture or optimization problem.

## 12. Why the code multiplies by `sqrt(2 / fan_in)`

For the code mapping, start with:

```math
G
\sim
\mathrm{N}(0,1)
```

Let:

```math
W=cG
```

Then:

```math
\mathrm{Var}(W)
=
c^2
\mathrm{Var}(G)
=
c^2
```

Set the target variance:

```math
c^2
=
\frac{2}
{n_{\mathrm{in}}}
```

Then:

```math
c
=
\sqrt{
\frac{2}
{n_{\mathrm{in}}}
}
```

This maps directly to the repository code:

```python
self.W1 = rng.standard_normal((n_features, hidden_dim)) * np.sqrt(
    2.0 / n_features
)
```

and:

```python
self.W2 = rng.standard_normal((hidden_dim, 1)) * np.sqrt(
    2.0 / hidden_dim
)
```

`rng.standard_normal()` creates standard-normal entries. Multiplying by the square-root factor changes standard deviation, so the resulting variance becomes the intended He-style value.

He-style scaling is especially motivated for weights feeding into or used with rectifier activations. The current educational implementation uses the same explicit scaling pattern for both weight matrices to keep initialization consistent and inspectable. Future experiments may compare initialization strategies for the sigmoid output layer.
## 13. Initialization is only the starting condition

Xavier and He initialization control signal scale at the beginning of training. Parameter updates change weight distributions over time, and signal statistics can drift during training. Initialization is necessary but not sufficient for very deep networks.

This motivates a brief preview of normalization layers, which dynamically influence activation statistics during training rather than only setting the initial parameter scale.
## 14. Normalization layers as a preview

Batch Normalization computes mini-batch-normalized activations during training:

```math
\hat{Z}
=
\frac{Z-\mu_{\mathrm{B}}}
{\sqrt{\sigma_{\mathrm{B}}^2+\epsilon}}
```

```math
Z_{\mathrm{out}}
=
\gamma\hat{Z}
+
\beta
```

Here $`\mu_{\mathrm{B}}`$ and $`\sigma_{\mathrm{B}}^2`$ are mini-batch statistics during training, while $`\gamma`$ and $`\beta`$ are learnable affine parameters. Normalization changes activation scale while learnable affine parameters restore representational flexibility. This does not guarantee exact reconstruction of every original activation tensor under changing batch statistics.

The original BatchNorm motivation emphasized reducing internal covariate shift. Later research argued that smoother optimization landscapes and more stable gradients are a more fundamental part of the explanation. It is also too strong to claim that BatchNorm always turns the landscape into a perfectly isotropic circular bowl. BatchNorm can improve optimization stability, but its behavior depends on architecture, batch size, data, and training regime.

LayerNorm and RMSNorm are related normalization methods. They normalize different dimensions or statistics, and they should be studied separately later rather than treated as identical to BatchNorm.

The current repository does not implement normalization yet; this section records the conceptual bridge from initialization to future deep-network engineering.
## 15. Initialization and normalization conceptual summary

- Affine layers alone collapse into one affine map.
- Nonlinear activations prevent collapse.
- ReLU creates piecewise-affine geometry and sparse activation masks.
- Pre-activation distributions can be approximated statistically under idealized assumptions.
- Initialization controls early signal scale.
- Xavier balances fan-in and fan-out for approximately symmetric activations.
- He compensates for ReLU second-moment reduction.
- `sqrt(2 / fan_in)` is a standard-deviation scaling factor derived from the target variance.
- Normalization layers dynamically influence activation statistics during training.
- All derivations rely on assumptions that should be stated explicitly.
## 16. Open questions

- How well do the idealized initialization assumptions match actual hidden activations after several training steps?
- How should initialization strategies differ for hidden ReLU layers and sigmoid output layers?
- When do ReLU units become genuinely dead across relevant inputs, rather than merely inactive for one batch?
- How should BatchNorm, LayerNorm, and RMSNorm be compared once deeper networks are implemented?

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Next: Engineering Validation →](05_engineering_validation.md)
