[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

# Multiclass MLP Backpropagation

## 1. Why multiclass MLP is the next step

Week 3 built a binary MLP for one-output sigmoid classification. That model is appropriate when there are two outcomes and the second probability is implied by `1 - p`.

Handwritten digits require ten mutually exclusive classes. A digit image cannot be both a `2` and a `7` at the same time, so the model must output a categorical probability distribution over all classes.

## 2. Architecture

For a batch of examples:

```math
Z_1 = XW_1 + b_1
```

```math
A_1 = ReLU(Z_1)
```

```math
Z_2 = A_1W_2 + b_2
```

```math
P = softmax(Z_2)
```

| Symbol | Shape    | Meaning                 |
| ------ | -------- | ----------------------- |
| `X`    | `(n, d)` | input batch             |
| `W1`   | `(d, h)` | input-to-hidden weights |
| `b1`   | `(h,)`   | hidden bias             |
| `Z1`   | `(n, h)` | hidden pre-activation   |
| `A1`   | `(n, h)` | hidden activation       |
| `W2`   | `(h, K)` | hidden-to-class weights |
| `b2`   | `(K,)`   | class bias              |
| `Z2`   | `(n, K)` | class logits            |
| `P`    | `(n, K)` | class probabilities     |
| `Y`    | `(n, K)` | one-hot labels          |

## 3. Loss function

The batch multiclass cross entropy is:

```math
L
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{k=1}^{K}
Y_{i,k}
\log P_{i,k}
```

This is the categorical negative log-likelihood. It measures the coding cost of the true labels under the model distribution. Because the code uses natural logarithms, the units are nats.

## 4. Output-layer gradient

From Task 6A:

```math
dZ_2
=
\frac{P-Y}{n}
```

`P` is the probability mass assigned by the model. `Y` is the probability mass demanded by the label. The difference `P - Y` is the probability-allocation error, and the division by `n` comes from the mean loss over the batch.

## 5. Second-layer gradients

The hidden-to-class weight gradient is:

```math
dW_2
=
A_1^T dZ_2
```

The class-bias gradient is:

```math
db_2
=
\sum_i dZ_{2,i}
```

The shape check is direct: `A1.T` has shape `(h, n)`, `dZ2` has shape `(n, K)`, and the result has shape `(h, K)`.

Each hidden unit contributes to each class logit. The entry `dW2[a,k]` accumulates hidden activation times class-probability residual for hidden unit `a` and class `k`.

## 6. Hidden-layer pullback

The gradient with respect to hidden activations is:

```math
dA_1
=
dZ_2 W_2^T
```

Output residuals are pulled back through the hidden-to-class weight matrix. Each hidden unit receives a weighted sum of downstream class errors.

## 7. ReLU mask

The hidden pre-activation gradient is:

```math
dZ_1
=
dA_1
\odot
1[Z_1 > 0]
```

ReLU transmits gradient only where the unit was active. Its derivative at zero is undefined, so the implementation chooses zero through the mask `Z1 > 0`. Gradient-checking examples should keep pre-activations away from zero to avoid kink ambiguity.

## 8. First-layer gradients

The input-to-hidden weight gradient is:

```math
dW_1
=
X^T dZ_1
```

The hidden-bias gradient is:

```math
db_1
=
\sum_i dZ_{1,i}
```

Input features are correlated with hidden-layer residuals. `dW1` tells how each input coordinate should change each hidden pre-activation.

## 9. Why divide by batch size only once

The loss is averaged over samples. The output residual `dZ2 = (P - Y) / n` already includes this averaging. Downstream gradients inherit that factor through the chain rule, so dividing again would shrink all gradients by an extra factor of `n`.

## 10. Numerical gradient checking

Analytical backpropagation is efficient but bug-prone. Central finite differences provide an independent check:

```math
\frac{
L(\theta_j+\epsilon)
-
L(\theta_j-\epsilon)
}{
2\epsilon
}
```

The check perturbs one parameter element at a time and compares the numerical estimate with analytical backpropagation. The model parameters must be restored afterward. ReLU pre-activations should be kept away from zero, and relative L2 error is a stable way to summarize agreement.

## 11. Relationship to Week 3 binary MLP

The binary MLP used a scalar logit, sigmoid probability, binary cross entropy, and residual `p - y`. The multiclass MLP uses `K` logits, softmax probabilities, multiclass cross entropy, and residual `P - Y`.

Both forms are predicted distribution minus target distribution. This is the core probability-gradient pattern.

## 12. Conceptual summary

The multiclass MLP extends the binary MLP by replacing sigmoid with softmax. Cross entropy measures wrong probability allocation. Backpropagation propagates class-probability residuals backward through the network. Gradient checking protects the implementation from silent matrix-calculus bugs.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
