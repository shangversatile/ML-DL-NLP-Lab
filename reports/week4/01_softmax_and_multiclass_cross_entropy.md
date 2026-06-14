[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)

# Softmax and Multiclass Cross Entropy

## 1. From binary classification to multiclass classification

A binary MLP uses one output logit and passes that value through a sigmoid function. The sigmoid maps the logit to a probability for the positive class, while the negative-class probability is implied by `1 - p`.

Multiclass classification uses `K` output logits. The classes are mutually exclusive, so the output must be a probability distribution whose entries are non-negative and sum to one. For handwritten-digit recognition, `K` will eventually be `10`, with one class for each digit from `0` through `9`.

## 2. Logits as unnormalized evidence

Logits are not probabilities. They are unconstrained real-valued scores that act as unnormalized evidence for each class. Larger logits represent stronger relative evidence, but their absolute values are not meaningful by themselves.

Only differences between logits matter for softmax. Adding the same constant to every logit shifts the whole evidence vector without changing the relative evidence between classes, so the resulting probabilities remain the same.

## 3. Softmax definition

For a single example with logits `z`, softmax defines class probability `p_k` as:

```math
p_k
=
\frac{
e^{z_k}
}{
\sum_{j=1}^{K}
e^{z_j}
}
```

For a batch with logit matrix `Z`, the row-wise definition is:

```math
P_{i,k}
=
\frac{
e^{Z_{i,k}}
}{
\sum_{j=1}^{K}
e^{Z_{i,j}}
}
```

Every softmax probability is positive because exponentials are positive. Each row sums to one because every entry is divided by the same row-wise normalization term. The output row is therefore a categorical distribution over the `K` possible classes.

## 4. Numerical stability and shift invariance

Softmax is invariant to subtracting the same constant `c` from every logit:

```math
\frac{
e^{z_k-c}
}{
\sum_j e^{z_j-c}
}
=
\frac{
e^{z_k}e^{-c}
}{
\sum_j e^{z_j}e^{-c}
}
=
\frac{
e^{z_k}
}{
\sum_j e^{z_j}
}
```

Set the shift to the largest logit:

```math
c
=
\max_j z_j
```

This prevents exponentiating huge positive numbers. The largest shifted logit becomes zero, and all other shifted logits are less than or equal to zero. Their exponentials remain finite in practical cases, while the probability distribution is mathematically unchanged.

## 5. Multiclass cross entropy

For a one-hot target vector `y`, multiclass cross entropy is:

```math
L
=
-
\sum_{k=1}^{K}
y_k
\log p_k
```

Because only the correct class has `y_k = 1`, this reduces to:

```math
L
=
-\log p_y
```

For a batch of `n` examples:

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

Cross entropy is the negative log-likelihood of a categorical distribution. It is also the coding cost of the observed label under the model distribution, measured in nats when natural logarithms are used. Confident correct predictions receive small loss, while confident incorrect predictions receive large loss.

## 6. Softmax Jacobian

The softmax Jacobian describes how changing one logit affects every output probability:

```math
\frac{
\partial p_k
}{
\partial z_j
}
=
p_k
(
1_{k=j}
-
p_j
)
```

Let:

```math
S
=
\sum_{\ell=1}^{K}
e^{z_\ell}
```

When `j = k`, increasing the class's own logit increases its probability:

```math
\frac{
\partial p_k
}{
\partial z_k
}
=
\frac{
e^{z_k}S
-
e^{z_k}e^{z_k}
}{
S^2
}
=
p_k
(
1-p_k
)
```

When `j != k`, increasing logit `z_j` decreases probability `p_k`:

```math
\frac{
\partial p_k
}{
\partial z_j
}
=
\frac{
0 \cdot S
-
e^{z_k}e^{z_j}
}{
S^2
}
=
-
p_kp_j
```

This happens because probability mass must sum to one. Increasing one logit increases its own probability but decreases other probabilities. Softmax therefore couples all output classes.

## 7. Deriving softmax plus cross entropy gradient

Start with one-example multiclass cross entropy:

```math
L
=
-
\sum_{k=1}^{K}
y_k
\log p_k
```

Use the chain rule:

```math
\frac{
\partial L
}{
\partial z_j
}
=
\sum_{k=1}^{K}
\frac{
\partial L
}{
\partial p_k
}
\frac{
\partial p_k
}{
\partial z_j
}
```

Since:

```math
\frac{
\partial L
}{
\partial p_k
}
=
-
\frac{
y_k
}{
p_k
}
```

Substitute the softmax Jacobian:

```math
\frac{
\partial L
}{
\partial z_j
}
=
\sum_{k=1}^{K}
\left(
-\frac{y_k}{p_k}
\right)
p_k
(
1_{k=j}
-
p_j
)
```

Cancel `p_k`:

```math
\frac{
\partial L
}{
\partial z_j
}
=
-
\sum_{k=1}^{K}
y_k
(
1_{k=j}
-
p_j
)
```

Split the sum:

```math
\frac{
\partial L
}{
\partial z_j
}
=
-y_j
+
p_j
\sum_{k=1}^{K}
y_k
```

For one-hot labels:

```math
\sum_{k=1}^{K}
y_k
=
1
```

Therefore:

```math
\frac{
\partial L
}{
\partial z_j
}
=
p_j
-
y_j
```

For a batch-mean loss:

```math
\frac{
\partial L
}{
\partial Z
}
=
\frac{
P-Y
}{
n
}
```

The intuition is direct: `p_j` is the probability mass assigned by the model, `y_j` is the probability mass demanded by the label, and `p_j - y_j` is the probability-allocation error.

## 8. Relationship to binary sigmoid BCE

Binary sigmoid BCE gave the output-layer residual `p - y`. Multiclass softmax CE gives the matrix residual `P - Y`.

The binary result is a special case of categorical probability modeling. The key structure is the same: predicted distribution minus target distribution. This residual is what drives output-layer learning.

## 9. Why the row-wise gradient sums to zero

For one example:

```math
\sum_{j=1}^{K}
(p_j-y_j)
=
\sum_{j=1}^{K}
p_j
-
\sum_{j=1}^{K}
y_j
=
1-1
=
0
```

Logits are shift-invariant: adding the same constant to all logits does not change softmax. The gradient therefore has no component in the all-ones direction.

## 10. Implementation notes

- Use stable softmax by subtracting the row-wise maximum before exponentiation.
- Validate class-label ranges before indexing probability arrays.
- Use natural logs in code so cross entropy is measured in nats.
- Clip probabilities in cross entropy to avoid `log(0)`.
- Divide by batch size exactly once when computing the batch-mean gradient.
- Do not use accuracy as a training loss.

## 11. Conceptual summary

Softmax converts relative evidence into a categorical distribution. Cross entropy measures the coding cost of the true class under predicted probabilities. Together, softmax and cross entropy create the clean residual `P - Y`, which directly drives output-layer learning.

Numerical stability is part of mathematical correctness, not an implementation afterthought.

[← Back to Week 4 Index](../week4_multiclass_digits_capstone.md)
