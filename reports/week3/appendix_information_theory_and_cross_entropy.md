# Appendix: Information Theory and Cross Entropy

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

This appendix explains why entropy measures the theoretical limit of lossless compression and why cross entropy naturally appears as a classification loss. The main chain of ideas is:

```text
symbol probability
→ ideal code length
→ self-information
→ entropy
→ mismatched coding cost
→ cross entropy
→ KL divergence
→ negative log-likelihood
→ neural-network classification loss
```

## 1. Lossless source coding setup

Let $`X`$ be a discrete random variable with alphabet:

```math
A
=
\{
x_1,
x_2,
\dots,
x_m
\}
```

and probabilities:

```math
P(X=x_i)
=
p_i
```

with:

```math
p_i
>
0
```

```math
\sum_{i=1}^{m}
p_i
=
1
```

A lossless code assigns a binary codeword to every symbol. Let $`l_i`$ denote the number of bits assigned to symbol $`x_i`$. The expected code length is:

```math
L
=
\sum_{i=1}^{m}
p_i l_i
```

The engineering intuition is direct: common symbols should receive shorter codewords, while rare symbols can receive longer codewords. The objective is to minimize the expected number of bits per symbol, not the maximum length of any single codeword.

An important correction is that code length does not equal probability. The ideal real-valued code length is proportional to the negative logarithm of probability:

```math
l_i^\star
=
-\log_2 p_i
```

High probability means short code. Low probability means long code. Halving a probability increases the ideal code length by one bit.

## 2. Prefix-free codes and the Kraft inequality

A prefix-free code has no codeword that is the prefix of another codeword. Prefix-free codes can be decoded sequentially without ambiguity because the decoder knows immediately when a complete codeword has ended.

Binary prefix-code lengths must satisfy the Kraft inequality:

```math
\sum_{i=1}^{m}
2^{-l_i}
\le
1
```

The tree intuition is useful. A binary codeword of length $`l_i`$ occupies a fraction $`2^{-l_i}`$ of the leaves in a full binary tree. The total occupied fraction cannot exceed one.

Prefix-free codes are sufficient for practical instantaneous decoding. The lower-bound argument also extends to uniquely decodable codes through the Kraft-McMillan inequality.

## 3. Why ideal code lengths equal negative log probabilities

Treat code lengths temporarily as positive real numbers rather than integers. Minimize:

```math
L
=
\sum_{i=1}^{m}
p_i l_i
```

subject to:

```math
\sum_{i=1}^{m}
2^{-l_i}
\le
1
```

At the optimum, the Kraft constraint is active. Otherwise all code lengths could be shortened slightly while keeping the code valid. Use the equality constraint:

```math
\sum_{i=1}^{m}
2^{-l_i}
=
1
```

Introduce the Lagrangian:

```math
J
=
\sum_{i=1}^{m}
p_i l_i
+
\lambda
\left(
\sum_{i=1}^{m}
2^{-l_i}
-
1
\right)
```

Differentiate with respect to $`l_i`$:

```math
\frac{\partial J}
{\partial l_i}
=
p_i
-
\lambda
\ln(2)
2^{-l_i}
```

Set the derivative to zero:

```math
p_i
=
\lambda
\ln(2)
2^{-l_i}
```

Therefore:

```math
2^{-l_i}
=
\frac{p_i}
{\lambda\ln(2)}
```

Sum over all symbols:

```math
\sum_{i=1}^{m}
2^{-l_i}
=
\frac{1}
{\lambda\ln(2)}
\sum_{i=1}^{m}
p_i
```

Using the active Kraft constraint and probability normalization:

```math
1
=
\frac{1}
{\lambda\ln(2)}
```

Therefore:

```math
\lambda\ln(2)
=
1
```

Substitute back:

```math
2^{-l_i}
=
p_i
```

Take the binary logarithm:

```math
l_i^\star
=
-\log_2 p_i
```

The deep intuition is that an ideal code allocates a fraction of the available binary tree equal to the symbol probability. A symbol that occurs twice as often receives one fewer bit. Optimal coding matches representation capacity to source frequency.

Real prefix-code lengths must be integers. The derivation establishes the ideal real-valued target, and practical coding methods approximate this target.

## 4. Self-information

Define the information content of observing symbol $`x_i`$:

```math
I(x_i)
=
-\log_2 p_i
```

Rare events carry more information. Common events carry less information. An event with probability one carries zero surprise. One bit corresponds to distinguishing between two equally likely outcomes.

| Probability | Self-information |
| ----------: | ---------------: |
|         `1` |         `0 bits` |
|       `1/2` |          `1 bit` |
|       `1/4` |         `2 bits` |
|       `1/8` |         `3 bits` |

The additivity argument explains why logarithms appear. For independent events $`A`$ and $`B`$:

```math
P(A,B)
=
P(A)P(B)
```

A reasonable information measure should satisfy:

```math
I(A,B)
=
I(A)+I(B)
```

The logarithm has exactly this property:

```math
-\log_2
\left(
P(A)P(B)
\right)
=
-\log_2P(A)
-
\log_2P(B)
```

The logarithm turns multiplication of probabilities into addition of information.

## 5. Shannon entropy

Entropy is the expected self-information:

```math
H(P)
=
\sum_{i=1}^{m}
p_i
\log_2
\frac{1}
{p_i}
```

Equivalently:

```math
H(P)
=
-
\sum_{i=1}^{m}
p_i
\log_2 p_i
```

Entropy measures the average number of bits needed per symbol in an ideal lossless code. It is not the information content of one particular observation. It is the distribution-level average.

### Fair coin

```math
H(P)
=
-
\frac{1}{2}
\log_2
\frac{1}{2}
-
\frac{1}{2}
\log_2
\frac{1}{2}
```

```math
H(P)
=
1
```

### Highly imbalanced binary source

Use probabilities $`0.99`$ and $`0.01`$:

```math
H(P)
=
-
0.99\log_2(0.99)
-
0.01\log_2(0.01)
```

The entropy is much smaller than one bit because a highly predictable source can be compressed more aggressively. Entropy quantifies uncertainty before observing the symbol.

## 6. Why no lossless prefix code can beat entropy on average

Let code lengths be $`l_i`$. Define:

```math
K
=
\sum_{i=1}^{m}
2^{-l_i}
```

By Kraft:

```math
K
\le
1
```

Define an auxiliary probability distribution:

```math
q_i
=
\frac{
2^{-l_i}
}
{K}
```

Then:

```math
\sum_{i=1}^{m}
q_i
=
1
```

Rearrange:

```math
l_i
=
-\log_2 q_i
-
\log_2 K
```

The expected code length is:

```math
L
=
\sum_{i=1}^{m}
p_i l_i
```

Substitute:

```math
L
=
-
\sum_{i=1}^{m}
p_i\log_2 q_i
-
\log_2K
```

Introduce cross entropy:

```math
H(P,Q)
=
-
\sum_{i=1}^{m}
p_i\log_2 q_i
```

Therefore:

```math
L
=
H(P,Q)
-
\log_2K
```

Use:

```math
H(P,Q)
=
H(P)
+
D_{KL}(P || Q)
```

Therefore:

```math
L
=
H(P)
+
D_{KL}(P || Q)
-
\log_2K
```

Since:

```math
D_{KL}(P || Q)
\ge
0
```

and:

```math
K
\le
1
```

so:

```math
-\log_2K
\ge
0
```

conclude:

```math
L
\ge
H(P)
```

Entropy is a fundamental lower bound: no valid lossless prefix code has expected length below entropy. Mismatch between code allocation and true source probability creates an extra KL-divergence penalty.

The non-negativity of KL divergence follows from Gibbs inequality. A direct derivation appears in Section 13.

## 7. Shannon coding achieves entropy within one bit

Choose integer code lengths:

```math
l_i
=
\lceil
-\log_2p_i
\rceil
```

Then:

```math
l_i
<
-\log_2p_i
+
1
```

Multiply by $`p_i`$ and sum:

```math
L
<
H(P)
+
1
```

Also:

```math
l_i
\ge
-\log_2p_i
```

Therefore:

```math
L
\ge
H(P)
```

Conclude:

```math
H(P)
\le
L
<
H(P)+1
```

Shannon coding creates a valid prefix code with average length less than one bit above entropy. Integer constraints create the gap. Block coding reduces the per-symbol gap.

## 8. Shannon's noiseless source coding theorem

This appendix discusses the noiseless source coding theorem, not the noisy-channel coding theorem.

For an independent identically distributed source block:

```math
X^n
=
(X_1,X_2,\dots,X_n)
```

state:

```math
H(X^n)
=
nH(X)
```

Apply Shannon coding to length-$`n`$ blocks:

```math
H(X^n)
\le
L_n
<
H(X^n)+1
```

Therefore:

```math
nH(X)
\le
L_n
<
nH(X)+1
```

Divide by $`n`$:

```math
H(X)
\le
\frac{L_n}{n}
<
H(X)
+
\frac{1}{n}
```

Take the limit:

```math
\lim_{n\to\infty}
\frac{L_n}{n}
=
H(X)
```

Entropy is the asymptotically achievable average number of bits per source symbol. No lossless uniquely decodable code can asymptotically beat entropy. Block coding allows fractional average bits per original symbol even though every physical codeword has integer length.

## 9. Typical-set intuition

For a long independent identically distributed sequence, most probability mass concentrates on a typical set. A typical sequence has probability approximately:

```math
2^{-nH(X)}
```

The number of typical sequences is approximately:

```math
2^{nH(X)}
```

Identifying one typical sequence therefore requires approximately:

```math
nH(X)
```

bits.

Compression is possible because almost all observed long sequences lie inside a much smaller structured subset of all possible raw sequences. Entropy measures the exponential growth rate of this effective set.

The earlier expected-length prefix-code proof is already sufficient for the coding bound used in this appendix. The typical-set view adds intuition for asymptotic block coding.

## 10. Practical coding methods

Huffman coding produces an optimal prefix code for known symbol probabilities under integer code lengths. Huffman lengths are typically close to:

```math
-\log_2p_i
```

Arithmetic coding encodes entire sequences and can approach fractional average lengths more closely. Modern compression systems combine probability modeling with efficient coding schemes.

The quality of compression depends on how accurately the probability model matches the source.

## 11. Cross entropy

Suppose data are generated by true distribution $`P`$, but the code is designed using assumed distribution $`Q`$. Assign ideal code length:

```math
l_Q(x_i)
=
-\log_2q_i
```

Then the average number of bits under the true source is:

```math
H(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2q_i
```

Entropy $`H(P)`$ is the best achievable average code length when the true distribution is known. Cross entropy $`H(P,Q)`$ is the average code length when data follow $`P`$ but coding decisions are based on $`Q`$. A wrong probability model wastes bits.

## 12. Cross entropy decomposition and KL divergence

Define:

```math
D_{KL}(P || Q)
=
\sum_{i=1}^{m}
p_i
\log_2
\frac{p_i}
{q_i}
```

Start from:

```math
H(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2q_i
```

Add and subtract the entropy term:

```math
H(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2p_i
+
\sum_{i=1}^{m}
p_i
\log_2
\frac{p_i}
{q_i}
```

Therefore:

```math
H(P,Q)
=
H(P)
+
D_{KL}(P || Q)
```

Entropy is the irreducible source uncertainty. KL divergence is the additional coding cost caused by using the wrong distribution. Cross entropy equals unavoidable uncertainty plus mismatch penalty.

## 13. Why KL divergence is non-negative

Use the inequality:

```math
\ln u
\le
u-1
```

for positive $`u`$. Apply it to:

```math
u
=
\frac{q_i}
{p_i}
```

Then:

```math
-\ln
\frac{q_i}
{p_i}
\ge
1
-
\frac{q_i}
{p_i}
```

Multiply by $`p_i`$:

```math
p_i
\ln
\frac{p_i}
{q_i}
\ge
p_i-q_i
```

Sum over $`i`$:

```math
\sum_i
p_i
\ln
\frac{p_i}
{q_i}
\ge
\sum_i
(p_i-q_i)
```

Since both probability distributions sum to one:

```math
\sum_i
(p_i-q_i)
=
0
```

Therefore:

```math
D_{KL}(P || Q)
\ge
0
```

The proof used natural logarithms, but changing the logarithm base only multiplies KL divergence by a positive constant. Equality holds when $`P = Q`$. Cross entropy is minimized when the modeled distribution matches the true distribution. This is why minimizing cross entropy trains the model to approximate the data-generating conditional distribution.

## 14. Cross entropy in neural-network classification

For an input $`x`$, the model predicts a categorical distribution:

```math
Q_\theta
(
Y=k
\mid
X=x
)
=
q_{\theta,k}(x)
```

For a one-hot target vector $`y`$, define multiclass cross entropy:

```math
L_{CE}
=
-
\sum_{k=1}^{K}
y_k
\log
q_{\theta,k}(x)
```

Since the target is one-hot, only the correct class remains:

```math
L_{CE}
=
-\log
q_{\theta,y}(x)
```

The loss is the code length assigned by the model to the correct label. Confident correct predictions receive a short code. Uncertain predictions receive a longer code. Confident incorrect predictions receive a very large penalty.

For a dataset:

```math
L(\theta)
=
-
\frac{1}{N}
\sum_{i=1}^{N}
\log
q_\theta
(
y_i
\mid
x_i
)
```

Minimizing cross entropy is equivalent to minimizing average negative log-likelihood. Minimizing negative log-likelihood is equivalent to maximizing likelihood. The model learns probabilities that compress observed labels efficiently conditioned on inputs.

## 15. Binary cross entropy

For binary labels:

```math
y
\in
\{
0,1
\}
```

and predicted positive-class probability:

```math
p
=
Q_\theta(Y=1 \mid X=x)
```

define:

```math
L_{BCE}
=
-
\left[
y\log p
+
(1-y)\log(1-p)
\right]
```

If:

```math
y=1
```

then:

```math
L_{BCE}
=
-\log p
```

If:

```math
y=0
```

then:

```math
L_{BCE}
=
-\log(1-p)
```

BCE is the negative log-likelihood of a Bernoulli model. BCE measures how many natural-log units are needed to encode the observed label under the predicted probability model.

Base-2 logarithms measure information in bits. Natural logarithms measure information in nats. Neural-network libraries usually use natural logarithms. Changing the logarithm base multiplies the loss by a positive constant, which changes scale but not the location of the optimum.

## 16. Why cross entropy is more informative than accuracy

For a positive label:

| Predicted probability | Accuracy at threshold `0.5` |             BCE |
| --------------------: | --------------------------: | --------------: |
|                `0.51` |                     correct | relatively high |
|                `0.80` |                     correct |           lower |
|                `0.99` |                     correct |        very low |
|                `0.01` |                   incorrect |  extremely high |

Accuracy only checks whether the probability crosses a threshold. Cross entropy also measures confidence quality. A differentiable loss is needed for gradient-based optimization. Accuracy is piecewise constant with respect to model parameters and provides almost no useful local gradient.

## 17. Why sigmoid and BCE simplify to `p - y`

Start with:

```math
p
=
\sigma(z)
```

and:

```math
L_{BCE}
=
-
\left[
y\log p
+
(1-y)\log(1-p)
\right]
```

Derive:

```math
\frac{\partial L_{BCE}}
{\partial p}
=
\frac{p-y}
{p(1-p)}
```

Use:

```math
\frac{\partial p}
{\partial z}
=
p(1-p)
```

Then:

```math
\frac{\partial L_{BCE}}
{\partial z}
=
p-y
```

Coding-theory interpretation: $`p - y`$ measures model-distribution mismatch for the observed binary label. Probabilistic interpretation: it is the predicted Bernoulli mean minus the observed target. Optimization interpretation: the residual directly drives the logit correction.

## 18. Interpretation boundaries

Shannon coding theorems describe asymptotic lossless compression under stated probabilistic assumptions. Real datasets may not be independent and identically distributed. Practical models only approximate the unknown source distribution.

Minimizing training cross entropy does not automatically guarantee generalization. Low cross entropy does not automatically guarantee perfect calibration. Cross entropy is a proper scoring rule, but finite data, optimization limits, distribution shift, and model misspecification still matter. Accuracy, calibration, robustness, and error analysis remain separate evaluation concerns.

## 19. Conceptual summary

- Frequent symbols should receive shorter descriptions.
- Ideal code length is negative log probability.
- Entropy is expected ideal code length.
- Entropy lower-bounds lossless compression.
- Block coding approaches the entropy rate asymptotically.
- Cross entropy measures coding cost under a mismatched model.
- KL divergence measures the mismatch penalty.
- Neural-network cross entropy is negative log-likelihood.
- Minimizing cross entropy trains predicted probabilities to assign shorter descriptions to observed labels.
- BCE is the Bernoulli special case.
- Sigmoid plus BCE produces the residual `p - y`.

## 20. Further reading

- A Mathematical Theory of Communication
- Elements of Information Theory
- Pattern Recognition and Machine Learning
- The Elements of Statistical Learning
- Stanford CS229 Notes: Generalized Linear Models
- Stanford CS231n Notes: Linear Classification and Loss Functions

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Back to MLP Forward Pass and Backpropagation](03_mlp_forward_and_backprop.md)
