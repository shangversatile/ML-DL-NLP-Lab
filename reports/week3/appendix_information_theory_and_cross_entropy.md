# Appendix: Information Theory and Cross Entropy

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

This appendix explains why entropy measures the theoretical limit of lossless compression and why cross entropy naturally appears as a classification loss. The main chain of ideas is:

```text
symbol probability
-> uniquely decodable code lengths
-> McMillan inequality
-> entropy lower bound
-> ideal code length
-> mismatched coding cost
-> cross entropy
-> KL divergence
-> negative log-likelihood
-> neural-network classification loss
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

A lossless source code assigns a finite codeword to every source symbol. Let the code alphabet contain $`D`$ symbols. A binary code is the special case $`D=2`$, where codeword lengths are measured in bits.

Let $`l_i`$ denote the number of $`D`$-ary code symbols assigned to source symbol $`x_i`$. The expected code length is:

```math
L
=
\sum_{i=1}^{m}
p_i l_i
```

The engineering intuition is direct: common symbols should receive shorter codewords, while rare symbols can receive longer codewords. The objective is to minimize the expected number of code symbols per source symbol, not the maximum length of any single codeword.

An important correction is that code length does not equal probability. The ideal real-valued code length is proportional to the negative logarithm of probability:

```math
l_i^\star
=
-\log_D p_i
```

High probability means short code. Low probability means long code. In binary coding, halving a probability increases the ideal code length by one bit.

## 2. Prefix-free codes and Kraft inequality

A prefix-free code has no codeword that is the prefix of another codeword. Prefix-free codes can be decoded sequentially without ambiguity because the decoder knows immediately when a complete codeword has ended.

For a $`D`$-ary prefix-free code, the code lengths must satisfy Kraft inequality:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
1
```

The tree intuition is useful. A complete $`D`$-ary tree has $`D`$ branches at each internal node. A codeword of length $`l_i`$ consumes a fraction $`D^{-l_i}`$ of the leaves of the complete $`D`$-ary tree at that depth. Prefix-free codewords occupy disjoint subtrees, so the total occupied fraction cannot exceed one.

This tree argument directly proves necessity for prefix-free codes. It does not yet prove necessity for all uniquely decodable codes, because uniquely decodable codes need not correspond to disjoint instantaneous subtrees. McMillan's theorem fills that gap.

## 3. Prefix-free codes versus uniquely decodable codes

### Prefix-free code

A code is prefix-free if no codeword is the prefix of another codeword.

Decoding can proceed immediately from left to right. No future symbols are required to determine the current symbol. Every prefix-free code is uniquely decodable.

### Uniquely decodable code

A code is uniquely decodable if every finite concatenation of codewords has at most one decomposition into source symbols.

A uniquely decodable code need not be prefix-free. Decoding may require reading additional symbols before the current source symbol is determined. Unique decipherability is therefore a weaker condition than prefix-freeness, but the code still must avoid collisions between distinct source-symbol sequences.

The hierarchy is:

```text
prefix-free codes
⊂
uniquely decodable codes
⊂
non-singular symbol codes
```

A non-singular symbol code assigns distinct codewords to individual symbols. This alone does not guarantee unique decoding after codewords are concatenated.

The mathematical question is:

> If uniquely decodable codes are more general than prefix-free codes, can they achieve a smaller average length by violating the Kraft inequality?

McMillan's inequality shows that the answer is no.

## 4. McMillan inequality for uniquely decodable codes

**Theorem.** Let the code alphabet contain $`D`$ symbols. Let the source alphabet contain $`m`$ symbols. Suppose source symbol $`x_i`$ receives a codeword of integer length $`l_i`$, and suppose the code is uniquely decodable. Then:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
1
```

Define the Kraft sum:

```math
K
=
\sum_{i=1}^{m}
D^{-l_i}
```

For prefix-free codes, the inequality follows from the tree argument above. For general uniquely decodable codes, the proof uses an algebraic counting argument.

## 5. Algebraic proof of the McMillan inequality

### 5.1 Extend the code from symbols to source blocks

Consider a source block of length $`n`$:

```math
s
=
(
i_1,
i_2,
\dots,
i_n
)
```

Its concatenated encoded length is:

```math
L(s)
=
l_{i_1}
+
l_{i_2}
+
\dots
+
l_{i_n}
```

Each $`i_j`$ identifies one source symbol. Encoding the block means concatenating the $`n`$ single-symbol codewords. Unique decipherability means distinct source blocks cannot produce the same concatenated code string.

### 5.2 Expand the n-th power of the Kraft sum

Start from:

```math
K
=
\sum_{i=1}^{m}
D^{-l_i}
```

Raise both sides to the $`n`$-th power:

```math
K^n
=
\left(
\sum_{i=1}^{m}
D^{-l_i}
\right)^n
```

Expand:

```math
K^n
=
\sum_{i_1=1}^{m}
\sum_{i_2=1}^{m}
\dots
\sum_{i_n=1}^{m}
D^{
-
(
l_{i_1}
+
l_{i_2}
+
\dots
+
l_{i_n}
)
}
```

Using the block-length notation:

```math
K^n
=
\sum_s
D^{-L(s)}
```

The expansion enumerates every possible source-symbol block of length $`n`$.

### 5.3 Group source blocks by total encoded length

Let:

```math
A_{n,r}
```

denote the number of source blocks of length $`n`$ whose concatenated encoding has total length $`r`$.

Then:

```math
K^n
=
\sum_r
A_{n,r}
D^{-r}
```

Define:

```math
l_min
=
\min_i
l_i
```

```math
l_max
=
\max_i
l_i
```

Every length-$`n`$ source block satisfies:

```math
n l_min
\le
r
\le
n l_max
```

Therefore:

```math
K^n
=
\sum_{r=n l_min}^{n l_max}
A_{n,r}
D^{-r}
```

### 5.4 Use unique decipherability as an injection condition

There are exactly $`D^r`$ possible code-alphabet strings of length $`r`$. If two distinct source blocks mapped to the same encoded string, that string would have two different decompositions into source symbols, so decoding would not be unique. Therefore unique decipherability implies:

```math
A_{n,r}
\le
D^r
```

Substitute this bound into the grouped expansion:

```math
K^n
\le
\sum_{r=n l_min}^{n l_max}
D^r
D^{-r}
```

Therefore:

```math
K^n
\le
\sum_{r=n l_min}^{n l_max}
1
```

There are at most:

```math
n
(
l_max-l_min
)
+
1
```

possible integer values of $`r`$. Hence:

```math
K^n
\le
n
(
l_max-l_min
)
+
1
```

### 5.5 Take the n-th root and pass to the limit

Take the $`n`$-th root:

```math
K
\le
\left[
n
(
l_max-l_min
)
+
1
\right]^{1/n}
```

Use:

```math
\lim_{n\to\infty}
\left(
cn+1
\right)^{1/n}
=
1
```

for every fixed non-negative constant $`c`$. Therefore:

```math
K
\le
1
```

Conclude:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
1
```

> This completes the algebraic proof of the McMillan inequality.

### 5.6 Proof intuition

The quantity $`K^n`$ counts the weighted code-space demand of all length-$`n`$ source blocks. Unique decipherability prevents more than $`D^r`$ blocks from occupying the $`D^r`$ available encoded strings of total length $`r`$. The number of possible total lengths grows only linearly with $`n`$. If $`K`$ were larger than one, $`K^n`$ would grow exponentially. Exponential growth cannot remain bounded by a linear function. Therefore $`K`$ cannot exceed one.

This is the central algebraic insight.

### 5.7 Converse relationship

McMillan inequality gives a necessary condition for uniquely decodable codes. Kraft's converse gives the matching existence result: if a set of integer lengths satisfies:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
1
```

then a prefix-free code exists with those lengths. Every prefix-free code is uniquely decodable.

Therefore the same length condition characterizes feasible length sets for prefix-free coding and uniquely decodable coding. Abandoning the prefix-free property does not improve the set of achievable codeword lengths. The actual codewords may differ, but the feasible length profiles are governed by the same inequality.

## 6. From McMillan inequality to the entropy lower bound

Let:

```math
K
=
\sum_{i=1}^{m}
D^{-l_i}
```

By McMillan:

```math
K
\le
1
```

Define:

```math
q_i
=
\frac{
D^{-l_i}
}{
K
}
```

Then:

```math
\sum_{i=1}^{m}
q_i
=
1
```

Therefore $`Q = (q_1, \dots, q_m)`$ is a probability distribution. Starting from:

```math
q_i
=
\frac{
D^{-l_i}
}{
K
}
```

derive:

```math
D^{-l_i}
=
Kq_i
```

Take logarithms:

```math
l_i
=
-\log_D q_i
-
\log_D K
```

Expected code length is:

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
p_i
\log_D q_i
-
\log_D K
```

Define cross entropy in base $`D`$:

```math
H_D(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_D q_i
```

Therefore:

```math
L
=
H_D(P,Q)
-
\log_D K
```

Define Shannon entropy in base $`D`$:

```math
H_D(P)
=
-
\sum_{i=1}^{m}
p_i
\log_D p_i
```

Define KL divergence in base $`D`$:

```math
D_D(P || Q)
=
\sum_{i=1}^{m}
p_i
\log_D
\frac{p_i}
{q_i}
```

Derive:

```math
H_D(P,Q)
=
H_D(P)
+
D_D(P || Q)
```

Therefore:

```math
L
=
H_D(P)
+
D_D(P || Q)
-
\log_D K
```

Use:

```math
D_D(P || Q)
\ge
0
```

and:

```math
K
\le
1
```

which implies:

```math
-\log_D K
\ge
0
```

Conclude:

```math
L
\ge
H_D(P)
```

Every uniquely decodable $`D`$-ary code has expected length at least equal to the Shannon entropy measured in $`D`$-ary code symbols. For binary codes:

```math
D
=
2
```

so entropy is measured in bits.

Self-information is the information content of observing symbol $`x_i`$:

```math
I(x_i)
=
-\log_D p_i
```

Rare events carry more information. Common events carry less information. An event with probability one carries zero surprise. In binary coding, one bit corresponds to distinguishing between two equally likely outcomes.

| Probability | Self-information in binary |
| ----------: | -------------------------: |
|         `1` |                   `0 bits` |
|       `1/2` |                   `1 bit`  |
|       `1/4` |                  `2 bits`  |
|       `1/8` |                  `3 bits`  |

Entropy is expected self-information:

```math
H_D(P)
=
\sum_{i=1}^{m}
p_i
\log_D
\frac{1}
{p_i}
```

Equivalently:

```math
H_D(P)
=
-
\sum_{i=1}^{m}
p_i
\log_D p_i
```

Entropy measures the average number of $`D`$-ary code symbols needed per source symbol in an ideal lossless code. It is not the information content of one particular observation. It is the distribution-level average.

For a fair binary coin:

```math
H_2(P)
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
H_2(P)
=
1
```

For a highly imbalanced binary source with probabilities $`0.99`$ and $`0.01`$:

```math
H_2(P)
=
-
0.99\log_2(0.99)
-
0.01\log_2(0.01)
```

The entropy is much smaller than one bit because a highly predictable source can be compressed more aggressively. Entropy quantifies uncertainty before observing the symbol.

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
-\log_D
\left(
P(A)P(B)
\right)
=
-\log_D P(A)
-
\log_D P(B)
```

The logarithm turns multiplication of probabilities into addition of information.

## 7. Equality conditions and ideal code lengths

Starting from:

```math
L
=
H_D(P)
+
D_D(P || Q)
-
\log_D K
```

equality in the entropy lower bound requires both:

```math
D_D(P || Q)
=
0
```

and:

```math
K
=
1
```

The first condition implies:

```math
q_i
=
p_i
```

The second condition and the definition of $`q_i`$ imply:

```math
D^{-l_i}
=
p_i
```

Therefore:

```math
l_i
=
-\log_D p_i
```

The fraction of code space assigned to a symbol matches its source probability, so its ideal code length equals the negative logarithm of that probability.

Common symbols receive larger code-space shares. Larger code-space shares require shorter paths in a coding tree. Rare symbols receive smaller shares and longer codewords. In binary coding, halving the probability increases ideal code length by one bit.

| Probability | Ideal code-space share | Ideal code length |
| ----------: | ---------------------: | ----------------: |
|       `1/2` |                  `1/2` |           `1 bit` |
|       `1/4` |                  `1/4` |          `2 bits` |
|       `1/8` |                  `1/8` |          `3 bits` |
|    `1/1024` |               `1/1024` |         `10 bits` |

The ideal lengths may not be integers. Exact equality is not always achievable with single-symbol prefix codes. Shannon coding and block coding address this issue.

## 8. Shannon coding achieves the entropy bound within one code symbol

Choose:

```math
l_i
=
\lceil
-\log_D p_i
\rceil
```

Then:

```math
l_i
\ge
-\log_D p_i
```

so:

```math
D^{-l_i}
\le
p_i
```

Sum:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
\sum_{i=1}^{m}
p_i
```

Therefore:

```math
\sum_{i=1}^{m}
D^{-l_i}
\le
1
```

By Kraft's converse, a prefix-free code exists with these integer lengths.

Now use:

```math
l_i
<
-\log_D p_i
+
1
```

Take expectations:

```math
L
<
H_D(P)
+
1
```

Combine with the entropy lower bound:

```math
H_D(P)
\le
L
<
H_D(P)
+
1
```

McMillan provides the lower bound for every uniquely decodable code. Kraft's converse provides the existence of a prefix-free code with Shannon lengths. The gap is caused by integer codeword lengths. In binary coding, the gap is less than one bit per encoded source symbol.

## 9. Shannon's noiseless source coding theorem

This is the noiseless source coding theorem, not the noisy-channel coding theorem.

For an independent identically distributed block:

```math
X^n
=
(
X_1,
X_2,
\dots,
X_n
)
```

use:

```math
H_D(X^n)
=
nH_D(X)
```

Apply Shannon coding to source blocks:

```math
H_D(X^n)
\le
L_n
<
H_D(X^n)
+
1
```

Substitute:

```math
nH_D(X)
\le
L_n
<
nH_D(X)
+
1
```

Divide by $`n`$:

```math
H_D(X)
\le
\frac{L_n}{n}
<
H_D(X)
+
\frac{1}{n}
```

Take the limit:

```math
\lim_{n\to\infty}
\frac{L_n}{n}
=
H_D(X)
```

Entropy is a lower bound for any uniquely decodable lossless code. Block coding makes the upper bound approach the same value. Entropy is therefore the asymptotically optimal average number of $`D`$-ary code symbols per source symbol. For $`D=2`$, this is measured in bits.

## 10. Typical-set intuition

For a long independent identically distributed sequence, most probability mass concentrates on a typical set. A typical sequence has probability approximately:

```math
2^{-nH_2(X)}
```

The number of typical sequences is approximately:

```math
2^{nH_2(X)}
```

Identifying one typical sequence therefore requires approximately:

```math
nH_2(X)
```

bits.

Compression is possible because almost all observed long sequences lie inside a much smaller structured subset of all possible raw sequences. Entropy measures the exponential growth rate of this effective set.

The expected-length proof above is already sufficient for the coding bound used in this appendix. The typical-set view adds intuition for asymptotic block coding.

## 11. Practical coding methods

Huffman coding produces an optimal prefix code for known symbol probabilities under integer code lengths. Huffman lengths are typically close to:

```math
-\log_2p_i
```

Arithmetic coding encodes entire sequences and can approach fractional average lengths more closely. Modern compression systems combine probability modeling with efficient coding schemes.

The quality of compression depends on how accurately the probability model matches the source.

## 12. Cross entropy

Suppose data are generated by true distribution $`P`$, but a code is designed using assumed distribution $`Q`$. Assign ideal code length:

```math
l_Q(x_i)
=
-\log_2 q_i
```

Then the average number of bits under the true source is:

```math
H_2(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2 q_i
```

Entropy $`H_2(P)`$ is the best achievable average code length when the true distribution is known. Cross entropy $`H_2(P,Q)`$ is the average code length when data follow $`P`$ but coding decisions are based on $`Q`$. A wrong probability model wastes bits.

The exact source-coding bridge was:

```math
L
=
H_D(P)
+
D_D(P || Q)
-
\log_D K
```

Source coding uses a code-induced model $`Q`$. Neural-network classification uses a learned conditional model $`Q_\theta(Y \mid X)`$. In both cases, mismatch between the true distribution and the modeled distribution creates excess expected coding cost. Minimizing cross entropy reduces this mismatch penalty.

## 13. Cross entropy decomposition and KL divergence

Define:

```math
D_2(P || Q)
=
\sum_{i=1}^{m}
p_i
\log_2
\frac{p_i}
{q_i}
```

Start from:

```math
H_2(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2 q_i
```

Add and subtract the entropy term:

```math
H_2(P,Q)
=
-
\sum_{i=1}^{m}
p_i
\log_2 p_i
+
\sum_{i=1}^{m}
p_i
\log_2
\frac{p_i}
{q_i}
```

Therefore:

```math
H_2(P,Q)
=
H_2(P)
+
D_2(P || Q)
```

Entropy is the irreducible source uncertainty. KL divergence is the additional coding cost caused by using the wrong distribution. Cross entropy equals unavoidable uncertainty plus mismatch penalty.

## 14. Why KL divergence is non-negative

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
D_D(P || Q)
\ge
0
```

The proof used natural logarithms, but changing the logarithm base only multiplies KL divergence by a positive constant. Equality holds when $`P = Q`$. Cross entropy is minimized when the modeled distribution matches the true distribution. This is why minimizing cross entropy trains the model to approximate the data-generating conditional distribution.

## 15. Cross entropy in neural-network classification

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
\sum_{k=1}^{C}
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

## 16. Binary cross entropy

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

## 17. Why cross entropy is more informative than accuracy

For a positive label:

| Predicted probability | Accuracy at threshold `0.5` |             BCE |
| --------------------: | --------------------------: | --------------: |
|                `0.51` |                     correct | relatively high |
|                `0.80` |                     correct |           lower |
|                `0.99` |                     correct |        very low |
|                `0.01` |                   incorrect |  extremely high |

Accuracy only checks whether the probability crosses a threshold. Cross entropy also measures confidence quality. A differentiable loss is needed for gradient-based optimization. Accuracy is piecewise constant with respect to model parameters and provides almost no useful local gradient.

## 18. Why sigmoid and BCE simplify to `p - y`

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

## 19. Interpretation boundaries

Shannon coding theorems describe asymptotic lossless compression under stated probabilistic assumptions. Real datasets may not be independent and identically distributed. Practical models only approximate the unknown source distribution.

Minimizing training cross entropy does not automatically guarantee generalization. Low cross entropy does not automatically guarantee perfect calibration. Cross entropy is a proper scoring rule, but finite data, optimization limits, distribution shift, and model misspecification still matter. Accuracy, calibration, robustness, and error analysis remain separate evaluation concerns.

## 20. Conceptual summary

```text
uniquely decodable code
→ McMillan inequality
→ normalized code-space distribution
→ entropy lower bound
→ ideal lengths equal negative log probabilities
→ Shannon coding approaches the bound
→ block coding reaches entropy asymptotically
→ mismatched code distribution creates cross entropy
→ mismatch penalty is KL divergence
→ neural-network cross entropy trains predicted probabilities
```

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

## 21. Further reading

- A Mathematical Theory of Communication
- Two Inequalities Implied by Unique Decipherability
- Elements of Information Theory
- Pattern Recognition and Machine Learning
- The Elements of Statistical Learning
- Stanford CS229 Notes: Generalized Linear Models
- Stanford CS231n Notes: Linear Classification and Loss Functions

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Back to MLP Forward Pass and Backpropagation](03_mlp_forward_and_backprop.md)
