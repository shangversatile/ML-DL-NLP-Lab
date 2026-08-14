# Radial Basis Functions: Local Representation and Similarity

[Back to Learning From Data Theory Notebook](../README.md)

This chapter corresponds to Caltech `Learning From Data` Lecture 16: Radial Basis Functions. The goal is to understand locality as a representation assumption, not to collapse every "RBF" phrase into the same algorithm.

Three objects must remain distinct:

```text
RBF basis model
RBF / Gaussian kernel
kernel SVM
```

They are related, but they are not identical.

![RBF local basis centers and widths](../assets/rbf_local_basis_centers_widths.png)

Figure 1: RBF basis functions respond locally around chosen centers. The width controls the spatial scale of the response in the chosen metric.

![RBF model versus kernel SVM](../assets/rbf_model_vs_kernel_svm.png)

Figure 2: an RBF model uses explicit finite centers as basis functions. A Gaussian-kernel SVM uses kernel evaluations against support vectors selected by the dual optimization.

## 0. Source Separation

### Caltech Core

Lecture 16 introduces radial basis functions, local representation, centers, widths, and the relationship among several learning models and techniques.

### Formal Derivation

This note defines RBF units, RBF models, the design matrix for fixed centers and widths, and the resulting linear fitting problem over basis responses.

### Stanford CS229 Extension

CS229 kernel material supports the comparison between explicit feature maps and kernelized dual prediction. The RBF network/model discussion remains separate from kernel SVM.

### Modern Perspective

The modern bridge is representation learning: locality is only meaningful in a representation where distance encodes relevant mechanisms. This note previews that point without turning into a deep representation-learning chapter.

## 1. Local Basis Representation

An RBF unit has the form

```math
\phi_k(x)
=
\exp
\left(
-
\frac{\|x-c_k\|^2}{2\sigma_k^2}
\right).
```

Here:

- $c_k$ is the center;
- $\sigma_k>0$ is the width;
- $\|x-c_k\|$ is the distance under the chosen metric;
- $\phi_k(x)$ is largest near $c_k$ and decays as $x$ moves away.

The word radial refers to dependence on distance from the center. The word basis means the unit becomes one coordinate of a transformed representation.

This is a similarity statement:

```text
points near the same center receive related activations
```

but the statement is only as meaningful as the distance used to define "near."

## 2. RBF Network / Model

An RBF model can be written as

```math
g(x)
=
\sum_{k=1}^{K}
w_k\phi_k(x)
+
b.
```

The model decomposes into:

```text
center selection
-> representation

width selection
-> spatial scale / locality

weights
-> prediction
```

Once centers and widths are fixed, the learner sees the transformed feature vector

```math
z(x)
=
(\phi_1(x),\ldots,\phi_K(x)).
```

Then the output layer is linear in $z(x)$:

```math
g(x)=w^\top z(x)+b.
```

This directly reconnects Lecture 16 to T1 Lecture 3: a nonlinear feature transform can make a linear-in-parameters model behave nonlinearly in original input space.

## 3. Why Locality Matters

An RBF representation says that nearby points under a chosen metric should have related basis activation. Therefore the metric itself contains an inductive assumption.

The important question is:

```text
What does "nearby" mean in the represented space?
```

If $x$ is a vector of well-scaled physical measurements, Euclidean distance may be defensible. If $x$ is raw pixels, token counts, mixed tabular fields, or measurements collected under changing conditions, Euclidean distance may mix irrelevant variation with relevant structure.

Locality is therefore not only a modeling detail. It is a claim about the representation:

```text
representation
-> distance
-> locality
-> prediction sharing
```

## 4. Center Selection

RBF centers can be chosen in several ways:

- fixed centers on a grid;
- selected training examples;
- clustering centers, such as k-means centers;
- learned centers optimized jointly or in stages.

Each choice changes the effective representation.

If centers are fixed independently of data, the representation is predetermined. If centers are selected from training data or by clustering, the representation is data-dependent. If centers are learned with the predictor, the boundary between representation learning and output fitting becomes less clean.

No method is universal. A grid may fail in high dimensions. Training-example centers may be too many or too sample-specific. Clustering may ignore labels. Learned centers may introduce a harder nonconvex optimization problem and additional selection risk.

## 5. Width Selection

The width $\sigma_k$ controls the spatial scale of the basis response.

Small width:

```text
high locality
high sensitivity
potentially fragmented fitting
```

Large width:

```text
broad smooth influence
lower spatial resolution
```

This is structural intuition, not a universal law. Whether small width overfits or large width underfits depends on sample size, noise, target smoothness, center placement, regularization, and the metric's meaning.

Width selection is a hyperparameter-selection problem when widths are tuned by validation or benchmark feedback. That links directly to T3.

## 6. Fit Output Weights

### Theorem: Fixed RBF Representation Makes Output Fitting Linear in Weights

#### Assumptions

- Centers $c_1,\ldots,c_K$ are fixed.
- Widths $\sigma_1,\ldots,\sigma_K$ are fixed.
- The RBF basis functions are

```math
\phi_k(x)
=
\exp
\left(
-
\frac{\|x-c_k\|^2}{2\sigma_k^2}
\right).
```

#### Claim

The RBF model

```math
g(x)
=
\sum_{k=1}^{K}
w_k\phi_k(x)
+
b
```

is linear in the output weights $w_1,\ldots,w_K,b$.

#### Derivation / Proof Idea

For a dataset $x_1,\ldots,x_N$, define the design matrix

```math
Z_{ik}
=
\phi_k(x_i).
```

Let $\tilde Z$ append a column of ones for the intercept:

```math
\tilde Z
=
\begin{bmatrix}
Z & \mathbf{1}
\end{bmatrix}.
```

Let

```math
\beta
=
(w_1,\ldots,w_K,b)^\top.
```

Then predictions on the training set are

```math
\hat y
=
\tilde Z\beta.
```

For squared loss, output fitting becomes linear least squares over the transformed representation:

```math
\min_{\beta}
\|\tilde Z\beta-y\|_2^2.
```

With ridge regularization on output weights, it becomes

```math
\min_{\beta}
\|\tilde Z\beta-y\|_2^2
+
\lambda\|w\|_2^2.
```

#### Interpretation

The nonlinearity is in representation construction, not in the final output layer once centers and widths are fixed.

#### What This Does NOT Imply

This does not make the whole RBF modeling pipeline linear if centers or widths are selected adaptively. Center choice, width choice, and validation loops remain part of the learning process.

#### Research Use

Separate representation design from output fitting. Ask which part was fixed before seeing data and which part was selected using data.

## 7. RBF Model versus Gaussian Kernel

### RBF Model

An RBF model uses explicit finite basis functions centered at selected centers:

```math
z(x)
=
(\phi_1(x),\ldots,\phi_K(x)).
```

Prediction has the finite explicit form

```math
g(x)
=
\sum_{k=1}^{K}
w_k\phi_k(x)
+
b.
```

The centers may be grid points, prototypes, training examples, cluster centers, or learned parameters.

### Gaussian / RBF Kernel

The Gaussian kernel is the pairwise function

```math
K(x,z)
=
\exp
\left(
-
\frac{\|x-z\|^2}{2\sigma^2}
\right).
```

It is used inside kernel methods to compute inner products in an implicit feature space.

The distinction is mandatory:

```text
RBF basis function:
one explicit feature response around a center

Gaussian/RBF kernel:
pairwise inner-product function used by a kernel method
```

Do not use the terms interchangeably.

## 8. RBF Model versus Kernel SVM

An RBF model and a Gaussian-kernel SVM can both rely on local similarity, but their solution structures differ.

```text
RBF model:
finite explicit basis centers

kernel SVM:
dual expansion involving support vectors / kernel evaluations
```

An RBF model predicts through chosen basis centers:

```math
g_{\mathrm{RBF}}(x)
=
\sum_{k=1}^{K}
w_k
\exp
\left(
-
\frac{\|x-c_k\|^2}{2\sigma_k^2}
\right)
+
b.
```

A Gaussian-kernel SVM predicts through support vectors:

```math
f_{\mathrm{SVM}}(x)
=
\sum_{i\in SV}
\alpha_i y_i
\exp
\left(
-
\frac{\|x_i-x\|^2}{2\sigma^2}
\right)
+
b.
```

In the RBF model, centers are part of the representation design. In the kernel SVM, support vectors are selected by the margin optimization and KKT structure.

## 9. Relation to Nearest-Neighbor Intuition

RBF methods share an intuition with nearest-neighbor methods: local similarity matters. A point is influenced more by nearby centers or nearby support vectors than by distant ones, especially with a narrow Gaussian width.

But RBF is not equivalent to nearest neighbor.

Nearest-neighbor methods use local decision rules based directly on nearby training examples. RBF models build a smooth basis representation and then fit output weights. Kernel SVMs solve a margin-regularized dual problem and use support-vector coefficients. The comparison is useful for locality intuition, not for algorithmic equivalence.

## 10. Curse of Dimensionality Caveat

Euclidean distance can become less informative in high-dimensional spaces. Distances may concentrate, irrelevant coordinates may dominate, and raw coordinate closeness may fail to correspond to semantic or causal similarity.

Therefore:

```text
distance-based locality
```

depends on meaningful representation.

This creates a direct bridge to modern representation learning. A learned embedding can be viewed as an attempt to construct a space where local distances, inner products, angles, and neighborhoods align better with the predictive mechanism.

### What This Does NOT Imply

Representation learning does not automatically solve locality. If the learned representation is biased by sampling artifacts, spurious features, or validation feedback, the induced geometry can still fail under deployment shift.

## 11. Research Lens

When reading an RBF or locality-based paper, ask:

- What metric defines locality?
- Was that metric chosen before seeing data?
- Are centers representative of the target population?
- How were widths selected?
- Is locality stable under distribution shift?
- Does the representation preserve the mechanism relevant to prediction?
- Could two points be close observationally but mechanistically different?
- Could two points be far observationally but predictively similar?
- Which part of the pipeline is explicit representation, and which part is learned solution selection?

### Existing Repository Links

- T1 Lecture 3 introduced nonlinear transforms and linear output fitting: [feature transforms](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md).
- T3 Lecture 10 introduced learned representations in neural networks: [neural networks and representation](../part3_fitting_regularization_validation/10_caltech_l10_neural_networks_backpropagation_representation.md).
- T3 Lecture 13 explains why tuning centers, widths, and kernel hyperparameters is a validation-selection issue: [validation](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md).
- Week 4 Canvas-Diagnostic-v1 shows a concrete case where raw input shift can make learned geometry unreliable: [Canvas diagnostic](../../../reports/week4/15_canvas_diagnostic_v1_inventory_and_failure_taxonomy.md).

[Back to Learning From Data Theory Notebook](../README.md)
