# Kernel Methods: Feature Spaces, Similarity, and Soft Margins

[Back to Learning From Data Theory Notebook](../README.md)

This chapter corresponds to Caltech `Learning From Data` Lecture 15: Kernel Methods. The central question is:

```text
How can a linear geometric algorithm produce nonlinear decision boundaries
without explicitly constructing the transformed feature vector?
```

![Kernel feature-space geometry](../assets/kernel_feature_space_geometry.png)

Figure 1: a nonlinear pattern in original input space can become linearly separable after a feature transformation. A kernel method can operate through the inner products in that feature space.

![Kernel Gram PSD geometry](../assets/kernel_gram_psd_geometry.png)

Figure 2: a valid inner-product kernel produces a positive semidefinite Gram matrix because every quadratic form equals a squared norm in feature space.

![Soft-margin slack and hinge geometry](../assets/soft_margin_slack_hinge_geometry.png)

Figure 3: soft-margin SVM introduces slack for margin violations. The hinge-loss view is the unconstrained regularized form of the same tradeoff, up to scaling conventions.

## 0. Source Separation

### Caltech Core

Lecture 15 extends SVMs through nonlinear transformations, kernels, the kernel trick, nonseparable data, and soft margins.

### Formal Derivation

This note derives the feature-space SVM dual, kernel prediction form, PSD Gram-matrix property, soft-margin primal, hinge-loss equivalence, and soft-margin dual box constraint.

### Stanford CS229 Extension

CS229 supplies the standard mathematical support for functional/geometric margins, primal/dual SVM, kernelization, Lagrange duality, KKT conditions, and soft-margin SVM.

### Stanford CS229M / Theory Extension

The theory bridge appears in the distinction between ambient feature dimension and effective statistical complexity. Infinite-dimensional feature maps do not automatically imply uncontrolled learning when the algorithm constrains norm, margin, or selected solution structure.

### Modern Perspective

RKHS language is used only as a preview: many kernels can be understood as inner products in a Hilbert space of functions. Full RKHS theory belongs to later study.

## 1. Return to Nonlinear Feature Transforms

T1 Lecture 3 introduced feature transforms:

```math
x
\mapsto
\Phi(x).
```

After transformation, a linear score becomes

```math
f(x)
=
w^\top\Phi(x)+b.
```

The model is linear in feature space, but the induced decision boundary in original input space can be nonlinear:

```text
linear in Phi(x)
!=
linear in x
```

This distinction is the conceptual bridge from Lecture 3 to kernels. The SVM still finds a separating hyperplane, but the hyperplane lives in the represented space rather than necessarily in the raw input space.

## 2. Why the Dual Matters

The hard-margin SVM dual from Lecture 14 is:

```math
\begin{aligned}
\max_{\alpha}\quad
&
\sum_i\alpha_i
-
\frac12
\sum_i\sum_j
\alpha_i\alpha_j y_i y_j x_i^\top x_j
\\
\text{subject to}\quad
&
\alpha_i\ge0,
\\
&
\sum_i\alpha_i y_i=0.
\end{aligned}
```

The training examples enter through inner products $x_i^\top x_j$.

After a feature transform, the primal score is

```math
f(x)
=
w^\top\Phi(x)+b,
```

and the stationarity relation becomes

```math
w
=
\sum_i\alpha_i y_i\Phi(x_i).
```

Substituting into the dual gives inner products

```math
\Phi(x_i)^\top\Phi(x_j).
```

That is the bridge to kernels.

## 3. Kernel Definition

A kernel is a function

```math
K(x,z)
=
\langle \Phi(x),\Phi(z)\rangle
```

for some feature representation $\Phi$ into an inner-product space.

It is tempting but imprecise to say:

```text
kernel = similarity function
```

A better statement is:

```text
a kernel is a similarity-like function with the mathematical structure
needed to behave as an inner product in some feature space
```

This structure matters because the SVM dual and prediction rule require inner-product geometry, not arbitrary pairwise scores.

## 4. Kernel Trick

Instead of explicitly computing $\Phi(x)$, compute

```math
K(x,z).
```

The feature-space dual becomes

```math
\begin{aligned}
\max_{\alpha}\quad
&
\sum_i\alpha_i
-
\frac12
\sum_i\sum_j
\alpha_i\alpha_j y_i y_j K(x_i,x_j)
\\
\text{subject to}\quad
&
\alpha_i\ge0,
\\
&
\sum_i\alpha_i y_i=0.
\end{aligned}
```

The prediction rule is

```math
f(x)
=
\sum_{i\in SV}
\alpha_i y_i K(x_i,x)
+
b.
```

The algorithm can therefore act as if it built a linear separator in feature space, while evaluating only pairwise kernel functions.

### What This Does NOT Imply

Do not write:

```text
kernel = nonlinear feature transform
```

The feature map and the kernel are different objects. The feature map sends inputs into a represented space. The kernel computes an inner product corresponding to such a representation.

## 5. High-Dimensional and Infinite-Dimensional Feature Spaces

Some kernels correspond to very high-dimensional feature maps. Some correspond to infinite-dimensional Hilbert-space representations. The Gaussian kernel is the standard example used to build this intuition.

The crucial T4 point is:

```text
infinite-dimensional representation
!=
uncontrolled learner
```

The reason is that representation dimension and effective statistical complexity are different objects. The actual learner is shaped by:

- norm control in feature space;
- margin constraints;
- soft-margin regularization;
- sample size;
- kernel hyperparameters;
- support-vector structure;
- the optimization problem used to select a solution.

T2 warned that raw hypothesis-class capacity can be too crude if it ignores the selected solution. T3 showed that regularization changes the solution preference without necessarily changing the nominal representable family. Kernel SVM makes this concrete: a rich or infinite feature representation can still produce a meaningful learner when the selected solution is controlled by norm/margin and the evidence is evaluated under appropriate sampling assumptions.

### What This Does NOT Imply

- Infinite feature space is not automatically overfitting.
- Infinite feature space is not automatically safe.
- A kernel method still makes representation and similarity assumptions.
- The training distribution, validation process, and hyperparameter search still determine what can be claimed.

## 6. Valid Kernels and PSD Gram Matrices

For points

```math
x_1,\dots,x_N,
```

define the Gram matrix

```math
K_{ij}
=
K(x_i,x_j).
```

### Theorem: Inner-Product Kernels Produce PSD Gram Matrices

#### Assumptions

- There exists a feature map $\Phi$ into an inner-product space.
- $K(x,z)=\langle \Phi(x),\Phi(z)\rangle$.
- $c\in\mathbb{R}^N$ is any finite coefficient vector.

#### Claim

The Gram matrix is positive semidefinite:

```math
c^\top K c\ge0.
```

#### Derivation / Proof Idea

Start with the quadratic form:

```math
c^\top K c
=
\sum_i\sum_j c_i c_j K(x_i,x_j).
```

Use the kernel definition:

```math
c^\top K c
=
\sum_i\sum_j
c_i c_j
\langle \Phi(x_i),\Phi(x_j)\rangle.
```

By bilinearity of the inner product,

```math
c^\top K c
=
\left\langle
\sum_i c_i\Phi(x_i),
\sum_j c_j\Phi(x_j)
\right\rangle.
```

Thus

```math
c^\top K c
=
\left\|
\sum_i c_i\Phi(x_i)
\right\|^2
\ge
0.
```

#### Interpretation

The PSD condition is not a technical afterthought. It is the finite-sample algebraic signature that kernel values can act like inner products.

#### What This Does NOT Imply

The derivation proves PSD when an inner-product feature representation exists. It does not prove that every symmetric similarity score is a valid kernel.

#### Research Use

When a paper proposes a new kernel-like similarity, ask whether finite Gram matrices are PSD or whether the method is using an indefinite similarity outside standard kernel-SVM theory.

## 7. Mercer Theorem Caveat

A common trap is to write:

```text
any similarity function is a kernel
```

This is false.

The precise landscape is:

- A valid inner-product kernel produces symmetric PSD Gram matrices on every finite set of inputs.
- For many modern kernel-method treatments, the finite Gram-matrix PSD condition is the operational condition needed for algorithms on finite data.
- Under suitable conditions, a symmetric PSD kernel has an associated Hilbert-space feature representation.
- Stronger classical Mercer theorem statements require additional assumptions, such as continuity and compactness conditions for integral-operator formulations.

The working lesson is:

```text
kernel validity is a mathematical property,
not a label attached to any intuitive similarity score
```

## 8. Common Kernels

### Linear Kernel

```math
K(x,z)=x^\top z.
```

This is the original input-space inner product. It assumes the raw or already engineered feature coordinates define the relevant geometry.

### Polynomial Kernel

A common form is

```math
K(x,z)
=
(x^\top z+c)^p.
```

The degree $p$ controls which polynomial interactions are made available. The offset $c$ changes how lower-order terms enter. This kernel changes the feature-space geometry by making certain interaction patterns linearly accessible.

### Gaussian / RBF Kernel

```math
K(x,z)
=
\exp
\left(
-
\frac{\|x-z\|^2}{2\sigma^2}
\right).
```

The width $\sigma$ controls how quickly similarity decays with distance. Smaller $\sigma$ makes similarity more local; larger $\sigma$ makes similarity broader. This is a geometric statement, not a universal bias-variance law. Its effect depends on data distribution, feature scaling, sample size, and regularization.

### Hyperparameters Alter Geometry

Kernel hyperparameters are not harmless implementation settings. They define which examples appear close, aligned, or similar in the induced feature geometry. Their selection therefore belongs to the T3 validation and adaptive-selection story.

## 9. Soft-Margin SVM

Real datasets are often not separable in the chosen feature space, or perfect separation may be undesirable because it fits noise. Soft-margin SVM introduces slack variables

```math
\xi_i\ge0
```

with constraints

```math
y_i(w^\top\Phi(x_i)+b)
\ge
1-\xi_i.
```

The primal is

```math
\begin{aligned}
\min_{w,b,\xi}\quad
&
\frac12\|w\|^2
+
C\sum_i\xi_i
\\
\text{subject to}\quad
&
y_i(w^\top\Phi(x_i)+b)\ge1-\xi_i,
\\
&
\xi_i\ge0.
\end{aligned}
```

The tradeoff is:

```text
margin preference
vs
training violations
```

The coefficient $C$ controls how expensive slack is relative to the norm penalty. Larger $C$ penalizes violations more strongly; smaller $C$ permits more margin violations to obtain a smaller norm. This is exactly a T3 regularization tradeoff.

## 10. Hinge-Loss View

Define the signed score

```math
s_i
=
y_i f(x_i).
```

The hinge loss is

```math
\ell_{\mathrm{hinge}}(s_i)
=
\max(0,1-s_i)
=
\max(0,1-y_i f(x_i)).
```

For fixed $w,b$, the smallest feasible slack satisfying

```math
\xi_i\ge0,
\qquad
\xi_i\ge 1-y_i f(x_i)
```

is

```math
\xi_i^*
=
\max(0,1-y_i f(x_i)).
```

Substituting optimal slacks into the soft-margin primal gives the regularized hinge-loss objective:

```math
\min_{w,b}
\frac12\|w\|^2
+
C\sum_i
\max(0,1-y_i f(x_i)).
```

Different texts may divide the empirical loss by $N$ or place the regularization coefficient differently. The conceptual equivalence is the same, subject to scaling conventions.

Interpretation:

- $y_i f(x_i)>1$: correctly classified and outside the margin; zero hinge loss.
- $y_i f(x_i)=1$: exactly on the margin; zero hinge loss but active constraint.
- $0<y_i f(x_i)<1$: correctly classified but inside the margin; positive hinge loss.
- $y_i f(x_i)<0$: misclassified; positive hinge loss greater than $1$.

### What This Does NOT Imply

Hinge loss is not log loss and does not by itself produce calibrated probabilities. A large positive SVM score is not automatically a calibrated probability confidence.

## 11. Soft-Margin Dual

### Theorem: Slack Variables Create Box Constraints

#### Assumptions

- Soft-margin primal with slack penalty $C\sum_i\xi_i$.
- Multipliers $\alpha_i\ge0$ for margin constraints.
- Multipliers $\mu_i\ge0$ for slack nonnegativity constraints $\xi_i\ge0$.

#### Claim

The dual has the same quadratic kernelized objective as the hard-margin dual, but the multipliers satisfy

```math
0\le\alpha_i\le C
```

instead of only

```math
\alpha_i\ge0.
```

#### Derivation / Proof Idea

Write the Lagrangian:

```math
L
=
\frac12\|w\|^2
+
C\sum_i\xi_i
-
\sum_i\alpha_i
\left[
y_i(w^\top\Phi(x_i)+b)-1+\xi_i
\right]
-
\sum_i\mu_i\xi_i.
```

Stationarity with respect to $\xi_i$ gives

```math
\frac{\partial L}{\partial \xi_i}
=
C-\alpha_i-\mu_i
=
0.
```

Because $\mu_i\ge0$,

```math
\alpha_i
=
C-\mu_i
\le
C.
```

Together with dual feasibility $\alpha_i\ge0$, this gives

```math
0\le\alpha_i\le C.
```

Stationarity with respect to $w$ and $b$ gives the same structural equations:

```math
w
=
\sum_i\alpha_i y_i\Phi(x_i),
\qquad
\sum_i\alpha_i y_i=0.
```

The dual objective is

```math
\begin{aligned}
\max_{\alpha}\quad
&
\sum_i\alpha_i
-
\frac12
\sum_i\sum_j
\alpha_i\alpha_j y_i y_j K(x_i,x_j)
\\
\text{subject to}\quad
&
0\le\alpha_i\le C,
\\
&
\sum_i\alpha_i y_i=0.
\end{aligned}
```

#### Interpretation

The upper bound $C$ limits how much any single training example can influence the dual solution. This is the dual footprint of allowing violations but penalizing them.

#### What This Does NOT Imply

Points with nonzero $\alpha_i$ in soft-margin SVM are not simply "the misclassified points." They include points on the margin, inside the margin, and sometimes misclassified points, with interpretation depending on whether $\alpha_i$ is $0$, between $0$ and $C$, or at $C$.

#### Research Use

The box constraint is a compact way to audit the fit-violation tradeoff. Ask how $C$ was selected, which data influenced that selection, and whether final evaluation remained independent.

## 12. Research Lens

When reading a kernel-method paper, ask:

- What notion of similarity does the kernel encode?
- Which invariances or smoothness assumptions are implied?
- Is the kernel appropriate for the domain representation?
- Are distances meaningful before applying the kernel?
- How sensitive is performance to kernel hyperparameters?
- Does high-dimensional feature space imply high effective complexity? Usually not by itself.
- What norm, margin, or regularization mechanism controls the selected solution?
- What happens under distribution shift when similarity geometry changes?
- Was the kernel chosen using validation or benchmark feedback?

### Existing Repository Links

- T1 Lecture 3: feature transforms and linear-in-feature-space models: [feature transforms](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md).
- T2: capacity control and the limits of raw parameter counting: [modern capacity control](../part2_generalization_theory/09_modern_uniform_convergence_and_capacity_control.md).
- T3: regularization and validation selection: [regularization](../part3_fitting_regularization_validation/12_caltech_l12_regularization_constraints_inductive_bias.md), [validation](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md).
- Week 4: canvas shift shows that a representation/similarity geometry can fail under real input shift: [Canvas diagnostic](../../../reports/week4/15_canvas_diagnostic_v1_inventory_and_failure_taxonomy.md).
- Week 5: SVM margin should not be confused with calibrated probability: [calibration metrics](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md).

[Back to Learning From Data Theory Notebook](../README.md)
