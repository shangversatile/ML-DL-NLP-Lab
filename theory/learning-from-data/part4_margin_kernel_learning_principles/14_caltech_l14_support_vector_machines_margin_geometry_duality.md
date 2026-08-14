# Support Vector Machines: Margin, Geometry, and Duality

[Back to Learning From Data Theory Notebook](../README.md)

This chapter corresponds to Caltech `Learning From Data` Lecture 14: Support Vector Machines. The goal is not to introduce one more classifier. The goal is to see SVM as a convergence point of geometry, optimization, regularization, capacity, and generalization.

![Functional versus geometric margin](../assets/svm_functional_vs_geometric_margin.png)

Figure 1: multiplying the parameters by a positive constant leaves the separating boundary unchanged but changes the functional margin. The geometric margin divides out this arbitrary scale.

![Maximum-margin support vectors](../assets/svm_maximum_margin_support_vectors.png)

Figure 2: the maximum-margin separator is controlled by the active margin constraints. These active points become support vectors in the dual representation.

![Primal-dual SVM structure](../assets/svm_primal_dual_structure.png)

Figure 3: the primal problem expresses geometry through $w,b$ and constraints. The dual expresses the solution through training examples and nonzero coefficients.

## 0. Source Separation

### Caltech Core

Lecture 14 introduces the support-vector-machine idea through separating hyperplanes, margin maximization, support vectors, and the geometric reason a maximum-margin classifier is not just another linear separator.

### Formal Derivation

This note gives the hard-margin primal, hyperplane distance, functional margin, geometric margin, Lagrangian dual, KKT conditions, and support-vector solution structure in a self-contained derivation.

### Stanford CS229 Extension

CS229 supplies the standard mathematical refinement: functional versus geometric margin, optimal-margin classifier, Lagrange duality, KKT conditions, and the support-vector interpretation. These derivations are not attributed to Caltech unless they are part of the Caltech lecture's conceptual core.

### Stanford CS229M / Theory Extension

The CS229M-style bridge appears only in the capacity discussion: modern generalization analysis often depends on the selected solution, norm, margin, stability, or compression, not only the nominal parameter count.

### Research Lens

The research use of SVM is to ask which geometry the representation induces and what kind of simple solution the learning algorithm prefers.

## 1. Start from the Separating Hyperplane

For binary labels,

```math
y_i\in\{-1,+1\},
```

start with the affine score

```math
f(x)=w^\top x+b
```

and classifier

```math
g(x)=\mathrm{sign}(f(x)).
```

The decision boundary is the set

```math
\{x:w^\top x+b=0\}.
```

This set is a hyperplane when $w\ne 0$. The vector $w$ is normal to the hyperplane because any two points $x_a,x_b$ on the hyperplane satisfy

```math
w^\top x_a+b=0,
\qquad
w^\top x_b+b=0.
```

Subtracting gives

```math
w^\top(x_a-x_b)=0.
```

The displacement $x_a-x_b$ lies inside the hyperplane, so $w$ is orthogonal to every direction tangent to the hyperplane.

### Derivation: Point-to-Hyperplane Distance

#### Assumptions

- $w\ne 0$.
- The hyperplane is $H=\{x:w^\top x+b=0\}$.
- $x_0$ is any point in input space.

#### Claim

The signed distance from $x_0$ to $H$ is

```math
\frac{w^\top x_0+b}{\|w\|_2},
```

and the unsigned distance is

```math
\frac{|w^\top x_0+b|}{\|w\|_2}.
```

#### Derivation / Proof Idea

Move from $x_0$ along the normal direction until the hyperplane is reached. Let

```math
x_\perp
=
x_0
-
t w.
```

We need $x_\perp$ to satisfy the hyperplane equation:

```math
w^\top(x_0-tw)+b=0.
```

Therefore

```math
w^\top x_0
-
t\|w\|_2^2
+
b
=
0,
```

so

```math
t
=
\frac{w^\top x_0+b}{\|w\|_2^2}.
```

The displacement length is

```math
\|x_0-x_\perp\|_2
=
\|t w\|_2
=
\frac{|w^\top x_0+b|}{\|w\|_2}.
```

#### Interpretation

The score $w^\top x_0+b$ is not a distance by itself. It becomes a geometric distance only after division by $\|w\|_2$.

#### What This Does NOT Imply

The distance is meaningful only in the geometry of the current representation. If $x$ is a bad representation, Euclidean distance to a hyperplane may be a poor description of semantic or mechanistic similarity.

#### Research Use

Whenever a paper invokes a margin, ask: margin in which representation, with which norm, and under which feature scaling?

## 2. Functional Margin

For a labeled training example $(x_i,y_i)$, define the functional margin

```math
\hat\gamma_i
=
y_i(w^\top x_i+b).
```

The sign determines correctness:

- if $\hat\gamma_i>0$, the example is classified correctly;
- if $\hat\gamma_i=0$, the example lies on the decision boundary;
- if $\hat\gamma_i<0$, the example is misclassified.

The magnitude records the signed numerical score after aligning the sign with the label. A larger value means the score is farther from zero in parameter units.

The problem is scale dependence. For any $c>0$,

```math
\mathrm{sign}(cw^\top x+cb)
=
\mathrm{sign}(w^\top x+b),
```

so $(w,b)$ and $(cw,cb)$ define the same classifier. But the functional margin becomes

```math
y_i((cw)^\top x_i+cb)
=
c\,y_i(w^\top x_i+b)
=
c\hat\gamma_i.
```

Thus functional margin alone cannot define the geometric quality of the classifier. It mixes boundary geometry with arbitrary parameter scale.

## 3. Geometric Margin

The geometric margin divides out parameter scale:

```math
\gamma_i
=
\frac{y_i(w^\top x_i+b)}{\|w\|_2}.
```

For $c>0$,

```math
\frac{y_i((cw)^\top x_i+cb)}{\|cw\|_2}
=
\frac{c\,y_i(w^\top x_i+b)}{c\|w\|_2}
=
\gamma_i.
```

The geometric margin is therefore invariant to positive rescaling of $(w,b)$. It measures the signed distance from $x_i$ to the decision boundary in the represented Euclidean geometry.

The key distinction is:

```text
parameter scale
!=
decision-boundary geometry
```

Functional margin is about the numerical scale of the score. Geometric margin is about the location of the boundary relative to points after fixing the representation and norm.

## 4. Maximum-Margin Principle

For separable data, there exists $(w,b)$ such that

```math
y_i(w^\top x_i+b)>0
\quad
\text{for all } i.
```

Because scaling is arbitrary, choose the canonical scale in which the smallest functional margin equals $1$:

```math
\min_i y_i(w^\top x_i+b)=1.
```

Equivalently,

```math
y_i(w^\top x_i+b)\ge 1
\quad
\text{for all } i.
```

Under this scale, the minimum geometric margin is

```math
\gamma
=
\min_i
\frac{y_i(w^\top x_i+b)}{\|w\|_2}
=
\frac{1}{\|w\|_2}.
```

The distance between the two margin planes

```math
w^\top x+b=1
```

and

```math
w^\top x+b=-1
```

is

```math
\frac{2}{\|w\|_2}.
```

Therefore maximizing the geometric margin is equivalent to minimizing $\|w\|_2$. The standard hard-margin SVM writes this as

```math
\min_{w,b}
\frac12\|w\|_2^2
```

subject to

```math
y_i(w^\top x_i+b)\ge 1
\quad
i=1,\ldots,N.
```

The factor $1/2$ is computational: it makes the derivative of $\frac12\|w\|_2^2$ equal to $w$, avoiding an unnecessary factor of $2$ in stationarity equations. It does not change the optimizer.

## 5. Why Margin Can Act as Complexity Control

T2 showed that generalization is not secured by low training error alone. A learning procedure must control the effective set of solutions it can select.

Large margin is not merely the claim that "farther points mean more confidence." SVM margin is a geometric constraint: among all separators that fit the data under the chosen representation, the algorithm prefers one with small $\|w\|_2$ under canonical scaling, equivalently large geometric separation from the closest training points.

This matters because many linear separators can classify the training set perfectly. Maximum margin selects a particular separator by imposing a norm/margin preference. In some settings, margin- or norm-controlled families can have better generalization analyses than raw parameter count would suggest. This connects to the T2 distinction between nominal class size and effective capacity, and to the T3 distinction between hypothesis family and solution-selection procedure.

### What This Does NOT Imply

- Large SVM margin is not automatically a calibrated probability.
- Margin alone does not universally determine generalization.
- A large margin on training data does not by itself prove robustness under distribution shift.
- The margin is measured in the chosen representation; feature scaling can change it.

## 6. Primal Optimization Problem

The hard-margin primal is:

```math
\begin{aligned}
\min_{w,b}\quad
&
\frac12\|w\|_2^2
\\
\text{subject to}\quad
&
y_i(w^\top x_i+b)\ge 1,
\quad i=1,\ldots,N.
\end{aligned}
```

Variables:

- $w\in\mathbb{R}^d$ controls the normal direction and norm;
- $b\in\mathbb{R}$ controls the offset;
- the data $(x_i,y_i)$ are fixed observations.

Objective:

- minimize squared norm;
- equivalently maximize the canonical geometric margin.

Constraints:

- every training point must be on the correct side of its margin plane.

Convexity:

- the objective is convex quadratic in $w$;
- the constraints are affine inequalities in $(w,b)$;
- the problem is a convex quadratic program.

This differs from logistic regression:

```text
Logistic regression:
probabilistic conditional model + likelihood

Hard-margin SVM:
geometric separation + constrained optimization
```

Logistic regression models $p(y\mid x)$ through a sigmoid score and optimizes likelihood or cross entropy. Hard-margin SVM does not output calibrated probabilities by default; it selects a separating boundary by maximizing margin. Neither is simply a better version of the other. They encode different modeling claims and different objective geometries.

## 7. Lagrangian Dual

### Theorem: Hard-Margin SVM Dual

#### Assumptions

- The data are linearly separable in the current representation.
- The primal problem is the hard-margin SVM above.
- Lagrange multipliers $\alpha_i\ge0$ correspond to the constraints

```math
y_i(w^\top x_i+b)-1\ge0.
```

#### Claim

The dual problem is

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

At optimality,

```math
w
=
\sum_i\alpha_i y_i x_i.
```

#### Derivation / Proof Idea

Start from the Lagrangian:

```math
L(w,b,\alpha)
=
\frac12\|w\|_2^2
-
\sum_i
\alpha_i
\left[
y_i(w^\top x_i+b)-1
\right],
```

with

```math
\alpha_i\ge0.
```

Expand:

```math
L(w,b,\alpha)
=
\frac12\|w\|_2^2
-
\sum_i\alpha_i y_i w^\top x_i
-
b\sum_i\alpha_i y_i
+
\sum_i\alpha_i.
```

Stationarity with respect to $w$ gives

```math
\nabla_w L
=
w
-
\sum_i\alpha_i y_i x_i
=
0,
```

so

```math
w
=
\sum_i
\alpha_i y_i x_i.
```

Stationarity with respect to $b$ gives

```math
\frac{\partial L}{\partial b}
=
-
\sum_i\alpha_i y_i
=
0,
```

so

```math
\sum_i\alpha_i y_i=0.
```

Substitute the stationarity relations back into the Lagrangian. Since

```math
\sum_i\alpha_i y_i w^\top x_i
=
w^\top
\sum_i\alpha_i y_i x_i
=
w^\top w
=
\|w\|_2^2
```

and $b\sum_i\alpha_i y_i=0$, the minimized Lagrangian over $w,b$ becomes

```math
\sum_i\alpha_i
-
\frac12\|w\|_2^2.
```

Using

```math
\|w\|_2^2
=
\left(
\sum_i\alpha_i y_i x_i
\right)^\top
\left(
\sum_j\alpha_j y_j x_j
\right)
=
\sum_i\sum_j
\alpha_i\alpha_j y_i y_j x_i^\top x_j,
```

the dual objective is

```math
\sum_i\alpha_i
-
\frac12
\sum_i\sum_j
\alpha_i\alpha_j y_i y_j x_i^\top x_j.
```

#### Interpretation

The primal describes a geometric separator through $w$. The dual describes the same solution through coefficients on training examples and inner products between examples.

#### What This Does NOT Imply

The dual is not just algebraic ornamentation. It changes the representation of the learned classifier and makes kernelization possible in Lecture 15. But the dual also does not eliminate the need to choose a meaningful representation or control the learning procedure.

#### Research Use

The dual asks which data points directly determine the final boundary and how the geometry enters through pairwise inner products.

## 8. KKT Conditions

For the hard-margin SVM, the Karush-Kuhn-Tucker conditions are:

Primal feasibility:

```math
y_i(w^\top x_i+b)-1\ge0
\quad
\text{for all } i.
```

Dual feasibility:

```math
\alpha_i\ge0
\quad
\text{for all } i.
```

Stationarity:

```math
w
=
\sum_i\alpha_i y_i x_i,
\qquad
\sum_i\alpha_i y_i=0.
```

Complementary slackness:

```math
\alpha_i
\left[
y_i(w^\top x_i+b)-1
\right]
=
0
\quad
\text{for all } i.
```

Complementary slackness explains support vectors. If a point is strictly outside the margin, then

```math
y_i(w^\top x_i+b)>1.
```

The bracket is positive, so the product can be zero only if

```math
\alpha_i=0.
```

If $\alpha_i>0$, then the bracket must be zero:

```math
y_i(w^\top x_i+b)=1.
```

The point lies exactly on an active margin constraint. These active points are the support vectors.

This answers the structural question:

```text
Only training points with nonzero dual coefficients appear directly in w.
```

The other points still matter indirectly because they constrain the feasible optimization problem. But once the optimum is fixed, they do not appear in the dual expansion of $w$.

## 9. Support Vectors as Solution Structure

Since

```math
w
=
\sum_i\alpha_i y_i x_i
```

and $\alpha_i=0$ for non-support vectors, the solution can be written as

```math
w
=
\sum_{i\in SV}
\alpha_i y_i x_i.
```

The classifier is

```math
f(x)
=
w^\top x+b
=
\sum_{i\in SV}
\alpha_i y_i x_i^\top x
+
b.
```

This is a compression-like structure: the final classifier can depend directly only on support vectors in the dual representation.

### What This Does NOT Imply

```text
few support vectors
=
automatic generalization guarantee
```

is not valid as a universal statement. The number and arrangement of support vectors, margin size, data distribution, feature scaling, kernel choice, regularization, and sampling process all affect what can be claimed.

## 10. Research Lens

When reading an SVM or margin-based paper, ask:

- What geometry does the representation induce?
- Which norm defines the margin?
- Which points determine the learned boundary?
- What notion of simplicity is being preferred?
- What happens if the representation changes?
- What happens under feature rescaling?
- Is the reported margin computed before or after hyperparameter selection?
- Does large training margin imply deployment robustness, or only a statement in the training geometry?
- Is confidence being treated as calibrated probability? If so, what calibration evidence supports that?

### Existing Repository Links

- T1 Lecture 3 introduced nonlinear feature transforms: [feature transforms](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md).
- T2 separated capacity from parameter count: [VC dimension and capacity](../part2_generalization_theory/07_caltech_l07_vc_dimension_capacity_and_sample_complexity.md).
- T3 showed regularization as solution preference: [regularization](../part3_fitting_regularization_validation/12_caltech_l12_regularization_constraints_inductive_bias.md).
- Week 2 logistic regression provides the contrast between probabilistic score modeling and geometric separation: [Week 2 report](../../../reports/week2_linear_logistic_regression.md).
- Week 5 calibration explains why margin and probability reliability are separate claims: [calibration metrics](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md).

[Back to Learning From Data Theory Notebook](../README.md)
