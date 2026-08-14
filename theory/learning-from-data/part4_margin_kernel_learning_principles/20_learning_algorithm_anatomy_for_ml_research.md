# Anatomy of a Learning Algorithm for ML Research

[Back to Learning From Data Theory Notebook](../README.md)

T2 asks:

```text
Is the generalization claim justified?
```

T3 asks:

```text
What adaptive selection process produced the model/result?
```

T4 asks:

```text
What structural assumptions make this algorithm learn the way it does?
```

This note is a reusable paper-reading tool.

## 0. Source Separation

### Caltech Core

The tool synthesizes the full Caltech `Learning From Data` arc: learning problem, generalization, fitting, regularization, validation, margin, kernels, RBFs, learning principles, Bayesian learning, and aggregation.

### Formal Derivation

The matrix below uses the formal objects derived across T1-T4: representation, geometry, hypothesis family, objective, regularization/constraint, optimizer, locality, probabilistic interpretation, and capacity-control mechanism.

### Stanford CS229 Extension

CS229 supports the logistic-regression, SVM, kernel, regularization, and validation entries.

### Stanford CS229M / Theory Extension

CS229M supports the algorithm-dependent complexity framing used for modern and overparameterized learners.

### Research Lens

The purpose is to audit what a paper's algorithm changes and what evidence would be needed to justify its claims.

## 1. Representation

Ask what object the model actually receives:

```text
raw observations?
handcrafted features?
kernel similarities?
learned embeddings?
local basis responses?
```

Examples:

- linear regression may receive raw feature vectors;
- kernel SVM may receive only kernel evaluations;
- an RBF model receives local basis activations;
- an MLP learns hidden representations from data.

Representation determines which distinctions are available to the learner.

## 2. Geometry

Ask what mathematical geometry is assumed:

- Euclidean distance;
- inner product;
- cosine-like relation;
- kernel-induced geometry;
- graph geometry;
- learned geometry.

Geometry determines the meaning of distance, angle, margin, and locality. It is not independent of representation.

## 3. Hypothesis Structure

Ask what functions can be represented.

Examples:

- affine functions;
- logistic scores passed through a sigmoid;
- finite basis expansions;
- kernel expansions;
- multilayer compositions;
- ensembles of base hypotheses.

Distinguish the nominal function family from the set of solutions the algorithm is likely to select.

## 4. Objective

Ask what is optimized:

- squared loss;
- negative log likelihood;
- hinge loss;
- margin objective;
- regularized empirical risk;
- posterior probability or evidence objective;
- validation-selected meta-objective.

The objective is not the same object as the evaluation metric.

## 5. Constraint / Regularization

Ask what solutions are discouraged or excluded:

- hard norm constraints;
- soft penalties;
- margin constraints;
- sparsity;
- priors;
- early stopping;
- data augmentation;
- support-vector sparsity;
- aggregation weights.

Regularization is a solution preference, not a magic guarantee.

## 6. Optimization Algorithm

Ask how one solution is selected among alternatives:

- closed-form solve;
- convex quadratic programming;
- gradient descent;
- stochastic gradient descent;
- Adam or other adaptive optimizers;
- coordinate descent;
- sequential boosting;
- approximate posterior inference.

The optimizer can matter even when the nominal objective is fixed.

## 7. Local versus Global Structure

Ask whether the model relies on:

- global linear structure;
- global margin geometry;
- local neighborhoods;
- basis centers;
- kernel similarity;
- compositional learned representation;
- aggregation of separate learners.

Locality and margin are geometry-dependent. They should be interpreted only after identifying the representation.

## 8. Sampling Assumption

Ask what population the training data represent.

Does the paper assume i.i.d. train/test sampling? Is there covariate shift, temporal shift, user self-selection, missingness, geographic bias, benchmark reuse, or label noise?

Sampling assumptions define the population to which evidence can generalize.

## 9. Selection Process

Ask what validation or research feedback shaped the final algorithm:

- feature engineering;
- preprocessing;
- architecture choice;
- optimizer tuning;
- regularization tuning;
- kernel/width/center choice;
- checkpoint selection;
- threshold selection;
- benchmark-driven revisions.

The final reported model is a product of the whole selection process, not only the last training run.

## 10. Evaluation Claim

Ask what exact population or distribution the evidence speaks about.

Possible claims:

- same-distribution test performance;
- shifted-distribution robustness;
- subgroup reliability;
- calibration;
- abstention/selective-risk reliability;
- computational efficiency;
- sample efficiency;
- stability across seeds;
- interpretability or mechanism.

Each claim requires evidence matched to that claim.

## 11. Failure Diagnosis

Map failures to:

```text
information / representation
approximation / specification
estimation / generalization
optimization / computation
sampling
adaptive selection / evaluation
irreducible uncertainty
```

Examples:

- If relevant information is absent from the input, the failure is information/representation.
- If the representation is available but the hypothesis family cannot express the target, the failure is approximation/specification.
- If the class can express the target but finite data select the wrong function, the failure is estimation/generalization.
- If the objective is appropriate but the optimizer fails, the failure is optimization/computation.
- If train data do not represent deployment, the failure is sampling.
- If benchmark feedback shaped the final procedure, the failure is adaptive selection/evaluation.
- If labels are inherently stochastic, irreducible uncertainty remains.

## 12. Algorithm Analysis Matrix

| Algorithm | Representation | Geometry | Hypothesis family | Objective | Regularization / constraint | Optimizer | Locality | Probabilistic interpretation | Capacity-control mechanism | Key failure modes |
| --------- | -------------- | -------- | ----------------- | --------- | --------------------------- | --------- | -------- | ---------------------------- | -------------------------- | ----------------- |
| Linear regression | Raw or engineered feature vector $x$ | Euclidean / inner-product geometry of features | Affine real-valued functions | Squared loss | Optional ridge/LASSO or constraints | Closed form or gradient methods | Global linear | Gaussian-noise likelihood under assumptions | Feature dimension, norm penalty, sample size | misspecified features, outliers, multicollinearity, shift |
| Logistic regression | Raw or engineered feature vector $x$ | Linear score geometry | Affine log-odds with sigmoid | Bernoulli negative log likelihood / cross entropy | Optional norm penalty | Convex optimization / gradient methods | Global linear score | Conditional probability model if specified correctly | Norm penalty, feature design, sample size | separation pathology, poor calibration under shift, wrong link/features |
| MLP | Learned hidden representation $\Phi_\theta(x)$ | Learned hidden-space geometry | Multilayer compositions | Usually empirical risk or likelihood surrogate | weight decay, augmentation, early stopping, architecture, implicit bias | SGD/Adam/backprop | Can learn local or global structure | Possible probabilistic output depending on loss/output layer | architecture, regularization, optimizer bias, data, validation | representation shortcut, overfitting, optimization instability, shift, calibration failure |
| SVM | Explicit feature vector $x$ | Margin in feature-space norm | Linear separators | maximize margin / minimize $\frac12\|w\|^2$ with constraints | hard margin or soft-margin $C$ | Convex quadratic program / dual solver | Global margin | Not probabilistic by default | margin, norm, support-vector structure, $C$ | bad feature scaling, nonseparable/noisy data, uncalibrated scores, shift |
| Kernel SVM | Implicit feature space via $K(x,z)$ | Kernel-induced inner product and margin | Linear separators in implicit feature space | kernelized hard/soft-margin dual | norm/margin control, box constraint $0\le\alpha_i\le C$ | Convex dual optimization | Depends on kernel; Gaussian is local | Not probabilistic by default | kernel choice, margin/norm, $C$, support vectors | invalid kernel, wrong similarity, hyperparameter snooping, scalability, shift |
| RBF model | Explicit RBF basis responses $\phi_k(x)$ | distance to centers under chosen metric | finite weighted sum of local basis units | squared loss, classification surrogate, or task loss on basis features | number of centers, widths, weight penalty | linear solve if centers/widths fixed; nonconvex if learned | Explicit local center-based | Depends on output/loss choice | center count, width, weight norm, selection process | meaningless distance, poor center coverage, width misselection, high-dimensional locality failure |

## 13. Use the Matrix on a New Paper

For a new algorithm, fill in the matrix before reading the results section too closely. Then ask:

- Which row entries are genuinely new?
- Which are inherited from a standard model?
- Which assumptions are stated?
- Which assumptions are only implied by implementation?
- Which hyperparameters were selected?
- Which data influenced selection?
- Which evaluation population is supported?
- Which failure modes were tested directly?

## 14. Relation to T1-T4

The full research audit is:

```text
T1:
What is the learning problem and representation?

T2:
Is the generalization claim justified?

T3:
What adaptive selection process produced the result?

T4:
What geometry, similarity, locality, margin, prior, or aggregation structure
makes the learner prefer this solution?
```

This tool is not a substitute for detailed proof or experiment. It is a disciplined first pass that prevents category mistakes.

### Existing Repository Links

- T2 generalization audit: [generalization claim audit](../part2_generalization_theory/10_generalization_claim_audit_for_ml_research.md).
- T3 selection protocol: [selection-aware research protocol](../part3_fitting_regularization_validation/15_selection_aware_ml_research_protocol.md).
- T4 unified lens: [geometry representation capacity lens](19_geometry_representation_capacity_unified_lens.md).

[Back to Learning From Data Theory Notebook](../README.md)
