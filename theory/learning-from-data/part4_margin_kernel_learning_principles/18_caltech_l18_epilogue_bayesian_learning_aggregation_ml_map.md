# Epilogue: Bayesian Learning, Aggregation, and the Map of Machine Learning

[Back to Learning From Data Theory Notebook](../README.md)

This chapter corresponds to Caltech `Learning From Data` Lecture 18: Epilogue. Its job is to close the classical course arc, not to become a full Bayesian ML, ensemble theory, or modern deep-learning theory course.

![Bayesian prior likelihood posterior](../assets/bayesian_prior_likelihood_posterior.png)

Figure 1: Bayesian learning combines a prior assumption with a likelihood from observed data to form a posterior over hypotheses.

![Aggregation variance correlation](../assets/aggregation_variance_correlation.png)

Figure 2: averaging reduces variance most when model errors are weakly correlated. Correlation limits the variance reduction available from adding more predictors.

## 0. Source Separation

### Caltech Core

Lecture 18 closes the course with a map of machine learning, Bayesian learning, aggregation, and final perspective on the learning problem.

### Formal Derivation

This note derives Bayes rule for posterior inference, introduces the posterior predictive, and derives the variance of an equally weighted average under equal variance and pairwise correlation assumptions.

### Stanford CS229 Extension

CS229 probabilistic-modeling and regularization material supports the likelihood/MAP contrast. CS229 methodology supports the validation discipline needed for stacking and blending.

### Stanford CS229M / Theory Extension

The theory bridge is limited to the idea that aggregation, priors, and algorithmic selection are additional ways of encoding inductive bias and controlling effective solution behavior.

### Modern Perspective

Modern ensemble and Bayesian language is used only for conceptual precision. Calibration, uncertainty quality, PAC-Bayes, Gaussian processes, and boosting theory are deferred.

## 1. Reconstruct the Machine-Learning Map

Machine learning concepts should not be collapsed into one axis. A model can be described along several dimensions.

### Paradigm

- supervised learning;
- unsupervised learning;
- reinforcement learning;
- online learning;
- active learning.

Paradigm describes the data/feedback structure.

### Representation / Model

- linear model;
- neural network;
- RBF model;
- kernel method;
- graphical/probabilistic model.

Representation/model describes the function family and input geometry.

### Learning / Selection Method

- empirical risk minimization;
- maximum likelihood;
- regularization;
- margin optimization;
- Bayesian inference;
- aggregation.

Learning/selection describes how the final hypothesis or predictive rule is chosen.

### Theory / Evaluation

- VC/generalization theory;
- bias-variance;
- validation;
- learning principles.

Theory/evaluation describes what evidence supports the claim.

These categories should not be collapsed. For example, a neural network can be trained by maximum likelihood, regularized by weight decay, selected by validation, aggregated in an ensemble, and evaluated under distribution shift. Each layer answers a different question.

## 2. Bayesian Learning

Bayesian learning starts with Bayes rule:

```math
p(h\mid D)
\propto
p(D\mid h)p(h).
```

Definitions:

- $h$ is a hypothesis, model, or parameterized explanation, depending on context;
- $p(h)$ is the prior;
- $p(D\mid h)$ is the likelihood;
- $p(h\mid D)$ is the posterior.

The prior is an assumption. It encodes which hypotheses are plausible before observing the dataset.

### Theorem: Bayes Rule for Hypotheses

#### Assumptions

- There is a hypothesis space with prior density or mass $p(h)$.
- The likelihood $p(D\mid h)$ is defined.
- The marginal probability $p(D)$ is positive and finite.

#### Claim

The posterior is

```math
p(h\mid D)
=
\frac{p(D\mid h)p(h)}{p(D)}.
```

Equivalently,

```math
p(h\mid D)
\propto
p(D\mid h)p(h)
```

as a function of $h$.

#### Derivation / Proof Idea

By the product rule,

```math
p(h,D)
=
p(D\mid h)p(h)
```

and also

```math
p(h,D)
=
p(h\mid D)p(D).
```

Equating the two expressions gives

```math
p(h\mid D)p(D)
=
p(D\mid h)p(h).
```

Divide by $p(D)$:

```math
p(h\mid D)
=
\frac{p(D\mid h)p(h)}{p(D)}.
```

Since $p(D)$ does not depend on $h$, posterior comparison across hypotheses uses proportionality:

```math
p(h\mid D)
\propto
p(D\mid h)p(h).
```

#### Interpretation

Bayesian learning updates assumptions using data. The posterior is not just the best single hypothesis; it is a distribution over hypotheses.

#### What This Does NOT Imply

Bayesian learning is not objective or assumption-free. The prior, likelihood, model class, and data-generation assumptions all matter.

#### Research Use

When a paper is Bayesian, ask what prior was used, what likelihood was assumed, whether inference was exact or approximate, and what population the data represent.

## 3. MAP versus Full Posterior Inference

Maximum a posteriori estimation chooses one hypothesis:

```math
h_{\mathrm{MAP}}
\in
\arg\max_h
p(h\mid D).
```

Using Bayes rule, this is equivalent to

```math
h_{\mathrm{MAP}}
\in
\arg\max_h
\left[
\log p(D\mid h)
+
\log p(h)
\right].
```

This connects to T3 regularization. For some likelihood-prior pairs, MAP estimation becomes a regularized optimization problem.

But:

```text
MAP point estimate
!=
full posterior inference
```

MAP returns one selected hypothesis. Full Bayesian inference keeps uncertainty over hypotheses, at least conceptually.

## 4. Posterior Predictive Idea

The posterior predictive distribution is

```math
p(y\mid x,D)
=
\int
p(y\mid x,h)
p(h\mid D)
\,dh.
```

For a discrete hypothesis set, the integral becomes a sum:

```math
p(y\mid x,D)
=
\sum_h
p(y\mid x,h)
p(h\mid D).
```

This propagates uncertainty over hypotheses into predictions.

### What This Does NOT Imply

Posterior predictive form does not automatically guarantee practical calibration. Calibration depends on model specification, prior, likelihood, approximate inference, sample representativeness, and evaluation.

## 5. Bayesian Learning and Inductive Bias

Prior information changes which hypotheses are plausible before observing the dataset. This is another formal expression of inductive bias.

Examples:

- a Gaussian prior over weights prefers smaller-norm weights in a MAP setting;
- a smoothness prior prefers functions that vary smoothly;
- a hierarchical prior shares statistical strength across related groups;
- a sparsity prior prefers solutions using fewer active components.

The prior is not automatically correct just because the method is Bayesian. It is an assumption that must be matched against domain knowledge and empirical evidence.

## 6. Aggregation

Let

```math
h_1,\ldots,h_T
```

be fitted predictors.

For regression, an aggregated predictor can be written as

```math
g(x)
=
\sum_{t=1}^{T}
\alpha_t h_t(x),
```

where the weights often satisfy

```math
\sum_{t=1}^{T}\alpha_t=1.
```

For classification, aggregation can use an unweighted vote, weighted vote, or averaged class score/probability followed by a decision rule.

Combining models can alter both bias and variance. It can reduce dataset-induced variability if individual predictors make partly independent errors. It can also preserve or amplify systematic bias if all predictors share the same wrong assumption.

## 7. Variance Reduction through Averaging

### Theorem: Variance of an Equally Weighted Average

#### Assumptions

- $h_1(x),\ldots,h_T(x)$ are random predictors at a fixed input $x$, where randomness comes from data sampling, resampling, initialization, or training variation.
- Each has equal variance:

```math
\mathrm{Var}(h_t)=\sigma^2.
```

- Each pair has the same correlation:

```math
\mathrm{Corr}(h_s,h_t)=\rho
\quad
\text{for } s\ne t.
```

Thus

```math
\mathrm{Cov}(h_s,h_t)
=
\rho\sigma^2
\quad
\text{for } s\ne t.
```

#### Claim

For the average

```math
\bar h
=
\frac1T
\sum_{t=1}^{T}
h_t,
```

the variance is

```math
\mathrm{Var}(\bar h)
=
\sigma^2
\left[
\rho
+
\frac{1-\rho}{T}
\right].
```

#### Derivation / Proof Idea

Start with

```math
\mathrm{Var}
\left(
\frac1T
\sum_t h_t
\right)
=
\frac{1}{T^2}
\mathrm{Var}
\left(
\sum_t h_t
\right).
```

Expand the variance of a sum:

```math
\mathrm{Var}
\left(
\sum_t h_t
\right)
=
\sum_t
\mathrm{Var}(h_t)
+
2
\sum_{s<t}
\mathrm{Cov}(h_s,h_t).
```

There are $T$ variance terms and $T(T-1)/2$ covariance pairs, so

```math
\mathrm{Var}
\left(
\sum_t h_t
\right)
=
T\sigma^2
+
T(T-1)\rho\sigma^2.
```

Divide by $T^2$:

```math
\mathrm{Var}(\bar h)
=
\frac{T\sigma^2+T(T-1)\rho\sigma^2}{T^2}.
```

Simplify:

```math
\mathrm{Var}(\bar h)
=
\sigma^2
\left[
\frac1T
+
\rho\frac{T-1}{T}
\right]
=
\sigma^2
\left[
\rho
+
\frac{1-\rho}{T}
\right].
```

#### Interpretation

Averaging independent predictors has $\rho=0$, giving variance $\sigma^2/T$. If predictors are highly correlated, the limiting variance is approximately $\rho\sigma^2$. Diversity matters because correlation limits the benefit of aggregation.

#### What This Does NOT Imply

Ensembles do not always reduce error. If predictors are systematically biased, poorly calibrated, trained on biased data, or highly correlated, aggregation may have limited benefit or may preserve the same failure.

#### Research Use

When an ensemble improves performance, ask whether the gain comes from variance reduction, bias change, different representation coverage, stronger selection, more computation, or data leakage through the aggregation protocol.

## 8. Bagging

Bagging follows the pattern:

```text
resample data
-> fit multiple learners
-> aggregate predictions
```

The goal is to reduce dataset-induced variance by fitting predictors on different bootstrap samples. This links to T2's view of a learned hypothesis as a random object depending on the training dataset.

Bagging is most useful when the base learner is unstable enough that resampling creates useful diversity. It is less useful when all resampled learners produce nearly identical predictions or share the same systematic bias.

## 9. Boosting

Boosting constructs learners sequentially, changing emphasis according to previous errors. Later learners focus on cases that earlier learners handled poorly.

This differs from simple independent averaging:

```text
bagging:
parallel or independent resampling-style variation

boosting:
sequentially shaped fitting process
```

Boosting can be interpreted through margins, additive modeling, reweighting, optimization, and regularization depending on the formal treatment. This note does not reduce boosting generalization to one slogan.

## 10. Blending / Stacking

Blending or stacking combines existing models through another learned layer. For example, a meta-model may learn weights over candidate predictors.

That extra layer is itself a selection or learning process. Therefore the aggregation data must have a clearly defined role:

- data for training base models;
- data for training the blender/stacker;
- data for selecting aggregation weights;
- data for final evaluation.

If the same validation or test feedback repeatedly shapes the stack, T3 data-snooping concerns apply.

## 11. Aggregation versus Joint Representation Learning

Preserve the distinction:

```text
ensemble:
learn solutions and combine them

multilayer model:
components are learned jointly as one model
```

An ensemble combines multiple fitted hypotheses or prediction rules. A multilayer neural network learns internal components jointly through one training objective. Both can contain many parts, but their learning structures are not the same.

This distinction matters for evidence. An ensemble's diversity, data reuse, and meta-selection can be audited separately. A jointly trained model's internal representation and optimizer trajectory must be analyzed as one coupled learning system.

## 12. Final Caltech Synthesis

The course arc can now be reconstructed as one question:

```text
What is learning?
Can learning generalize?
How do we fit models?
Why do they fail?
How do we control selection?
How do geometry and representation matter?
How should evidence be interpreted?
```

T1 built the ontology:

```text
World
-> Observations
-> Representation
-> Hypothesis Set
-> Learning Algorithm
-> Learned Hypothesis
-> Error / Noise
```

T2 added the finite-data generalization discipline:

```text
Finite Data
-> Data-dependent Selection
-> Capacity
-> Uniform Control
-> Generalization
-> Research Claim Discipline
```

T3 added the fitting and selection discipline:

```text
Objective
-> Optimization
-> Fitting
-> Overfitting
-> Regularization
-> Validation
-> Adaptive Selection
-> Credible Final Evaluation
```

T4 closes the classical spine with:

```text
Input Geometry
-> Representation
-> Similarity / Inner Product
-> Decision Geometry
-> Margin / Norm
-> Effective Complexity
-> Generalization
-> Sampling Assumptions
-> Selection Discipline
-> Credible Learning
```

The final lesson is not that one algorithm is supreme. It is that a credible learning claim must identify the representation, geometry, hypothesis structure, objective, regularization, sampling process, and evaluation discipline that make the learned prediction meaningful.

### Existing Repository Links

- T3 regularization/MAP boundary: [regularization](../part3_fitting_regularization_validation/12_caltech_l12_regularization_constraints_inductive_bias.md).
- T3 validation and stacking discipline: [validation](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md).
- T2 bias-variance and dataset-induced variability: [bias-variance](../part2_generalization_theory/08_caltech_l08_bias_variance_learning_curves.md).
- Week 5 calibration is a reminder that predictive probabilities and uncertainty claims require direct evaluation: [calibration metrics](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md).

[Back to Learning From Data Theory Notebook](../README.md)
