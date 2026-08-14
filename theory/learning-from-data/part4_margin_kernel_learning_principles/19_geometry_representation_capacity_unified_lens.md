# Geometry, Representation, Capacity, and Inductive Bias: A Unified Lens

[Back to Learning From Data Theory Notebook](../README.md)

这篇 note 不是机械总结 Lectures 14-18。它回答一个更高层问题：

```text
为什么许多看似不同的 algorithms，
实际上主要区别在于它们怎样编码 representation、geometry、similarity 与 solution preference？
```

![T4 geometry representation capacity map](../assets/t4_geometry_representation_capacity_map.png)

图 1：T4 的统一链条。representation 诱导 geometry；geometry 定义 distance、inner product、margin 与 locality；learning algorithm 再通过 objective、constraint 与 selection process 选出 solution。

## 0. Source Separation

### Caltech Core

Caltech Lectures 14-18 提供 classical arc：margin geometry、kernel representation、RBF locality、learning principles、Bayesian learning 与 aggregation。

### Formal Derivation

本篇不新增 theorem，而是重组前面已经推导过的 SVM margin/dual、kernel PSD、RBF design matrix、Bayesian posterior 与 aggregation variance。

### Stanford CS229 Extension

CS229 的作用是支撑 SVM 与 kernel 的数学结构。这里把这些结构作为 unified lens 的部件，而不是重新推导。

### Stanford CS229M / Theory Extension

现代桥接只用于一个原则：effective complexity 与 selected solution 有关，不能由 ambient dimension 或 raw parameter count 单独决定。

### Research Lens

读新论文时，不要只问“用了什么模型名”。要问：这篇论文实际改变的是哪个 arrow？

```text
World
-> Observation
-> Representation
-> Geometry
-> Hypothesis Structure
-> Objective + Constraint
-> Selected Solution
-> Evaluation Claim
```

## 1. One Prediction Problem, Different Structures

同一个 supervised prediction problem 可以被多种结构表达。

### Logistic Regression

logistic regression 使用 probabilistic score model：

```text
linear score
-> sigmoid / softmax
-> conditional probability model
```

它的核心语言是 likelihood、conditional probability 与 calibration-oriented evidence，而不是 margin geometry 本身。

### SVM

SVM 使用 margin geometry：

```text
separating hyperplane
-> geometric margin
-> norm-constrained maximum-margin solution
```

它的核心偏好是 large margin / small norm，而不是 probability likelihood。

### Kernel SVM

kernel SVM 使用 implicit feature-space geometry：

```text
kernel
-> feature space 中的 inner product
-> 不显式构造 Phi 的 margin optimization
```

它可以在原始 input space 中产生 nonlinear decision boundary，但 algorithm 仍在 feature geometry 中做 linear separation。

### RBF Model

RBF model 使用 explicit local basis representation：

```text
centers
-> localized basis responses
-> linear output fitting
```

它把 locality 假设写进 basis functions 与 metric。

### Neural Network

neural network 学习 data-dependent hierarchical representation：

```math
\Phi_\theta(x).
```

因此 geometry 本身也被 data、objective、optimizer 与 regularization 共同塑造。

## 2. Representation Determines Geometry

给定 representation

```math
z=\Phi(x),
```

以下对象都在 represented space 中定义：

- distance；
- angle；
- inner product；
- margin；
- locality。

因此 geometry 不是独立于 representation 的客观背景。换一个 $\Phi$，同一对 raw inputs 的 distance、angle、similarity 与 margin 都可能改变。

这个点对 T4 至关重要：

```text
raw observation
!=
learner 实际使用的 geometry
```

例如，SVM 的 geometric margin 是 $\Phi(x)$ 空间中的 hyperplane distance；Gaussian kernel 的 locality 是输入 metric 通过 kernel 诱导的 similarity；neural network 的 hidden representation 会让几何关系变成 learned object。

## 3. Kernel as Explicit Assumption about Similarity

kernel 声明了：

```text
哪些 examples 应该被视为 similar
```

更精确地说，valid kernel 定义了某个 feature-space inner product：

```math
K(x,z)
=
\langle\Phi(x),\Phi(z)\rangle.
```

因此 kernel method 并没有逃离 representation assumptions。它只是把 representation 通过 kernel function 隐式表达。

研究上要问：

- kernel 是否表达了 domain 中合理的 invariance？
- distance 或 dot product 是否受 feature scaling 支配？
- distribution shift 后，相同 kernel geometry 是否仍然合理？
- kernel hyperparameters 是否通过 adaptive benchmark feedback 选出？

## 4. RBF as Explicit Locality Assumption

RBF model 假设 predictive structure 可以通过围绕 centers 的 localized responses 表示：

```math
\phi_k(x)
=
\exp
\left(
-
\frac{\|x-c_k\|^2}{2\sigma_k^2}
\right).
```

这不是中性的技术选择。它要求：

```text
在 chosen metric 下 nearby
-> basis activation 相似
-> 共享某些 predictive structure
```

若 metric 不表达 prediction-relevant mechanism，locality 就可能失效。尤其在 high-dimensional raw inputs 中，Euclidean distance 不应被默认当成有意义的 similarity。

## 5. Neural Representation as Learned Geometry

连接 T1 与 T3：neural network 学习

```math
\Phi_\theta(x),
```

所以 learner 不只是在 fixed feature space 中选 boundary；它也在用 data 形成 feature space。

这会带来两面性：

- 好处：model 可能学习比 handcrafted features 更适合任务的 geometry；
- 风险：geometry 也可能吸收 sampling artifacts、spurious correlations、label noise 或 validation feedback。

因此 modern representation analysis 重要，不是因为“deep model 神秘”，而是因为 model 的 similarity structure 本身变成 data-dependent selected object。

## 6. Capacity Cannot Be Read from Dimension Alone

kernel methods 是关键反例：

```text
very high-dimensional / infinite-dimensional feature space
```

不等于

```text
uncontrolled effective complexity
```

需要分清三类概念。

### Structural / Generalization Control

这些机制直接限制或偏好某些 solutions：

- restricted hypothesis family；
- norm control；
- margin control；
- explicit regularization；
- algorithmic stability；
- compression，只有在明确调用 compression/generalization theorem 时才作为 argument；
- optimizer 或 learning rule 对 selected solution 的偏好。

这些可以构成 generalization-control story，但仍需要对应 assumptions。

### Statistical Conditions

这些条件决定 evidence 的统计精度和适用目标：

- sample size；
- i.i.d. / sampling assumptions；
- source distribution；
- target distribution。

sample size 会影响 estimation error 与 generalization precision，但它不是 hypothesis capacity 本身。

### Selection / Evaluation Discipline

这些机制保护最终 evidence 的可信度：

- validation protocol；
- hyperparameter search accounting；
- benchmark feedback control；
- held-out final evaluation。

validation discipline 不是 capacity-control mechanism；它是防止 adaptive selection 污染 final claim 的 evidence discipline。

同理，support-vector sparsity 是 solution structure。除非明确连接到 formal compression/generalization argument，否则不要把 “support vectors 少” 直接叫作 capacity control。

## 7. Three Axes of Inductive Bias

为了读论文，可以把 inductive bias 分成三条轴。

### Representation Bias

模型实际看见什么对象？

```text
raw observations
handcrafted features
kernel similarities
learned embeddings
local basis responses
```

representation bias 决定哪些 distinctions 可被表达，哪些 distinctions 可能被压掉。

### Geometric / Similarity Bias

什么算 nearby、aligned、similar？

```text
distance
inner product
angle
kernel geometry
graph neighborhood
learned embedding geometry
```

这决定 model 如何在 examples 之间共享 evidence。

### Algorithmic Solution-Selection Bias

learning algorithm 在多个可拟合 solutions 中偏好哪一个？

```text
minimum norm
maximum margin
regularized optimum
early-stopped solution
stable solution
compressed solution
```

这里使用 `algorithmic solution-selection bias`，避免与 statistical sampling-selection bias 混淆。

## 8. Reliability Perspective

当 model 失败时，不要只问“模型是不是太小/太大”。沿着链条定位 failure：

- representation 是否压掉了 relevant distinctions？
- similarity geometry 是否在 deployment 中失效？
- hypothesis class 是否 misspecified，无法表达 mechanism？
- optimizer 是否选了 unstable solution？
- collection / sampling mechanism 是否偏置了进入 dataset 的 observations？
- deployment environment / distribution 是否相对 source environment 改变？
- evaluation 是否被 validation reuse、test inspection 或 benchmark feedback 污染？
- error 是否来自 irreducible stochastic uncertainty？

这里要保持 Lecture 17 的区分：sampling / selection mechanism failure 与 distribution / environment shift 可以重叠，但不是同一个概念。

## 9. Full T4 Chain

T4 的完整 research chain 是：

```text
World
↓
Observation
↓
Representation Phi
↓
Induced Geometry
↓
Similarity / Distance / Margin / Locality
↓
Hypothesis Structure
↓
Objective + Constraint
↓
Learning Algorithm
↓
Selected Solution
↓
Generalization under sampling assumptions
↓
Evaluation under selection discipline
```

读新 ML paper 时，长线问题是：

```text
这篇 paper 实际修改的是哪一个 arrow？
```

可能答案包括：

- 改 observation process；
- 改 representation $\Phi$；
- 改 geometry / similarity；
- 改 hypothesis structure；
- 改 objective 或 constraint；
- 改 optimizer；
- 改 sampling assumption；
- 改 validation / evaluation protocol；
- 改 final claim 的 target population。

这个问题能防止把 method name 当成解释本身。

## 10. Cross-Links to the Existing Theory Map

- T1 建立 world、observation、representation、hypothesis 与 learning algorithm 的基本 ontology：[theory ontology](../00_learning_theory_ontology_world_data_generalization_research_lens.md)。
- T2 说明 finite data、capacity 与 selected hypothesis 的关系：[generalization theory](../part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md)。
- T3 说明 objective、regularization、validation 与 adaptive selection：[validation and model selection](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md)。
- Lecture 14 给出 SVM margin geometry：[SVM margin geometry](14_caltech_l14_support_vector_machines_margin_geometry_duality.md)。
- Lecture 15 给出 kernel-induced geometry：[kernel methods](15_caltech_l15_kernel_methods_feature_spaces_soft_margins.md)。
- Lecture 16 给出 locality as representation assumption：[RBF local representation](16_caltech_l16_radial_basis_functions_local_representation.md)。
- Lecture 17 给出 learning principles and evidence discipline：[three learning principles](17_caltech_l17_three_learning_principles_occam_sampling_snooping.md)。
- Week 4 与 Week 5 分别提供 shift 与 calibration 的应用语境：[Canvas diagnostic](../../../reports/week4/15_canvas_diagnostic_v1_inventory_and_failure_taxonomy.md)，[calibration metrics](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md)。

[Back to Learning From Data Theory Notebook](../README.md)
