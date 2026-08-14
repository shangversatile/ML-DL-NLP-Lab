# Anatomy of a Learning Algorithm for ML Research

[Back to Learning From Data Theory Notebook](../README.md)

这是 T4 的可复用 paper-reading 工具。T2 问：

```text
generalization claim 是否被证据支持？
```

T3 问：

```text
final model / result 是由什么 adaptive selection process 产生的？
```

T4 继续追问：

```text
哪些 structural assumptions 让这个 algorithm 以这种方式学习？
```

## 0. Source Separation

### Caltech Core

本工具来自 Caltech `Learning From Data` 的完整 classical arc：learning problem、generalization、regularization/validation、margin/kernel/RBF、learning principles 与 epilogue。

### Formal Derivation

本篇不新增推导；它把已有 derivations 转成 diagnostic checklist。

### Stanford CS229 Extension

CS229 支撑 SVM、kernel 与 optimization vocabulary，如 primal/dual、KKT、kernel trick、soft margin。

### Stanford CS229M / Theory Extension

现代理论只作为解释 effective solution complexity 的背景。这里不展开 full modern theory。

### Research Lens

使用这份 anatomy 时，目标不是给 algorithm 贴标签，而是定位：

```text
representation：模型实际接收什么
geometry：representation 诱导什么几何
hypothesis structure：能表达哪些 functions
objective：优化什么
constraint：排除或惩罚什么 solutions
optimizer：怎样选出一个 solution
sampling / distribution relation：数据与目标环境怎样对应
selection / evaluation discipline：evidence 怎样被保护
```

## 1. Representation

先问 model 实际收到什么对象：

```text
raw observations？
handcrafted features？
kernel similarities？
learned embeddings？
local basis responses？
```

representation 决定信息入口。若 relevant information 没进入 representation，后续 optimizer 再好也无法无条件恢复。

## 2. Geometry

再问 algorithm 假设什么 geometry：

- Euclidean distance；
- inner product；
- cosine-like relation；
- kernel-induced geometry；
- graph geometry；
- learned geometry。

geometry 不是自动来自 raw data；它由 representation 与 metric / kernel / architecture 共同决定。

## 3. Hypothesis Structure

问 model 能表达哪些 functions：

- linear functions；
- affine separators；
- polynomial interactions；
- local basis expansions；
- kernel expansions；
- multilayer compositional functions；
- probabilistic graphical structure。

这里分析的是 approximation / specification：即使 estimation 很好，hypothesis family 也可能 miss mechanism。

## 4. Objective

问被优化的目标是什么：

- squared loss；
- logistic / cross-entropy loss；
- hinge loss；
- likelihood；
- margin objective；
- reconstruction objective；
- contrastive objective；
- combined multi-term objective。

objective 决定 fit 的含义。相同 hypothesis family，在不同 objective 下可能选择不同 solution。

## 5. Constraint / Regularization

问什么 solutions 被 discouraged 或 excluded：

- norm penalty；
- margin constraint；
- sparsity penalty；
- smoothness penalty；
- early stopping；
- architecture constraint；
- prior in Bayesian model。

这是 structural / generalization control 的主要入口。但要注意：regularization claim 需要说明 assumptions 和实际 selection process。

## 6. Optimization Algorithm

问一个 solution 是怎样从 alternatives 中被选出的：

- closed-form solver；
- convex optimization；
- gradient descent / SGD；
- coordinate descent；
- dual solver；
- alternating optimization；
- nonconvex training with initialization choices。

optimizer 不是中性的实现细节。它可能引入 implicit bias，例如 minimum-norm preference、path dependence 或 sensitivity to initialization。

## 7. Local versus Global Structure

问 model 依赖哪类结构：

- global linear relation；
- margin geometry；
- local neighborhoods；
- basis centers；
- kernel similarities；
- compositional learned representation。

local structure 与 global structure 不是优劣排序，而是不同 inductive assumptions。

## 8. Sampling, Distribution, and Noise Conditions

这一节不能把所有 failure 都叫 sampling assumption。需要分三类。

### Sampling / Selection Mechanism

这些问题关心 observed dataset 是怎样从 population 中进入数据集的：

- observations 是否 representative？
- 是否存在 self-selection？
- 是否存在 selective inclusion / selective labeling？
- missingness 是否 non-random？
- collection mechanism 是否排除了某些 subpopulation 或 condition？

若 collection / inclusion mechanism 让 observed sample 不能代表 intended source 或 target population，这是 sampling / selection failure。

### Distribution / Environment Relation

这些问题关心 source/train distribution 与 target/deployment distribution 的关系：

- source/train distribution 是否匹配 target/deployment distribution？
- 是否存在 covariate shift？
- 是否存在 temporal drift？
- 是否存在 spatial / domain shift？
- 是否存在 conditional / mechanism change？

distribution / environment shift 不自动等于 sampling bias。training data 可能在收集时代表 source environment，但 deployment environment 后来改变。

### Target / Observation Noise

label noise、measurement noise 与 irreducible target stochasticity 不应列为 sampling assumption。它们属于 noise / uncertainty category，并影响 attainable error 与 evaluation interpretation。

adaptive evaluation 也不属于 sampling assumption。benchmark reuse、validation reuse 与 test inspection 应放在 selection / evaluation discipline 中。

## 9. Selection Process

问 final algorithm 或 result 是怎样通过 feedback 被塑造的：

- feature engineering 是否看过 validation/test behavior？
- hyperparameters 如何搜索？
- architecture revisions 是否受 benchmark 反馈影响？
- preprocessing 是否在 full dataset 上 fit？
- validation set 是否被反复使用？
- test set 是否被检查后继续修改 method？

这对应 T3 的 adaptive selection。它保护的不是 capacity 本身，而是 final evidence 的 credibility。

## 10. Evaluation Claim

问 evidence 实际支持哪个 claim：

- 同一 distribution 下的 expected performance？
- 特定 target population 上的 performance？
- robustness under distribution shift？
- calibrated probability？
- subgroup performance？
- deployment decision utility？

评价指标、data split、sampling process 与 selection history 必须与 claim 对齐。

## 11. Failure Diagnosis

把 failure 映射到不同来源：

```text
information / representation
approximation / specification
estimation / generalization
optimization / computation
sampling / selection mechanism
distribution / environment shift
adaptive selection / evaluation
irreducible stochastic uncertainty
```

对应解释：

- information / representation：relevant information 没有进入 features、kernel similarities 或 embeddings；
- approximation / specification：hypothesis family 不能表达 target mechanism；
- estimation / generalization：finite data 不足以稳定估计 selected solution；
- optimization / computation：solver 没有找到目标要求的 solution，或训练过程不稳定；
- sampling / selection mechanism：collection / inclusion mechanism 让 observed sample 不能代表 intended source / target population；
- distribution / environment shift：training/source data 收集时可能有代表性，但 target/deployment environment 后来不同；
- adaptive selection / evaluation：validation reuse、test inspection、benchmark feedback 污染 final evidence；
- irreducible stochastic uncertainty：即使 representation 与 algorithm 合理，target 本身仍有不可消除的不确定性。

不要写成：

```text
train data do not represent deployment = sampling failure
```

除非明确指出原因是 collection / inclusion mechanism。若原因是 deployment environment 改变，应归为 distribution / environment shift。

## 12. Algorithm Analysis Matrix

下面的 matrix 用于快速比较算法。重点是把 structural control 与 selection/evaluation discipline 分开。

| Algorithm | Representation | Geometry | Hypothesis family | Objective | Structural / generalization control | Optimizer | Locality | Probabilistic interpretation | Selection / evaluation issues | Key failure modes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear regression | raw / engineered numeric features | feature space 中的 Euclidean / inner-product geometry | affine real-valued functions | squared loss | restricted linear family；可加 ridge/lasso | closed form 或 convex optimization | global | assumptions 成立时有 Gaussian-noise interpretation | feature choice、regularization tuning、validation split | misspecification、outliers、collinearity、shift |
| Logistic regression | raw / engineered features | linear score geometry | sigmoid/softmax 上的 affine decision boundary | conditional likelihood / cross-entropy | restricted linear score；可加 norm penalty | standard form 下是 convex optimization | global | conditional probability model，但 calibration 仍需验证 | threshold choice、regularization tuning、calibration evaluation | nonlinear mechanism missed、imbalance、calibration failure |
| MLP | learned hidden representation | hidden layers 中的 learned geometry | compositional nonlinear functions | task loss，常见为 cross-entropy | architecture、explicit regularization、implicit optimizer bias | nonconvex gradient-based training | learned local/global mix | 只在 output/loss assumptions 下有 probabilistic interpretation | architecture search、early stopping、benchmark feedback | spurious features、instability、overfit、shift |
| SVM | explicit features | hyperplane margin geometry | affine separators | maximum margin / hinge variant | margin 与 norm control | convex primal/dual optimization | chosen features 中 global | 默认不是 probability model | $C$ 或 feature scaling selection；若声称概率需 calibration evidence | wrong geometry、uncalibrated scores、shift |
| Kernel SVM | kernel 给出的 implicit feature space | kernel-induced inner product | implicit feature space 中的 linear separator | margin objective / hinge loss | feature-space norm、margin control、soft-margin regularization | convex dual optimization | 取决于 kernel | 默认不是 calibrated probability model | kernel choice、kernel hyperparameters、validation reuse | invalid similarity assumptions、kernel sensitivity、shift |
| RBF model | explicit local basis responses | distance-to-centers geometry | finite basis expansion + linear output | basis features 上的 output loss | finite centers；可加 output-weight regularization | centers/widths 固定时 linear solver；若学习 centers 则可能 nonconvex | explicit locality | 取决于 output loss / model | center/width selection、clustering choices、validation loops | meaningless metric、bad centers、curse of dimensionality |

sample size 不在 structural-control column 中，因为它影响 estimation/generalization precision，而不是 hypothesis capacity 本身。held-out evaluation 也不在该列中，因为它保护 evidence credibility，而不是限制 hypothesis family。

## 13. Use the Matrix on a New Paper

读新论文时按顺序填：

1. model 实际接收什么 representation？
2. 这个 representation 诱导什么 geometry？
3. hypothesis family 能表达哪些 functions？
4. objective 优化什么？
5. 哪些 constraints 或 regularizers 塑造 solution？
6. optimizer 怎样从 alternatives 中选出一个 solution？
7. 哪些 sampling / distribution assumptions 让 evidence 与 claim 相关？
8. 哪些 adaptive selection 塑造了 final reported result？
9. claim 面向哪个 target population 或 deployment condition？
10. failure 可能从链条的哪一环进入？

这比只写 “the paper proposes a new model” 更可审计。

## 14. Relation to T1-T4

T1 给出 learning ontology：

```text
World
-> Observations
-> Representation
-> Hypothesis Set
-> Learning Algorithm
-> Learned Hypothesis
-> Error / Noise
```

T2 问 selected hypothesis 是否有 generalization support。

T3 问 validation / adaptive selection 是否污染 final evidence。

T4 问 structural assumptions 如何决定 learning behavior：

```text
Representation
-> Geometry
-> Similarity / Distance / Margin / Locality
-> Objective + Constraint
-> Selected Solution
-> Evidence Claim
```

这份 anatomy 的核心用途是：把 algorithm name 拆成可检查的 assumptions。

### Existing Repository Links

- T1 ontology：[world-data-generalization lens](../00_learning_theory_ontology_world_data_generalization_research_lens.md)。
- T2 selection/generalization：[finite data and selection](../part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md)。
- T3 validation/adaptive selection：[validation and data contamination](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md)。
- T4 unified lens：[geometry representation capacity](19_geometry_representation_capacity_unified_lens.md)。

[Back to Learning From Data Theory Notebook](../README.md)
