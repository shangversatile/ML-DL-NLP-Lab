# Three Learning Principles: Occam, Sampling Bias, and Data Snooping

[Back to Learning From Data Theory Notebook](../README.md)

本章对应 Caltech `Learning From Data` Lecture 17: Three Learning Principles。这里的三条原则不是松散的 practical advice，而是把 T1、T2、T3 组织成 ML research 的 evidence discipline。

```text
Occam
-> 控制什么 explanations / models 可以被选择

Population / Distribution Mismatch
-> 控制 data 代表什么 population / distribution

Data Snooping
-> 控制 evidence 怎样进入 selection
```

## 0. Source Separation

### Caltech Core

Lecture 17 的核心是 Occam's razor、sampling bias、data snooping。Caltech 用这三条原则提醒：learning claim 不只取决于 fit，也取决于 model search、data source 与 evidence accounting。

### Formal Derivation

本章只使用 T2 已经建立的 finite-hypothesis / simultaneous-control intuition，不重新发展完整 generalization theory。

### Stanford CS229 Extension

CS229 不作为本章主源。它只在必要时支撑 train/test distribution agreement 与 supervised-learning notation。

### Stanford CS229M / Theory Extension

CS229M / modern theory 的作用是提醒：complexity 不等于 parameter count；simplicity 可以通过 norm、margin、stability、compression 或 algorithmic selection preference 表达。

### Modern Perspective

modern benchmark contamination、pretraining contamination 与 repeated benchmark feedback 作为现代扩展处理。它们与 classical data snooping 相通，但不能被说成完全同一个数学机制。

## 1. Principle 1 - Occam's Razor

Caltech 的原则可以写成：

```text
当多个 explanations / models 都能充分解释 evidence 时，优先选择更 simple 的那个。
```

关键问题是：什么叫 simple？

不能把 simplicity 简化成：

```text
参数更少（fewer parameters）
```

在学习理论中，simplicity 可能由多种结构表达：

- finite hypothesis count；
- VC dimension；
- margin；
- norm；
- explicit regularization；
- compression，仅在有 formal compression/generalization connection 时使用；
- description length；
- algorithmic solution-selection preference。

不同 simplicity notion 对应不同 inductive bias。选择哪一种，不是语言问题，而是理论假设问题。

### What This Does NOT Imply

Occam's razor 不意味着：

- nature 总是 simple；
- smaller neural network 一定 generalizes better；
- parameter count 是 universal complexity measure；
- 只要模型“看起来简单”，evidence 就自动可靠。

## 2. Occam as Statistical Reasoning

Occam 的统计直觉是：在 constrained/simple family 中找到 good fit，与在 enormous family 中经过大量搜索后找到 good fit，证据含义不同。

核心链条是：

```text
selection opportunity 增多
-> accidental fit 的概率上升
```

搜索空间越大，越可能碰到 accidental pattern。因此同样的 training fit，来自不同 selection process 时，evidence strength 不同。

### Evidence-Control Result

#### Assumptions

- 有有限 hypothesis set $\mathcal{H}$；
- 对每个固定 $h\in\mathcal{H}$，有 concentration bound 控制 $E_{\mathrm{in}}(h)$ 与 $E_{\mathrm{out}}(h)$ 的差距；
- data 在该 bound 的 sampling assumptions 下产生。

#### Claim

对整个 $\mathcal{H}$ 同时控制时，通常会出现与 $|\mathcal{H}|$ 相关的 penalty。T2 的 union-bound intuition 是：

```math
\Pr\left[
\exists h\in\mathcal{H}:
|E_{\mathrm{in}}(h)-E_{\mathrm{out}}(h)|>\epsilon
\right]
\le
2|\mathcal{H}|e^{-2N\epsilon^2}.
```

#### Derivation / Proof Idea

对每个 fixed $h$ 使用 concentration；再对 $\mathcal{H}$ 中所有可被选择的 hypotheses 取 union bound。

#### Interpretation

这不是说大模型一定失败，而是说“经过多少 selection opportunities 后得到这个 fit”会影响证据解释。

#### What This Does NOT Imply

这个 finite-class 形式不是现代 ML 的唯一复杂度理论；它只是把 Occam 的核心统计逻辑显式化。

#### Research Use

读论文时问：reported model 是 fixed before data，还是从许多 candidates 中 selected？如果是后者，selection process 是否被 validation / final evaluation 正确隔离？

## 3. What Occam Does NOT Imply

Occam 不是审美原则，也不是“参数少就是好”。它是一条 evidence-accounting 原则：当 fit 是从巨大选择机会中找到的，必须为这种 search 付出解释成本。

现代模型经常 overparameterized，但依然可能因 norm、margin、optimizer bias、data augmentation、early stopping、implicit regularization 或 representation structure 而表现出受控的 selected solution。T4 的重点是：不要把 nominal size 与 effective solution complexity 混在一起。

## 4. Principle 2 - Population / Distribution Mismatch

第二个原则涉及 data 到底代表什么。为了避免概念混乱，本章使用更宽的 umbrella：

```text
Population / Distribution Mismatch
├── Sampling / Selection Bias
└── Distribution Shift
```

这两类可能重叠，但不是同义词。

### Sampling / Selection Bias

sampling / selection bias 指 collection 或 inclusion mechanism 使 observed data 不能代表 intended target population。

典型例子包括：

- self-selection；
- selective labeling / inclusion；
- non-random missingness；
- nonrepresentative sampling；
- 只记录某类用户、设备、地区或时间窗口。

这里的问题发生在：

```text
World
-> Sampling / inclusion mechanism
-> Dataset
```

即使 learning algorithm 完美，也无法从系统性缺失的信息中无条件恢复 target population 的结构。

### Distribution Shift

distribution shift 指 training/source distribution 与 target/deployment distribution 不一致。它不一定由 biased sampling 造成。

例子包括：

- covariate shift；
- temporal drift；
- spatial / domain shift；
- conditional 或 mechanism change。

covariate shift 的常见形式是：

```math
P_{\mathrm{train}}(X)
\ne
P_{\mathrm{target}}(X),
```

同时在 standard covariate-shift assumption 下：

```math
P_{\mathrm{train}}(Y\mid X)
=
P_{\mathrm{target}}(Y\mid X).
```

注意：不是每个 temporal 或 spatial change 都是 covariate shift。时间、地点或 domain 改变也可能改变 $P(Y\mid X)$，也就是 conditional / mechanism change。

### Relationship between the Categories

biased sampling 可能制造 train/deployment distribution mismatch，但不是每个 distribution shift 都由 biased sampling 引起。例如，training data 当时可能代表 source environment；后来 deployment environment 发生变化，这属于 environment shift，而不是原始 sampling mechanism 的错误。

因此 failure taxonomy 中要保留两个位置：

```text
sampling failure
distribution / environment shift
```

不要把所有 “train data do not represent deployment” 都粗略写成 sampling bias。

## 5. Train/Test Distribution Agreement

classical generalization theory 通常假设 train 与 test examples 来自同一个 population / distributional process。这个假设让 finite-sample generalization statement 有明确目标：

```text
training evidence
-> 关于同一 data-generating process 的 claim
```

需要区分：

```text
finite-sample generalization
```

和

```text
distribution shift generalization
```

前者问：在同一分布假设下，有限样本能否支持 out-of-sample claim？后者问：当 target/deployment distribution 改变时，claim 是否还能成立？

Week 4 的 Canvas-Diagnostic 与 synthetic shift experiments 正好说明：模型可能在 held-out data 上表现正常，但 representation / similarity geometry 在新的 input conditions 下失效。

## 6. Population Mismatch as World-Representation Failure

连接 T1：

```text
World
-> Sampling mechanism
-> Dataset
```

如果 sampling mechanism 系统性排除了某些 subpopulation、condition 或 label pattern，dataset 就不是 intended population 的充分 representation。

再连接 T4：

```text
Dataset
-> Representation Phi
-> Geometry / Similarity
```

即使 training labels 在 observed sample 上拟合良好，representation 也可能只保存了 sample-specific regularities，而没有保存 deployment mechanism。这样失败的根源不是 optimizer，而是 world-to-data 或 data-to-representation 的箭头。

## 7. Principle 3 - Data Snooping

data snooping 指：在解释 final evidence 时，没有把数据曾经如何参与 selection 计入。它远不只是 test leakage。

T3 中已经看到 snooping 可以发生在：

- feature choice；
- preprocessing；
- architecture revision；
- hyperparameter tuning；
- validation reuse；
- test inspection；
- benchmark feedback；
- post-hoc subgroup selection。

关键不是“有没有坏意”，而是 data information 是否以未记账的方式进入了 final model 或 final claim。

## 8. Data Snooping as Hidden Selection

数学直觉是：

```text
adaptive choices 越多
-> effective search process 越大
-> 越可能发现 accidental patterns
```

这直接连接 T2：

```text
fixed hypothesis
vs
selected hypothesis
```

若 hypothesis、features、preprocessing、architecture、hyperparameters 或 reported subgroup 都是看数据后选出的，那么 final evidence 必须反映这条 adaptive path。

### What This Does NOT Imply

不要为 arbitrary researcher adaptivity 发明一个 universal numeric correction。不同 workflow 的 selection process 结构不同；修正方式可能需要 nested validation、pre-registration、held-out final test、benchmark governance 或更明确的 uncertainty accounting。

## 9. Modern Benchmark Contamination

现代 ML 中，data snooping 的形式扩展到 benchmark 生态。

需要区分：

- direct train-test leakage：test examples 或 labels 直接进入 training；
- repeated benchmark feedback：多轮调参受同一 benchmark 结果影响；
- benchmark-aware model development：模型设计长期围绕公开 leaderboard 或 known benchmark quirks 调整；
- pretraining contamination：大规模预训练数据中包含 evaluation examples 或其近似变体。

这些机制都削弱 evidence independence，但不能简单说成同一个数学现象。它们影响的入口不同：有的污染 training set，有的污染 model-selection process，有的污染 benchmark-level research feedback。

## 10. Three Principles as One Structure

三条原则可以合成一个结构：

```text
Occam
-> 控制什么可以被选择

Sampling / Distribution
-> 控制 data 代表什么

Data Snooping
-> 控制 evidence 怎样进入 selection
```

Occam 管的是可被选择的 explanation space；sampling / distribution 管的是 data 与 target world 的关系；data snooping 管的是 evidence 是否被 adaptive selection 偷偷使用。

这就是 Lecture 17 的核心：learning theory 不只是 bound，也是一套 claim discipline。

## 11. Research Lens

读 ML paper 时，问：

- model simplicity 用什么概念表达：dimension、VC、margin、norm、description length、stability，还是 optimizer bias？
- fit 是来自 constrained family，还是来自 large adaptive search？
- dataset 代表哪个 population？
- mismatch 是 sampling / selection mechanism 造成的，还是 deployment environment 改变造成的？
- validation set 是否被反复用到 architecture 或 hyperparameter decisions？
- test set 是否保持 final evidence 的角色？
- benchmark feedback 是否已经成为 training-like signal？
- final claim 面向同一 distribution，还是声称 robustness under shift？

### Existing Repository Links

- T1 的 world-data-representation chain：[ontology map](../00_learning_theory_ontology_world_data_generalization_research_lens.md)。
- T2 的 selected hypothesis 与 uniform control：[finite data and selection](../part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md)。
- T3 的 validation / data contamination：[validation and model selection](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md)。
- Week 4 的 Canvas shift 提供 distribution/environment shift 的实例：[Canvas diagnostic](../../../reports/week4/15_canvas_diagnostic_v1_inventory_and_failure_taxonomy.md)。
- Week 5 的 calibration / abstention 说明 final evidence 要明确 claim type：[calibration metrics](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md)。

[Back to Learning From Data Theory Notebook](../README.md)
