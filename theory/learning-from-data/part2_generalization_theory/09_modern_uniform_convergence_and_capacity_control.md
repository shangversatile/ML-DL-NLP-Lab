# Uniform Convergence and Modern Capacity Control

[← Back to Learning From Data Theory Notebook](../README.md)

## Source Separation

### Caltech Core

承接 Lecture 5-7 的 fixed hypothesis、finite-class union bound、growth function 与 VC dimension。

### Formal Derivation

本 note 形式化 population risk、empirical risk、ERM、excess risk，并推导 uniform convergence 控制 ERM excess risk 的 `2 epsilon` consequence。

### Stanford / Theory Extension

把这些概念放入 ERM、excess risk、PAC-style learnability 与 modern capacity measures 的统一语言。

### Modern Perspective

说明 classical uniform convergence 回答什么问题，以及为什么 overparameterized deep learning 需要 margin、norm、stability、compression、algorithm-dependent 等额外视角。

### Research Lens

把 theory result 转化为 research claim audit：claim 控制的是 generalization gap、excess risk、optimization quality 还是 shifted deployment performance？

### What This Does NOT Imply

modern capacity control preview 不把任何单一工具宣称为 universal explanation；uniform convergence 是 foundational control logic，但不是所有 modern generalization 的完整机制。

## 1. ERM Formalism

给定 input-output space $\mathcal{X}\times\mathcal{Y}$、data-generating distribution $P$、loss $\ell$ 与 hypothesis class $\mathcal{H}$，population risk 定义为：

```math
R(h)
=
\mathbb{E}_{(X,Y)\sim P}
\left[
\ell(h(X),Y)
\right]
```

给定 dataset：

```math
D=\{(x_i,y_i)\}_{i=1}^{N}
```

empirical risk 定义为：

```math
\hat R_D(h)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell(h(x_i),y_i)
```

Empirical Risk Minimization (ERM) 选择：

```math
\hat h
\in
\arg\min_{h\in\mathcal{H}}
\hat R_D(h)
```

population-best member of $\mathcal{H}$ 是：

```math
h^*_{\mathcal{H}}
\in
\arg\min_{h\in\mathcal{H}}
R(h)
```

若存在更大的 reference class 或 Bayes-optimal rule $h^*$，excess risk 常写成：

```math
R(\hat h)-R(h^*)
```

但 research 中必须说明 $h^*$ 是什么 reference：unrestricted Bayes rule、best model in a larger class、human baseline、or deployment policy。

### What This Does NOT Imply

ERM formalism 不意味着实际 algorithm 一定精确求解 ERM；deep learning 常见的是 stochastic optimization、early stopping、regularization、architecture constraints 与 implicit selection。

## 2. Excess-Risk Decomposition

本 note 固定使用以下 notation：

- $h^*$：unrestricted/reference population optimum；
- $h^*_{\mathcal{H}}$：population-best member in $\mathcal{H}$；
- $\hat h$：exact empirical risk minimizer；
- $\tilde h$：actual algorithm output。

一种 useful population-level reference decomposition 是：

```math
R(\tilde h)-R(h^*)
=
\left[
R(h^*_{\mathcal{H}})-R(h^*)
\right]
+
\left[
R(\tilde h)-R(h^*_{\mathcal{H}})
\right]
```

其中：

- 第一项是 **approximation/specification**：chosen class $\mathcal{H}$ 相对 unrestricted/reference optimum 的限制；
- 第二项是 actual output $\tilde h$ 相对 population-best-in-class 的 excess risk，需要由 generalization/estimation control 与 optimization analysis 一起 upper bound。

不要把：

```math
R(\tilde h)-R(\hat h)
```

直接称为非负 optimization error。$\hat h$ 是 empirical minimizer，不一定是 population minimizer；因此 $R(\tilde h)-R(\hat h)$ 可以为正、为负或无法由 optimization logs 直接解释。

### Empirical Optimization Suboptimality

actual algorithm output 的 empirical optimization suboptimality 定义为：

```math
\epsilon_{\mathrm{opt}}
=
\hat R_D(\tilde h)-\hat R_D(\hat h)
\ge
0
```

非负性来自 $\hat h$ 是 exact ERM：

```math
\hat R_D(\hat h)
\le
\hat R_D(h)
\quad
\text{for all }h\in\mathcal{H}
```

因此如果 $\tilde h\in\mathcal{H}$，就有 $\hat R_D(\tilde h)-\hat R_D(\hat h)\ge0$。

### Bound with Generalization and Optimization

#### Assumptions

- $\mathcal{H}$ 固定，且 $\tilde h,\hat h,h^*_{\mathcal{H}}\in\mathcal{H}$；
- $D$ 的 sampling assumptions 足以支持同一个 population risk $R$ 与 empirical risk $\hat R_D$ 之间的 uniform control；
- $\hat h$ 是 exact empirical risk minimizer of $\hat R_D$ over $\mathcal{H}$；
- $\tilde h$ 是 actual algorithm output；
- empirical objective 与 population risk 使用一致的 loss，或已明确说明二者的关系；
- $\epsilon_{\mathrm{opt}}$ 按 empirical objective 定义，而不是按 population-risk difference 定义。

定义 uniform generalization control：

```math
\epsilon_{\mathrm{gen}}
=
\sup_{h\in\mathcal{H}}
\left|
R(h)-\hat R_D(h)
\right|
```

若 $\tilde h,\hat h,h^*_{\mathcal{H}}\in\mathcal{H}$，则：

```math
R(\tilde h)-R(h^*_{\mathcal{H}})
\le
2\epsilon_{\mathrm{gen}}
+
\epsilon_{\mathrm{opt}}
```

推导：

```math
R(\tilde h)
\le
\hat R_D(\tilde h)+\epsilon_{\mathrm{gen}}
```

```math
\hat R_D(\tilde h)
=
\hat R_D(\hat h)+\epsilon_{\mathrm{opt}}
```

```math
\hat R_D(\hat h)
\le
\hat R_D(h^*_{\mathcal{H}})
```

```math
\hat R_D(h^*_{\mathcal{H}})
\le
R(h^*_{\mathcal{H}})+\epsilon_{\mathrm{gen}}
```

合并得到：

```math
R(\tilde h)
\le
R(h^*_{\mathcal{H}})
+
2\epsilon_{\mathrm{gen}}
+
\epsilon_{\mathrm{opt}}
```

相对于 unrestricted/reference optimum：

```math
R(\tilde h)-R(h^*)
\le
\left[
R(h^*_{\mathcal{H}})-R(h^*)
\right]
+
2\epsilon_{\mathrm{gen}}
+
\epsilon_{\mathrm{opt}}
```

三项分别是：

- **approximation/specification**：$R(h^*_{\mathcal{H}})-R(h^*)$；
- **generalization/estimation control**：$2\epsilon_{\mathrm{gen}}$；
- **empirical optimization suboptimality**：$\epsilon_{\mathrm{opt}}$。

### Important Caveat

这不是唯一 analysis template。若使用 regularized ERM，$\hat h$ 可能应定义为 regularized empirical objective 的 minimizer；若 algorithm 输出不在同一个 $\mathcal{H}$，uniform control set 也要随之改变。population excess risk 的分解还可能加入 regularization bias、algorithmic stability terms、optimization noise 或 approximation to Bayes risk。研究笔记中必须先定义 reference predictors，再解释每一项。

### What This Does NOT Imply

decomposition 和 inequality 不说明每一项都可观测。$R(h)$、$h^*$、$h^*_{\mathcal{H}}$ 与 $\epsilon_{\mathrm{gen}}$ 通常不可直接访问；实际研究只能用 held-out evidence、bounds、simulation、ablations 或 controlled tasks 来间接诊断。它也不说明 $R(\tilde h)-R(\hat h)$ 是非负 optimization error。

## 3. Uniform Convergence

uniform convergence formalizes：

```math
\sup_{h\in\mathcal{H}}
\left|
\hat R_D(h)-R(h)
\right|
```

若该 quantity 小，则 empirical risk landscape 与 population risk landscape 在整个 $\mathcal{H}$ 上接近。ERM 的 population performance 可以由 empirical performance 推出。

### Theorem: Uniform Convergence Controls ERM Excess Risk

#### Assumptions

- $\mathcal{H}$ 固定；
- $D$ i.i.d. from $P$；
- loss 与风险定义一致；
- with probability at least $1-\delta$：

```math
\sup_{h\in\mathcal{H}}
\left|
\hat R_D(h)-R(h)
\right|
\le
\epsilon
```

- $\hat h$ 是 exact ERM；
- $h^*_{\mathcal{H}}$ 是 population-best in class。

#### Claim

在该 high-probability event 上：

```math
R(\hat h)-R(h^*_{\mathcal{H}})
\le
2\epsilon
```

#### Derivation / Proof Idea

```math
R(\hat h)
\le
\hat R_D(\hat h)+\epsilon
```

ERM gives：

```math
\hat R_D(\hat h)
\le
\hat R_D(h^*_{\mathcal{H}})
```

uniform convergence again gives：

```math
\hat R_D(h^*_{\mathcal{H}})
\le
R(h^*_{\mathcal{H}})+\epsilon
```

因此：

```math
R(\hat h)
\le
R(h^*_{\mathcal{H}})+2\epsilon
```

#### Interpretation

uniform convergence 把 “minimize empirical risk” 连接到 “near-minimize population risk in class”。它解决的是 data-dependent selection 的核心问题。

#### What This Does NOT Imply

- 不控制 approximation error $R(h^*_{\mathcal{H}})-R(h^*)$；
- 不控制 actual optimizer output $\tilde h$，除非加入 optimization analysis；
- 不适用于 distribution shift；
- 不保证 bound tight；
- 不解释所有 modern overparameterized generalization；
- 不等于 model is calibrated/interpretable/causal。

#### Research Use

论文若只报告 low training loss，需要额外证据说明：为什么 empirical selection 没有利用 finite sample noise？uniform convergence 是一种答案，但不是唯一答案。

## 4. Finite $\mathcal{H}$ versus VC Control

finite hypothesis class 给出：

```math
\left|
\hat R(h)-R(h)
\right|
\lesssim
\sqrt{
\frac{\log|\mathcal{H}|+\log(1/\delta)}{N}
}
```

VC control 给出：

```math
\left|
\hat R(h)-R(h)
\right|
\lesssim
\sqrt{
\frac{
d_{\mathrm{VC}}\log(N/d_{\mathrm{VC}})
+
\log(1/\delta)
}{N}
}
```

二者的共同逻辑是：

```text
capacity term
competes with
sample size
under a confidence requirement
```

finite $\mathcal{H}$ 用 global count；VC dimension 用 finite-sample dichotomy capacity。后者可以处理 infinite classes，但仍是 worst-case class-level control。

### What This Does NOT Imply

不能把 $\log|\mathcal{H}|$、$d_{\mathrm{VC}}$、parameter count、norm、margin 或 Rademacher complexity 混为同一个 “complexity”。它们回答相近问题，但数学对象不同。

## 5. PAC-Style Interpretation

PAC-style language 关注 Probably Approximately Correct：

- $\epsilon$：tolerance，允许 learned predictor 的 risk 离目标多远；
- $\delta$：failure probability，允许 theorem 不成立的概率；
- sample complexity：达到 $(\epsilon,\delta)$ 所需的 $N$；
- learnability：是否存在 algorithm，使 sample size 随 $1/\epsilon$、$\log(1/\delta)$ 和 complexity measure 以可控方式增长。

在 agnostic setting 中，通常目标是与 best-in-class 比较：

```math
R(\hat h)
\le
\inf_{h\in\mathcal{H}}R(h)+\epsilon
```

在 realizable setting 中，假设某个 $h\in\mathcal{H}$ 可以 achieve zero population error 或 labels truly generated by class member。这个 assumption 更强，bounds 形式也可不同。

### What This Does NOT Imply

PAC vocabulary 不自动给出实际 bound。必须说明 $\mathcal{H}$、loss、sampling、realizable/agnostic condition、algorithm 和 capacity measure。

## 6. Beyond VC Dimension

以下概念是 T5 会深入的 preview；T2 只给 research-reading vocabulary。

### Rademacher Complexity

Rademacher complexity 衡量 $\mathcal{H}$ 对 random signs 的 fit 能力。它比 VC dimension 更 data-dependent，可用于 real-valued function classes 和 bounded losses。

### Margin-Based Bounds

对于 classifiers，尤其是 SVM 与 neural networks，margin 可能比 raw classification capacity 更 relevant。大 margin 表示 decision boundary 离 training points 更远，通常支持更强 robustness intuition。

### Norm-Based Capacity

在 linear models 和 neural networks 中，parameter norm 或 path norm 等 constraints 可以控制 function class 的 effective size。parameter count 大不一定表示 norm-controlled class 大。

### Algorithmic Stability

stability 分析问：移除或替换一个 training example，algorithm output 或 loss 是否变化很小？如果 algorithm 稳定，generalization gap 可被控制。

### Compression

compression 观点问：trained model 或 its predictions 是否可以由少量 information 描述？若可以，effective selection complexity 可能远小于 parameter count。

### Algorithm-Dependent Generalization

classical uniform convergence controls all of $\mathcal{H}$。现代 deep learning 中，actual optimizer 可能只访问特定 solution subset。algorithm-dependent theory 尝试控制这个 subset，而不是 worst-case class。

## 7. Why Classical Uniform Convergence Can Be Insufficient

classical uniform convergence 回答的问题是：

```text
Does empirical risk approximate population risk uniformly over a chosen class?
```

它剩下的问题包括：

- chosen class 太大时 worst-case bound 可能 vacuous；
- overparameterized models 可以 shatter training data，但 SGD 仍选择 structured solutions；
- population distribution 可能有 benign geometry；
- interpolation 不必然导致 poor generalization；
- implicit regularization 可能比 explicit class capacity 更关键；
- adaptive benchmark use 可能违反独立 evaluation assumptions。

### What This Does NOT Imply

uniform convergence 不是 useless。它提供了最清晰的 fixed-vs-selected logic、capacity/sample/confidence vocabulary、ERM guarantee template 和 overclaim prevention framework。现代限制说明需要更细工具，不是基础逻辑失效。

## 8. Research Lens

读 modern ML theory 或 empirical paper 时，将 claim 放入四个层级：

1. **Representation**：input 是否包含 target-relevant information？
2. **Class/Algorithm**：可选择的 functions 是什么，实际 algorithm 偏向哪里？
3. **Statistical Evidence**：empirical performance 如何支持 population claim？
4. **Deployment Scope**：claim 覆盖哪个 distribution、loss 与 decision setting？

如果一篇论文没有回答这些问题，它的 generalization claim 应被看作 empirical observation，而不是理论上充分支撑的 conclusion。
