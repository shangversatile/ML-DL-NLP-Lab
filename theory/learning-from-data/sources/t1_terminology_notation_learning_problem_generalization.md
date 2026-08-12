# Terminology and Notation

[← Back to Learning From Data Theory Notebook](../README.md)

本文件为 `theory/learning-from-data/` 提供稳定 notation。不同课程、论文和代码库会使用不同符号；本目录优先保持概念角色清楚，而不是追求所有来源的符号完全一致。

## 1. Core notation table

| Symbol / term | Meaning | Notes |
| ------------- | ------- | ----- |
| $\mathcal{X}$ | input space | 所有可能输入的集合；代码中常对应 feature vectors、images、tokens 或 structured objects |
| $\mathcal{Y}$ | output space | labels、real-valued targets、classes、actions 或 structured outputs |
| $x$ | one input | realized observation 的 input component |
| $y$ | one output / label / target observation | realized observation 的 output component，可能 noisy |
| $X$ | input random variable or design matrix | 上下文决定；作为 random variable 时大写，作为矩阵时通常有 shape `(N,d)` |
| $Y$ | output random variable | 与 $X$ 共同服从 $P$ |
| $f$ | target function | deterministic target setting 中的 unknown mapping |
| $h$ | hypothesis | hypothesis set 中任意候选函数 |
| $g$ | final / selected hypothesis | learning algorithm 在 dataset 上训练后输出的 hypothesis |
| $\mathcal{H}$ | hypothesis set / hypothesis class | learner 被允许选择的函数集合 |
| $A$ | learning algorithm | 从 dataset 到 selected hypothesis 的 procedure |
| $D$ | dataset | finite realized sample，通常 $D=\{(x_i,y_i)\}_{i=1}^{N}$ |
| $P$ | data-generating distribution | 定义 train/future sampling 与 population risk 的 distribution |
| $\ell$ | loss | per-example error measure |
| $E_{\mathrm{in}}$ | in-sample error / empirical error | finite dataset 上的 average loss |
| $E_{\mathrm{out}}$ | out-of-sample error / population risk | distribution $P$ 上的 expected loss |
| risk | expected loss | 有时指 population risk；需看上下文 |
| empirical risk | finite-sample average loss | 与 $E_{\mathrm{in}}$ 基本同义 |
| $\hat R_D(h)$ | empirical risk | T2 中常用 notation，等价于在 dataset $D$ 上的 $E_{\mathrm{in}}(h)$ |
| $R(h)$ | population risk | T2 中常用 notation，等价于 $E_{\mathrm{out}}(h)$ |
| target function | unknown ideal mapping | noisy setting 中可能要替换为 conditional distribution |
| hypothesis | candidate predictor | 函数层对象，不等于参数向量 |
| hypothesis set | allowed family of hypotheses | 表达 inductive bias |
| learning algorithm | selection procedure | 可包含 optimization、regularization、early stopping、model selection |
| generalization | out-of-sample performance | 不能只由 training error 判断 |
| $\epsilon$ | tolerance / deviation parameter | generalization bound 中允许的误差或 excess-risk tolerance |
| $\delta$ | confidence failure probability | theorem 不成立的概率上界，confidence 是 $1-\delta$ |
| dichotomy | labeling pattern on finite points | $\mathcal{H}$ 在一组 points 上诱导的 binary labels |
| $m_{\mathcal{H}}(N)$ | growth function | $\mathcal{H}$ 在任意 $N$ points 上最多能实现的 dichotomies 数量 |
| shattering | realizing all labelings | 若 $\mathcal{H}$ 能实现 $N$ points 上所有 $2^N$ labelings，则 shatter 该点集 |
| break point | first unshatterable size | 若 $m_{\mathcal{H}}(k)<2^k$，则 $k$ 是 break point |
| $d_{\mathrm{VC}}$ | VC dimension | $\mathcal{H}$ 可 shatter 的最大 point-set size |
| uniform convergence | simultaneous empirical-population control | 控制 $\sup_{h\in\mathcal{H}}|\hat R_D(h)-R(h)|$ |
| ERM | empirical risk minimization | 选择 empirical risk 最小的 hypothesis |
| excess risk | risk above a reference predictor | 通常为 $R(\hat h)-R(h^*)$ 或 $R(\hat h)-R(h^*_{\mathcal{H}})$，需定义 reference |
| sample complexity | samples needed for target guarantee | 达到给定 $\epsilon,\delta$ 所需的 $N$ |
| approximation error | best-in-class limitation | $\mathcal{H}$ 与 reference class/Bayes rule 的 population-risk 差距 |
| estimation error | finite-sample selection error | selected empirical hypothesis 与 population-best-in-class 的差距 |
| optimization error | algorithmic search error | actual algorithm output 与 intended optimum 的差距 |
| bias | average predictor's systematic deviation | squared-loss bias-variance setup 中定义 |
| variance | dataset-induced predictor variability | over possible training datasets 的 prediction fluctuation |
| capacity | effective flexibility of a class/procedure | 可能由 VC dimension、growth function、norm、margin、stability 等衡量 |

## 2. Empirical error vs population risk

empirical error 是已观察 training dataset 上的 average：

```math
E_{\mathrm{in}}(h)
=
\frac{1}{N}
\sum_{i=1}^{N}
\ell(h(x_i),y_i)
```

population risk 是对 data-generating distribution 的 expectation：

```math
E_{\mathrm{out}}(h)
=
\mathbb{E}_{(X,Y)\sim P}
\left[
\ell(h(X),Y)
\right]
```

区别：

- $E_{\mathrm{in}}$ 依赖 realized dataset；
- $E_{\mathrm{out}}$ 依赖 unknown $P$；
- 低 $E_{\mathrm{in}}$ 不自动意味着低 $E_{\mathrm{out}}$；
- learning theory 的核心问题是何时可以用 finite-sample evidence 约束 population behavior。

## 3. Hypothesis vs parameter

parameter 是某个 model family 内的坐标，例如：

```math
\theta=(w,b)
```

hypothesis 是由 parameter 决定的函数：

```math
h_\theta(x)=w^\top x+b
```

在 simple parameterized families 中，parameter 与 hypothesis 常一一对应。但概念上仍应区分：

- parameter 是 numerical representation；
- hypothesis 是 input-output mapping；
- 多个 parameter settings 可能表示同一个 function；
- learning theory 通常关心 functions 的 behavior，而不是只关心 parameter count。

## 4. Model family vs learning algorithm

model family / hypothesis set：

```math
\mathcal{H}=\{h_\theta:\theta\in\Theta\}
```

learning algorithm：

```math
A(D)=h_{\hat{\theta}}
```

同一个 $\mathcal{H}$ 可以配不同 algorithms：normal equations、batch gradient descent、SGD、Adam、regularized solver、early stopping pipeline。不同 algorithms 可能选择不同 $g$，即使 hypothesis set 相同。

## 5. Representation vs hypothesis

representation 是 learner 实际接收或构造的 input form：

```math
\Phi(x)
```

hypothesis 在 representation 上定义：

```math
h(x)=w^\top\Phi(x)+b
```

representation 决定哪些 distinctions 被保留、放大或丢失；hypothesis set 决定在这些 represented inputs 上允许哪些 mapping。二者共同决定 expressivity 与 generalization behavior。

## 6. Random variable vs realized observation

random variable 表示抽样前的不确定对象：

```math
(X,Y)\sim P
```

realized observation 是抽样后的具体值：

```math
(x_i,y_i)
```

区分二者非常重要：

- $E_{\mathrm{out}}$ 是关于 random variables 的 expectation；
- $E_{\mathrm{in}}$ 是关于 realized observations 的 average；
- dataset $D$ 本身可以被视为 random object；
- selected hypothesis $g=A(D)$ 也依赖 random dataset。

## 7. Deterministic target vs probabilistic target

deterministic target：

```math
Y=f(X)
```

probabilistic target：

```math
Y\mid X=x \sim P(\cdot\mid x)
```

在 noisy setting 中，应优先说明 target 是 deterministic function、conditional expectation、conditional mode、conditional probability、quantile，还是 full predictive distribution。不同 loss 对应不同 target functional。

## 8. Generalization

generalization 是 selected hypothesis 在未见 samples 上表现良好的性质。最常见数学对象是 generalization gap：

```math
E_{\mathrm{out}}(g)-E_{\mathrm{in}}(g)
```

但研究中还必须说明：

- out-of-sample distribution 是什么；
- evaluation set 是否独立；
- validation 是否被 adaptive reuse；
- metric 是否匹配 problem definition；
- subgroup、shift、calibration、abstention 等是否需要单独评估。

## 9. Growth function, shattering, and VC dimension

对于 binary classification，$\mathcal{H}$ 在 $N$ 个 points 上诱导的 dichotomy set 是：

```math
\Pi_{\mathcal{H}}(x_1,\ldots,x_N)
=
\left\{
(h(x_1),\ldots,h(x_N)):h\in\mathcal{H}
\right\}
```

growth function 是 worst-case dichotomy count：

```math
m_{\mathcal{H}}(N)
=
\max_{x_1,\ldots,x_N}
\left|
\Pi_{\mathcal{H}}(x_1,\ldots,x_N)
\right|
```

若存在某组 $N$ points 被 $\mathcal{H}$ shatter，则：

```math
m_{\mathcal{H}}(N)=2^N
```

VC dimension 是最大可 shatter size：

```math
d_{\mathrm{VC}}
=
\max
\{N:m_{\mathcal{H}}(N)=2^N\}
```

break point 是第一个无法达到 maximal dichotomy count 的 size：

```math
m_{\mathcal{H}}(k)<2^k
```

若 $d_{\mathrm{VC}}$ 有限，通常 $k=d_{\mathrm{VC}}+1$。

## 10. Uniform convergence and ERM

uniform convergence 控制的是：

```math
\sup_{h\in\mathcal{H}}
\left|
\hat R_D(h)-R(h)
\right|
```

它与 fixed-h concentration 的区别在 quantifier：

- fixed-h concentration：一个预先固定的 $h$；
- finite-class bound：所有 $h\in\mathcal{H}$，但 $\mathcal{H}$ finite；
- uniform convergence：一般 hypothesis class 中所有 $h$，在满足 capacity/control 条件时成立。

ERM:

```math
\hat h
\in
\arg\min_{h\in\mathcal{H}}
\hat R_D(h)
```

若 uniform deviation 不超过 $\epsilon$，则 exact ERM 满足：

```math
R(\hat h)
\le
\inf_{h\in\mathcal{H}}R(h)+2\epsilon
```

该结论只比较 $\mathcal{H}$ 内的 population-best member，不自动处理 $\mathcal{H}$ 外的 Bayes risk。

## 11. Generalization gap vs excess risk

generalization gap 比较同一个 hypothesis 的 empirical 与 population quantities：

```math
R(h)-\hat R_D(h)
```

excess risk 比较两个 hypotheses 或 policies 的 population risks：

```math
R(\hat h)-R(h^*)
```

二者不是同一个对象。small generalization gap 可能与 high excess risk 同时成立：例如一个很弱的 constant classifier 在 train 与 test 上表现都差，但 gap 很小。low empirical risk 也不等于 low excess risk，除非再加入 uniform convergence、approximation 与 optimization 条件。

## 12. Capacity vs parameter count

capacity 是 hypothesis class 或 algorithm-selected family 的 effective flexibility。parameter count 只是某种 parameterization 的维度。

不能写成：

```text
capacity = parameter count
```

原因：

- 多个 parameter vectors 可能表示同一个 function；
- constraints、regularization、norm、margin 会改变 effective class；
- feature map $\Phi$ 改变 original input domain 上的 induced family；
- algorithmic stability 或 implicit bias 可能比 raw class size 更 relevant；
- VC dimension 是 worst-case shattering capacity，不是 dataset-specific performance。

## 13. Approximation, estimation, optimization

给定 reference predictor $h^*$、population-best-in-class $h^*_{\mathcal{H}}$、exact ERM $\hat h$ 与 actual output $\tilde h$，一种 useful decomposition 是：

```math
R(\tilde h)-R(h^*)
=
\left[
R(h^*_{\mathcal{H}})-R(h^*)
\right]
+
\left[
R(\hat h)-R(h^*_{\mathcal{H}})
\right]
+
\left[
R(\tilde h)-R(\hat h)
\right]
```

这里的三个 bracket 分别对应 approximation/specification、estimation/generalization、optimization/computation。但这不是唯一 decomposition；必须在每篇 note 或论文中说明 reference objects。

## 14. Bias and variance

在 squared-loss deterministic-target setup 中，固定 $x$：

```math
\bar g(x)
=
\mathbb{E}_{D}[g_D(x)]
```

bias:

```math
\bar g(x)-f(x)
```

variance:

```math
\mathbb{E}_{D}
\left[
\left(g_D(x)-\bar g(x)\right)^2
\right]
```

若 $Y=f(x)+\eta$ 且 $\mathbb{E}[\eta|x]=0$，则 squared loss 下还会出现 noise term：

```math
\mathbb{V}[\eta|X=x]
```

不要把 bias 简化成 underfitting，也不要把 variance 简化成 overfitting。那些是经验症状，不是定义。

[← Back to Learning From Data Theory Notebook](../README.md)
