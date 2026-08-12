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
| target function | unknown ideal mapping | noisy setting 中可能要替换为 conditional distribution |
| hypothesis | candidate predictor | 函数层对象，不等于参数向量 |
| hypothesis set | allowed family of hypotheses | 表达 inductive bias |
| learning algorithm | selection procedure | 可包含 optimization、regularization、early stopping、model selection |
| generalization | out-of-sample performance | 不能只由 training error 判断 |

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

[← Back to Learning From Data Theory Notebook](../README.md)

