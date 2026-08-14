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
| $h^*$ | unrestricted/reference population optimum | reference class 或 Bayes rule 下的 population optimum，必须按上下文定义 |
| $h^*_{\mathcal{H}}$ | population-best member of $\mathcal{H}$ | $\mathcal{H}$ 内 population risk 最小的 hypothesis |
| $\hat h$ | exact empirical risk minimizer | 精确最小化 $\hat R_D$ 的 idealized ERM solution |
| $\tilde h$ | actual algorithm output | 实际 optimizer / training procedure 输出的 hypothesis |
| excess risk | risk above a reference predictor | 通常为 $R(\tilde h)-R(h^*)$、$R(\hat h)-R(h^*)$ 或 $R(\hat h)-R(h^*_{\mathcal{H}})$，需定义 reference |
| sample complexity | samples needed for target guarantee | 达到给定 $\epsilon,\delta$ 所需的 $N$ |
| approximation error | best-in-class limitation | $\mathcal{H}$ 与 reference class/Bayes rule 的 population-risk 差距 |
| estimation error | finite-sample selection error | selected empirical hypothesis 与 population-best-in-class 的差距 |
| $\epsilon_{\mathrm{opt}}$ | empirical optimization suboptimality | $\hat R_D(\tilde h)-\hat R_D(\hat h)\ge0$，不是 $R(\tilde h)-R(\hat h)$ |
| optimization error | algorithmic search error | 应先在 empirical objective 或明确 optimization objective 上定义 |
| bias | average predictor's systematic deviation | squared-loss bias-variance setup 中定义 |
| variance | dataset-induced predictor variability | over possible training datasets 的 prediction fluctuation |
| capacity | effective flexibility of a class/procedure | 可能由 VC dimension、growth function、norm、margin、stability 等衡量 |
| likelihood | probability of observed data under model | T3 logistic regression 中通常指 conditional likelihood $p(y\mid x;w)$ 的乘积 |
| log likelihood | logarithm of likelihood | 把 product 转成 sum，便于 optimization |
| negative log likelihood | loss from maximizing likelihood | 最小化 NLL 等价于最大化 likelihood |
| cross entropy | common classification surrogate loss | binary case 下与 Bernoulli conditional NLL 对应 |
| surrogate loss | optimizable proxy objective | 例如 cross entropy 代理 0/1 classification objective，但二者不是同一 metric |
| parameterization | numerical coordinates for functions | 例如 $w$ 或 neural-network $\theta$；不等于 function space 本身 |
| selected parameter vector | actual fitted parameters | 例如训练后得到的 $\tilde w$ 或 $\tilde\theta$ |
| selected function | function represented by selected parameters | 例如 $g=h_{\tilde\theta}$；不等于 parameter vector 本身 |
| optimizer | update/search procedure | 使用 gradients 或其他信号更新 parameters；不等于 objective |
| optimization trajectory | sequence of iterates | 例如 $\theta_0,\theta_1,\ldots,\theta_T$，可影响 selected solution |
| hidden representation | intermediate learned state | 例如 $z_\ell=\Phi_{\theta,\ell}(x)$ |
| computational graph | dependency graph of operations | backpropagation 在图上复用 local derivatives |
| backpropagation | gradient-computation algorithm | 计算 gradients；不等于 gradient descent |
| overfitting | selected procedure exploits sample-specific structure | low training error alone is not the definition |
| deterministic noise | residual target structure relative to chosen class | target deterministic 但 outside effective hypothesis family 时出现 |
| stochastic noise | randomness in observations or labels | 来自 $Y\mid X$ 的随机性、measurement 或 label variation |
| regularizer | explicit or implicit preference mechanism | 严格地说 explicit regularizer 需给出 $\Omega$ 或 constraint |
| $\lambda$ | regularization coefficient | soft penalty 中控制 empirical fit 与 penalty tradeoff |
| hard constraint | restricted feasible region | 例如 $\Omega(w)\le C$ |
| soft constraint / penalty | augmented objective | 例如 $\hat R_D(w)+\lambda\Omega(w)$ |
| validation set / dev set | data used for model selection | 可选择 hyperparameters/checkpoints，但被使用后不是 final independent test |
| test set | data for final evaluation | 若要保持 final-test role，不应影响 development choices |
| model selection | choosing among candidate procedures/models | selection layer 可依赖 validation data |
| hyperparameter selection | choosing non-fitted controls | 如 $\lambda$、learning rate、architecture width、threshold |
| data leakage | held-out information enters fitting pipeline | 包括 direct leakage 与 preprocessing leakage |
| data contamination | evaluation data influences development | 包括 selection leakage 与 benchmark adaptation |
| cross-validation | repeated train/validation resampling estimate | 用于 selection 后不自动提供独立 final test |
| nested cross-validation | inner selection plus outer evaluation | outer loop 评估 selection procedure |
| selection procedure | data-dependent rule selecting final candidate | 可包括 researcher iteration 与 automated search |
| adaptive data reuse | repeated use of data to guide future choices | 会改变 evaluation estimate 的 interpretation |
| selection / evaluation failure | research-process evidence failure | evaluation data 影响 final procedure 后仍被解释为 independent evidence |

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

给定 unrestricted/reference optimum $h^*$、population-best-in-class $h^*_{\mathcal{H}}$、exact ERM $\hat h$ 与 actual output $\tilde h$，一种 useful reference decomposition 是：

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

第一项是 approximation/specification。第二项不是纯 optimization error；它需要用 generalization/estimation control 与 empirical optimization suboptimality 一起 upper bound。

empirical optimization suboptimality 定义为：

```math
\epsilon_{\mathrm{opt}}
=
\hat R_D(\tilde h)-\hat R_D(\hat h)
\ge
0
```

若：

```math
\epsilon_{\mathrm{gen}}
=
\sup_{h\in\mathcal{H}}
\left|
R(h)-\hat R_D(h)
\right|
```

且 $\tilde h,\hat h,h^*_{\mathcal{H}}\in\mathcal{H}$，则：

```math
R(\tilde h)-R(h^*_{\mathcal{H}})
\le
2\epsilon_{\mathrm{gen}}
+
\epsilon_{\mathrm{opt}}
```

因此：

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

三项分别对应 approximation/specification、generalization/estimation control、empirical optimization suboptimality。不要把 $R(\tilde h)-R(\hat h)$ 称为非负 optimization error；它是 population-risk difference，不由 empirical minimization 保证非负。

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

## 15. Parameter vs hyperparameter

parameter 是在 fitting objective 中由 training algorithm 直接更新或估计的 quantity：

```math
w,\theta
```

hyperparameter 控制 learning procedure，但通常不由同一个 empirical objective 直接拟合，例如：

```math
\lambda,\eta,\text{network width},\text{batch size},\text{threshold}
```

区别不是绝对的数学本体，而是 procedure role：

- parameter 通常由 train set fitting；
- hyperparameter 通常由 validation/dev selection；
- 若 hyperparameter search 使用 validation feedback，则 final selected model 依赖 validation data；
- reported final performance 必须说明 hyperparameter selection 是否已完成并冻结。

## 16. Training objective vs evaluation metric

training objective 是 optimizer 最小化或最大化的 quantity：

```math
\hat R_D(w)
```

evaluation metric 是报告或比较模型的 quantity，例如 accuracy、NLL、ECE、AUROC、selective risk 或 cost-sensitive utility。

二者不必相同：

```text
cross entropy
!=
classification error
!=
calibration error
!=
deployment utility
```

surrogate loss 可使 optimization 更可行，但不能自动证明 reported metric 会改善，也不能自动证明 model calibrated 或 robust。

## 17. Validation error vs final test error

validation error 是 development signal：

```math
\hat R_{\mathrm{val}}(g_m)
```

它可用于选择 model、hyperparameters、thresholds 或 checkpoints。选择后：

```math
g_{\mathrm{selected}}
=
A(D_{\mathrm{train}},D_{\mathrm{val}})
```

因此 selected model 的 validation error 不再是 untouched final estimate。

final test error 应在 procedure 冻结后计算：

```math
\hat R_{\mathrm{test}}(g_{\mathrm{frozen}})
```

如果 test result 反馈到 model/procedure design，它就失去原先 final-test role。

## 18. Model fitting vs model selection

model fitting 是在 fixed procedure 内估计 parameters：

```math
\tilde h
=
A_m(D_{\mathrm{train}})
```

model selection 是在多个 fitted candidates 或 procedures 中选择：

```math
\hat m
=
\arg\min_{m\in\mathcal{M}}
\hat R_{\mathrm{val}}(g_m)
```

然后：

```math
g_{\hat m}
```

是 validation-dependent selected model。credible evaluation 必须同时审计 fitting layer 与 selection layer。

## 19. T4 margin, kernel, locality, and evidence terminology

| Term | Meaning | Notes |
| ---- | ------- | ----- |
| separating hyperplane | affine decision boundary $w^\top x+b=0$ | $w$ is normal to the hyperplane |
| functional margin | $y_i(w^\top x_i+b)$ | scale-dependent; changes under $(w,b)\mapsto(cw,cb)$ for $c>0$ |
| geometric margin | $y_i(w^\top x_i+b)/\|w\|_2$ | scale-invariant distance to the boundary in the chosen representation |
| hard-margin SVM | maximum-margin classifier with no training margin violations | requires separability in the chosen representation |
| soft-margin SVM | SVM allowing slack variables for violations | trades margin/norm preference against violations via $C$ |
| slack variable | $\xi_i\ge0$ measuring margin-constraint violation | not identical to misclassification |
| support vector | training example with nonzero dual coefficient | hard-margin support vectors lie on active margin constraints |
| primal problem | optimization in original variables such as $w,b,\xi$ | expresses geometry and constraints directly |
| dual problem | optimization in Lagrange multipliers such as $\alpha_i$ | exposes inner products and support-vector structure |
| Lagrangian | objective plus multiplier-weighted constraints | bridge from constrained primal to dual |
| KKT conditions | feasibility, stationarity, and complementary slackness conditions | characterize convex SVM optima under standard regularity |
| complementary slackness | product of multiplier and constraint residual equals zero | explains why inactive constraints have zero multipliers |
| hinge loss | $\max(0,1-yf(x))$ | margin-aware surrogate loss; not log loss |
| kernel | function $K(x,z)=\langle\Phi(x),\Phi(z)\rangle$ | similarity-like inner-product function, not arbitrary similarity |
| kernel trick | using $K(x,z)$ without explicitly forming $\Phi(x)$ | relies on algorithms depending on inner products |
| Gram matrix | finite matrix $K_{ij}=K(x_i,x_j)$ | valid kernels yield PSD Gram matrices on finite samples |
| positive semidefinite | $c^\top Kc\ge0$ for all finite $c$ | finite-sample signature of inner-product behavior |
| feature space | represented space containing $\Phi(x)$ | geometry is defined there, not necessarily in raw input space |
| RKHS | reproducing-kernel Hilbert space | preview only; full RKHS theory is deferred |
| radial basis function | localized basis response based on distance to a center | common Gaussian form uses $\exp(-\|x-c\|^2/(2\sigma^2))$ |
| RBF center | point $c_k$ around which a basis unit is localized | may be fixed, selected from data, clustered, or learned |
| RBF width | scale $\sigma_k$ controlling spatial spread | smaller/larger widths have contextual effects, not universal laws |
| locality | assumption that nearby points or centers should have related behavior | depends on metric and representation |
| Occam's razor | prefer simpler explanations when fit is adequate | simplicity may mean count, VC dimension, norm, margin, compression, prior, or algorithmic preference |
| sampling bias | sample fails to represent the intended target population | more biased samples do not repair the mechanism by themselves |
| data snooping | unaccounted data use influences final evidence or selection | broader than direct test leakage |
| prior | pre-data assumption over hypotheses or parameters | Bayesian methods are not assumption-free |
| likelihood | probability of observed data under a hypothesis/model | connects data to hypotheses |
| posterior | updated distribution $p(h\mid D)$ | proportional to likelihood times prior |
| posterior predictive | predictive distribution integrating over posterior hypotheses | does not automatically guarantee calibration |
| aggregation | combining multiple hypotheses or predictors | may reduce variance or alter bias depending on correlation and errors |
| bagging | resample, fit multiple learners, aggregate | targets dataset-induced variance for unstable learners |
| boosting | sequentially fit learners with changed emphasis on prior errors | not merely independent averaging |
| blending / stacking | learn a combination layer over existing models | introduces another selection/evaluation layer |

Required distinctions:

```text
margin
!=
probability confidence
```

An SVM score or margin is geometric unless a separate calibration model/evidence justifies probability interpretation.

```text
kernel
!=
feature map
```

The feature map $\Phi$ defines a represented space. The kernel computes inner products corresponding to that representation.

```text
RBF basis function
!=
RBF kernel
```

An RBF basis function is an explicit localized feature around a center. A Gaussian/RBF kernel is a pairwise inner-product function used inside kernel methods.

[← Back to Learning From Data Theory Notebook](../README.md)
