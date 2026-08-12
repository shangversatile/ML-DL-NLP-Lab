# What Does It Mean to Learn?

[← Back to Learning From Data Theory Notebook](../README.md)

本章主要对应 Caltech `Learning From Data` Lecture 1, `The Learning Problem`。Lecture 1 的贡献不是提出一个复杂算法，而是给出 machine learning 的基本 diagram：unknown target function、training examples、hypothesis set、learning algorithm、final hypothesis。这个 diagram 会在后续 feasibility、generalization、VC dimension、regularization 与 validation 中反复出现。

## 0. Source Separation

- **Caltech Core**：learning from data versus explicit programming、target function、training examples、learning algorithm、hypothesis set、final hypothesis、supervised/unsupervised/reinforcement learning。
- **Stanford / Theory Extension**：把 learning formulation 看作 statistical learning problem 的形式化入口。
- **Modern Perspective**：inductive bias、memorization、representation 与 evaluation discipline 在现代 ML systems 中仍然是同一个基础问题。
- **Research Reflection**：说明如何把一个 research idea operationalize 成可审计的 learning problem。

## 1. Learning from data versus explicit programming

### Caltech Core

传统 programming 的理想形式是：程序员知道规则，然后把规则写成 code。对于排序、精确算术、数据库查询、确定性业务规则，这种模式有效，因为 desired mapping 本身可以被清楚指定。

Machine learning 的出发点不同：我们通常知道 examples，知道希望系统在某种意义上表现好，但无法直接写出完整 mapping。例如：

- 给定一张手写数字图像，输出数字类别；
- 给定用户行为历史，预测是否会流失；
- 给定医学影像，判断是否存在病灶；
- 给定一句话，预测下一个 token 或语义标签。

这些任务并非没有结构，而是结构很难被人工完整编码。学习的目标是用 data 让 algorithm 在 hypothesis set 中选择一个 approximation。

### Formal Setup

设 input space 为 $\mathcal{X}$，output space 为 $\mathcal{Y}$。如果存在 deterministic target function：

```math
f:\mathcal{X}\to\mathcal{Y}
```

training examples 是有限集合：

```math
D=\{(x_1,y_1),\ldots,(x_N,y_N)\}
```

在无噪声 supervised setting 中：

```math
y_i=f(x_i)
```

learner 无法直接访问完整的 $f$，只能访问 $D$。learning algorithm $A$ 接收 $D$，从 hypothesis set $\mathcal{H}$ 中选择 final hypothesis：

```math
g=A(D),\quad g\in\mathcal{H}
```

learning 的基本希望是：

```math
g(x)\approx f(x)
```

尤其是在 training set 之外的 unseen $x$ 上。

### Intuition

这一定义强调：learning 的重点不是把 training examples 存下来，而是用 examples 推断未见区域的 behavior。若只要求在已见 examples 上正确，一个 lookup table 就足够；若要求在未来输入上表现好，必须引入 generalization 问题。

## 2. Components of a learning problem

### Target Function

`target function` 是我们希望 learner 模仿或服务的未知 mapping。它不是模型，不是参数，也不是数据文件。它代表任务中“正确输出”与输入之间的关系。

在 classification 中：

```math
f(x)\in\{-1,+1\}
```

在 regression 中：

```math
f(x)\in\mathbb{R}
```

在 noisy setting 中，Lecture 4 会把 target 扩展为 probabilistic target 或 target distribution。

### Training Examples

training examples 是 learner 实际看到的 evidence：

```math
(x_i,y_i),\quad i=1,\ldots,N
```

它们通常被假设来自某个 unknown distribution $P$。这个 distribution 在 Lecture 2 中非常关键，因为 out-of-sample error 必须相对于某个 population 或 future-sampling mechanism 才有意义。

### Hypothesis Set

`hypothesis set` $\mathcal{H}$ 是 learner 被允许输出的函数集合。例如 linear classifiers：

```math
\mathcal{H}
=
\{h_w(x)=\mathrm{sign}(w^\top x+b): w\in\mathbb{R}^d,b\in\mathbb{R}\}
```

如果 $\mathcal{H}$ 太小，它可能无法 represent target；如果太大，finite data 可能无法可靠区分好 hypothesis 与偶然拟合 training set 的 hypothesis。这一 tension 是后续 generalization theory 的核心。

### Learning Algorithm

`learning algorithm` $A$ 是从 data 到 hypothesis 的映射。它可以是 perceptron learning algorithm、least squares、gradient descent、regularized ERM、neural-network training pipeline，也可以包含 preprocessing、initialization、early stopping 与 model selection。

形式上：

```math
A:
\left(\mathcal{X}\times\mathcal{Y}\right)^N
\to
\mathcal{H}
```

这强调 algorithm 与 hypothesis 不是同一个对象。linear regression family 是 hypothesis set；normal equation 或 gradient descent 是 algorithm；最后得到的一组 weights 定义 selected hypothesis。

### Final Hypothesis

`final hypothesis` $g$ 是训练后实际用于 prediction 的函数：

```math
g(x)=h_{\hat{\theta}}(x)
```

其中 $\hat{\theta}$ 是 algorithm 从 data 中选出的 realized parameter。写成 $\hat{\theta}$ 是为了提醒：它依赖 finite sample，因此是 random 的；换一个 training set，最终 hypothesis 可能不同。

## 3. Supervised, unsupervised, reinforcement learning

### Caltech Core

Lecture 1 区分了 supervised learning、unsupervised learning 与 reinforcement learning。区别不只是任务名字，而是 learner 得到的信息信号不同。

### Supervised Learning

supervised learning 给出 input-output examples：

```math
D=\{(x_i,y_i)\}_{i=1}^{N}
```

其中 $y_i$ 是针对 $x_i$ 的 label、target value 或 response。learner 的 evidence 是“这个输入应该对应这个输出”。classification、regression、sequence labeling 都属于这种范式。

### Unsupervised Learning

unsupervised learning 通常只观察：

```math
D=\{x_i\}_{i=1}^{N}
```

没有外部给定的 $y_i$。目标不是拟合 target function $f:\mathcal{X}\to\mathcal{Y}$，而是发现 data distribution 中的 structure，例如 clusters、latent factors、density、manifold、embedding 或 compressed representation。

这并不意味着 unsupervised learning 没有目标。它仍然需要 objective 或 criterion，例如 reconstruction error、likelihood、contrastive objective、clustering distortion。差别在于 supervision signal 不再是 direct label，而是来自数据自身结构与建模假设。

### Reinforcement Learning

reinforcement learning 中，learner 与 environment interaction。它观察 state、选择 action，并收到 delayed reward。数据不是预先固定的 iid examples，而是受 policy 影响的 trajectory：

```math
(s_t,a_t,r_t,s_{t+1})
```

信息信号也不同：reward 不告诉 learner 每个 state 下的正确 action，只评价某些 action sequence 的长期后果。因此 RL 的 learning problem 同时包含 exploration、credit assignment、off-policy evaluation 与 distribution shift between policies。

### Information Signal View

三类 learning 的关键区别可以用 evidence type 描述：

| Paradigm | Observed signal | Main inference pressure |
| -------- | --------------- | ----------------------- |
| supervised learning | paired labels or targets | 从 labeled examples 推断 input-output mapping |
| unsupervised learning | inputs only | 从 marginal structure 推断 representation or latent organization |
| reinforcement learning | rewards through interaction | 从 delayed feedback 推断 policy under environment dynamics |

## 4. Unknown target and finite observations

### Problem

为什么 target 不直接可得？因为如果完整 target 已知，learning problem 通常已经消失。真正困难的是：

```math
D \text{ finite},\quad \mathcal{X} \text{ often enormous or continuous}
```

即使 $D$ 中所有 examples 都正确，未见输入仍然无限多或极多。finite observations 无法唯一确定 target function。

### Mathematical Point

设 $\mathcal{X}$ 包含 $M$ 个可能输入，training set 只覆盖 $N<M$ 个点。对于 training set 上一致的任何函数 $h$：

```math
h(x_i)=y_i,\quad i=1,\ldots,N
```

在未观察点 $\mathcal{X}\setminus\{x_1,\ldots,x_N\}$ 上仍可任意改变取值，而不影响 training consistency。若 $\mathcal{X}$ 是连续空间，这种不确定性更大。

### Consequence

因此 learning 必须依赖 data 之外的约束：

- hypothesis set 限制；
- smoothness、linearity、margin、sparsity、low-dimensional structure 等 inductive assumptions；
- loss 与 objective；
- optimization bias；
- data collection assumptions；
- validation protocol。

这不是缺陷，而是 finite learning 的基本条件。

## 5. Learning versus memorization

### Definition

memorization 指 learner 在 training inputs 上记录正确 outputs，但不形成可靠的 out-of-sample rule。一个 extreme memorizer 可以定义为：

```math
h_{\mathrm{mem}}(x)
=
\begin{cases}
y_i, & x=x_i \text{ for some training point } x_i,\\
c, & \text{otherwise}.
\end{cases}
```

其中 $c$ 是任意默认输出。这个 hypothesis 可以让 training error 很低甚至为零：

```math
E_{\mathrm{in}}(h_{\mathrm{mem}})=0
```

但它对未见输入没有结构性保证。

### Generalization Criterion

真正的 learning 至少要求 low out-of-sample error：

```math
E_{\mathrm{out}}(g)
=
\mathbb{E}_{(X,Y)\sim P}
\left[
\ell(g(X),Y)
\right]
```

如果 $E_{\mathrm{in}}$ 低但 $E_{\mathrm{out}}$ 高，说明 learner 主要适应了 training sample 的偶然细节，而非 population-level structure。

### Intuition

memorization 并不总是无用。nearest-neighbor methods、retrieval systems、large-scale language models 都可能存储大量 training information。关键区别在于：系统是否只依赖 exact recall，还是通过 representation、metric、architecture 或 objective 形成了对未见 inputs 有效的 inductive structure。理论上，training fit 只是必要 evidence 的一部分，不是 generalization guarantee。

## 6. Inductive bias

### Caltech Core

Lecture 1 已经暗含 inductive bias：learning algorithm 必须从 many possible hypotheses 中选择一个。finite data 本身无法唯一决定 unseen behavior，因此 learner 的选择规则必然偏向某些 functions。

### Definition

`inductive bias` 是 learner 在 data 不足以唯一决定答案时偏好的假设结构。它可以来自：

- hypothesis set，例如只允许 linear separators；
- regularization，例如偏好 small norm；
- architecture，例如 convolutional structure 偏好 local translation patterns；
- optimizer，例如 gradient descent 的 implicit bias；
- preprocessing，例如 feature scaling 或 image centering；
- objective，例如 cross entropy 与 squared loss 对 errors 的权重不同。

### Formal Consequence

若两个 hypotheses 在 training set 上完全一致：

```math
h_1(x_i)=h_2(x_i),\quad i=1,\ldots,N
```

但在 unseen region 上不同：

```math
h_1(x)\neq h_2(x)
```

仅凭 $D$ 无法判定哪个更接近 target。algorithm 必须有 selection preference：

```math
A(D)=h_1 \quad \text{or} \quad A(D)=h_2
```

这个 preference 就是 inductive bias 的体现。

### Failure Mode

inductive bias 可以帮助 generalization，也可以制造 systematic failure。如果 bias 与 target structure 对齐，有限数据可以被有效放大；如果 bias 与 deployment environment 错位，模型会稳定地学到错误规则。Week 4 real canvas distribution shift 就是一个 concrete example：训练 distribution 的 representation 与真实 canvas 输入之间的差异，让模型在某些实际书写方式上出现高置信错误。

## 7. Connections to existing repository work

### Linear Regression

[Week 2 Linear / Logistic Regression](../../../reports/week2_linear_logistic_regression.md) 中的 scratch linear regression 已经体现了 canonical learning system：

- hypothesis set 是 affine functions；
- algorithm 是 batch gradient descent；
- loss 是 MSE；
- synthetic dataset 暗含 linear target 加 Gaussian noise；
- validation loss 用来初步估计 out-of-sample behavior。

### Logistic Regression

同一 Week 2 note 中 logistic regression 展示了 target interpretation 的变化：模型输出 probability-like score，binary cross entropy 与 Bernoulli likelihood 有自然联系。这为 Lecture 4 的 probabilistic target 与 loss discussion 做准备。

### MLP and Evaluation

[Week 3 Optimization and MLP Notes](../../../reports/week3_optimization_and_mlp.md) 与 [Week 4 Multiclass MLP Capstone](../../../reports/week4_multiclass_digits_capstone.md) 显示，hypothesis set、optimizer 与 representation 一起决定 final hypothesis。Week 5 的 calibration 与 abstention work 则提醒：prediction correctness、confidence reliability、selective prediction 是不同 evaluation questions。

## 8. Research reflection

### Operationalizing a Research Problem

把一个模糊研究想法变成 machine learning problem 时，至少要明确：

1. 输入空间 $\mathcal{X}$ 是什么；
2. 输出空间 $\mathcal{Y}$ 是什么；
3. target 是 deterministic function、conditional distribution，还是 decision rule；
4. observations 如何采样；
5. representation 丢失了什么信息；
6. hypothesis set 或 architecture 限制了什么；
7. loss 是否对应真实成本；
8. evaluation distribution 是否等于 deployment distribution；
9. algorithm 的 optimization failure 与 statistical failure 如何区分。

### Research Lens

很多 ML 论文的争议并不来自公式错误，而是来自 problem formulation 不清楚。例如，某方法可能在 benchmark 上提升 accuracy，但提升来自更强 data augmentation；某模型可能降低 average loss，但增加 rare subgroup risk；某 representation 可能在 iid test 上有效，但依赖 spurious correlation。Lecture 1 的 learning-problem diagram 是审计这些问题的最低层工具。

## 9. Conceptual conclusion

`learn` 的最小含义不是“training error 下降”，而是：

```math
\text{finite examples}
\xrightarrow{A,\mathcal{H},\ell}
g
\quad
\text{such that}
\quad
E_{\mathrm{out}}(g)
\text{ is acceptably small under an explicit distribution.}
```

Lecture 2 将进一步追问：为什么 finite examples 能告诉我们 anything about $E_{\mathrm{out}}$？

[← Back to Learning From Data Theory Notebook](../README.md)
