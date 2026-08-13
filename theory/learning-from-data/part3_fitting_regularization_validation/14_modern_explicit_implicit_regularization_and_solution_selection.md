# Explicit Regularization, Implicit Bias, and Solution Selection

[← Back to Learning From Data Theory Notebook](../README.md)

本章是 T3 的 Stanford CS229M / modern theory bridge。Caltech Lecture 12 讲清楚 explicit regularization；现代 deep learning 进一步迫使我们问：即使没有显式 penalty，optimizer、initialization、trajectory 与 parameterization 是否也会影响 selected solution？

本章不把 T3 扩展成完整 modern deep-learning theory，只建立研究阅读所需的边界：architecture alone does not define the effective learner。

## 0. Source Separation

### Caltech Core

Caltech Core 来自 Lecture 12：hard constraints、soft penalties、augmented error 与 weight decay。它提供 classical regularization picture。

### Stanford CS229M / Theory Extension

CS229M 关注 overparameterization、non-convex optimization、implicit / algorithmic regularization 与 theory of deep learning。本章使用这些主题作为扩展视角，解释为什么只看 $\mathcal{H}$ 的 worst-case capacity 不足以描述现代 learner。

### Modern Research

separable logistic regression 的 implicit bias、Neural Tangent Kernel、double descent、benign overfitting 与 adaptive evaluation 都来自现代文献。本章只给概念性、受限表述；精确定理必须回到原论文假设。

## 1. Classical Regularization Picture

classical picture 可以压缩为：

```text
large family
+ explicit constraint / penalty
→ controlled solution
```

hard constraint：

```math
\min_w \hat R_D(w)
\quad
\text{subject to}
\quad
\Omega(w)\le C
```

soft penalty：

```math
\min_w
\hat R_D(w)+\lambda\Omega(w)
```

这条路线回答的问题是：当很多 hypotheses 都能 fit training data 时，如何用 explicit preference 排除或惩罚不想要的 solutions？

## 2. Optimization Can Itself Induce Bias

现代问题是：即使 objective 中没有 explicit penalty，optimizer 也可能偏好某些 solutions。

概念结构：

```text
same nominal hypothesis family
same empirical objective
different initialization / optimizer / batch order / stopping rule
→ potentially different selected solutions
```

这类现象常被称为 algorithmic bias 或 implicit regularization。但应谨慎使用：并非任何 optimizer effect 都已被形式化为 regularization theorem。

### Object Separation

一个 complete learner 更像：

```math
\tilde h
=
A(D;\mathcal{H},\Theta,\hat R,\theta_0,\mathcal{O},T)
```

其中：

- $\mathcal{H}$ 是 represented function family；
- $\Theta$ 是 parameter space；
- $\hat R$ 是 empirical objective；
- $\theta_0$ 是 initialization；
- $\mathcal{O}$ 是 optimizer；
- $T$ 是 stopping rule / training horizon；
- $A$ 输出 actual hypothesis $\tilde h$。

因此 generalization analysis 可以从 class-level property 转向 procedure-level property。

## 3. Separable Logistic Regression Example

### Setup

对 binary linearly separable data，unregularized logistic regression 的 finite MLE 可能不存在。训练 loss 可以沿着 separating direction 继续下降，weight norm 继续增大。

### Modern Result, Carefully Stated

现代 implicit-bias 文献表明，在特定假设下，gradient descent 对 separable logistic-type losses 的方向可能收敛到 max-margin separator。这里的关键词是“方向”：norm 可能发散，但 normalized direction 有限制。

### What This Does NOT Imply

这不意味着：

- 所有 optimizers 都选择 max-margin solution；
- 所有 losses 都有同样 implicit bias；
- 所有 neural networks 的 generalization 都由这个现象解释；
- finite-time training 等同于 asymptotic theorem；
- max-margin direction 一定是 deployment-optimal。

### Research Use

这个例子说明：当 explicit regularizer 缺席时，training trajectory 仍可能有 solution preference。研究者必须报告 optimizer、learning rate、stopping rule 与 initialization，而不是只报告 model family。

## 4. Neural Networks

overparameterized neural networks 常有大量 interpolating solutions：

```math
\hat R_D(h_\theta)=0
```

或接近 0。此时只问：

```text
How large is H?
```

可能太粗。更有解释力的问题是：

```text
Which interpolating solution is selected by the training procedure?
```

selected solution 可能由以下因素影响：

- initialization scale；
- optimizer；
- batch noise；
- normalization layers；
- architecture symmetries；
- data augmentation；
- implicit norm or margin effects；
- early stopping；
- loss shape。

这不取消 T2 的 capacity theory，而是把关注点从 $\mathcal{H}$ alone 扩展到 $(\mathcal{H},A,D)$。

## 5. Explicit versus Implicit Regularization

| Mechanism | Where preference enters | Careful wording |
| --------- | ----------------------- | --------------- |
| explicit penalty | objective | formal regularizer when $\Omega$ is specified |
| hard constraint | feasible region | restricts effective feasible set under parameterization |
| early stopping | optimization trajectory | inductive influence; implicit regularization when theory supports it |
| initialization | starting state | affects reachable trajectory and symmetry breaking |
| optimizer | trajectory / solution selection | algorithmic bias; not automatically a generalization proof |
| data augmentation | empirical objective / invariance structure | changes training distribution and desired invariances |

### Boundary

不要把所有对 solution 有影响的因素都随意称为 regularizer。更稳健的词是：

```text
inductive influence
algorithmic bias
solution-selection mechanism
```

只有当有明确数学关系时，再使用 implicit regularization 的强说法。

## 6. Non-Convex Optimization

non-convexity 使下面对象相互纠缠：

```text
parameterization
optimization landscape
initialization
algorithm
selected solution
```

但不应把 modern generalization 简化为“local minima”。很多 overparameterized settings 中，训练可以找到 low empirical risk；真正的问题转向 low-risk solutions 中哪一类被 selected，以及它们为什么 generalize。

### What This Does NOT Imply

- non-convex objective 不等于不可训练；
- 找到 low training loss 不等于 generalization；
- local-minimum story 不足以解释 representation、margin、norm、stability 与 data geometry；
- optimizer-specific behavior 需要 empirical evidence 或 theorem，而不是直觉宣称。

## 7. NTK as a Theory Bridge

Neural Tangent Kernel (NTK) theory 研究一种 regime：在特定宽度、initialization 与 training 条件下，neural-network training dynamics 可近似为围绕 initialization 的线性化模型，并与 kernel-like dynamics 联系。

它帮助隔离一个问题：

```text
When can a nonlinear neural network be analyzed through a tractable kernel-like training dynamics?
```

### Boundary

NTK 不是所有 neural-network generalization 的完整解释。它常研究特定 infinite-width 或 lazy-training regimes；feature learning 强、finite-width effects 显著、optimizer/architecture 更复杂时，需要其他理论工具。

## 8. Why This Matters for Research Credibility

T2 的 classical chain 是：

```text
capacity of H
→ uniform control
→ selected hypothesis generalization
```

T3 将其扩展为：

```text
representation
+ objective
+ optimizer
+ regularization
+ validation selection
→ selected solution
→ credible evaluation
```

因此，当论文声称“architecture X generalizes”，研究者应追问：

- 是 architecture 本身，还是 optimizer/regularizer/augmentation/early stopping 共同产生结果？
- 同一个 architecture 换 optimizer 是否仍成立？
- selected checkpoint 是否由 validation feedback 决定？
- reported gain 是否来自 model class，还是 outer model-selection loop？
- theory claim 控制的是 $\mathcal{H}$、solution norm、margin、stability，还是 full algorithm？

[← Back to Learning From Data Theory Notebook](../README.md)
