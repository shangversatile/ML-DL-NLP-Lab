# Generalization Claim Audit for Machine Learning Research

[← Back to Learning From Data Theory Notebook](../README.md)

## Purpose

这份 note 是一个 research reasoning tool。它把 T1/T2 的理论链：

```text
Data
→ Hypothesis Set
→ Learning Algorithm
→ Selected Hypothesis
→ In-sample Error
→ Out-of-sample Error
→ Generalization Conditions
→ Credible Research Claim
```

转化为阅读 ML/AI papers 时可执行的审计框架。它不是 checklist 形式主义；每个问题都对应一个 learning-theory failure mode。

## Source Separation

### Caltech Core

审计框架直接来自 Lecture 5-8 的主线：fixed hypothesis 与 selected hypothesis 的区别、simultaneous control、growth function、VC dimension、sample complexity、bias-variance 与 learning curves。

### Formal Derivation

本 note 本身不新增 central theorem；它复用 T2 theorem consequences，将它们转化为 population、sampling、selection、capacity 与 evaluation-protocol audit questions。

### Stanford / Theory Extension

ERM、excess risk、PAC-style tolerance/confidence、capacity-control vocabulary 来自 Stanford / statistical learning theory extension layer。

### Modern Perspective

现代 research 中的 validation reuse、benchmark overfitting、prompt search、calibration、abstention 与 distribution shift 被作为 evidence-discipline 问题处理。

### Research Lens

整篇 note 是长期 paper-reading checklist：判断一个 empirical improvement 支持多强的 generalization claim。

### What This Does NOT Imply

这份 audit 不把 empirical papers 都降格为无效；它只要求 claim strength 与 evidence strength 匹配。

## 1. What Is the Population of Interest?

任何 generalization claim 都必须指向某个 population、distribution 或 environment。

需要明确：

- claim 是关于 benchmark distribution、deployment users、future time period、new domain，还是某个 simulated environment；
- metric 的 expectation 是 over examples、users、tasks、prompts、random seeds，还是 interventions；
- population 是否与 data collection process 一致。

如果 population 没有定义，reported performance 只能说明 observed dataset 上的 behavior。

### What This Does NOT Imply

定义 population 不保证 sample 代表该 population；它只是让 claim 变得可检验。

## 2. How Were Data Sampled?

检查 sampling mechanism：

- i.i.d.；
- temporal split；
- spatial/geographic clustering；
- user-level clustering；
- adaptive data collection；
- self-selected users；
- filtered benchmark；
- synthetic generation；
- annotation pipeline。

经典 generalization bounds 常依赖 i.i.d. sampling。若样本 clustered 或 time-dependent，effective sample size 可能小于 raw count。若 data 被 model failures 触发收集，sampling distribution 已经 adaptive。

### What This Does NOT Imply

非 i.i.d. 不等于研究无效；但它要求不同的 evidence protocol，例如 group split、time split、cluster-robust uncertainty、domain-specific validation 或 shift analysis。

## 3. What Is the Hypothesis Family?

问清楚 learner 允许输出哪些 functions：

- architecture；
- feature representation；
- pretrained model family；
- prompt/template space；
- decoding or decision policy；
- regularization constraints；
- fine-tuning scope；
- post-processing rules。

在现代系统中，hypothesis family 不只是 “model class”。preprocessing、retrieval system、prompt search、thresholding、abstention policy 都可能改变 selected predictor。

### What This Does NOT Imply

知道 hypothesis family 不等于知道 capacity。parameter count、VC dimension、norm bound、margin、Rademacher complexity、stability 都是不同 complexity lenses。

## 4. How Was the Hypothesis Selected?

selection mechanism 可能是：

- ERM；
- regularized ERM；
- early stopping；
- hyperparameter search；
- architecture search；
- prompt search；
- human-in-the-loop selection；
- benchmark-driven iteration；
- threshold tuning；
- ensemble selection。

理论上最关键的问题是：

```math
g = A(D)
```

selected hypothesis 是否依赖被用来报告 performance 的 data？若是，fixed-test concentration 不再直接适用。

### What This Does NOT Imply

data-dependent selection 并不违法；training 本来就是 selection。问题是：selection class 或 selection protocol 是否被控制？

## 5. Which Data Influenced Selection?

画出 influence map：

```text
training
validation
test
benchmark
researcher feedback
```

逐项问：

- 哪些 data 用于 gradient updates？
- 哪些 data 用于 hyperparameter tuning？
- 哪些 data 用于 early stopping？
- 哪些 benchmark 分数影响了 method design？
- 哪些 examples 被人工查看后改变了 preprocessing 或 prompts？
- final test 是否在所有 design decisions 后才打开？

### Research Logic

data 的角色不是由文件名决定的，而是由它是否影响 selection 决定的。一个叫 “test” 的 set 如果被用于反复调参，在理论上已经承担 validation/selection 角色。

## 6. Is the Evaluation Truly Out of Sample?

区分三种 evidence：

| Evaluation type | Meaning | Supports |
| --- | --- | --- |
| mathematically independent test | selected $g$ fixed before evaluation | risk estimate under same $P$ |
| repeatedly consulted benchmark | evaluation influenced selection over time | weaker benchmark-conditioned evidence |
| shifted deployment data | different environment or sampling process | shift-specific evidence only if sampled properly |

independent test performance 与 distribution shift 是不同问题。前者问 fixed $g$ 在同一 $P$ 下的 risk；后者问 $P_{\mathrm{train}}$ 与 $P_{\mathrm{deploy}}$ 不同时 claim 如何变化。

### What This Does NOT Imply

out-of-sample 不等于 out-of-distribution。held-out i.i.d. test set 不能自动证明 robustness under arbitrary shift。

## 7. What Capacity-Control Argument Exists?

可能的 control mechanisms：

- finite hypothesis class；
- VC dimension；
- growth function；
- Rademacher complexity；
- norm bound；
- margin；
- explicit regularization；
- algorithmic stability；
- compression；
- validation protocol；
- independent replication；
- purely empirical evidence。

如果论文没有 formal bound，也可以是有效 empirical research，但 claim strength 应相应降低。经验提升支持 observed benchmark improvement；它不自动支持 high-probability guarantee。

### What This Does NOT Imply

capacity control 只处理 estimation/generalization 的一部分。它不保证 representation sufficient、loss aligned、optimization successful 或 deployment safe。

## 8. What Kind of Claim Is Supported?

把 claim 分类：

- **empirical observation**：在某些 datasets/runs 上 metric improves；
- **confidence interval**：对 finite evaluation sample 的 uncertainty estimate；
- **expected-risk claim**：关于 population expectation；
- **high-probability guarantee**：以 $1-\delta$ 概率控制 deviation；
- **worst-case guarantee**：对 class/distribution family 的统一保证；
- **robustness claim**：对 shift、perturbation 或 environment changes 的声明；
- **mechanistic claim**：关于 model 使用了何种 causal/semantic structure；
- **decision claim**：关于 downstream utility/cost/safety。

不同 claim 需要不同 evidence。test accuracy 不能直接证明 calibration、fairness、mechanistic understanding 或 arbitrary robustness。

### What This Does NOT Imply

一个 claim 较弱不等于没价值。严谨 research 的关键是 claim strength 与 evidence strength 匹配。

## 9. What Assumptions Make the Claim Valid?

建立 assumption ledger：

| Assumption | Why it matters | Evidence needed |
| --- | --- | --- |
| i.i.d. sampling | concentration and risk estimates | sampling design |
| same train/test distribution | target $R(h)$ unchanged | split protocol, shift checks |
| independent test | fixed-h evaluation valid | test isolation |
| bounded or controlled loss | concentration applies | metric definition |
| hypothesis class fixed or controlled | selected $g$ covered | search space documentation |
| optimization reaches intended solution | algorithm output matches theory | training diagnostics |
| labels match target | loss estimates intended error | annotation audit |
| evaluation metric aligned with cost | metric supports decision claim | domain cost analysis |

每个 assumption 都可能成为 failure condition。

## 10. What Does the Evidence Fail to Establish?

常见 non-implications：

- causal mechanism；
- robustness under arbitrary distribution shift；
- calibration；
- fairness across subpopulations；
- interpretability；
- deployment safety；
- resistance to adaptive attacks；
- reliability under feedback loops；
- correctness outside measured tasks；
- absence of shortcut learning。

这些不是附加哲学问题，而是不同 target quantities。若论文没测，就不应暗示。

## 11. Research Credibility Checklist

对每篇论文或项目报告，最后做一次 concise audit：

1. **Population**：claim 指向哪个 distribution or environment？
2. **Sampling**：data 如何采样，i.i.d. assumption 是否合理？
3. **Representation**：model 观察到的信息是否足以支持 target？
4. **Hypothesis family**：哪些 functions/policies 可能被选中？
5. **Selection path**：training、validation、test、benchmark 哪些影响了 $g$？
6. **Capacity control**：有什么 formal 或 empirical control 防止 data-dependent overclaim？
7. **Risk quantity**：报告的是 empirical risk、population risk estimate、generalization gap、excess risk 还是 selective risk？
8. **Independence**：final evaluation 是否独立于 model/researcher decisions？
9. **Distribution match**：train/test/deployment 是否同分布，若不同有何证据？
10. **Uncertainty**：是否报告 confidence interval、seed variance、split variance 或 bound？
11. **Non-implications**：哪些常见强结论没有被 evidence 支持？

### Logic Behind the Checklist

这些检查项对应 T1/T2 的理论链。Population 与 sampling 定义 $R(h)$；representation 与 hypothesis family 定义可学对象；selection path 决定 fixed-h theorem 是否适用；capacity control 决定 selected $g$ 是否能被 generalization theory 覆盖；risk quantity 防止把 generalization gap、excess risk、calibration 和 robustness 混为一谈；non-implications 防止从单一 metric 推出过强 research claim。

## Cross-links to Existing Repository Evidence

- [Week 3: empirical risk and overfitting](../../../reports/week3/02_gradient_risk_and_sampling.md) 是 low training loss 不等于 low population risk 的本地例子。
- [Week 4: shift and confidence diagnostics](../../../reports/week4/07_shift_and_confidence_diagnostics.md) 展示同一 model 在 configured distribution changes 下 behavior 改变。
- [Week 4: Canvas-Diagnostic-v1 protocol](../../../reports/week4/13_canvas_dataset_protocol_and_next_stage_experiment_design.md) 明确要求 diagnostic、train、validation、test 角色分离。
- [Week 5: calibration](../../../reports/week5/02_calibration_metrics_and_reliability_diagrams.md) 说明 accuracy generalization 不等于 calibrated probabilities。
- [Week 5: abstention](../../../reports/week5/03_confidence_thresholding_and_abstention_policy.md) 说明 selective prediction 是 decision-policy change，不是 representation failure 的自动修复。
