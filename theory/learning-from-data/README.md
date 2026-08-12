# Learning From Data 理论笔记总览

这个目录是 `ML-DL-NLP-Lab` 的长期 Machine Learning Theory knowledge base。它不是 Caltech `Learning From Data` 的逐字转录，也不是考试速查表；它的目标是把课程的 logic spine 转化为可复读、可引用、可扩展的研究型理论笔记。

## 目的

这套笔记研究一个核心问题：learner 如何从有限 observations 中得到一个在 unseen samples 上仍然可靠的 prediction 或 decision。

T1 先建立 learning problem 的基础 ontology。T2 建立 generalization theory 的核心推理链。后续 T3-T5 会继续展开 regularization、validation、kernels、SVM，以及 modern ML theory bridges。

本目录长期关注：

- 什么叫从 finite data 中 learn；
- 为什么 generalization 在特定 assumptions 下可能；
- representation 与 hypothesis set 如何共同定义 learning problem；
- probability、optimization、data、loss 与 inductive assumptions 如何共同决定学习结果；
- 这些概念如何支撑后续 ML、DL、NLP、trustworthy ML、representation learning 与 research methodology。

## 知识来源

**Caltech Core**：Yaser Abu-Mostafa 的 Caltech CS/CNS/EE 156 / `Learning From Data` 提供主线。T1 对应 Lecture 1-4：learning problem、learning feasibility、linear models、error and noise。T2 对应 Lecture 5-8：training versus testing、theory of generalization、VC dimension、bias-variance。

**Stanford / Theory Extension**：Stanford STATS214 / CS229M `Machine Learning Theory` 提供更深的 statistical-learning-theory 视角，尤其是 uniform convergence、generalization bounds、ERM、hypothesis complexity 与现代 learning theory 的研究问题。

**Modern Perspective**：现代研究解释与工程联系会被显式标出，不与原课程内容混在一起。现代部分用于连接 representation learning、distribution shift、calibration、abstention、deep learning 与 evaluation methodology。

## 五阶段路线图

T1：learning problem、feasibility、representation、error/noise。

T2：training/testing、generalization theory、VC dimension、bias-variance、research claim discipline。

T3：logistic regression、neural networks、overfitting、regularization、validation。

T4：SVM、kernels、RBF、learning principles、epilogue。

T5：modern theory bridges 与 research synthesis。

## 阅读方法

每章尽量遵循同一条阅读路径：

1. problem；
2. formal setup；
3. derivation；
4. intuition；
5. assumptions；
6. failure modes；
7. modern extension；
8. research reflection。

读者不应只记公式。更重要的是追问：公式回答什么问题、依赖什么 assumptions、assumptions 失效时会发生什么、它允许推出多强的 generalization claim，以及它如何改变我们设计实验和解释论文结论的方式。

## T1 笔记

- [理论地图：world、data、hypothesis、generalization 与 research lens](00_learning_theory_ontology_world_data_generalization_research_lens.md)
- [Lecture 1：learning problem、target、hypothesis、inductive bias](part1_learning_problem/01_caltech_l01_learning_problem_target_hypothesis_inductive_bias.md)
- [Lecture 2：finite-sample generalization、Hoeffding、uniform convergence](part1_learning_problem/02_caltech_l02_finite_sample_generalization_hoeffding_uniform_convergence.md)
- [Lecture 3：hypothesis spaces、linear models、feature transforms](part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md)
- [Lecture 4：error measures、noise、target distribution](part1_learning_problem/04_caltech_l04_error_measures_noise_target_distribution.md)

## T2 笔记

T2 建立核心 generalization-theory chain：

```text
fixed hypothesis
→ selected hypothesis
→ simultaneous control
→ growth function
→ VC dimension
→ sample complexity
→ bias/variance
→ research claim discipline
```

- [Lecture 5：training/testing、fixed evaluation 与 model selection](part2_generalization_theory/05_caltech_l05_training_testing_and_model_selection.md)
- [Lecture 6：growth function、break point 与 uniform control](part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md)
- [Lecture 7：VC dimension、capacity 与 sample complexity](part2_generalization_theory/07_caltech_l07_vc_dimension_capacity_and_sample_complexity.md)
- [Lecture 8：bias-variance、learning curves 与 error sources](part2_generalization_theory/08_caltech_l08_bias_variance_learning_curves.md)
- [Modern extension：uniform convergence 与 modern capacity control](part2_generalization_theory/09_modern_uniform_convergence_and_capacity_control.md)
- [Research tool：generalization claim audit for ML research](part2_generalization_theory/10_generalization_claim_audit_for_ml_research.md)

## Sources and notation

- [Source traceability：Caltech、Stanford 与 modern theory map](sources/t1_source_traceability_caltech_stanford_map.md)
- [Terminology and notation：learning problem、generalization theory 与 research audit](sources/t1_terminology_notation_learning_problem_generalization.md)
