# Source Map

[← Back to Learning From Data Theory Notebook](../README.md)

本文件记录 `theory/learning-from-data/` notes 的 source traceability。它不声称逐页复现课程内容；它说明每个 note 的 Caltech Core、Stanford / Theory Extension、Modern Perspective 与 Research Reflection 来自何处或如何扩展。

## 1. Sources verified for T1

### Caltech primary sources

- Official Learning From Data lecture page: <https://home.work.caltech.edu/lectures.html>
- Official AML slides page: <https://amlbook.com/slides.html>
- Lecture 1 slides: <https://work.caltech.edu/slides/slides01.pdf>
- Lecture 2 slides: <https://work.caltech.edu/slides/slides02.pdf>
- Lecture 3 slides: <https://work.caltech.edu/slides/slides03.pdf>
- Lecture 4 slides: <https://work.caltech.edu/slides/slides04.pdf>
- Official Machine Learning Video Library: <https://home.work.caltech.edu/library/>
- Official supporting-material page: <https://amlbook.com/support.html>
- Official slides page: <https://amlbook.com/slides.html>

### Stanford / theory extension sources

- Official STATS214 / CS229M Machine Learning Theory page: <https://web.stanford.edu/class/stats214/>
- Official page-linked public notes repository: <https://github.com/tengyuma/cs229m_notes/blob/main/master.pdf>

The Stanford page explicitly frames the course around why ML algorithms work, formalizing learning from data, statistical properties of learning algorithms, and topics including generalization bounds via uniform convergence. T1 uses Stanford material only as a conceptual extension layer; detailed Stanford-style proofs are deferred to T2.

## 2. Sources verified for T2

### Caltech primary sources

- Official Learning From Data lecture page: <https://home.work.caltech.edu/lectures.html>
- Lecture 5 slides: <https://work.caltech.edu/slides/slides05.pdf>
- Lecture 6 slides: <https://work.caltech.edu/slides/slides06.pdf>
- Lecture 7 slides: <https://work.caltech.edu/slides/slides07.pdf>
- Lecture 8 slides: <https://work.caltech.edu/slides/slides08.pdf>
- Official AML slides page: <https://amlbook.com/slides.html>
- Official supporting-material page: <https://amlbook.com/support.html>

### Stanford / theory extension sources

- Official STATS214 / CS229M Machine Learning Theory page: <https://web.stanford.edu/class/stats214/>
- Official page-linked public notes repository: <https://github.com/tengyuma/cs229m_notes/blob/main/master.pdf>

The Stanford material is used as an extension layer for ERM, uniform convergence, sample complexity, capacity control, and modern statistical-learning-theory vocabulary. The T2 narrative remains organized by the Caltech Lecture 5-8 sequence.

### Modern-theory clarification sources

- Zhang, Bengio, Hardt, Recht, and Vinyals, `Understanding deep learning requires rethinking generalization`: <https://arxiv.org/abs/1611.03530>
- Belkin, Hsu, Ma, and Mandal, `Reconciling modern machine-learning practice and the classical bias-variance trade-off`: <https://arxiv.org/abs/1812.11118>
- Nakkiran et al., `Deep Double Descent`: <https://arxiv.org/abs/1912.02292>

These sources are used only for the labeled Modern Perspective sections on overparameterization, interpolation, double descent, and the limits of worst-case classical capacity bounds. They are not treated as Caltech Core.

## 3. Sources verified for T3

### Caltech primary sources

- Official Learning From Data lecture page: <https://home.work.caltech.edu/lectures.html>
- Lecture 9 slides: <https://work.caltech.edu/slides/slides09.pdf>
- Lecture 10 slides: <https://work.caltech.edu/slides/slides10.pdf>
- Lecture 11 slides: <https://work.caltech.edu/slides/slides11.pdf>
- Lecture 12 slides: <https://work.caltech.edu/slides/slides12.pdf>
- Lecture 13 slides: <https://work.caltech.edu/slides/slides13.pdf>
- Official AML slides page: <https://amlbook.com/slides.html>
- Official supporting-material page: <https://amlbook.com/support.html>

### Stanford CS229 derivational / methodological sources

- Official Stanford CS229 course page and materials index: <https://cs229.stanford.edu/>
- Official CS229 materials page: <https://cs229.stanford.edu/materials.html-withcomments>

CS229 is used for derivational and methodological support: logistic-regression likelihood, cross entropy, gradients, Hessian/convexity, neural-network forward/backward propagation, regularization, train/dev/test methodology, model selection, and cross-validation. These are not labeled CS229M.

### Stanford CS229M / theory extension sources

- Official STATS214 / CS229M Machine Learning Theory page: <https://web.stanford.edu/class/stats214/>
- Official page-linked public notes repository: <https://github.com/tengyuma/cs229m_notes/blob/main/master.pdf>

CS229M is used as a theory extension for non-convex optimization, overparameterization, implicit / algorithmic regularization, algorithm-dependent generalization, NTK as a theory bridge, and limits of purely classical explicit-capacity explanations.

### Modern-theory clarification sources

- Soudry, Hoffer, Nacson, Gunasekar, and Srebro, `The Implicit Bias of Gradient Descent on Separable Data`: <https://arxiv.org/abs/1710.10345>
- Jacot, Gabriel, and Hongler, `Neural Tangent Kernel: Convergence and Generalization in Neural Networks`: <https://arxiv.org/abs/1806.07572>
- Loshchilov and Hutter, `Decoupled Weight Decay Regularization`: <https://arxiv.org/abs/1711.05101>
- Dwork et al., `The Reusable Holdout: Preserving Validity in Adaptive Data Analysis`: <https://arxiv.org/abs/1507.02629>
- Dwork et al., `Preserving Statistical Validity in Adaptive Data Analysis`: <https://arxiv.org/abs/1411.2664>
- Zhang, Bengio, Hardt, Recht, and Vinyals, `Understanding deep learning requires rethinking generalization`: <https://arxiv.org/abs/1611.03530>
- Belkin, Hsu, Ma, and Mandal, `Reconciling modern machine-learning practice and the classical bias-variance trade-off`: <https://arxiv.org/abs/1812.11118>
- Nakkiran et al., `Deep Double Descent`: <https://arxiv.org/abs/1912.02292>

These sources are used only for labeled Modern Perspective sections and conceptual caveats. T3 does not turn implicit bias, NTK, double descent, benign overfitting, or adaptive-data-analysis methods into fully derived chapters.

## 4. Sources verified for T4

### Caltech primary sources

- Official Learning From Data lecture page: <https://home.work.caltech.edu/lectures.html>
- Official AML slides page: <https://amlbook.com/slides.html>
- Lecture 14 slides: <https://work.caltech.edu/slides/slides14.pdf>
- Lecture 15 slides: <https://work.caltech.edu/slides/slides15.pdf>
- Lecture 16 slides: <https://work.caltech.edu/slides/slides16.pdf>
- Lecture 17 slides: <https://work.caltech.edu/slides/slides17.pdf>
- Lecture 18 slides: <https://work.caltech.edu/slides/slides18.pdf>
- Official Machine Learning Video Library: <https://home.work.caltech.edu/library/>
- Official supporting-material page: <https://amlbook.com/support.html>

### Stanford CS229 derivational sources

- Official Stanford CS229 course page: <https://cs229.stanford.edu/>
- Official Stanford SEE CS229 materials page: <https://see.stanford.edu/Course/CS229>
- Stanford SEE CS229 SVM notes: <https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf>
- Official CS229 syllabus/materials index: <https://cs229.stanford.edu/syllabus-new.html>

CS229 is used for functional margin, geometric margin, optimal-margin classifier, primal/dual SVM, Lagrange duality, KKT conditions, support vectors, kernels, and soft-margin SVM. It is not labeled as Caltech content.

### Stanford CS229M / theory extension sources

- Official STATS214 / CS229M Machine Learning Theory page: <https://web.stanford.edu/class/stats214/>
- Official page-linked public notes repository: <https://github.com/tengyuma/cs229m_notes/blob/main/master.pdf>

CS229M is used only as a light theory bridge for algorithm-dependent complexity, norm/margin control, and the distinction between ambient representation dimension and effective statistical complexity.

### Modern primary / canonical sources used sparingly

- Cortes and Vapnik, `Support-Vector Networks`: <https://doi.org/10.1007/BF00994018>
- Bickel, Bruckner, and Scheffer, `Discriminative Learning Under Covariate Shift`: <https://www.jmlr.org/papers/v10/bickel09a.html>
- Breiman, `Bagging Predictors`: <https://doi.org/10.1007/BF00058655>
- Freund and Schapire, `A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting`: <https://doi.org/10.1006/jcss.1997.1504>
- Dwork et al., `The Reusable Holdout: Preserving Validity in Adaptive Data Analysis`: <https://arxiv.org/abs/1507.02629>
- Dwork et al., `Preserving Statistical Validity in Adaptive Data Analysis`: <https://arxiv.org/abs/1411.2664>

These sources support labeled modern or canonical clarification only. T4 does not become full RKHS theory, PAC-Bayes, Gaussian processes, boosting theory, NTK theory, or deep margin theory.

## 5. Sources verified for T5

### Stanford CS229M / primary course spine

- Official STATS214 / CS229M Machine Learning Theory page: <https://web.stanford.edu/class/stats214/>
- Official page-linked public notes repository: <https://github.com/tengyuma/cs229m_notes/blob/main/master.pdf>

T5 uses Stanford STATS214 / CS229M as a topic spine for modern generalization, deep-learning theory, non-convex optimization, implicit / algorithmic regularization, NTK-style regimes, and domain adaptation. T5 does not reproduce the course; it organizes repository-level synthesis around source-scoped primary papers.

### Primary modern theory papers and scope

| Source | Topic/theorem used | Mathematical regime | Notes using it | Directly sourced | Repository synthesis / limitation |
| --- | --- | --- | --- | --- | --- |
| Bousquet and Elisseeff, `Stability and Generalization`: <https://www.jmlr.org/papers/v2/bousquet02a.html> | uniform stability definition and stability-generalization theorem | bounded loss, i.i.d. source sampling, abstract learning algorithms | 22, 27, 28 | stability convention, high-probability bound structure, stability intuition | Used only for training-sample perturbation sensitivity; not adversarial or distributional robustness |
| Hardt, Recht, and Singer, `Train faster, generalize better`: <https://proceedings.mlr.press/v48/hardt16.html> | SGD stability dependence on learning rates, steps, smoothness | convex/smooth and nonconvex/smooth SGD settings under theorem assumptions | 22, 27, 28 | trajectory-stability framing and convex smooth bound scaling | Does not imply SGD always generalizes or that every neural-network training run is stable |
| Bartlett and Mendelson, `Rademacher and Gaussian Complexities`: <https://www.jmlr.org/papers/v3/bartlett02a.html> | Rademacher complexity and risk bounds | bounded/Lipschitz function or loss classes under i.i.d. source sampling | 23, 27, 28 | empirical complexity definition, symmetrization lens, source-generalization bound family | Constants depend on convention; not a distribution-shift theorem |
| Bartlett, Foster, and Telgarsky, `Spectrally-normalized margin bounds for neural networks`: <https://arxiv.org/abs/1706.08498> | neural-network margin/norm structural message | specified feedforward networks, margin loss, spectral/Frobenius norm quantities | 23, 27, 28 | margin/norm dependence beyond parameter count | Used as scoped example; not a claim that all trained networks have small non-vacuous bounds |
| Zhang et al., `Understanding deep learning requires rethinking generalization`: <https://arxiv.org/abs/1611.03530> | random-label experiment and modern generalization puzzle | empirical deep-network training experiments | 21, 24, 27, 28 | distinction between expressive/memorization capacity and generalization explanation | Does not prove VC theory false or regularization/data structure irrelevant |
| Belkin et al., `Reconciling modern machine-learning practice and the classical bias-variance trade-off`: <https://arxiv.org/abs/1812.11118> | interpolation/double-descent motivation | modern interpolation regimes across model families | 24, 27, 28 | conceptual role of interpolation threshold and bias-variance revision | Not used as universal law that larger models always improve |
| Nakkiran et al., `Deep Double Descent`: <https://arxiv.org/abs/1912.02292> | model-wise, sample-wise, and epoch-wise double descent phenomena | empirical and conceptual deep-learning settings | 24, 27, 28 | double-descent taxonomy and caution around interpolation threshold | Not treated as theorem for every architecture or dataset |
| Bartlett, Long, Lugosi, and Tsigler, `Benign Overfitting in Linear Regression`: <https://www.pnas.org/doi/10.1073/pnas.1907378117> | benign overfitting in minimum-norm linear regression | high-dimensional linear regression with covariance/noise spectral assumptions | 24, 27, 28 | minimum-norm interpolation can have small population risk under conditions | Used only for linear-regression regime; not generalized directly to arbitrary deep networks |
| Soudry et al., `The Implicit Bias of Gradient Descent on Separable Data`: <https://www.jmlr.org/papers/v19/18-188.html> | directional convergence to max-margin separator | separable linear classification, logistic/exponential-tail losses, gradient descent | 25, 27, 28 | $w_t/\|w_t\|\to w_{\mathrm{SVM}}/\|w_{\mathrm{SVM}}\|$ | Used only for separable linear logistic-regression directional implicit-bias result |
| Jacot, Gabriel, and Hongler, `Neural Tangent Kernel`: <https://arxiv.org/abs/1806.07572> | NTK definition and kernel-like dynamics | infinite-width / appropriate scaling regimes, tangent-kernel analysis | 25, 27, 28 | tangent feature kernel, $K=JJ^\top$, function-space dynamics | NTK is a regime, not all neural-network training or feature learning |
| Ben-David et al., `A Theory of Learning from Different Domains`: <https://link.springer.com/article/10.1007/s10994-009-5152-4> | $\mathcal H\Delta\mathcal H$ divergence and target-risk bound | binary classification domain adaptation | 26, 27, 28 | source risk + domain discrepancy + joint error bound structure | Does not solve arbitrary mechanism shift or calibration |
| Zhao et al., `On Learning Invariant Representations for Domain Adaptation`: <https://proceedings.mlr.press/v97/zhao19a.html> | limits of domain-invariant representation | domain adaptation / representation alignment settings | 26, 27, 28 | invariance can conflict with joint predictive performance under conditions | Does not say invariance is always harmful; says invariance alone is insufficient |

## 6. T1 note mapping

| Note | Primary Caltech source | Stanford extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| [00 Theory Map](../00_learning_theory_ontology_world_data_generalization_research_lens.md) | Lecture 1-4 page topics; video-library topics on Learning Diagram, Bin Model, Error Measures, Noisy Targets, Nonlinear Transformation | STATS214 / CS229M official description and uniform-convergence topic | learning diagram ontology; finite data; input distribution; hypothesis set; error/noise; nonlinear transformation | world-to-representation ontology; refined four-way failure taxonomy; research-paper audit questions; links to repo evaluation methodology |
| [01 What Does It Mean to Learn?](../part1_learning_problem/01_caltech_l01_learning_problem_target_hypothesis_inductive_bias.md) | Lecture 1: The Learning Problem; Lecture 1 slides | STATS214 framing of formalizing learning from data | target function, training examples, learning algorithm, hypothesis set, final hypothesis; supervised/unsupervised/reinforcement distinction | inductive bias framing; memorization distinction; operationalizing research problems; links to scratch linear/logistic/MLP work |
| [02 From Finite Data to Generalization](../part1_learning_problem/02_caltech_l02_finite_sample_generalization_hoeffding_uniform_convergence.md) | Lecture 2: Is Learning Feasible; Lecture 2 slides; video-library Bin Model segments | STATS214 uniform-convergence topic and linked notes repository | in-sample/out-of-sample error; Hoeffding-style concentration; coin/bin analogy; fixed hypothesis vs selected hypothesis | union-bound preview; uniform convergence as selected-hypothesis control; validation/data-snooping research interpretation |
| [03 Hypothesis Spaces and Linear Models](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md) | Lecture 3: The Linear Model I; Lecture 3 slides; video-library Linear Classification, Linear Regression, Nonlinear Transformation topics | STATS214 general lens on hypothesis classes and statistical learning | linear classification geometry; linear regression; nonlinear transforms; linear-in-parameters but nonlinear-in-input distinction | feature maps as induced effective hypothesis families; separation of information loss and specification error; continuity to neural representation learning; links to Week 2/3/4 implementation notes |
| [04 Error, Noise, and Target Distribution](../part1_learning_problem/04_caltech_l04_error_measures_noise_target_distribution.md) | Lecture 4: Error and Noise; Lecture 4 slides; video-library Error Measures and Noisy Targets topics | STATS214 statistical-learning framing of population risk and loss | error measure as part of learning problem; noisy targets; target distribution | conditional distribution interpretation; Bayes classifier and conditional mean derivations; loss-likelihood connection; calibration/abstention/distribution-shift implications |
| [Terminology and Notation](t1_terminology_notation_learning_problem_generalization.md) | Lecture 1-4 notation conventions | STATS214 population/empirical risk vocabulary | symbols for target, hypothesis, algorithm, dataset, in/out error | explicit distinctions among parameter/function, representation/hypothesis, empirical/population, random/realized |

## 7. T2 note mapping

| Note | Primary Caltech source | Stanford extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| [05 Training versus Testing](../part2_generalization_theory/05_caltech_l05_training_testing_and_model_selection.md) | Lecture 5: Training versus Testing; Lecture 5 slides | STATS214 ERM/generalization framing | distinction between training and testing; fixed hypothesis testing; training selection dependence | finite-class union-bound derivation; validation reuse and leaderboard-overfitting methodology; links to Week 3-5 repo evidence |
| [06 Theory of Generalization](../part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md) | Lecture 6: Theory of Generalization; Lecture 6 slides | STATS214 uniform convergence and sample-induced complexity | dichotomies; growth function; break point; finite dichotomy control | Sauer-type proof skeleton; uniform convergence supremum; ERM `2 epsilon` consequence; pointwise/uniform research-reading table |
| [07 VC Dimension](../part2_generalization_theory/07_caltech_l07_vc_dimension_capacity_and_sample_complexity.md) | Lecture 7: The VC Dimension; Lecture 7 slides | STATS214 capacity/sample-complexity framing | shattering; VC dimension; VC-style generalization bound; sample complexity | capacity vs parameter count; modern overparameterization caveat; margin/norm/stability preview |
| [08 Bias-Variance](../part2_generalization_theory/08_caltech_l08_bias_variance_learning_curves.md) | Lecture 8: Bias-Variance Tradeoff; Lecture 8 slides | STATS214-style excess-risk and estimator decomposition vocabulary | dataset-dependent hypothesis; bias; variance; learning curves | squared-loss derivation with stochastic noise term; approximation-estimation-optimization distinction; double descent preview; repo experiment links |
| [09 Modern Uniform Convergence](../part2_generalization_theory/09_modern_uniform_convergence_and_capacity_control.md) | Lecture 5-7 concepts synthesized | STATS214 / CS229M official page and linked notes | fixed-to-uniform logic; ERM consequence | excess-risk decomposition variants; PAC vocabulary; beyond-VC capacity preview; limits of worst-case class control |
| [10 Generalization Claim Audit](../part2_generalization_theory/10_generalization_claim_audit_for_ml_research.md) | Lecture 5-8 generalization chain | STATS214 research-level generalization vocabulary | training/testing distinction; capacity control; sample complexity; bias-variance caveats | original research audit framework for population, sampling, selection, evidence, assumptions, and non-implications |

## 8. T3 note mapping

| Note | Primary Caltech source | Stanford / modern extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ---------------------------------- | -------------------- | ---------------------------------------- |
| [09 Logistic Regression](../part3_fitting_regularization_validation/09_caltech_l09_logistic_regression_likelihood_gradient_descent.md) | Lecture 9: The Linear Model II; Lecture 9 slides | Stanford CS229 logistic/GLM derivational material; Soudry et al. for separable-data implicit-bias caveat | logistic model; likelihood; cross entropy; gradient descent | Bernoulli conditional likelihood derivation; gradient/Hessian details; objective vs metric vs decision rule; perfect-separation caveat; Week 2/5 links |
| [10 Neural Networks](../part3_fitting_regularization_validation/10_caltech_l10_neural_networks_backpropagation_representation.md) | Lecture 10: Neural Networks; Lecture 10 slides | Stanford CS229 neural-network/backprop material; CS229M deep-learning-theory topics | hidden layers; neural-network hypothesis; backpropagation | scalar-to-matrix backprop derivation; learned representation lens; function-space vs parameter-space distinction; non-convexity boundary; Week 3/4 links |
| [11 Overfitting](../part3_fitting_regularization_validation/11_caltech_l11_overfitting_noise_and_effective_complexity.md) | Lecture 11: Overfitting; Lecture 11 slides | CS229M overparameterization perspective; Zhang et al., Belkin et al., Nakkiran et al. for modern caveats | fitting data too well; stochastic noise; deterministic noise | overfitting as selection phenomenon; effective complexity beyond parameter count; overfitting vs distribution shift distinction; Week 3/4 links |
| [12 Regularization](../part3_fitting_regularization_validation/12_caltech_l12_regularization_constraints_inductive_bias.md) | Lecture 12: Regularization; Lecture 12 slides | Stanford CS229 regularization/MAP material; CS229M implicit-solution-selection lens; Loshchilov and Hutter for AdamW/decoupled weight decay | hard constraints; soft constraints; augmented error; weight decay | Lagrangian caveat; nominal H vs feasible family vs solution preference; L1 comparison; L2 penalty vs weight-decay distinction for adaptive optimizers; MAP boundary; Week 2/3/5 links |
| [13 Validation](../part3_fitting_regularization_validation/13_caltech_l13_validation_model_selection_data_contamination.md) | Lecture 13: Validation; Lecture 13 slides | Stanford CS229 model selection / cross-validation material; adaptive-data-analysis literature for caveats | validation; model selection; data contamination; cross-validation | validation-dependent selection formalization; effective hypothesis set of research process; nested CV; Canvas-Diagnostic-v1 role; final test isolation |
| [14 Modern Explicit/Implicit Regularization](../part3_fitting_regularization_validation/14_modern_explicit_implicit_regularization_and_solution_selection.md) | Lecture 12 synthesized | CS229M official page/topics; Soudry et al.; Jacot et al.; Zhang/Belkin/Nakkiran caveats | explicit regularization picture | implicit / algorithmic bias boundary; separable logistic-regression example; NTK preview; architecture vs full learner distinction |
| [15 Selection-Aware Research Protocol](../part3_fitting_regularization_validation/15_selection_aware_ml_research_protocol.md) | Lecture 13 synthesized | Stanford CS229 methodology; Dwork et al. adaptive data-analysis papers | validation and contamination principles | selection ledger; dataset-role ledger; researcher as adaptive loop; freeze point; credibility levels; Canvas protocol connection |

## 9. T4 note mapping

| Note | Primary Caltech source | Stanford / modern extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ---------------------------------- | -------------------- | ---------------------------------------- |
| [14 Support Vector Machines](../part4_margin_kernel_learning_principles/14_caltech_l14_support_vector_machines_margin_geometry_duality.md) | Lecture 14: Support Vector Machines; Lecture 14 slides | Stanford CS229 SVM notes; CS229M algorithm-dependent complexity framing | separating hyperplanes; maximum margin; support vectors | functional/geometric margin derivation; hard-margin primal/dual/KKT; margin as norm-controlled solution preference; margin/probability caveat |
| [15 Kernel Methods](../part4_margin_kernel_learning_principles/15_caltech_l15_kernel_methods_feature_spaces_soft_margins.md) | Lecture 15: Kernel Methods; Lecture 15 slides | Stanford CS229 kernel/SVM notes; CS229M dimension-vs-effective-complexity framing; Cortes/Vapnik canonical SVM source | nonlinear feature spaces; kernel trick; soft margins | PSD Gram-matrix derivation; Mercer caveat; kernel validity; infinite-dimensional representation versus effective complexity |
| [16 Radial Basis Functions](../part4_margin_kernel_learning_principles/16_caltech_l16_radial_basis_functions_local_representation.md) | Lecture 16: Radial Basis Functions; Lecture 16 slides | Stanford CS229 kernel comparison | RBF units; centers; widths; local representation | design-matrix derivation; RBF model versus Gaussian kernel versus kernel SVM; high-dimensional distance caveat |
| [17 Three Learning Principles](../part4_margin_kernel_learning_principles/17_caltech_l17_three_learning_principles_occam_sampling_snooping.md) | Lecture 17: Three Learning Principles; Lecture 17 slides | CS229 methodology; CS229M capacity vocabulary; Bickel et al. for covariate-shift/sample-selection-bias precision; Dwork et al. adaptive data-analysis papers | Occam's razor; sampling bias; data snooping | Occam beyond parameter count; population/distribution mismatch; sampling bias versus distribution shift; data snooping as hidden selection; benchmark contamination distinctions |
| [18 Epilogue](../part4_margin_kernel_learning_principles/18_caltech_l18_epilogue_bayesian_learning_aggregation_ml_map.md) | Lecture 18: Epilogue; Lecture 18 slides | CS229 probabilistic/MAP support; Breiman bagging; Freund/Schapire boosting | ML map; Bayesian learning; aggregation | Bayes-rule derivation; posterior predictive; aggregation variance derivation; ensemble versus joint representation learning |
| [19 Unified Lens](../part4_margin_kernel_learning_principles/19_geometry_representation_capacity_unified_lens.md) | Lecture 14-18 synthesis | CS229/CS229M support across SVM, kernel, and capacity | representation, geometry, similarity, margin, locality, principles | original unified chain and paper-reading question: which arrow does a new ML paper modify |
| [20 Learning Algorithm Anatomy](../part4_margin_kernel_learning_principles/20_learning_algorithm_anatomy_for_ml_research.md) | Full Learning From Data arc | CS229 and CS229M support for formal categories | map of model, representation, learning, and evidence | original analysis matrix comparing linear regression, logistic regression, MLP, SVM, kernel SVM, and RBF models |

## 10. T5 note mapping

| Note | Primary source families | Mathematical regime | Directly sourced in note | Repository synthesis |
| ---- | ----------------------- | ------------------- | ------------------------ | ------------------- |
| [21 Classical Theory Meets Modern Deep Learning](../part5_modern_learning_theory_bridge/21_classical_theory_meets_modern_deep_learning.md) | Stanford CS229M; Zhang et al. | empirical random-label deep-network experiments plus classical capacity framing | random-label phenomenon and non-implications | nominal $H$ versus $A(S)$ selected-solution bridge |
| [22 Algorithmic Stability](../part5_modern_learning_theory_bridge/22_algorithmic_stability_and_algorithm_dependent_generalization.md) | Bousquet-Elisseeff; Hardt-Recht-Singer | bounded-loss stable algorithms; convex/smooth SGD stability regimes | uniform stability definition, stability theorem structure, SGD stability scaling | class-dependent versus algorithm-dependent distinction and Week 3 optimizer connection |
| [23 Data-Dependent Complexity](../part5_modern_learning_theory_bridge/23_data_dependent_complexity_rademacher_margin_norm.md) | Bartlett-Mendelson; Bartlett-Foster-Telgarsky | bounded/Lipschitz loss classes; linear norm-bounded classes; scoped neural margin/norm bounds | empirical Rademacher definition, source-generalization bound family, spectral margin/norm structural message | full $BR/\sqrt n$ derivation and T4 representation-geometry connection |
| [24 Overparameterization](../part5_modern_learning_theory_bridge/24_overparameterization_interpolation_double_descent_benign_overfitting.md) | Belkin et al.; Nakkiran et al.; Bartlett-Long-Lugosi-Tsigler | interpolation phenomena; minimum-norm linear regression; benign overfitting under spectral assumptions | double-descent phenomena, benign-overfitting formal scope | minimum-norm interpolator derivation and bias-variance reinterpretation |
| [25 Implicit Bias and NTK](../part5_modern_learning_theory_bridge/25_implicit_bias_optimization_dynamics_and_ntk.md) | Soudry et al.; Jacot-Gabriel-Hongler | separable linear logistic regression; NTK/lazy infinite-width regimes | directional max-margin convergence, tangent kernel definition | optimization-geometry bridge and lazy versus feature-learning distinction |
| [26 Distribution Shift](../part5_modern_learning_theory_bridge/26_distribution_shift_domain_adaptation_and_representation.md) | Ben-David et al.; Zhao et al. | binary domain adaptation; invariant representation limitations | $\mathcal H\Delta\mathcal H$ divergence, target-risk bound, invariance limitation | Week 4 Canvas / source-target research taxonomy |
| [27 Unified Map](../part5_modern_learning_theory_bridge/27_modern_generalization_theory_unified_map.md) | all T5 source families | synthesis across regimes | no new theorem | matrix of explanatory lenses and explanation-versus-bound distinction |
| [28 Claim Audit](../part5_modern_learning_theory_bridge/28_modern_ml_theory_claim_audit.md) | all T5 source families | research-audit framework | no new theorem | theorem-evidence ladder and conceptual-trap audit |

## 11. What is original in this repository

The following T1 content is original synthesis for this repository, not copied from course slides:

- the world → measurement → representation → model input framing;
- the refined four-way failure taxonomy: information/representation failure, approximation/specification error, estimation/generalization error, optimization/computation error;
- research-paper audit questions in the theory map;
- connections to this repository's Week 2-5 scratch implementations and reports;
- all PNG figures under `theory/learning-from-data/assets/`;
- the terminology table and source-separation structure.

The following T2 content is original synthesis for this repository:

- theorem templates that force Assumptions, Claim, Derivation / Proof Idea, Interpretation, What This Does NOT Imply, and Research Use;
- the generalization credibility layer: sample → adaptive selection → capacity control → uniform guarantee → out-of-sample claim;
- the research claim audit note and assumption ledger;
- cross-links from theory to Week 3 overfitting, Week 4 shift/canvas diagnostics, and Week 5 calibration/abstention evidence;
- all T2 PNG figures under `theory/learning-from-data/assets/`;
- the explicit separation between generalization gap, excess risk, bias-variance, and approximation-estimation-optimization decomposition.

The following T3 content is original synthesis for this repository:

- the T3 chain from objective, optimization, fitting, overfitting, regularization, validation, selection, to credible evaluation;
- the explicit separation among hypothesis family, parameterization, empirical objective, optimizer, selected parameter vector, selected function, regularizer, validation procedure, model-selection procedure, and final evaluation procedure;
- the Selection / Evaluation Failure category as a research-process failure mode distinct from the T2 risk decomposition terms;
- the selection-aware research protocol, selection ledger, dataset-role ledger, and freeze-point framing;
- cross-links to Week 2 logistic regression, Week 3 MLP/overfitting/optimization, Week 4 Canvas-Diagnostic-v1, and Week 5 calibration/abstention;
- all T3 PNG figures and the reproducible T3 figure-generation script under `theory/learning-from-data/assets/`.

The following T4 content is original synthesis for this repository:

- the chain from input geometry through representation, similarity, margin/norm, effective complexity, sampling assumptions, selection discipline, and credible learning;
- the unified representation-geometry-capacity lens and the question "which arrow does a new ML paper actually modify?";
- the learning-algorithm anatomy matrix for research reading;
- the explicit conceptual-trap audits around margin/probability, kernel/similarity, RBF model/kernel SVM, data snooping, sampling bias, Bayesian priors, and ensembles;
- cross-links to T1-T3 and Week 2-5 reports;
- all T4 PNG figures and the reproducible T4 figure-generation script under `theory/learning-from-data/assets/`.

The following T5 content is original synthesis for this repository:

- the modern-theory master chain from class-level capacity through data-dependent complexity, algorithm-dependent behavior, implicit bias, representation regime, source generalization, and distribution shift;
- the class-dependent versus algorithm-dependent generalization distinction as a reusable research lens;
- the explanation-versus-bound distinction separating valid upper bounds, tight bounds, predictive theories, and mechanistic explanations;
- the T5 theorem-evidence ladder and modern ML theory claim-audit framework;
- the source-versus-target reliability distinction connected to Week 4 Canvas diagnostics and Week 5 calibration/abstention;
- the reproducible T5 figures and figure-generation script under `theory/learning-from-data/assets/`.

## 12. Source-use boundaries

No copyrighted slide screenshots were copied. Notes use original wording, diagrams, and derivations. Formulae such as Hoeffding inequality, least-squares normal equations, empirical/population risk, Bayes classifier under 0/1 loss, and likelihood-derived cross entropy are standard mathematical material and are written here in original explanatory context.

Formulae such as finite-class union bounds, Sauer-type growth control, VC-style uniform bounds, ERM excess-risk consequences, and squared-loss bias-variance decompositions are standard mathematical material. T2 gives original explanatory derivations and proof skeletons rather than copying source notes.

Formulae such as Bernoulli likelihood, logistic negative log likelihood, binary cross entropy, logistic gradient/Hessian, chain-rule backpropagation, constrained/penalized regularization, L2 gradient, MAP/L2 correspondence, and hold-out validation selection are standard mathematical material. T3 writes original derivations and distinguishes objective, optimizer, model selection, and evaluation roles.

Formulae such as hyperplane distance, functional margin, geometric margin, hard-margin SVM primal/dual, KKT complementary slackness, support-vector expansions, soft-margin slack variables, hinge loss, kernelized dual objectives, PSD Gram-matrix derivations, RBF design matrices, Bayes rule, posterior predictive distributions, and variance of an averaged ensemble are standard mathematical material. T4 writes original explanatory derivations and source-separated interpretation.

Formulae such as uniform stability, expected ghost-sample stability arguments, strongly convex regularized-ERM stability scaling, empirical Rademacher complexity, the $BR/\sqrt n$ norm-bounded linear-class derivation, minimum-norm interpolation via Lagrange multipliers, separable logistic-regression directional max-margin convergence, NTK first-order linearization, $K=JJ^\top$, squared-loss function-space dynamics, $\mathcal H\Delta\mathcal H$ divergence, and Ben-David-style target-risk bounds are standard or source-attributed mathematical material. T5 writes original explanatory derivations and explicitly records model-regime boundaries.

Unresolved source issue: direct video transcript-level attribution is not recorded section by section. The official lecture page, official slides, and official/linked theory notes were used for mapping, while the notes synthesize the concepts rather than quote lecture speech.

[← Back to Learning From Data Theory Notebook](../README.md)
