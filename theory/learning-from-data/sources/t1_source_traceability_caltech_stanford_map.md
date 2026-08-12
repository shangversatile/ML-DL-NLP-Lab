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

## 3. T1 note mapping

| Note | Primary Caltech source | Stanford extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| [00 Theory Map](../00_learning_theory_ontology_world_data_generalization_research_lens.md) | Lecture 1-4 page topics; video-library topics on Learning Diagram, Bin Model, Error Measures, Noisy Targets, Nonlinear Transformation | STATS214 / CS229M official description and uniform-convergence topic | learning diagram ontology; finite data; input distribution; hypothesis set; error/noise; nonlinear transformation | world-to-representation ontology; refined four-way failure taxonomy; research-paper audit questions; links to repo evaluation methodology |
| [01 What Does It Mean to Learn?](../part1_learning_problem/01_caltech_l01_learning_problem_target_hypothesis_inductive_bias.md) | Lecture 1: The Learning Problem; Lecture 1 slides | STATS214 framing of formalizing learning from data | target function, training examples, learning algorithm, hypothesis set, final hypothesis; supervised/unsupervised/reinforcement distinction | inductive bias framing; memorization distinction; operationalizing research problems; links to scratch linear/logistic/MLP work |
| [02 From Finite Data to Generalization](../part1_learning_problem/02_caltech_l02_finite_sample_generalization_hoeffding_uniform_convergence.md) | Lecture 2: Is Learning Feasible; Lecture 2 slides; video-library Bin Model segments | STATS214 uniform-convergence topic and linked notes repository | in-sample/out-of-sample error; Hoeffding-style concentration; coin/bin analogy; fixed hypothesis vs selected hypothesis | union-bound preview; uniform convergence as selected-hypothesis control; validation/data-snooping research interpretation |
| [03 Hypothesis Spaces and Linear Models](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md) | Lecture 3: The Linear Model I; Lecture 3 slides; video-library Linear Classification, Linear Regression, Nonlinear Transformation topics | STATS214 general lens on hypothesis classes and statistical learning | linear classification geometry; linear regression; nonlinear transforms; linear-in-parameters but nonlinear-in-input distinction | feature maps as induced effective hypothesis families; separation of information loss and specification error; continuity to neural representation learning; links to Week 2/3/4 implementation notes |
| [04 Error, Noise, and Target Distribution](../part1_learning_problem/04_caltech_l04_error_measures_noise_target_distribution.md) | Lecture 4: Error and Noise; Lecture 4 slides; video-library Error Measures and Noisy Targets topics | STATS214 statistical-learning framing of population risk and loss | error measure as part of learning problem; noisy targets; target distribution | conditional distribution interpretation; Bayes classifier and conditional mean derivations; loss-likelihood connection; calibration/abstention/distribution-shift implications |
| [Terminology and Notation](t1_terminology_notation_learning_problem_generalization.md) | Lecture 1-4 notation conventions | STATS214 population/empirical risk vocabulary | symbols for target, hypothesis, algorithm, dataset, in/out error | explicit distinctions among parameter/function, representation/hypothesis, empirical/population, random/realized |

## 4. T2 note mapping

| Note | Primary Caltech source | Stanford extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| [05 Training versus Testing](../part2_generalization_theory/05_caltech_l05_training_testing_and_model_selection.md) | Lecture 5: Training versus Testing; Lecture 5 slides | STATS214 ERM/generalization framing | distinction between training and testing; fixed hypothesis testing; training selection dependence | finite-class union-bound derivation; validation reuse and leaderboard-overfitting methodology; links to Week 3-5 repo evidence |
| [06 Theory of Generalization](../part2_generalization_theory/06_caltech_l06_generalization_theory_growth_function_uniform_control.md) | Lecture 6: Theory of Generalization; Lecture 6 slides | STATS214 uniform convergence and sample-induced complexity | dichotomies; growth function; break point; finite dichotomy control | Sauer-type proof skeleton; uniform convergence supremum; ERM `2 epsilon` consequence; pointwise/uniform research-reading table |
| [07 VC Dimension](../part2_generalization_theory/07_caltech_l07_vc_dimension_capacity_and_sample_complexity.md) | Lecture 7: The VC Dimension; Lecture 7 slides | STATS214 capacity/sample-complexity framing | shattering; VC dimension; VC-style generalization bound; sample complexity | capacity vs parameter count; modern overparameterization caveat; margin/norm/stability preview |
| [08 Bias-Variance](../part2_generalization_theory/08_caltech_l08_bias_variance_learning_curves.md) | Lecture 8: Bias-Variance Tradeoff; Lecture 8 slides | STATS214-style excess-risk and estimator decomposition vocabulary | dataset-dependent hypothesis; bias; variance; learning curves | squared-loss derivation with stochastic noise term; approximation-estimation-optimization distinction; double descent preview; repo experiment links |
| [09 Modern Uniform Convergence](../part2_generalization_theory/09_modern_uniform_convergence_and_capacity_control.md) | Lecture 5-7 concepts synthesized | STATS214 / CS229M official page and linked notes | fixed-to-uniform logic; ERM consequence | excess-risk decomposition variants; PAC vocabulary; beyond-VC capacity preview; limits of worst-case class control |
| [10 Generalization Claim Audit](../part2_generalization_theory/10_generalization_claim_audit_for_ml_research.md) | Lecture 5-8 generalization chain | STATS214 research-level generalization vocabulary | training/testing distinction; capacity control; sample complexity; bias-variance caveats | original research audit framework for population, sampling, selection, evidence, assumptions, and non-implications |

## 5. What is original in this repository

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

## 6. Source-use boundaries

No copyrighted slide screenshots were copied. Notes use original wording, diagrams, and derivations. Formulae such as Hoeffding inequality, least-squares normal equations, empirical/population risk, Bayes classifier under 0/1 loss, and likelihood-derived cross entropy are standard mathematical material and are written here in original explanatory context.

Formulae such as finite-class union bounds, Sauer-type growth control, VC-style uniform bounds, ERM excess-risk consequences, and squared-loss bias-variance decompositions are standard mathematical material. T2 gives original explanatory derivations and proof skeletons rather than copying source notes.

Unresolved source issue: direct video transcript-level attribution is not recorded section by section. The official lecture page, official slides, and official/linked theory notes were used for mapping, while the notes synthesize the concepts rather than quote lecture speech.

[← Back to Learning From Data Theory Notebook](../README.md)
