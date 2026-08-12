# Source Map

[← Back to Learning From Data Theory Notebook](../README.md)

本文件记录 T1 notes 的 source traceability。它不声称逐页复现课程内容；它说明每个 note 的 Caltech Core、Stanford / Theory Extension、Modern Perspective 与 Research Reflection 来自何处或如何扩展。

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

## 2. T1 note mapping

| Note | Primary Caltech source | Stanford extension source | Caltech Core in note | Extension / modern / original reflection |
| ---- | ---------------------- | ------------------------- | -------------------- | ---------------------------------------- |
| [00 Theory Map](../00_learning_theory_ontology_world_data_generalization_research_lens.md) | Lecture 1-4 page topics; video-library topics on Learning Diagram, Bin Model, Error Measures, Noisy Targets, Nonlinear Transformation | STATS214 / CS229M official description and uniform-convergence topic | learning diagram ontology; finite data; input distribution; hypothesis set; error/noise; nonlinear transformation | world-to-representation ontology; three-gap framework; research-paper audit questions; links to repo evaluation methodology |
| [01 What Does It Mean to Learn?](../part1_learning_problem/01_caltech_l01_learning_problem_target_hypothesis_inductive_bias.md) | Lecture 1: The Learning Problem; Lecture 1 slides | STATS214 framing of formalizing learning from data | target function, training examples, learning algorithm, hypothesis set, final hypothesis; supervised/unsupervised/reinforcement distinction | inductive bias framing; memorization distinction; operationalizing research problems; links to scratch linear/logistic/MLP work |
| [02 From Finite Data to Generalization](../part1_learning_problem/02_caltech_l02_finite_sample_generalization_hoeffding_uniform_convergence.md) | Lecture 2: Is Learning Feasible; Lecture 2 slides; video-library Bin Model segments | STATS214 uniform-convergence topic and linked notes repository | in-sample/out-of-sample error; Hoeffding-style concentration; coin/bin analogy; fixed hypothesis vs selected hypothesis | union-bound preview; uniform convergence as selected-hypothesis control; validation/data-snooping research interpretation |
| [03 Hypothesis Spaces and Linear Models](../part1_learning_problem/03_caltech_l03_hypothesis_spaces_linear_models_feature_transforms.md) | Lecture 3: The Linear Model I; Lecture 3 slides; video-library Linear Classification, Linear Regression, Nonlinear Transformation topics | STATS214 general lens on hypothesis classes and statistical learning | linear classification geometry; linear regression; nonlinear transforms; linear-in-parameters but nonlinear-in-input distinction | representation gap; feature geometry; continuity to neural representation learning; links to Week 2/3/4 implementation notes |
| [04 Error, Noise, and Target Distribution](../part1_learning_problem/04_caltech_l04_error_measures_noise_target_distribution.md) | Lecture 4: Error and Noise; Lecture 4 slides; video-library Error Measures and Noisy Targets topics | STATS214 statistical-learning framing of population risk and loss | error measure as part of learning problem; noisy targets; target distribution | conditional distribution interpretation; Bayes classifier and conditional mean derivations; loss-likelihood connection; calibration/abstention/distribution-shift implications |
| [Terminology and Notation](t1_terminology_notation_learning_problem_generalization.md) | Lecture 1-4 notation conventions | STATS214 population/empirical risk vocabulary | symbols for target, hypothesis, algorithm, dataset, in/out error | explicit distinctions among parameter/function, representation/hypothesis, empirical/population, random/realized |

## 3. What is original in this repository

The following T1 content is original synthesis for this repository, not copied from course slides:

- the world → measurement → representation → model input framing;
- the three-gap framework: representation gap, estimation/generalization gap, optimization/computation gap;
- research-paper audit questions in the theory map;
- connections to this repository's Week 2-5 scratch implementations and reports;
- all PNG figures under `theory/learning-from-data/assets/`;
- the terminology table and source-separation structure.

## 4. Source-use boundaries

No copyrighted slide screenshots were copied. Notes use original wording, diagrams, and derivations. Formulae such as Hoeffding inequality, least-squares normal equations, empirical/population risk, Bayes classifier under 0/1 loss, and likelihood-derived cross entropy are standard mathematical material and are written here in original explanatory context.

Unresolved source issue: direct video transcript-level attribution is not recorded section by section. The official lecture page and video-library topic index were used for mapping, while the notes synthesize the concepts rather than quote lecture speech.

[← Back to Learning From Data Theory Notebook](../README.md)
