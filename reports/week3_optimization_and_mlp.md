# Week 3 Optimization and MLP Notes

## 1. Current scope

Complete Week 3 modules:

- SGD
- Momentum
- Adam
- mini-batch iteration
- optimizer comparison
- MLP forward pass
- MLP backpropagation derivation
- MLP analytical backpropagation implementation
- MLP numerical gradient checking
- generic parameter-dictionary optimizer API
- reusable XOR MLP training loop
- controlled MLP optimizer comparison
- controlled MLP overfitting analysis

Next modules:

- [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](week4_multiclass_digits_capstone.md)

## 2. Learning map

```text
Optimization algorithms
        ↓
Gradient estimation and statistical foundations
        ↓
MLP forward pass
        ↓
Activation statistics and initialization
        ↓
MLP backpropagation
        ↓
Engineering validation
```

## 3. Table of contents

- [Optimization Algorithms](week3/01_optimization_algorithms.md): SGD, Momentum, Adam, optimizer-comparison experiments, and the generic parameter-dictionary optimizer API.
- [Gradient, Risk, and Sampling](week3/02_gradient_risk_and_sampling.md): loss terminology, empirical risk, expected risk, stochastic gradients, sampling assumptions, and overfitting.
- [MLP Forward Pass and Backpropagation](week3/03_mlp_forward_and_backprop.md): one-hidden-layer MLP equations, shapes, caches, and backpropagation derivations.
- [Activation, Initialization, and Normalization](week3/04_activation_initialization_normalization.md): nonlinear geometry, activation statistics, Xavier/He initialization, and a normalization preview.
- [Engineering Validation](week3/05_engineering_validation.md): mini-batch iterator behavior, optimizer-state testing, numerical stability, shape tests, gradient checking, parameter-optimizer tests, and controlled experiment validation.
- [Appendix: Adam Derivation](week3/appendix_adam_derivation.md): detailed Adam moment, bias-correction, implementation, and testing derivations.
- [Appendix: Information Theory and Cross Entropy](week3/appendix_information_theory_and_cross_entropy.md) — derives the entropy limit of lossless source coding and explains why cross entropy is used as a neural-network classification loss.

## 4. Key Week 3 takeaways

- BGD and SGD share the same parameter-update formula; gradient sampling changes the algorithmic behavior.
- Momentum smooths recent gradients.
- Adam combines direction smoothing with coordinate-wise adaptive scaling.
- Mini-batch gradients estimate empirical-risk gradients.
- MLP nonlinear activations prevent affine-layer collapse.
- Backpropagation is repeated application of the chain rule.
- Numerical gradient checking validates MLP backpropagation on a deterministic local case.
- Parameter-dictionary optimizers can update multiple MLP tensors without depending on layer semantics.
- Controlled validation curves distinguish optimization progress from generalization behavior.
- Correct shapes, numerical stability, and state-transition tests are part of the mathematical implementation.

## 5. Next stage

Completed:

- `Task 5F-A: reusable XOR MLP training loop`
- `Task 5F-B: controlled optimizer comparison`
- `Task 5F-C: controlled overfitting analysis`

Next:

- [Week 4 Multiclass MLP and Handwritten-Digit Recognition Capstone](week4_multiclass_digits_capstone.md)

Week 3 closes the binary MLP foundation; Week 4 generalizes the same probability-and-gradient logic from sigmoid/BCE to softmax/multiclass cross entropy on a real handwritten-digit task.

The capstone extends the NumPy MLP from binary classification to multiclass softmax classification, trains on a real handwritten-digit dataset, and connects the model to a lightweight local prediction application.

## 6. Open questions

Detailed open questions are tracked in the modular notes:

- [Optimization open questions](week3/01_optimization_algorithms.md#23-open-questions)
- [Gradient, risk, and sampling open questions](week3/02_gradient_risk_and_sampling.md#13-open-questions)
- [MLP forward/backpropagation open questions](week3/03_mlp_forward_and_backprop.md#21-open-questions)
- [Activation and initialization open questions](week3/04_activation_initialization_normalization.md#16-open-questions)
- [Engineering validation open questions](week3/05_engineering_validation.md#30-open-questions)
