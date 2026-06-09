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

Next modules:

- MLP analytical gradient implementation
- numerical gradient checking
- MLP training loop
- optimizer API generalization for multiple parameter tensors

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

- [Optimization Algorithms](week3/01_optimization_algorithms.md): SGD, Momentum, Adam, and the optimizer-comparison experiment.
- [Gradient, Risk, and Sampling](week3/02_gradient_risk_and_sampling.md): loss terminology, empirical risk, expected risk, stochastic gradients, and sampling assumptions.
- [MLP Forward Pass and Backpropagation](week3/03_mlp_forward_and_backprop.md): one-hidden-layer MLP equations, shapes, caches, and backpropagation derivations.
- [Activation, Initialization, and Normalization](week3/04_activation_initialization_normalization.md): nonlinear geometry, activation statistics, Xavier/He initialization, and a normalization preview.
- [Engineering Validation](week3/05_engineering_validation.md): mini-batch iterator behavior, optimizer-state testing, numerical stability, shape tests, and future gradient checking.
- [Appendix: Adam Derivation](week3/appendix_adam_derivation.md): detailed Adam moment, bias-correction, implementation, and testing derivations.

## 4. Key Week 3 takeaways

- BGD and SGD share the same parameter-update formula; gradient sampling changes the algorithmic behavior.
- Momentum smooths recent gradients.
- Adam combines direction smoothing with coordinate-wise adaptive scaling.
- Mini-batch gradients estimate empirical-risk gradients.
- MLP nonlinear activations prevent affine-layer collapse.
- Backpropagation is repeated application of the chain rule.
- Correct shapes, numerical stability, and state-transition tests are part of the mathematical implementation.

## 5. Next task

`Task 5C: Implement BinaryMLPScratch.compute_loss() and BinaryMLPScratch.compute_gradients().`

## 6. Open questions

Detailed open questions are tracked in the modular notes:

- [Optimization open questions](week3/01_optimization_algorithms.md#14-open-questions)
- [Gradient, risk, and sampling open questions](week3/02_gradient_risk_and_sampling.md#9-open-questions)
- [MLP forward/backpropagation open questions](week3/03_mlp_forward_and_backprop.md#20-open-questions)
- [Activation and initialization open questions](week3/04_activation_initialization_normalization.md#16-open-questions)
- [Engineering validation open questions](week3/05_engineering_validation.md#12-open-questions)
