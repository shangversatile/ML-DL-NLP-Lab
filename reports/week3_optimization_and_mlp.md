# Week 3 Optimization and MLP Notes

## 1. SGD optimizer

I implemented `SGD` in `src/optimization/sgd.py` and added unit tests in `tests/test_sgd.py`. The optimizer applies the update rule:

w_new = w_old - learning_rate * dw

b_new = b_old - learning_rate * db

The update formula looks the same as Batch Gradient Descent. The key difference is not the `step()` formula itself, but how gradients are computed. Batch Gradient Descent uses the full training set, classic SGD uses one sample at a time, and mini-batch SGD uses a subset of samples.

## 2. Why mini-batch sampling belongs to the training loop

The SGD optimizer should only update parameters from the gradients it receives, while the training loop should decide how samples are shuffled and grouped into mini-batches. This separation keeps the optimizer reusable: the same update rule can operate on gradients computed from one sample, a mini-batch, or the full dataset.

If mini-batch sampling were embedded inside the optimizer, data iteration logic and optimization logic would become tightly coupled. Keeping them separate makes the system easier to test, debug, and extend to Momentum and Adam.

## 3. Momentum optimizer

I implemented `Momentum` in `src/optimization/momentum.py` and added unit tests in `tests/test_momentum.py`. Unlike SGD, Momentum is a stateful optimizer. It stores `velocity_weights` and `velocity_bias`, which summarize recent gradient directions.

The update rule is:

v_t = beta * v_(t-1) + (1 - beta) * g_t

theta_t = theta_(t-1) - learning_rate * v_t

where `g_t` is the current gradient, `v_t` is the exponentially weighted moving average of recent gradients, and `beta` controls how strongly historical gradients are retained.

## 4. Why Momentum reduces oscillation

Momentum helps reduce oscillation because gradients that repeatedly change sign partially cancel each other in the moving average, while gradients that point in a stable direction accumulate. This is useful in narrow optimization valleys, where gradients may oscillate strongly across the steep direction while remaining consistent along the direction toward the minimum.

If the gradient direction remains stable, velocity gradually increases toward that direction. If gradients alternate between positive and negative values, the moving average becomes smaller than the raw gradients. This creates smoother and more stable parameter updates.

## 5. Interpreting beta

The coefficient `beta` controls the effective memory length of Momentum. A useful approximation is:

effective memory length ~= 1 / (1 - beta)

For example:

- beta = 0.0 behaves like ordinary SGD
- beta = 0.9 remembers roughly 10 recent steps
- beta = 0.99 remembers roughly 100 recent steps

A larger beta creates smoother but slower-reacting updates. A smaller beta responds more strongly to current gradients but performs less smoothing. The common default beta = 0.9 is a practical compromise rather than a universal mathematical optimum.

## 6. Open questions

- How does batch size affect gradient noise and convergence?
- Why can noisy SGD updates sometimes help optimization?
- How should Momentum store and update optimizer state?
- How does Adam combine first-moment and second-moment estimates?
