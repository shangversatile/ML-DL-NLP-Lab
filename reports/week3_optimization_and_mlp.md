# Week 3 Optimization and MLP Notes

## 1. SGD optimizer

I implemented `SGD` in `src/optimization/sgd.py` and added unit tests in `tests/test_sgd.py`. The optimizer applies the update rule:

w_new = w_old - learning_rate * dw

b_new = b_old - learning_rate * db

The update formula looks the same as Batch Gradient Descent. The key difference is not the `step()` formula itself, but how gradients are computed. Batch Gradient Descent uses the full training set, classic SGD uses one sample at a time, and mini-batch SGD uses a subset of samples.

## 2. Why mini-batch sampling belongs to the training loop

The SGD optimizer should only update parameters from the gradients it receives, while the training loop should decide how samples are shuffled and grouped into mini-batches. This separation keeps the optimizer reusable: the same update rule can operate on gradients computed from one sample, a mini-batch, or the full dataset.

If mini-batch sampling were embedded inside the optimizer, data iteration logic and optimization logic would become tightly coupled. Keeping them separate makes the system easier to test, debug, and extend to Momentum and Adam.

## 3. Open questions

- How does batch size affect gradient noise and convergence?
- Why can noisy SGD updates sometimes help optimization?
- How should Momentum store and update optimizer state?
- How does Adam combine first-moment and second-moment estimates?
