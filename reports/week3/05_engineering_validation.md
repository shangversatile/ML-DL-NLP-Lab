[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

# Engineering Validation

## 1. Mini-batch iterator implementation

`iterate_minibatches()` belongs to the data/preprocessing layer rather than the optimizer layer. It creates sample indices with `np.arange(n_samples)`, optionally shuffles those indices, and slices the index array into batches instead of modifying `X` and `y` in place. Indexing both arrays with the same `batch_indices` keeps features and labels aligned.

The final batch may contain fewer than `batch_size` examples. In the optimizer comparison, the training loop passes `seed + epoch`, so each epoch has a different but reproducible order. The same iterator can be reused by SGD, Momentum, Adam, and future MLP training because it only controls data access, not parameter-update logic.
## 2. Reproducible shuffling with `seed + epoch`

The mini-batch iterator uses `seed + epoch` to create different but reproducible shuffling each epoch. The shuffle order changes from epoch to epoch, but rerunning the experiment with the same base seed produces the same sequence of epoch-level shuffles.

This gives two useful properties at once: each epoch sees a different order, and test or experiment reruns remain reproducible.
## 3. Why optimizer state must be isolated

Each optimizer comparison run gets a fresh `LogisticRegressionScratch` model initialized with zero weights and zero bias. This prevents one optimizer from benefiting from parameters learned by a previous optimizer.

Momentum and Adam also receive fresh optimizer objects so that historical state does not leak between experiments. Momentum gets a new velocity state, and Adam gets new first-moment, second-moment, and time-step state. Stateful optimizers must be isolated because their update at time $`t`$ depends on previous gradients, not only on the current gradient.
## 4. Testing stateful optimizers

One-step tests are necessary but insufficient for stateful optimizers. A faulty optimizer may produce the correct first update while incorrectly resetting or misusing internal state later.

Adam tests should verify the transition from $`state_0`$ to $`state_1`$ to $`state_2`$. The strengthened second-step test checks `first_moment_weights`, `second_moment_weights`, `first_moment_bias`, `second_moment_bias`, `time_step`, updated weights, and updated bias. This protects against errors such as resetting moments each step, mixing `beta1` and `beta2`, omitting gradient squaring, or applying bias correction incorrectly.
## 5. Adam second-step transition test

A first-step Adam test can pass even when the optimizer mishandles later state. For example, an implementation might initialize moments correctly on step one but reset them before step two.

The stronger test is a transition check:

```text
state_0 -> state_1 -> state_2
```

The second-step test should verify `first_moment_weights`, `second_moment_weights`, `first_moment_bias`, `second_moment_bias`, `time_step`, updated weights, and updated bias. This catches errors such as resetting moments each step, mixing `beta1` and `beta2`, omitting gradient squaring, or applying bias correction incorrectly.
## 6. Shape-based testing

The forward-pass tests cover parameter shapes, reproducible initialization, different initialization under different seeds, ReLU values, numerically stable sigmoid behavior, forward-pass shapes, manually verifiable forward-pass values, invalid dimensions, and invalid input shape.

Shape tests catch interface errors. Manual-value tests validate the complete mathematical path from input features through hidden activations to output probabilities. Stable-sigmoid tests catch numerical errors that would not necessarily appear in purely symbolic formula checks.

| Parameter |            Parameter shape | Gradient |             Gradient shape |
| --------- | -------------------------: | -------- | -------------------------: |
| `W1`      | `(n_features, hidden_dim)` | `dW1`    | `(n_features, hidden_dim)` |
| `b1`      |            `(hidden_dim,)` | `db1`    |            `(hidden_dim,)` |
| `W2`      |          `(hidden_dim, 1)` | `dW2`    |          `(hidden_dim, 1)` |
| `b2`      |                     `(1,)` | `db2`    |                     `(1,)` |

Each parameter gradient must have exactly the same shape as its corresponding parameter. This invariant is what allows the optimizer to update parameters element by element without broadcasting mistakes or shape-dependent special cases.
## 7. Manual-value testing

Manual-value tests validate the complete mathematical path on small examples where the expected values can be computed by hand. They are especially useful for forward-pass equations, simple optimizer updates, and backpropagation formulas with known intermediate values.

Because floating-point arithmetic is approximate, tests should use tolerances rather than exact equality for non-integer numeric results. `np.allclose()` is appropriate for arrays, and `pytest.approx()` is appropriate for scalar floating-point values.
## 8. Numerical-stability testing

Numerical-stability tests exercise inputs that can expose overflow, underflow, or divide-by-zero behavior. The stable sigmoid test with `0.0`, `1000.0`, and `-1000.0` checks ordinary, very positive, and very negative logits.

Adam's $`\epsilon`$ term is another numerical-stability mechanism: it prevents division by zero or division by an extremely small second-moment scale.
## 9. Headless plotting with Matplotlib `Agg`

Plotting code should use a headless-safe Matplotlib backend such as `Agg` when figures are generated in scripts, CI jobs, or remote environments without a display server.

This keeps experiment visualization reproducible and prevents plotting from depending on a local interactive GUI session.
## 10. Future task: numerical gradient checking

Numerical gradient checking is the next major validation step for MLP backpropagation. The idea is to compare analytical gradients from `compute_gradients()` with finite-difference approximations of the loss.

A centered finite difference for one parameter coordinate is:

```math
\frac{L(\theta+h)-L(\theta-h)}{2h}
```

If the analytical gradient and numerical gradient agree within tolerance, the chain-rule implementation is much more credible. Gradient checking is slow, so it should run on tiny synthetic inputs and small parameter tensors, not as a full training test.

## 11. Engineering checklist

- Keep data iteration outside optimizer classes.
- Shuffle indices instead of mutating `X` and `y` in place.
- Preserve the final smaller mini-batch unless a specific experiment intentionally drops it.
- Give each optimizer comparison run fresh model and optimizer state.
- Test stateful optimizers across at least two transitions.
- Use shape tests to catch broadcasting and orientation errors.
- Use manual-value tests for small deterministic examples.
- Use tolerance-based floating-point assertions with `np.allclose()` or `pytest.approx()`.
- Include numerical-stability tests for sigmoid and adaptive optimizers.
- Add numerical gradient checking before trusting MLP backpropagation in a training loop.
## 12. Open questions

- How can numerical gradient checking validate an MLP backpropagation implementation?
- Which tolerances should be used for finite-difference checks in float64 experiments?
- How should tests distinguish shape bugs from mathematically wrong gradient values?
- What plotting outputs should be saved for optimizer and MLP training diagnostics?

[← Back to Week 3 Index](../week3_optimization_and_mlp.md)

[Appendix: Adam Derivation →](appendix_adam_derivation.md)
