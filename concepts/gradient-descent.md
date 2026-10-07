---
title: Gradient descent (batch, mini-batch, stochastic)
category: concepts
tags: [gradient-descent, sgd, mini-batch, learning-rate, convergence, feature-scaling, convex-optimization, numpy, optimization]
use_cases:
  - "understand how a model's weights are actually fitted before using a framework"
  - "choose between full-batch, mini-batch and stochastic gradient descent"
  - "fix a model whose loss diverges or crawls because of the learning rate or unscaled features"
  - "implement linear or logistic regression from scratch for teaching or an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://www.deeplearningbook.org/contents/numerical.html
  - https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
  - https://arxiv.org/abs/1609.04747
  - https://scikit-learn.org/stable/modules/sgd.html
---

# Gradient descent (batch, mini-batch, stochastic)

## Summary
Gradient descent fits a model by repeatedly moving its parameters a small step against the gradient of the loss: `w ← w − lr · ∇L(w)`. It is how almost every neural network and many classical models (logistic regression, linear SVMs, matrix factorisation) are trained. The variants differ in how many samples each step's gradient uses. That one choice sets the cost per step, the noise in the updates and how well the method scales to big data. Feature scaling and the learning rate decide whether it converges at all.

## Key concepts
- **Gradient.** The vector of partial derivatives of the loss with respect to each parameter. It points uphill, so step the other way. For MSE linear regression: `∂L/∂w = (2/n) Xᵀ(Xw + b − y)`.
- **Learning rate (step size).** Too small and training crawls. Too large and it oscillates or diverges (loss → inf/NaN). For a convex quadratic, steps above `2 / λ_max` (largest Hessian eigenvalue) diverge.
- **Batch (full) GD.** Uses all n samples per step. Exact gradient, smooth descent, O(n) cost per step, and the data must be iterated fully for every update.
- **Stochastic GD (batch = 1).** One sample per step. Very cheap, noisy updates, needs a decaying learning rate to settle. The noise can help escape poor regions in non-convex problems.
- **Mini-batch GD (batch = 16 to 1024).** The default in deep learning. It averages out much of the noise and maps onto vectorised GPU maths. "SGD" in frameworks almost always means mini-batch.
- **Epoch.** One pass over the training set. Shuffle every epoch so batches are not correlated.
- **Conditioning and feature scaling.** When features have very different scales, the loss surface is a long thin valley. One learning rate is too big for one direction and too small for another. Standardising features makes the valley round.
- **Convex vs non-convex.** Linear/logistic regression losses are convex, so GD reaches the global minimum. Neural network losses are not, and GD finds a good local region. That works well in practice ([[generalization-in-deep-learning]]).

## When to use / scenarios
- Data too large for a closed-form solve or second-order method (millions of rows, streaming data): mini-batch or `SGDClassifier`/`SGDRegressor` with `partial_fit` ([[online-learning-and-concept-drift]]).
- Any neural network: mini-batch GD with momentum or Adam ([[optimizers]]).
- Teaching, interviews and debugging: a 20-line NumPy implementation makes it clear what the framework does.
- Not for small tabular regression. `LinearRegression` (least squares) or L-BFGS solves it exactly and faster. Do not use plain GD there.

## Setup & code
```bash
pip install numpy
```
```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000
X = np.c_[rng.normal(50, 10, n), rng.normal(0.5, 0.1, n)]  # very different scales
w_true, b_true = np.array([0.3, -4.0]), 2.0
y = X @ w_true + b_true + rng.normal(0, 0.5, n)


def mse(w, b, X, y):
    return np.mean((X @ w + b - y) ** 2)


def gd(X, y, lr, epochs, batch=None, seed=0):
    """batch=None: full-batch GD; batch=1: SGD; else mini-batch."""
    r = np.random.default_rng(seed)
    w, b = np.zeros(X.shape[1]), 0.0
    n = len(y)
    bs = n if batch is None else batch
    for _ in range(epochs):
        order = r.permutation(n)
        for s in range(0, n, bs):
            i = order[s:s + bs]
            err = X[i] @ w + b - y[i]               # residuals
            w -= lr * 2 * X[i].T @ err / len(i)     # dMSE/dw
            b -= lr * 2 * err.mean()                # dMSE/db
    return w, b


# Closed form for reference.
A = np.c_[X, np.ones(n)]
sol = np.linalg.lstsq(A, y, rcond=None)[0]
print("closed form  w=", sol[:2].round(3), "b=", round(sol[2], 3), "mse=", round(mse(sol[:2], sol[2], X, y), 4))

# Raw features: any lr big enough to move w2 makes w1 diverge.
with np.errstate(over="ignore", invalid="ignore"):
    for lr in [1e-3, 1e-4]:
        w, b = gd(X, y, lr, 200)
        print(f"raw  lr={lr:g}  mse={mse(w, b, X, y):.4g}")

# Standardised features: one lr works for all weights.
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd
for name, batch, lr, ep in [("full-batch", None, 0.1, 200), ("mini-batch 32", 32, 0.01, 20), ("SGD (batch=1)", 1, 0.001, 5)]:
    w, b = gd(Xs, y, lr, ep, batch)
    w_orig, b_orig = w / sd, b - (w / sd) @ mu   # map back to raw units
    print(f"scaled {name:14s} lr={lr:<5g} epochs={ep:3d} w={w_orig.round(3)} b={b_orig:.3f} mse={mse(w, b, Xs, y):.4f}")
```
Output (NumPy 2.5):

| Run | Result |
|---|---|
| Closed form | w = [0.297, −3.963], b = 2.090, MSE 0.2403 |
| Raw features, lr 1e-3, 200 epochs | diverged (MSE 2e+247) |
| Raw features, lr 1e-4, 200 epochs | MSE 0.406: stable but far from the optimum |
| Scaled, full batch, lr 0.1, 200 steps | exact match with the closed form |
| Scaled, mini-batch 32, lr 0.01, 20 epochs | MSE 0.2403, w₂ = −3.955 |
| Scaled, SGD, lr 0.001, 5 epochs | MSE 0.2407, w₂ = −3.821 (still noisy) |

On raw features no single learning rate works: the large-scale feature diverges before the small-scale one has moved. After standardising, all three variants reach the least-squares solution. Mini-batch and SGD make many more updates per epoch, so they need far fewer passes over the data.

## Choosing / trade-offs
| | Full batch | Mini-batch | SGD (1 sample) |
|---|---|---|---|
| Cost per step | O(n) | O(batch) | O(1) |
| Gradient noise | none | moderate | high |
| Needs LR decay to converge | no | helps | yes |
| Hardware fit | memory-bound for big n | best on GPU | poor vectorisation |
| Typical use | small convex problems, L-BFGS | deep learning default | online / streaming |

- **Batch size and learning rate move together.** Bigger batches have less noise and allow a larger learning rate (roughly linear scaling with warm-up, see [[learning-rate-schedules]]).
- **Plain GD vs adaptive optimisers.** Momentum and Adam fix much of the ill-conditioning problem automatically ([[optimizers]]), but scaled inputs still help.
- **Find the learning rate empirically.** Try values a factor of ~3 apart (1e-4, 3e-4, 1e-3, ...) and keep the largest that decreases loss smoothly, or use an LR range test ([[debugging-neural-network-training]]).

## Gotchas
- **Unscaled features** are the classic reason a from-scratch GD diverges or stalls. Standardise with statistics from the training set only ([[feature-engineering]]).
- **Forgetting to shuffle.** Data sorted by label or time gives batches that pull the weights back and forth.
- **Summing instead of averaging the batch gradient** ties the effective learning rate to the batch size. Divide by the batch size (as above) so lr means the same thing for any batch.
- **Judging convergence on the last batch.** Mini-batch loss is noisy. Track the full training or validation loss per epoch.
- **Constant learning rate with SGD** settles into a noise ball around the minimum, not on it. Decay the rate or average the iterates.
- **Gradient bugs fail quietly.** A wrong sign or a missing factor still lowers the loss a little. Check against a numerical gradient ([[backpropagation-and-autograd]]).

## Related
- [[optimizers]] - momentum, RMSprop, Adam and AdamW built on top of plain gradient descent.
- [[learning-rate-schedules]] - warm-up, decay and batch-size scaling.
- [[math-for-machine-learning]] - derivatives, gradients and the chain rule.
- [[linear-models]] - closed-form and solver-based alternatives for linear models.
- [[neural-network-from-scratch-numpy]] - gradient descent applied to a two-layer network.
- [[backpropagation-and-autograd]] - how gradients are computed for deep models.

## References
- Goodfellow, Bengio, Courville, *Deep Learning*, chapter 4 (numerical computation, gradient-based optimisation): https://www.deeplearningbook.org/contents/numerical.html
- Google ML Crash Course, gradient descent: https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent
- Ruder, "An overview of gradient descent optimization algorithms" (2016): https://arxiv.org/abs/1609.04747
- scikit-learn, stochastic gradient descent: https://scikit-learn.org/stable/modules/sgd.html
