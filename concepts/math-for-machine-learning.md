---
title: Math for machine learning (linear algebra, calculus, probability, optimization)
category: concepts
tags: [math, linear-algebra, calculus, gradients, probability, bayes, optimization, svd, numerical-stability, numpy]
use_cases:
  - "understand what a model is doing well enough to debug it, not just call fit()"
  - "derive and check a gradient for a custom loss"
  - "know why training diverges at a given learning rate"
  - "read an ML paper or library doc without getting lost in the notation"
  - "avoid NaNs and overflow in softmax, log-likelihoods and probabilities"
status: draft
last_verified: 2026-10-04
sources:
  - https://mml-book.github.io/
  - https://www.deeplearningbook.org/
  - https://numpy.org/doc/stable/reference/routines.linalg.html
---

# Math for machine learning (linear algebra, calculus, probability, optimization)

## Summary
Four areas of math carry almost all of ML: linear algebra (data as matrices,
models as matrix products), calculus (gradients tell the optimizer which way
to move), probability (losses are negative log-likelihoods, predictions are
distributions) and optimization (gradient descent and its limits). You do not
need proofs to use ML, but this core lets you debug divergence, NaNs, slow
training and wrong-looking probabilities instead of guessing.

## Key concepts
- **Vectors and matrices.** A dataset is an `n x d` matrix `X`; a linear model
  is `X @ w`; a neural layer is `activation(X @ W + b)`. Shapes are the first
  thing to check when code breaks.
- **Dot product and norms.** Similarity (cosine = dot of unit vectors, see
  [[embeddings]]); L2 norm for distance and weight decay, L1 for sparsity.
- **Eigenvalues / SVD.** SVD `X = U S V'` gives PCA (directions of most
  variance, see [[dimensionality-reduction]]), low-rank approximations (LoRA in
  [[fine-tuning-and-peft]]) and the condition number (largest / smallest
  singular value): high condition number = ill-posed, slow, unstable fits.
- **Gradients and the chain rule.** The gradient points uphill; training steps
  the other way. Backpropagation is the chain rule applied layer by layer
  (see [[neural-network-fundamentals]]). The Hessian (second derivatives)
  measures curvature.
- **Probability.** Random variables, expectation, variance, conditional
  probability and Bayes' rule `p(a|b) = p(b|a) p(a) / p(b)`. Common
  distributions: Bernoulli/categorical (classification), Gaussian
  (regression noise), Poisson (counts).
- **Maximum likelihood.** MSE = Gaussian negative log-likelihood;
  cross-entropy = categorical negative log-likelihood. L2 regularization =
  Gaussian prior on weights (MAP estimation).
- **Optimization.** Convex problems (linear/logistic regression) have one
  minimum; deep nets do not but SGD works anyway. On a quadratic, gradient
  descent is stable only for learning rate `< 2 / L`, where `L` is the
  largest Hessian eigenvalue.
- **Numerics.** Floats overflow and underflow: work in log space, use
  log-sum-exp, never invert a matrix you can solve against.

## When to use / scenarios
- Before or alongside [[ml-fundamentals]] and [[neural-network-fundamentals]]
  when concepts feel like magic.
- Debugging: loss goes to NaN/inf, training diverges, a custom loss does not
  learn (gradient check it), probabilities that do not sum to 1.
- Writing custom losses, layers or samplers (see [[pytorch-basics]]).
- Interpreting results: why a 95% accurate test still gives mostly false
  positives on a rare condition (base rates, see [[model-evaluation-and-metrics]]).
- NOT a prerequisite for calling a well-tested library on a standard problem;
  learn it when the problem forces you to.

## Setup & code
```bash
pip install numpy
```
```python
import numpy as np
rng = np.random.default_rng(0)

# --- Linear algebra: least squares ---
n, d = 200, 3
X = np.c_[np.ones(n), rng.normal(size=(n, d - 1))]
w_true = np.array([1.0, 2.0, -3.0])
y = X @ w_true + rng.normal(0, 0.5, n)
w_normal = np.linalg.solve(X.T @ X, X.T @ y)          # normal equations
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]        # QR/SVD, numerically safer
print("normal:", w_normal.round(3), "lstsq:", w_lstsq.round(3))

# --- Calculus: analytic gradient of MSE, checked by finite differences ---
def loss(w):
    r = X @ w - y
    return r @ r / n

def grad(w):
    return 2 * X.T @ (X @ w - y) / n

w0 = rng.normal(size=d)
eps = 1e-6
num = np.array([(loss(w0 + eps * e) - loss(w0 - eps * e)) / (2 * eps) for e in np.eye(d)])
print("grad check max abs diff:", f"{np.abs(num - grad(w0)).max():.2e}")

# --- Optimization: gradient descent; step size limited by curvature ---
H = 2 * X.T @ X / n                                  # Hessian of the MSE
L = np.linalg.eigvalsh(H).max()                      # largest curvature
for lr in (1.0 / L, 2.1 / L):
    w = np.zeros(d)
    for _ in range(200):
        w -= lr * grad(w)
    print(f"lr={lr * L:.1f}/L -> loss={loss(w):.4f}")
print(f"condition number of H: {np.linalg.cond(H):.2f}")

# --- SVD / PCA: variance explained by top components ---
Z = rng.normal(size=(500, 2)) @ np.array([[3.0, 0, 0, 0], [0, 1.0, 0, 0]]) + 0.1 * rng.normal(size=(500, 4))
Zc = Z - Z.mean(0)
s = np.linalg.svd(Zc, compute_uv=False)
print("variance explained:", (s**2 / (s**2).sum()).round(3))

# --- Probability: Bayes rule for a rare-condition test ---
prior, sens, spec = 0.01, 0.95, 0.95
post = sens * prior / (sens * prior + (1 - spec) * (1 - prior))
print(f"P(condition | positive) = {post:.3f}")

# --- Numerics: log-sum-exp keeps softmax finite ---
logits = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(logits) / np.exp(logits).sum()
m = logits.max()
stable = np.exp(logits - m) / np.exp(logits - m).sum()
print("naive softmax:", naive, "stable:", stable.round(3))
```
Output (numpy 2.5.1): both solvers give `[0.999 1.975 -2.94]` (true
`[1, 2, -3]`); the gradient check agrees to `1.91e-09`. Gradient descent at
`1.0/L` reaches loss 0.2458, at `2.1/L` it blows up to ~7e13, just past the
`2/L` limit. PCA finds 89.8% + 10.0% of the variance in two components (the
data was built from two). A 95%-sensitive, 95%-specific test for a 1%
condition gives only a 16.1% chance a positive is real. Naive softmax of
large logits is `[nan nan nan]`; the shifted version gives `[0.09 0.245 0.665]`.

## Choosing / trade-offs
- Learning order that pays off fastest: shapes and matrix products, then
  gradients and the chain rule, then likelihood and Bayes, then
  eigenvalues/SVD. Learn measure theory and convex analysis only if you do research.
- Solve, do not invert: `np.linalg.solve`/`lstsq` over `inv(A) @ b`
  (faster and more accurate). For ill-conditioned problems, add ridge
  regularization ([[linear-models]]) rather than trusting a raw solve.
- Analytic gradients are fast but easy to get wrong; autograd
  ([[pytorch-basics]]) is the default, finite differences are for checking.

## Gotchas
- Finite-difference checks need float64 and a central difference; in float32
  with `eps=1e-6` the check is pure noise.
- Feature scales set the curvature: one feature in millions and one in units
  gives a huge condition number and a learning rate no single value can fit.
  Standardize first ([[feature-engineering]]).
- `log(0)` and `exp(large)` are the usual sources of NaN; clip probabilities
  or use library losses that take logits (`BCEWithLogitsLoss`,
  `cross_entropy`) instead of probabilities.
- Base-rate neglect: accuracy, sensitivity and specificity do not tell you
  the probability a positive is real without the prior.
- Broadcasting silently creates `(n, n)` matrices from `(n,)` and `(n, 1)`
  arrays; assert shapes on anything you subtract.

## Related
- [[ml-fundamentals]] - the ML workflow this math sits under.
- [[neural-network-fundamentals]] - backpropagation is the chain rule.
- [[linear-models]] - least squares, ridge and logistic regression in practice.
- [[dimensionality-reduction]] - PCA via SVD.
- [[bayesian-and-gaussian-processes]] - probability used end to end.
- [[deep-learning-training]] - optimizers and learning-rate schedules.

## References
- Deisenroth, Faisal, Ong, Mathematics for Machine Learning (free book): https://mml-book.github.io/
- Goodfellow, Bengio, Courville, Deep Learning, Part I (math background): https://www.deeplearningbook.org/
- NumPy linear algebra reference: https://numpy.org/doc/stable/reference/routines.linalg.html
