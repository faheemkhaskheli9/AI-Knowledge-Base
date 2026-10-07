---
title: Sinkhorn optimal transport from scratch (entropic OT, log-domain stability, soft assignment)
category: concepts
tags: [optimal-transport, sinkhorn, wasserstein-distance, earth-movers-distance, entropic-regularization, soft-assignment, logsumexp, numpy, scipy, from-scratch]
use_cases:
  - "compare two distributions or histograms by how far mass has to move, not bin by bin"
  - "compute a differentiable Wasserstein-style loss or soft matching inside a model"
  - "choose the entropic regularisation epsilon and avoid NaNs from Sinkhorn at small epsilon"
  - "turn a cost matrix into a balanced soft assignment (clusters, tokens to experts, prototypes)"
status: stable
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/1306.0895
  - https://arxiv.org/abs/1803.00567
  - https://pythonot.github.io/
---

# Sinkhorn optimal transport from scratch (entropic OT, log-domain stability, soft assignment)

## Summary
Optimal transport (OT) asks for the cheapest way to move one distribution of mass onto another, given a cost per unit moved between each pair of locations. Its optimal cost is the Wasserstein distance (earth mover's distance), which respects geometry: two histograms shifted by one bin are close, while bin-wise distances like KL see them as unrelated. Exact OT is a linear program. Cuturi (2013) added an entropy term, which turns the problem into alternately rescaling the rows and columns of a matrix (the Sinkhorn algorithm): a few lines of NumPy, GPU-friendly and differentiable. This file implements plain and log-domain Sinkhorn, checks them against the exact 1D answer, and shows the trade-off epsilon controls: accuracy against iterations and numerical stability.

## Key concepts
- **Transport plan.** A matrix `P ≥ 0` whose rows sum to the source weights `a` and columns to the target weights `b`. `P[i, j]` is the mass moved from `i` to `j`. Cost is `⟨P, C⟩ = Σ P[i,j]·C[i,j]`.
- **Entropic OT.** Minimise `⟨P, C⟩ − ε·H(P)`. The solution has the form `P = diag(u)·K·diag(v)` with the Gibbs kernel `K = exp(−C/ε)`, so only the scaling vectors `u`, `v` need to be found.
- **Sinkhorn iterations.** `v = b / (Kᵀu)`, `u = a / (Kv)`, repeat until the marginals match. Each step is a matrix-vector product.
- **Epsilon.** Large ε: blurry plan, fast convergence, biased cost. Small ε: plan approaches the exact (sparse) solution, but iterations grow roughly like `1/ε` and `exp(−C/ε)` underflows to 0.
- **Log domain.** Iterate on dual potentials `f = ε log u`, `g = ε log v` with `logsumexp`. Same updates, no underflow, needed for ε small relative to the costs.
- **Exact 1D case.** On a line the optimal plan is monotone (sorted to sorted), so `W2² = ∫(F_a⁻¹(t) − F_b⁻¹(t))² dt`. Useful as ground truth, and as the fast path whenever data are 1D.
- **Soft assignment.** With uniform `a`, `b` and small ε, the plan approaches a permutation: Sinkhorn is a differentiable relaxation of the Hungarian algorithm ([[hungarian-algorithm-tracking-from-scratch]]).

## When to use / scenarios
- Comparing histograms or point clouds where geometry matters: colour histograms, word-embedding sets (Word Mover's Distance), distribution shift between a training set and production data in embedding space.
- Losses between distributions in generative models and domain adaptation, where a differentiable distance with sensible gradients is needed.
- Balanced assignment inside models: equal-size cluster assignment in self-supervised learning (SwAV), balanced token-to-expert routing ([[mixture-of-experts-from-scratch-numpy]]), matching predictions to targets.
- Not for exact assignment of a few hundred items: use the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), which is exact and fast at that size.
- Not when data are 1D: sort and use quantiles, exact and `O(n log n)`. `scipy.stats.wasserstein_distance` does this.
- Not when a plain divergence works: if the supports overlap well and geometry does not matter, KL or cross-entropy are cheaper.

## Setup & code
NumPy plus SciPy (`logsumexp`, `linear_sum_assignment`), runs in a few seconds. Two 1D histograms on 60 bins (a two-bump source and a one-bump target), squared-distance cost so the exact answer is `W2²`.

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

n = 60
x = np.linspace(0, 1, n)
a = np.exp(-(x - 0.25) ** 2 / 0.005) + 0.5 * np.exp(-(x - 0.7) ** 2 / 0.002)
b = np.exp(-(x - 0.6) ** 2 / 0.01)
a, b = a / a.sum(), b / b.sum()
C = (x[:, None] - x[None]) ** 2                     # squared distance cost -> W2^2


def exact_w2sq_1d(a, b, x):
    """In 1D the optimal plan is monotone: W2^2 = integral of (F_a^-1(t) - F_b^-1(t))^2 dt."""
    t = np.union1d(np.cumsum(a), np.cumsum(b))
    t = np.r_[0, t[t < 1 - 1e-12], 1]
    mid = (t[:-1] + t[1:]) / 2
    qa = x[np.searchsorted(np.cumsum(a), mid)]
    qb = x[np.searchsorted(np.cumsum(b), mid)]
    return np.sum(np.diff(t) * (qa - qb) ** 2)


def sinkhorn(a, b, C, eps, iters=10000, tol=1e-9):
    """Plain Sinkhorn: alternately rescale rows and columns of K = exp(-C/eps)."""
    K = np.exp(-C / eps)
    u = np.ones_like(a)
    for i in range(iters):
        v = b / (K.T @ u)
        u = a / (K @ v)
        P = u[:, None] * K * v[None]
        if np.abs(P.sum(0) - b).sum() < tol:
            break
    return P, i + 1


def sinkhorn_log(a, b, C, eps, iters=10000, tol=1e-9):
    """Same updates on dual potentials f, g with logsumexp: stable for small eps."""
    f, g = np.zeros_like(a), np.zeros_like(b)
    la, lb = np.log(a), np.log(b)
    for i in range(iters):
        g = eps * (lb - logsumexp((f[:, None] - C) / eps, axis=0))
        f = eps * (la - logsumexp((g[None] - C) / eps, axis=1))
        P = np.exp((f[:, None] + g[None] - C) / eps)
        if np.abs(P.sum(0) - b).sum() < tol:
            break
    return P, i + 1


exact = exact_w2sq_1d(a, b, x)
print(f"exact W2^2 (1D quantiles): {exact:.5f}")
print("   eps   | plain: iters  cost     | log: iters  cost     rel.err  plan entropy")
with np.errstate(all="ignore"):
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        P, it = sinkhorn(a, b, C, eps)
        plain = f"{it:5d}  {np.sum(P * C):.5f}" if np.isfinite(P).all() else "  NaN (K underflowed)"
        Pl, itl = sinkhorn_log(a, b, C, eps)
        cost = np.sum(Pl * C)
        H = -np.sum(Pl[Pl > 0] * np.log(Pl[Pl > 0]))
        print(f"{eps:8.0e} | {plain:22s} | {itl:5d}  {cost:.5f}  {abs(cost - exact) / exact:6.1%}  {H:6.2f}")

# Soft assignment: with uniform weights and small eps the plan approaches a permutation
rng = np.random.default_rng(0)
p, q = rng.random((8, 2)), rng.random((8, 2))
D = ((p[:, None] - q[None]) ** 2).sum(-1)
w = np.full(8, 1 / 8)
r, c = linear_sum_assignment(D)
for eps in (0.05, 0.005, 0.0005):
    P, _ = sinkhorn_log(w, w, D, eps)
    print(f"eps {eps}: row argmax = Hungarian {np.mean(P.argmax(1) == c):4.0%}, "
          f"weakest row's top share {P.max(1).min() * 8:.2f}")
```

Output (Python 3.14, NumPy 2.5, SciPy 1.18):
```
exact W2^2 (1D quantiles): 0.07852
   eps   | plain: iters  cost     | log: iters  cost     rel.err  plan entropy
   1e-01 |     9  0.09534         |     9  0.09534   21.4%    5.75
   1e-02 |    70  0.08177         |    70  0.08177    4.1%    5.32
   1e-03 |   544  0.07892         |   544  0.07892    0.5%    4.52
   1e-04 |   NaN (K underflowed)  |  3797  0.07853    0.0%    3.53
eps 0.05: row argmax = Hungarian  88%, weakest row's top share 0.25
eps 0.005: row argmax = Hungarian 100%, weakest row's top share 0.51
eps 0.0005: row argmax = Hungarian 100%, weakest row's top share 0.63
```

How to read it:
- The transport cost of the entropic plan overestimates the exact `W2²` by 21% at ε = 0.1 and converges to it as ε shrinks: 4.1%, 0.5%, then within 0.01% at ε = 1e-4. The plan entropy falls in step: a smaller ε means a sharper, sparser plan.
- The price is iterations, which grow about 7–8× per 10× decrease in ε (9, 70, 544, 3797). This is the main cost knob in practice.
- At ε = 1e-4 the plain version returns NaN: `exp(−C/ε)` underflows to exactly 0 for every pair farther apart than about 0.27, the scaling vectors overflow to `inf` trying to compensate, and `inf · 0` gives NaN. The log-domain version computes the same thing without the overflow. Plain and log agree exactly wherever the plain one works.
- For matching 8 points to 8 points, the row-wise argmax of the plan agrees with the Hungarian assignment in 7 of 8 rows at ε = 0.05 and in all rows at smaller ε. Even at ε = 0.0005 the least decided row still splits its mass (top share 0.63), because two targets cost almost the same for that point. The soft plan exposes ambiguity that a hard assignment hides.

## Choosing / trade-offs
- **ε relative to the costs.** What matters is `C/ε`. Normalise the cost (divide by its median or max) so one ε setting transfers between problems. Start around 1–5% of the typical cost and lower it only if the bias matters.
- **Plain vs log domain.** Plain is a pair of matrix-vector products per iteration, fastest on GPU. Log domain costs a `logsumexp` per iteration but never underflows. Use log domain unless ε is large and speed matters. ε-scaling (start large, decrease over iterations, warm-start the potentials) cuts the iteration count at small ε.
- **Debiasing.** Entropic OT between a distribution and itself is not 0. The Sinkhorn divergence `S(a,b) = OT_ε(a,b) − ½OT_ε(a,a) − ½OT_ε(b,b)` removes that bias and is the version to use as a loss.
- **Exact vs entropic.** For a one-off distance between moderate-size histograms, an exact LP solver (POT's `ot.emd`) is fine. Sinkhorn wins when it must run many times, in batches, on GPU, or inside backpropagation.
- **Unbalanced OT.** When the total masses differ or outliers should not be forced to move, relax the marginal constraints (POT's `ot.unbalanced`). Plain Sinkhorn insists that every unit of mass goes somewhere.
- **Library.** POT (`pip install pot`) provides `ot.sinkhorn`, `ot.sinkhorn2`, log-domain and stabilised variants, and exact solvers. GeomLoss provides GPU Sinkhorn divergences for PyTorch.

## Gotchas
- Reporting `⟨P, C⟩` from the entropic plan and calling it a Wasserstein distance: it is biased upward at any practical ε (21% above at ε = 0.1 here). State ε, or debias.
- A zero entry in `a` or `b` makes `log(a)` equal `−inf`. Drop empty bins or add a tiny floor before the log-domain version.
- Convergence checks on the wrong quantity. After the `u` update the rows match exactly by construction; check the column marginals (or the change in potentials).
- Cost scale: a cost in pixels squared with ε = 0.01 is effectively ε = 0, which underflows immediately. Normalise first.
- Memory: `C` and `P` are `n × m` dense. For tens of thousands of points use a library with online (KeOps) or multiscale implementations instead of materialising the matrix.
- Sinkhorn gives a balanced plan by design. If the task needs some sources to stay unmatched (false detections, new tracks), add dummy rows or columns or use unbalanced OT.

## Related
- [[hungarian-algorithm-tracking-from-scratch]] - the exact hard assignment that Sinkhorn relaxes.
- [[mixture-of-experts-from-scratch-numpy]] - balanced routing, one place a Sinkhorn step is used.
- [[contrastive-learning-infonce-from-scratch-numpy]] - self-supervised learning, where SwAV uses Sinkhorn for balanced cluster codes.
- [[gumbel-softmax-from-scratch-numpy]] - another differentiable relaxation of a discrete choice.
- [[split-conformal-prediction-from-scratch]] - another distribution-level tool for checking shift.

## References
- Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", NeurIPS: https://arxiv.org/abs/1306.0895
- Peyré & Cuturi (2019), "Computational Optimal Transport", Foundations and Trends in ML: https://arxiv.org/abs/1803.00567
- POT: Python Optimal Transport, documentation: https://pythonot.github.io/
