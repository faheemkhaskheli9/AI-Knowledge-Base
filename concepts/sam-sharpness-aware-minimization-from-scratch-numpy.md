---
title: Sharpness-aware minimisation (SAM) from scratch in NumPy (ascent step, flat minima, Hessian sharpness)
category: concepts
tags: [sam, sharpness-aware-minimization, flat-minima, sharpness, hessian, power-iteration, generalization, label-noise, optimizer, numpy, from-scratch]
use_cases:
  - "understand how SAM finds flatter minima and what its two gradient passes per step buy"
  - "decide whether SAM is worth the 2x training cost for a small or noisy-label dataset"
  - "measure the sharpness (top Hessian eigenvalue) of a trained network without forming the Hessian"
status: stable
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/2010.01412
  - https://arxiv.org/abs/2102.11600
---

# Sharpness-aware minimisation (SAM) from scratch in NumPy (ascent step, flat minima, Hessian sharpness)

## Summary
SAM (Foret et al. 2021) trains for weights whose whole neighbourhood has low loss, not just the weights themselves. Each step first moves the weights uphill by a radius `rho` along the normalised gradient, computes the gradient at that worst-case point, and then applies that gradient to the original weights. That costs two forward/backward passes per step and steers training towards flat minima, which tend to generalise better, especially with label noise. This file trains a small tanh MLP in NumPy on a two-moons task with 20% flipped labels. It compares SGD and SAM at several `rho`, measures sharpness as the top Hessian eigenvalue by power iteration, and checks against early stopping as the cheap alternative.

## Key concepts
- **Objective.** `min_w max_{||e|| <= rho} L(w + e)`. A first-order approximation gives the worst perturbation in closed form: `e* = rho * g / ||g||`.
- **Update.** `g = ∇L(w)`, `e = rho * g/||g||`, `g_sam = ∇L(w + e)`, `w ← w - lr * g_sam`. The perturbation is thrown away after the second gradient; only the base optimiser state changes. Any optimiser (SGD, momentum, Adam) can sit underneath.
- **Why it flattens.** `∇L(w + e) ≈ g + rho * H g/||g||`, so SAM adds a term that pushes away from directions of high curvature `H`.
- **Sharpness.** The largest eigenvalue `λ_max` of the Hessian of the training loss. Power iteration on Hessian-vector products estimates it. Each `Hv` here is a central finite difference of two gradients, so the Hessian is never formed.
- **`rho`.** The neighbourhood radius in weight space. The paper's typical values are 0.05–0.1 for image classification from scratch. Too large and training underfits, because every step optimises a pessimistic point.
- **m-sharpness.** SAM computes `e` per mini-batch (or per device shard), not over the full dataset. Smaller batches for the perturbation tend to work better, so its effect is not just "minimise full-batch sharpness".

## When to use / scenarios
- Training from scratch on small or noisy-label datasets (medical images, user-labelled data, weak labels) where overfitting the noise is the main failure. SAM was strongly robust to label noise in the original paper.
- Image classifiers and ViTs trained without huge pre-training, where SAM-family methods gave consistent accuracy gains in the literature.
- Fine-tuning when the budget allows 2x compute and a fraction of a point matters.
- Not first: try early stopping, weight decay, augmentation and label cleaning ([[regularization-in-deep-learning]]) first. They are cheaper, and in the run below early stopping almost matches SAM.
- Not for LLM pre-training at scale. The 2x cost is rarely spent there; optimiser choice ([[muon-optimizer-from-scratch-numpy]], [[optimizers]]) and data matter more.

## Setup & code
NumPy only, about 40 seconds on a laptop CPU. The network is a 2-64-1 tanh MLP with all weights in one flat vector, so SAM's perturbation is one vector operation. Training uses 200 points, 20% of them with flipped labels. Each setting is averaged over 3 seeds; test accuracy is measured against the clean labels.

<!-- check-timeout: 600 -->
```python
import numpy as np

H = 64                                            # hidden units of a 2-H-1 tanh MLP


def make_data(n, rng, noise=0.2, flip=0.2):
    """Two interleaved moons; `flip` of the training labels are flipped to give the net something to overfit."""
    t = rng.uniform(0, np.pi, n)
    y = rng.integers(0, 2, n)
    X = np.c_[np.cos(t) - y, np.sin(t) * (1 - 2 * y) + 0.5 * y] + noise * rng.standard_normal((n, 2))
    yn = np.where(rng.random(n) < flip, 1 - y, y)
    return X, yn.astype(float), y.astype(float)


def unpack(w):
    W1, b1 = w[:2 * H].reshape(2, H), w[2 * H:3 * H]
    W2, b2 = w[3 * H:4 * H], w[4 * H]
    return W1, b1, W2, b2


def loss_grad(w, X, y):
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(X @ W1 + b1)
    z = h @ W2 + b2
    p = 1 / (1 + np.exp(-z))
    loss = np.mean(np.logaddexp(0, z) - y * z)          # stable binary cross-entropy
    dz = (p - y) / len(y)
    dh = np.outer(dz, W2) * (1 - h ** 2)
    g = np.concatenate([(X.T @ dh).ravel(), dh.sum(0), h.T @ dz, [dz.sum()]])
    return loss, g


def accuracy(w, X, y):
    W1, b1, W2, b2 = unpack(w)
    return np.mean(((np.tanh(X @ W1 + b1) @ W2 + b2) > 0) == y)


def sharpness(w, X, y, iters=50, eps=1e-4, seed=0):
    """Top Hessian eigenvalue by power iteration on finite-difference Hessian-vector products."""
    v = np.random.default_rng(seed).standard_normal(w.size)
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        Hv = (loss_grad(w + eps * v, X, y)[1] - loss_grad(w - eps * v, X, y)[1]) / (2 * eps)
        lam = v @ Hv
        v = Hv / np.linalg.norm(Hv)
    return lam


def train(X, y, rho, steps=3000, lr=0.5, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    w = np.concatenate([rng.standard_normal(2 * H) * 1.0, np.zeros(H), rng.standard_normal(H) / np.sqrt(H), [0.0]])
    for _ in range(steps):
        idx = rng.choice(len(y), batch, replace=False)
        _, g = loss_grad(w, X[idx], y[idx])
        if rho > 0:                                    # SAM: climb to the worst nearby point, take its gradient
            e = rho * g / (np.linalg.norm(g) + 1e-12)
            _, g = loss_grad(w + e, X[idx], y[idx])
        w -= lr * g
    return w


rng = np.random.default_rng(1)
Xtr, ytr, _ = make_data(200, rng)
Xte, _, yte = make_data(5000, rng, flip=0.0)

# Gradient check against central finite differences
w0 = np.random.default_rng(2).standard_normal(4 * H + 1) * 0.5
_, g = loss_grad(w0, Xtr, ytr)
fd = np.array([(loss_grad(w0 + 1e-6 * e, Xtr, ytr)[0] - loss_grad(w0 - 1e-6 * e, Xtr, ytr)[0]) / 2e-6
               for e in np.eye(w0.size)[:20]])
assert np.allclose(g[:20], fd, atol=1e-6)
print("analytic gradient matches finite differences")

print("run                     train loss  train acc  test acc  sharpness")
runs = [(f"SGD  rho=0    {s:5d} steps", 0.0, s) for s in (5000, 20000, 40000)]
runs += [(f"SAM  rho={r:<4} 20000 steps", r, 20000) for r in (0.05, 0.1, 0.2, 0.5)]
for name, rho, steps in runs:
    res = []
    for seed in range(3):
        w = train(Xtr, ytr, rho, steps=steps, lr=0.1, seed=seed)
        res.append((loss_grad(w, Xtr, ytr)[0], accuracy(w, Xtr, ytr), accuracy(w, Xte, yte), sharpness(w, Xtr, ytr)))
    L, a_tr, a_te, lam = np.mean(res, 0)
    print(f"{name}  {L:9.3f}  {a_tr:9.3f}  {a_te:8.3f}  {lam:9.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
analytic gradient matches finite differences
run                     train loss  train acc  test acc  sharpness
SGD  rho=0     5000 steps      0.487      0.787     0.917       3.66
SGD  rho=0    20000 steps      0.412      0.812     0.823       3.69
SGD  rho=0    40000 steps      0.376      0.817     0.823       3.47
SAM  rho=0.05 20000 steps      0.416      0.807     0.842       2.51
SAM  rho=0.1  20000 steps      0.420      0.807     0.854       1.87
SAM  rho=0.2  20000 steps      0.431      0.808     0.884       1.16
SAM  rho=0.5  20000 steps      0.498      0.783     0.928       0.49
```

How to read it:
- With 20% flipped labels, a classifier that ignores the noise scores at most about 80% on the noisy training set (a little less, since the moons overlap). Training accuracy above that means the net is fitting the noise. Plain SGD at 20k–40k steps does that (81–82% train) and its clean test accuracy drops from 91.7% at 5k steps to 82.3%.
- SAM at the same 20k steps reaches about the same training loss for small `rho`, but its sharpness drops steadily with `rho` (3.69 → 0.49), and clean test accuracy rises with it (82.3% → 92.8%). Flatter solution, less memorised noise.
- Compute is not equal: SAM at 20k steps costs 40k gradient evaluations. SGD at 40k steps (the same budget) does no better than at 20k. The gain comes from the objective, not from extra compute.
- Early stopping (SGD at 5k steps, 91.7%) gets most of the way for a quarter of SAM's compute. On this toy problem the honest conclusion is "SAM ≈ a well-tuned early stop, but less sensitive to when you stop".
- Large `rho` (0.5) fits the training set least (78.3%, the highest training loss). It happened to help here because the noise is the main problem; on clean data, sweep `rho` and expect the best value to be small.

## Choosing / trade-offs
- **Cost.** 2x gradient computations per step. Variants reduce it: apply SAM every k steps, perturb only part of the batch, or use the efficient variants (ESAM, LookSAM) from the literature.
- **SAM vs ASAM.** Plain SAM's ball is not scale-invariant: rescaling ReLU layers changes sharpness without changing the function. Adaptive SAM (Kwon et al. 2021) scales the perturbation per parameter by `|w|` and is usually more robust to the `rho` choice.
- **Batch norm.** In PyTorch implementations, the second forward pass updates BN running statistics again. Common practice is to disable BN momentum for the ascent pass.
- **Where it pays.** Biggest wins are with label noise, small data or no pre-training. With large pre-trained models and clean data, gains are small and early stopping or weight decay may be the better use of compute.

## Gotchas
- Normalise by the gradient norm of the whole parameter vector (or use ASAM's per-parameter scaling). Per-layer normalisation is a different method.
- Use the same mini-batch for both gradient passes. A fresh batch for the second pass turns SAM into a noisy look-ahead method.
- Weight decay belongs in the base optimiser step, not inside the perturbed loss, or it is applied at the wrong point.
- Sharpness numbers are only comparable between the same architecture and parameterisation. Rescaling weights can make any minimum look arbitrarily sharp ([[loss-landscapes-and-flat-minima]]).
- Power iteration finds the eigenvalue largest in magnitude. Near saddle points that can be a large negative curvature, so check the sign of `v @ Hv`.
- Finite-difference Hessian-vector products need `eps` large enough to avoid round-off (1e-4 here in float64). In float32 frameworks use autodiff double-backward instead.

## Related
- [[loss-landscapes-and-flat-minima]] - the theory behind sharp vs flat minima, edge of stability and reparameterisation caveats.
- [[optimizers-from-scratch-numpy]] - the base optimisers SAM wraps.
- [[regularization-in-deep-learning]] - cheaper alternatives to try first (early stopping, weight decay, augmentation).
- [[label-noise-and-data-cleaning]] - fixing the labels instead of making training robust to them.
- [[generalization-in-deep-learning]] - why flat minima are linked to generalisation, and the limits of that link.

## References
- Foret et al. (2021), "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR: https://arxiv.org/abs/2010.01412
- Kwon et al. (2021), "ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks", ICML: https://arxiv.org/abs/2102.11600
