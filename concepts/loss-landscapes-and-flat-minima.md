---
title: Loss landscapes, sharp vs flat minima and sharpness-aware training
category: concepts
tags: [loss-landscape, flat-minima, sharp-minima, hessian, sharpness, sam, mode-connectivity, edge-of-stability, pytorch, deep-learning-basics]
use_cases:
  - "explain why some minima of a neural net generalise better than others"
  - "measure the sharpness of a trained model (top Hessian eigenvalue)"
  - "plot a 1-D or 2-D slice of a network's loss surface"
  - "decide whether sharpness-aware minimisation (SAM) is worth trying"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1712.09913
  - https://arxiv.org/abs/1609.04836
  - https://arxiv.org/abs/1703.04933
  - https://arxiv.org/abs/2010.01412
  - https://arxiv.org/abs/2103.00065
  - https://arxiv.org/abs/1802.10026
  - https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
---

# Loss landscapes, sharp vs flat minima and sharpness-aware training

## Summary
A neural network's training loss is a function over millions of weights. Its shape around the solution, the *loss landscape*, helps explain what training finds. A **flat** minimum is one where the loss stays low when the weights move a little. A **sharp** minimum is one where the loss rises quickly. Flat minima are widely linked to better generalisation and robustness, large batches tend to find sharper minima, and methods such as SAM explicitly seek flat regions. The link is a useful heuristic, not a law: sharpness depends on how the weights are parameterised.

## Key concepts
- **Hessian and curvature.** The second derivatives of the loss. Its largest eigenvalue `λ_max` is the standard sharpness measure: the curvature along the steepest direction. It can be computed without forming the Hessian, by power iteration on Hessian-vector products.
- **Flat vs sharp.** Intuition: test data shifts the loss surface slightly, and a wide basin keeps the loss low under that shift. A bound in the PAC-Bayes style formalises "weights robust to noise generalise".
- **Batch size and learning rate.** SGD noise scales roughly with `lr / batch_size`. Small batches and large learning rates cannot settle in narrow basins, so they end up in flatter ones (Keskar et al. 2017) ([[batch-size-and-gradient-noise]]).
- **Edge of stability.** With full-batch GD at learning rate `η`, sharpness rises until `λ_max ≈ 2/η` and then hovers there. The step size itself caps how sharp a reachable minimum can be (Cohen et al. 2021).
- **Visualising.** Plot `L(θ + α·d)` along a random direction `d`, or `L(θ + α·d1 + β·d2)` on a plane. Each direction must be rescaled per layer or per filter to the weights' norm; otherwise scale-invariant layers make the plots meaningless (Li et al. 2018).
- **Reparameterisation caveat.** With ReLU and normalisation layers, the weights can be rescaled to make any minimum arbitrarily sharp without changing the function (Dinh et al. 2017). Compare sharpness only between comparable models.
- **Mode connectivity.** Different minima found from different seeds are usually joined by simple curved paths of low loss (Garipov et al. 2018). The landscape is less "many isolated valleys" than it looks.
- **SAM.** Sharpness-Aware Minimisation minimises the worst loss within a radius `ρ`: take a gradient step up to `θ + ρ·g/‖g‖`, compute the gradient there, and apply it at `θ`. It costs two forward/backward passes per step (Foret et al. 2021).

## When to use / scenarios
- Explaining a generalisation gap: a large-batch run with the same training loss but worse test accuracy than the small-batch run.
- Choosing a training recipe: flatness is one reason to prefer warmup + higher learning rate, weight averaging (SWA/EMA) and moderate batch sizes ([[learning-rate-schedules]]).
- Trying SAM when a vision or fine-tuning model overfits and you can afford about twice the compute per step. It often helps with label noise and small datasets.
- Research and debugging: checking whether a method truly changes curvature, or whether training is at the edge of stability (loss spikes, oscillation).
- Do NOT use sharpness as a model-selection metric across architectures or parameterisations. Use held-out validation ([[model-selection-and-comparison]]).

## Setup & code
`pip install torch scikit-learn`. Runs on CPU in about a minute. Two identical MLPs on two-moons, trained for the same ~7,600 SGD steps with batch 16 vs full batch 600. Sharpness is measured as `λ_max` by power iteration, plus a 1-D loss slice along a layer-normalised random direction.

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

torch.manual_seed(0)
X, y = make_moons(n_samples=1000, noise=0.25, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
X_tr, y_tr, X_te, y_te = X[:600], y[:600], X[600:], y[600:]


def make_model():
    return nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


def train(batch_size, lr, epochs):
    torch.manual_seed(0)
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            nn.functional.cross_entropy(model(X_tr[idx]), y_tr[idx]).backward()
            opt.step()
    return model


@torch.no_grad()
def loss_along_direction(model, alphas):
    """1-D slice: theta + a * d, with d a random direction rescaled per layer to the weights' norm."""
    params = [p.clone() for p in model.parameters()]
    g = torch.Generator().manual_seed(1)
    dirs = []
    for p in params:
        d = torch.randn(p.shape, generator=g)
        dirs.append(d * p.norm() / (d.norm() + 1e-10))
    out = []
    for a in alphas:
        for p, p0, d in zip(model.parameters(), params, dirs):
            p.copy_(p0 + a * d)
        out.append(nn.functional.cross_entropy(model(X_tr), y_tr).item())
    for p, p0 in zip(model.parameters(), params):
        p.copy_(p0)
    return out


def top_hessian_eigenvalue(model, iters=50):
    """Sharpness: largest Hessian eigenvalue, by power iteration on Hessian-vector products."""
    params = list(model.parameters())
    loss = nn.functional.cross_entropy(model(X_tr), y_tr)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    eig = 0.0
    for _ in range(iters):
        norm = torch.sqrt(sum((vi ** 2).sum() for vi in v))
        v = [vi / norm for vi in v]
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        eig = sum((h * vi).sum() for h, vi in zip(hv, v)).item()
        v = [h.detach() for h in hv]
    return eig


@torch.no_grad()
def acc(model, X_, y_):
    return (model(X_).argmax(1) == y_).float().mean().item()


alphas = [0.0, 0.25, 0.5]
for name, bs, lr, epochs in [("batch 16 ", 16, 0.05, 200), ("batch 600", 600, 0.05, 7600)]:
    m = train(bs, lr, epochs)
    curve = loss_along_direction(m, alphas)
    print(f"{name}: train acc {acc(m, X_tr, y_tr):.3f} test acc {acc(m, X_te, y_te):.3f} "
          f"top Hessian eig {top_hessian_eigenvalue(m):7.2f}  "
          f"loss at alpha {alphas}: " + ", ".join(f"{v:.3f}" for v in curve))
```

Output (torch 2.13 CPU, scikit-learn 1.9):
```
batch 16 : train acc 0.945 test acc 0.942 top Hessian eig    5.39  loss at alpha [0.0, 0.25, 0.5]: 0.121, 0.188, 0.379
batch 600: train acc 0.947 test acc 0.945 top Hessian eig   16.36  loss at alpha [0.0, 0.25, 0.5]: 0.119, 0.187, 0.407
```

Both runs reach the same training loss. The full-batch solution is about 3x sharper by `λ_max`, yet its test accuracy is the same. On an easy 2-D problem the sharpness gap does not turn into a generalisation gap, and a single random 1-D slice barely distinguishes the two. Random directions mostly miss the few sharp ones; `λ_max` finds them.

A minimal SAM step, for reference (one optimizer step, two backward passes):

```python
def sam_step(model, loss_fn, xb, yb, opt, rho=0.05):
    loss_fn(model(xb), yb).backward()
    grads = [p.grad for p in model.parameters()]
    scale = rho / (torch.sqrt(sum((g ** 2).sum() for g in grads)) + 1e-12)
    eps = []
    with torch.no_grad():
        for p, g in zip(model.parameters(), grads):
            e = g * scale
            p.add_(e)           # climb to the worst nearby point
            eps.append(e)
    opt.zero_grad()
    loss_fn(model(xb), yb).backward()   # gradient at the perturbed weights
    with torch.no_grad():
        for p, e in zip(model.parameters(), eps):
            p.sub_(e)           # return to the original weights
    opt.step()                  # apply the perturbed gradient there
    opt.zero_grad()
```

## Choosing / trade-offs
- **Measuring sharpness**: `λ_max` (power iteration, cheap and standard) vs the Hessian trace (Hutchinson's estimator, an average curvature) vs worst-case loss within a radius (what SAM optimises). All need a fixed parameterisation to be comparable.
- **SAM vs cheaper flatness tricks**: SAM roughly doubles step cost. SWA/EMA weight averaging, a larger learning rate with warmup, and a smaller batch are free or cheap and often get much of the benefit. `ρ` is one more hyperparameter (0.05 is the usual start).
- **Visualisation**: 1-D slices and 2-D planes are good for intuition and papers, but are low-dimensional projections of a million-dimensional surface. They can hide sharp directions and show convexity that is not there.

## Gotchas
- Comparing models at different training losses. A model stopped early is in a high, flat region; "flatter" then just means "less trained". Match training loss (or steps) before comparing.
- Plotting along unnormalised random directions. Batch norm and ReLU scale invariance make the same function look sharp or flat depending on the weights' scale.
- Treating flatness as causal for generalisation. It correlates in many settings, fails in others, and can be changed by reparameterisation without changing predictions.
- Hessian-vector products through batch norm in train mode mix in batch statistics. Put the model in `eval()` or use a fixed batch.
- Power iteration returns the eigenvalue largest in magnitude. Early in training it can be a large negative one (a saddle direction).
- SAM with batch norm: the perturbed pass updates running statistics twice. Many implementations disable the running-stat update in the second pass.

## Related
- [[generalization-in-deep-learning]] - the wider picture: double descent, implicit regularisation, benign overfitting.
- [[batch-size-and-gradient-noise]] - why the batch size changes which minima SGD finds.
- [[optimizers]] - SGD, Adam and how their dynamics differ.
- [[learning-rate-schedules]] - warmup, cycles and weight averaging that steer toward flat regions.
- [[regularization-in-deep-learning]] - other ways to improve generalisation.
- [[gradient-descent]] - curvature, condition number and step-size limits.

## References
- Li et al., Visualizing the Loss Landscape of Neural Nets (2018): https://arxiv.org/abs/1712.09913
- Keskar et al., On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (2017): https://arxiv.org/abs/1609.04836
- Dinh et al., Sharp Minima Can Generalize For Deep Nets (2017): https://arxiv.org/abs/1703.04933
- Foret et al., Sharpness-Aware Minimization for Efficiently Improving Generalization (2021): https://arxiv.org/abs/2010.01412
- Cohen et al., Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability (2021): https://arxiv.org/abs/2103.00065
- Garipov et al., Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs (2018): https://arxiv.org/abs/1802.10026
- PyTorch `torch.autograd.grad` (Hessian-vector products): https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
