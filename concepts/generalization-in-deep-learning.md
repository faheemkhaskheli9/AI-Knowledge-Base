---
title: Generalization in deep learning (memorization, double descent, grokking, lottery tickets, flat minima)
category: concepts
tags: [deep-learning, generalization, overfitting, memorization, double-descent, benign-overfitting, grokking, lottery-ticket-hypothesis, flat-minima, sharpness, sam, implicit-regularization, pytorch]
use_cases:
  - "understand why a huge network that can fit random labels still works on new data"
  - "decide whether to make a model bigger or smaller when validation error is bad"
  - "explain a validation curve that gets worse then better again as model size or training time grows"
  - "judge whether a model has memorised its training data or learned the task"
  - "decide how long to keep training after training loss reaches zero"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1611.03530
  - https://arxiv.org/abs/1812.11118
  - https://arxiv.org/abs/1912.02292
  - https://arxiv.org/abs/2201.02177
  - https://arxiv.org/abs/1803.03635
  - https://arxiv.org/abs/2010.01412
  - https://docs.pytorch.org/docs/stable/nn.html#module-torch.nn.utils
---

# Generalization in deep learning (memorization, double descent, grokking, lottery tickets, flat minima)

## Summary
Classical learning theory says a model with more parameters than data points should overfit. Deep networks routinely have far more, can fit pure noise, and still generalise well on real data. Several empirical findings explain parts of this: the training procedure (SGD, initialisation, architecture) prefers simple solutions (implicit regularisation); test error can follow a *double descent* curve, peaking when the model is just big enough to interpolate the training set and falling again beyond it; networks can *grok*, generalising long after they first memorise; large networks contain small sub-networks that train as well (lottery tickets); and solutions in flat regions of the loss tend to generalise better than sharp ones. This file turns those findings into practical decisions about model size, training length and regularisation.

## Key concepts
- **Memorisation capacity.** A plain MLP or CNN reaches 100% training accuracy on randomly shuffled labels (Zhang et al. 2017). So capacity alone cannot explain generalisation; the data's structure and the optimiser's bias do.
- **Implicit regularisation.** Gradient descent from a small initialisation tends toward low-norm, simple solutions; small batches add noise that favours wide minima. Architecture adds priors (convolution = locality and translation equivariance).
- **Interpolation threshold.** The model size (or training time) at which the model can just fit the training data exactly. Near it the fit is forced through every noisy point and test error spikes.
- **Double descent.** Test error vs size: U-shaped in the classical regime, peak at the interpolation threshold, then a second descent in the over-parameterised regime (Belkin et al. 2019; Nakkiran et al. 2019). Also appears vs training epochs (epoch-wise) and vs dataset size (more data can briefly hurt near the threshold). Label noise makes the peak much larger; tuned regularisation flattens it.
- **Benign overfitting.** Interpolating noisy labels can still give near-optimal test error when the model is very over-parameterised, because the noise is absorbed in many unimportant directions.
- **Grokking.** On small algorithmic datasets, validation accuracy jumps from chance to near-perfect long after training accuracy hit 100%, typically with weight decay (Power et al. 2022). Lesson: "train loss is zero" does not mean learning has stopped.
- **Lottery ticket hypothesis.** A dense network contains a sparse sub-network that, reset to its original initial weights, trains to similar accuracy (Frankle & Carbin 2019). Found by iterative magnitude pruning; motivates pruning and explains part of why over-parameterisation helps optimisation.
- **Flat vs sharp minima.** Minima where the loss changes slowly around the weights tend to generalise better. Sharpness-Aware Minimization (SAM) explicitly seeks them; large batches and high learning rates without warmup tend toward sharp ones.
- **Shortcut learning.** Generalising on the test split does not mean learning the intended concept: models pick spurious cues (background, watermark, hospital ID) that hold in-distribution and fail elsewhere ([[out-of-distribution-detection]]).

## When to use / scenarios
- Choosing model size: in deep learning, "bigger plus regularisation and enough training" usually beats "small so it cannot overfit" ([[pretraining-and-scaling-laws]]).
- Reading a validation curve that worsens and then recovers with more epochs or parameters: likely epoch- or model-wise double descent, not a bug ([[debugging-neural-network-training]]).
- Deciding whether to stop at the first validation plateau: on small, clean, algorithmic or structured tasks with weight decay, extra training can still help; on noisy-label data, early stopping is usually right.
- Auditing memorisation: privacy (training data extraction, membership inference) and benchmark contamination ([[federated-learning-and-differential-privacy]]).
- Pruning and compression decisions: lottery-ticket results justify training big and pruning after ([[knowledge-distillation-and-compression]]).
- NOT a replacement for a held-out test set and proper evaluation: these phenomena describe tendencies, not guarantees for your data ([[model-evaluation-and-metrics]]).

## Setup & code
```bash
pip install torch numpy
```
Two small experiments on CPU: (1) an MLP fits random labels perfectly but learns nothing transferable; (2) min-norm least squares on p random ReLU features shows model-wise double descent with n = 100 training points:
```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)
rng = np.random.default_rng(0)

# 1) Memorisation: a big enough MLP fits random labels perfectly (Zhang et al. 2017).
X = torch.randn(1000, 20)
w = torch.randn(20)
y_real = (X @ w > 0).long()
y_rand = torch.randint(0, 2, (1000,))
X_te = torch.randn(1000, 20)
y_te = (X_te @ w > 0).long()

for name, y in [("real labels", y_real), ("random labels", y_rand)]:
    net = nn.Sequential(nn.Linear(20, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(1500):                    # full-batch, no regularisation
        opt.zero_grad()
        nn.functional.cross_entropy(net(X), y).backward()
        opt.step()
    with torch.no_grad():
        tr = (net(X).argmax(1) == y).float().mean().item()
        te = (net(X_te).argmax(1) == y_te).float().mean().item()
    print(f"{name:13s} train acc {tr:.3f}  test acc {te:.3f}")

# 2) Double descent: min-norm least squares on p random ReLU features, n = 100 samples.
n, d = 100, 10
beta = rng.normal(size=d)
Xtr, Xte = rng.normal(size=(n, d)), rng.normal(size=(2000, d))
ytr = Xtr @ beta + rng.normal(0, 0.5, n)
yte = Xte @ beta
W = rng.normal(size=(d, 2000)) / np.sqrt(d)
for p in [10, 50, 90, 100, 110, 200, 500, 2000]:
    Ftr, Fte = np.maximum(Xtr @ W[:, :p], 0), np.maximum(Xte @ W[:, :p], 0)
    coef = np.linalg.pinv(Ftr) @ ytr          # least squares, minimum-norm when p > n
    print(f"p={p:5d}  test MSE {np.mean((Fte @ coef - yte) ** 2):10.2f}")
```
Output with torch 2.13.0 (CPU) and numpy 2.5.1:
```text
real labels   train acc 1.000  test acc 0.959
random labels train acc 1.000  test acc 0.510
p=   10  test MSE       2.17
p=   50  test MSE       0.61
p=   90  test MSE       2.52
p=  100  test MSE      21.55
p=  110  test MSE       4.15
p=  200  test MSE       1.09
p=  500  test MSE       0.83
p= 2000  test MSE       0.64
```
Same network, same 100% training accuracy, chance-level test accuracy on random labels. Test error peaks sharply at p = n = 100 (the interpolation threshold) and falls again as features grow far past n. Adding a ridge penalty (`np.linalg.solve(F.T@F + lam*I, F.T@y)`) removes most of the peak.

Lottery-ticket-style magnitude pruning uses `torch.nn.utils.prune` (for example `prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=0.8)`), then rewinding the surviving weights to their early-training values and retraining.

## Choosing / trade-offs
- **Bigger vs smaller.** If the model sits near the interpolation threshold (training error just reaches zero, validation error spikes), go clearly bigger with regularisation or clearly smaller; the middle is the worst place.
- **More data can hurt briefly** near the threshold; if adding data makes things worse, check model size relative to the new dataset.
- **Early stopping vs long training.** Early stopping is the safe default with noisy labels. With weight decay and clean, structured data, monitor longer before concluding the model has peaked ([[regularization-in-deep-learning]]).
- **SAM** costs about two forward/backward passes per step; worth it for vision models trained from scratch on limited data, rarely for LLM fine-tuning.
- **Weight decay and data augmentation** are the cheapest ways to flatten the double-descent peak and push toward simpler solutions ([[data-augmentation]], [[optimizers]]).

## Gotchas
- "Training accuracy is 100%, so it overfits" is not a diagnosis; compare validation error to a baseline and look at the trend.
- Random-label results show capacity, not that your model memorised your data; test memorisation with held-out duplicates, canaries or membership-inference checks.
- Double-descent peaks are easy to miss with coarse size sweeps or with regularisation already tuned; they are also easy to mistake for a training bug.
- Grokking results come from small algorithmic tasks; do not assume a stalled real-world model will suddenly generalise if you just train longer.
- Lottery tickets at large scale need rewinding to early-training weights, not to initialisation; winning tickets also do not usually speed up dense GPU training without sparse kernels.
- Sharpness measures depend on parameter scale (re-parameterising a ReLU net changes sharpness without changing the function); compare them only within one architecture.

## Related
- [[ml-fundamentals]] - bias-variance and classical overfitting.
- [[regularization-in-deep-learning]] - weight decay, dropout, early stopping.
- [[pretraining-and-scaling-laws]] - how loss falls with model and data size.
- [[debugging-neural-network-training]] - telling real bugs from odd but normal curves.
- [[knowledge-distillation-and-compression]] - pruning and small models from large ones.
- [[optimizers]] - SGD noise, batch size and SAM.
- [[out-of-distribution-detection]] - when in-distribution generalisation is not enough.

## References
- Zhang et al., Understanding Deep Learning Requires Rethinking Generalization (ICLR 2017): https://arxiv.org/abs/1611.03530
- Belkin et al., Reconciling Modern Machine Learning Practice and the Bias-Variance Trade-off (PNAS 2019): https://arxiv.org/abs/1812.11118
- Nakkiran et al., Deep Double Descent (ICLR 2020): https://arxiv.org/abs/1912.02292
- Power et al., Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (2022): https://arxiv.org/abs/2201.02177
- Frankle & Carbin, The Lottery Ticket Hypothesis (ICLR 2019): https://arxiv.org/abs/1803.03635
- Foret et al., Sharpness-Aware Minimization (ICLR 2021): https://arxiv.org/abs/2010.01412
- PyTorch, `torch.nn.utils.prune`: https://docs.pytorch.org/docs/stable/nn.html#module-torch.nn.utils
