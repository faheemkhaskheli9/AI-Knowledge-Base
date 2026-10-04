---
title: Bayesian deep learning and uncertainty estimation
category: concepts
tags: [uncertainty, bayesian-neural-networks, mc-dropout, deep-ensembles, laplace-approximation, epistemic, aleatoric, calibration, temperature-scaling, pytorch]
use_cases:
  - "make a neural network report how sure it is about each prediction"
  - "separate 'noisy data' uncertainty from 'model has not seen this' uncertainty"
  - "calibrate an overconfident deep classifier so 0.9 means right 90% of the time"
  - "pick which unlabelled samples to label next based on model uncertainty"
  - "predict a value with error bars from a neural network regressor"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1506.02142
  - https://arxiv.org/abs/1612.01474
  - https://arxiv.org/abs/1706.04599
  - https://arxiv.org/abs/1703.04977
  - https://arxiv.org/abs/2106.14806
---

# Bayesian deep learning and uncertainty estimation

## Summary
A standard neural network gives one answer and, for classifiers, a softmax score that is usually overconfident. Uncertainty estimation adds a measure of how much to trust each prediction. Bayesian deep learning treats the weights as a distribution instead of a point, and practical approximations make that cheap: **deep ensembles** (train 5 models, look at their spread) are the most reliable, **MC dropout** (keep dropout on at test time) is nearly free, and the **Laplace approximation** adds uncertainty to an already-trained network. For a calibrated confidence score alone, **temperature scaling** fixes most overconfidence in one parameter.

## Key concepts
- **Two kinds of uncertainty.**
  - **Aleatoric:** noise in the data itself (blurry image, inherently random outcome). More data does not reduce it. Model it by predicting a variance (regression) or reading the predictive entropy.
  - **Epistemic:** the model does not know, because the input is unlike training data or data is scarce. More data reduces it. Shows up as *disagreement* between plausible models.
- **Bayesian neural network (BNN).** Put a prior on weights, compute the posterior given data, predict by averaging over it. Exact inference is intractable; everything below is an approximation.
- **Deep ensembles** (Lakshminarayanan 2017). Train M ≈ 5 networks from different random seeds; average predictions; their variance is epistemic uncertainty. Strongest practical baseline; cost is M× training and inference.
- **MC dropout** (Gal & Ghahramani 2016). Leave dropout active at inference, run T ≈ 20-50 forward passes, use mean and variance. Only works if the model has dropout; uncertainty is often underestimated.
- **Laplace approximation.** Fit a Gaussian around the trained weights using the curvature (Hessian); "last-layer Laplace" is cheap and post-hoc (`laplace-torch` library).
- **Variational inference / Bayes by Backprop.** Learn a mean and variance per weight. Principled but doubles parameters and is harder to train; rarely beats ensembles in practice.
- **Heteroscedastic regression.** Network outputs mean *and* variance, trained with Gaussian negative log-likelihood (`nn.GaussianNLLLoss`); captures aleatoric noise that varies by input.
- **Calibration.** Expected calibration error (ECE), reliability diagrams, negative log-likelihood, Brier score. **Temperature scaling** (Guo 2017): divide logits by a scalar T fitted on validation data; fixes confidence without changing accuracy.
- **Uncertainty decomposition for classifiers.** Total = entropy of the mean prediction; aleatoric ≈ mean entropy of each member; epistemic = their difference (mutual information, "BALD").

## When to use / scenarios
- Medical imaging and diagnosis aids: show confidence, refer uncertain cases to a clinician ([[healthcare]]).
- Autonomous systems and robotics: slow down or hand over when epistemic uncertainty spikes.
- Active learning: label the samples with highest epistemic uncertainty (BALD) ([[semi-supervised-and-active-learning]]).
- Scientific and engineering surrogates: neural regressors that need error bars ([[physics-informed-neural-networks]]).
- Credit, pricing and forecasting where a probability or interval drives a decision ([[finance]]).
- NOT the tool when you need a *guaranteed* coverage rate: wrap any model with [[conformal-prediction-and-uncertainty]]. NOT a full OOD detector by itself: combine with [[out-of-distribution-detection]] scores. Small tabular data with a need for exact Bayesian uncertainty: Gaussian processes ([[bayesian-and-gaussian-processes]]).

## Setup & code
```bash
pip install torch numpy
```
1-D regression with a gap in the training data. A deep ensemble of mean-variance networks gives aleatoric noise (per-model variance) and epistemic uncertainty (spread of means); MC dropout is shown for comparison. Epistemic uncertainty should be low inside the data and high in the gap and beyond:
```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(0); rng = np.random.default_rng(0)
x = np.concatenate([rng.uniform(-4, -1, 200), rng.uniform(1, 4, 200)])   # gap in (-1, 1)
y = np.sin(x) + rng.normal(0, 0.1 + 0.1 * np.abs(x), x.shape)            # noise grows with |x|
X = torch.tensor(x, dtype=torch.float32)[:, None]
Y = torch.tensor(y, dtype=torch.float32)[:, None]

def make(p_drop=0.0):
    return nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p_drop),
                         nn.Linear(64, 2))                  # outputs: mean, log-variance

def train(model, epochs=1500):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.GaussianNLLLoss()
    for _ in range(epochs):
        out = model(X)
        loss = loss_fn(out[:, :1], Y, out[:, 1:].exp())
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# Deep ensemble: 5 seeds.
ensemble = []
for s in range(5):
    torch.manual_seed(s); ensemble.append(train(make()).eval())

grid = torch.tensor([[-3.0], [0.0], [3.0], [6.0]])          # in-data, gap, in-data, far away
with torch.no_grad():
    outs = torch.stack([m(grid) for m in ensemble])          # (M, n, 2)
mu, var = outs[..., 0], outs[..., 1].exp()
aleatoric = var.mean(0)
epistemic = mu.var(0)
for g, a, e in zip(grid[:, 0], aleatoric, epistemic):
    print(f"x={g:5.1f}  aleatoric={a:.3f}  epistemic={e:.3f}")

# MC dropout: one model, dropout left on at test time.
torch.manual_seed(0); mc = train(make(p_drop=0.1)).train()   # .train() keeps dropout active
with torch.no_grad():
    mc_mu = torch.stack([mc(grid)[:, 0] for _ in range(50)])
print("MC-dropout epistemic std:", mc_mu.std(0).numpy().round(3))
```
For classifiers, the cheapest win is temperature scaling on a validation set: learn one scalar `T` by minimising NLL of `logits / T`, then report `softmax(logits / T)`.

## Choosing / trade-offs
- **Just need calibrated confidence?** Temperature scaling. One parameter, no retraining, keeps accuracy.
- **Need reliable epistemic uncertainty and can pay 5× compute?** Deep ensembles. Best quality, simplest to get right; distil the ensemble into one model if inference cost matters ([[knowledge-distillation-and-compression]]).
- **Model already trained and has dropout?** MC dropout; cheap but tends to underestimate uncertainty and costs T forward passes.
- **Large pretrained model, post-hoc?** Last-layer Laplace; one extra pass over training data, no retraining.
- **Need guaranteed coverage (intervals that hold 90% of the time)?** Conformal prediction on top of any of the above.
- **Full BNN (variational, MCMC/HMC):** research-grade; worth it only for small networks where principled posteriors matter.

## Gotchas
- Softmax probability is not uncertainty: modern networks are confidently wrong, especially far from training data. Measure calibration (ECE, reliability diagram) before trusting it.
- Forgetting `model.train()` for MC dropout (or leaving BatchNorm in train mode while doing it) gives either zero variance or corrupted statistics; enable only the dropout layers if the model uses BatchNorm.
- Ensemble members trained from the same checkpoint or with the same seed agree too much; vary initialisation and data order.
- Temperature fitted on the training set does nothing useful; fit it on held-out data, and refit after any distribution shift.
- Uncertainty estimates are themselves uncalibrated under shift; evaluate on shifted data, not only on the test split.
- Predicted variance collapsing to tiny values early in training with `GaussianNLLLoss`; warm up on MSE first or clamp the log-variance.
- High uncertainty is only useful if something acts on it: define the abstain / escalate threshold and measure accuracy at a given coverage.

## Related
- [[conformal-prediction-and-uncertainty]] - distribution-free coverage guarantees for any model.
- [[bayesian-and-gaussian-processes]] - exact Bayesian uncertainty for small data.
- [[out-of-distribution-detection]] - scoring inputs unlike the training data.
- [[model-evaluation-and-metrics]] - calibration metrics and reliability plots.
- [[semi-supervised-and-active-learning]] - uncertainty-driven labelling.
- [[deep-learning-training]] - dropout, ensembles and regularisation basics.

## References
- Gal & Ghahramani, Dropout as a Bayesian Approximation (2016): https://arxiv.org/abs/1506.02142
- Lakshminarayanan et al., Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles (2017): https://arxiv.org/abs/1612.01474
- Guo et al., On Calibration of Modern Neural Networks (2017): https://arxiv.org/abs/1706.04599
- Kendall & Gal, What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? (2017): https://arxiv.org/abs/1703.04977
- Daxberger et al., Laplace Redux (2021), `laplace-torch`: https://arxiv.org/abs/2106.14806
