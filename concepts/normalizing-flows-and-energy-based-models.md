---
title: Normalizing flows and energy-based models
category: concepts
tags: [generative-models, normalizing-flows, realnvp, glow, flow-matching, energy-based-models, density-estimation, likelihood, pytorch]
use_cases:
  - "estimate an exact probability density for tabular or sensor data"
  - "score how likely a sample is, for anomaly or out-of-distribution detection"
  - "sample from a learned distribution and also compute its likelihood"
  - "understand flow matching, the training method behind recent image and audio generators"
  - "learn an unnormalized score or energy over inputs"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1605.08803
  - https://arxiv.org/abs/1912.02762
  - https://arxiv.org/abs/2210.02747
  - https://arxiv.org/abs/2101.03288
---

# Normalizing flows and energy-based models

## Summary
Normalizing flows build an invertible neural network that maps data to a
simple base distribution (a standard Gaussian); the change-of-variables
formula then gives an exact log-likelihood, and running the network backwards
gives samples. Energy-based models (EBMs) instead learn a scalar energy
`E(x)` where `p(x) ∝ exp(-E(x))`: very flexible, but the normalizing constant
is intractable, so training and sampling need tricks. Flow matching, a
successor training method, is now a mainstream alternative to diffusion for
image, video and audio generation.

## Key concepts
- **Change of variables:** if `z = f(x)` is invertible,
  `log p(x) = log p_z(f(x)) + log |det df/dx|`. Every flow layer must be
  invertible and have a cheap Jacobian determinant.
- **Coupling layers (RealNVP, Glow):** split the input; transform one half with
  scale/shift computed from the other half. Triangular Jacobian, so the
  log-det is just the sum of log-scales. Alternate which half is transformed.
- **Autoregressive flows (MAF, IAF):** each dimension conditioned on the
  previous ones; MAF has fast density, slow sampling; IAF the reverse.
- **Continuous flows / flow matching:** learn a velocity field that transports
  noise to data along a path; flow matching trains it by simple regression
  without simulating the ODE. Rectified flow is a straight-path variant.
- **Energy-based models:** any network outputs `E(x)`; low energy = likely.
  Trained with contrastive divergence, score matching or noise-contrastive
  estimation; sampled with Langevin dynamics (MCMC). Diffusion models learn
  the score `grad log p(x)`, closely related.
- **Trade-off triangle:** exact likelihood (flows), sample quality (diffusion,
  flow matching, GANs), flexibility (EBMs).

## When to use / scenarios
- Density estimation and anomaly scoring on low/medium-dimensional data
  (sensor readings, transactions) where you need a real likelihood
  ([[anomaly-detection]]).
- Simulation-based inference in science: a flow as a learned posterior over
  simulator parameters.
- Image/video/audio generation: through flow matching / rectified flow models
  (see [[diffusion-models]], [[image-generation-models]]).
- Variational inference: flows make a richer approximate posterior.
- NOT the first choice for high-quality image generation from a classic
  coupling flow (use diffusion/flow matching); NOT for plain tabular anomaly
  detection when an Isolation Forest already works. Flow likelihoods are also
  known to rate some out-of-distribution images as more likely than training
  data, so do not use raw likelihood alone for image OOD detection.

## Setup & code
```bash
pip install torch scikit-learn
```
A RealNVP flow on the 2D two-moons dataset, trained by exact maximum likelihood:
```python
import math
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

torch.manual_seed(0)
X = torch.tensor(make_moons(4000, noise=0.05, random_state=0)[0], dtype=torch.float32)
X = (X - X.mean(0)) / X.std(0)

class Coupling(nn.Module):
    """RealNVP affine coupling: transform one half conditioned on the other."""
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x):                       # data -> latent, returns log|det J|
        s, t = self.net(x * self.mask).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)     # bounded scales keep training stable
        t = t * (1 - self.mask)
        return x * torch.exp(s) + t, s.sum(1)

    def inverse(self, z):                       # latent -> data
        s, t = self.net(z * self.mask).chunk(2, dim=1)
        s = torch.tanh(s) * (1 - self.mask)
        t = t * (1 - self.mask)
        return (z - t) * torch.exp(-s)

masks = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])] * 3
flow = nn.ModuleList(Coupling(m) for m in masks)

def log_prob(x):
    logdet = torch.zeros(len(x))
    for layer in flow:
        x, ld = layer(x)
        logdet = logdet + ld
    base = -0.5 * (x**2).sum(1) - math.log(2 * math.pi)   # standard normal in 2D
    return base + logdet                                    # change of variables

def sample(n):
    z = torch.randn(n, 2)
    for layer in reversed(flow):
        z = layer.inverse(z)
    return z

opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
for step in range(3001):
    batch = X[torch.randint(len(X), (256,))]
    loss = -log_prob(batch).mean()              # exact maximum likelihood
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"step {step} nll={loss.item():.3f}")

with torch.no_grad():
    gauss_nll = -torch.distributions.MultivariateNormal(X.mean(0), torch.cov(X.T)).log_prob(X).mean()
    print(f"flow nll={-log_prob(X).mean():.3f}  single-gaussian nll={gauss_nll:.3f}")
    s = sample(2000)
    print(f"sample->data nearest dist={torch.cdist(s, X).min(1).values.mean():.3f}  "
          f"gaussian noise->data={torch.cdist(torch.randn(2000, 2), X).min(1).values.mean():.3f}")
```
Output (torch 2.13.0 CPU, under a minute): NLL falls from 3.039 to 1.218 over
3,000 steps; on the full data the flow scores NLL 1.221 against 2.726 for the
best single Gaussian. Samples land on the moons (mean distance to the nearest
training point 0.024, vs 0.256 for plain Gaussian noise). Running
`layer(x)` through all layers and then `layer.inverse` back reconstructs the
inputs to within `6e-07`: the model is exactly invertible.

For real work use a library: `normflows` or `zuko` (PyTorch) for flows,
`torchcfm` for conditional flow matching; Hugging Face `diffusers` ships flow
matching schedulers for image models.

## Choosing / trade-offs
- Need exact likelihood plus sampling, low/medium dimension: coupling or
  autoregressive flow (neural spline flows are a strong default).
- Need the best sample quality for images/audio/video: diffusion or flow
  matching; flow matching is simpler to train and samples in fewer steps.
- Need fast density evaluation only (scoring): MAF. Need fast sampling only: IAF.
- Need an unconstrained, composable score (combine constraints by adding
  energies): EBM, accepting slow MCMC sampling and finicky training.
- Plain tabular density/anomaly work: try a Gaussian mixture or KDE first
  ([[clustering]], [[anomaly-detection]]); reach for a flow only if they underfit.

## Gotchas
- Invertibility forces the latent to have the same dimension as the data:
  flows are memory-heavy for images and cannot compress like a VAE.
- Unbounded scales (`exp(s)`) blow up early in training; bound them (`tanh`,
  as above) or use spline transforms.
- Discrete data (pixel values, counts) must be dequantized (add uniform noise)
  or the likelihood goes to infinity on the data points.
- Compare likelihoods in bits per dimension, and only on the same
  preprocessing; a change of scaling changes the log-det term.
- EBM training with short-run MCMC can learn a sampler that works while the
  energy itself is meaningless; check the energies, not just the samples.

## Related
- [[diffusion-models]] - score-based generation, close relative of EBMs and flow matching.
- [[autoencoders-and-self-supervised-learning]] - VAEs, the other likelihood-based family.
- [[generative-adversarial-networks]] - sharp samples, no likelihood.
- [[anomaly-detection]] - density as an anomaly score.
- [[image-generation-models]] - production models built on flow matching.
- [[math-for-machine-learning]] - the change-of-variables and Jacobian background.

## References
- Dinh et al., Density estimation using Real NVP: https://arxiv.org/abs/1605.08803
- Papamakarios et al., Normalizing Flows for Probabilistic Modeling and Inference: https://arxiv.org/abs/1912.02762
- Lipman et al., Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Song and Kingma, How to Train Your Energy-Based Models: https://arxiv.org/abs/2101.03288
