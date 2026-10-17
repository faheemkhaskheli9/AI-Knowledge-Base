---
title: Flow matching and rectified flow (the objective behind current image and video generators)
category: concepts
tags: [flow-matching, rectified-flow, continuous-normalizing-flows, velocity-prediction, ode-sampling, euler, reflow, diffusion-models, generative-models, numpy, from-scratch]
use_cases:
  - "understand the training objective that current open image and video generators (Stable Diffusion 3, FLUX, Wan) use instead of DDPM noise prediction"
  - "implement flow matching end to end in NumPy: straight-line path, velocity regression, Euler sampling"
  - "explain why rectified flow and reflow let a generator sample in few steps"
  - "pick a sampler, step count and timestep schedule for a flow-matching model in diffusers"
status: draft
last_verified: 2026-10-17
sources:
  - https://arxiv.org/abs/2210.02747
  - https://arxiv.org/abs/2209.03003
  - https://arxiv.org/abs/2209.15571
  - https://arxiv.org/abs/2403.03206
  - https://arxiv.org/abs/2412.06264
  - https://github.com/facebookresearch/flow_matching
  - https://github.com/atong01/conditional-flow-matching
  - https://github.com/black-forest-labs/flux2
  - https://github.com/Wan-Video/Wan2.1
---

# Flow matching and rectified flow (the objective behind current image and video generators)

## Summary
Flow matching trains a network `v_θ(x, t)` to predict a velocity field that
carries samples from a simple source (Gaussian noise) to the data along a
chosen path; sampling integrates the ODE `dx/dt = v_θ(x, t)` from `t = 0` to
`t = 1`. With the straight-line path `x_t = (1 − t)·x₀ + t·x₁` (rectified flow)
the training target is just `x₁ − x₀`, so the loss is a plain regression with
no noise schedule to design and no simulation during training. It is the
objective behind Stable Diffusion 3 ("rectified flow transformers"), FLUX.2
("flow matching transformer") and Wan 2.1 ("Flow Matching framework"). The
NumPy demo below trains one on an 8-mode 2-D ring in about 20 s on a CPU and
shows the few-step gain from one reflow round.

## Key concepts
- **Probability path.** Pick a way to interpolate each noise sample `x₀ ~ N(0, I)`
  with a data sample `x₁`. Rectified flow / linear path: `x_t = (1 − t)x₀ + t x₁`.
  Other choices (variance-preserving, cosine) recover diffusion-style paths.
- **Conditional flow matching (CFM) loss.** `E_{t, x₀, x₁} ‖v_θ(x_t, t) − (x₁ − x₀)‖²`.
  Regressing on the per-pair velocity gives the same gradient as regressing on
  the intractable marginal velocity field (Lipman et al.), so this simple loss
  is enough.
- **Sampling is an ODE solve.** Euler: `x ← x + Δt · v_θ(x, t)`. Heun or other
  higher-order solvers trade more network calls per step for fewer steps.
  There is no injected noise; the map from noise to sample is deterministic.
- **Why paths curve.** Pairs are drawn independently, so paths from different
  pairs cross; the learned field averages them and the resulting trajectories
  bend. Curved trajectories need many Euler steps.
- **Reflow.** Sample `x₁' = ODE(x₀)` with the trained model, then retrain on the
  coupled pairs `(x₀, x₁')`. Those paths no longer cross, so the new field is
  nearly straight and one or two Euler steps work. Usually followed by
  distillation for production few-step models.
- **Relation to DDPM.** Gaussian diffusion and flow matching with a Gaussian
  path are the same family seen through different parameterisations:
  ε-prediction, x₀-prediction, v-prediction and flow velocity convert into each
  other given the path coefficients. Differences in practice come from the path,
  the timestep weighting and the sampler, not from a different model class.
- **Timestep sampling.** Uniform `t` is the baseline; SD3 reports gains from
  sampling middle timesteps more often (logit-normal), and inference schedulers
  often apply a "shift" that spends more steps near the noise end for high
  resolutions.
- **Conditioning and guidance** work as in diffusion: text or image embeddings
  as inputs, classifier-free guidance on the predicted velocity.

## When to use / scenarios
- Understanding or modifying a current open image/video model: their schedulers
  (`FlowMatchEulerDiscreteScheduler` in diffusers), "shift" settings and step
  counts only make sense in flow-matching terms. See [[image-generation-models]],
  [[video-generation-models]].
- Training a new continuous generator (images, audio, molecules, robot actions,
  latent codes) from scratch: flow matching is a simpler default than DDPM,
  with fewer schedule hyperparameters.
- Few-step generation: rectified flow + reflow, then distillation.
- Interviews: "flow matching vs diffusion", "what does the network predict",
  "why is sampling deterministic", "why does reflow straighten paths".
- Not for: exact likelihoods cheaply (use a normalizing flow, see
  [[normalizing-flows-and-energy-based-models]]; a flow-matching ODE gives a
  likelihood only through an expensive trace integral), discrete data such as
  text without a discrete variant (see [[diffusion-language-models]]), or
  production image generation from scratch in NumPy (use pretrained models via
  `diffusers`, see [[diffusion-models]]).

## Setup & code
`pip install numpy`. Trains two small MLPs (flow matching, then one reflow) and
samples with Euler steps; about 20 s on one CPU core.

```python
import numpy as np

rng = np.random.default_rng(0)

# target: 8 Gaussians on a ring of radius 3; source: standard normal
angles = np.arange(8) * np.pi / 4
centers = 3 * np.c_[np.cos(angles), np.sin(angles)]
def sample_data(n):
    return centers[rng.integers(8, size=n)] + rng.normal(0, 0.1, (n, 2))

def temb(t):  # sinusoidal time features, t in [0, 1]
    f = np.arange(1, 9) * np.pi
    return np.c_[np.sin(np.outer(t, f)), np.cos(np.outer(t, f))]

# velocity MLP v_theta(x_t, t): [x, temb(t)] -> 128 -> 128 -> 2, ReLU
H, D_IN = 128, 2 + 16
def init():
    return {"W1": rng.normal(0, np.sqrt(2 / D_IN), (D_IN, H)), "b1": np.zeros(H),
            "W2": rng.normal(0, np.sqrt(2 / H), (H, H)), "b2": np.zeros(H),
            "W3": rng.normal(0, np.sqrt(1 / H), (H, 2)), "b3": np.zeros(2)}

def forward(p, x, t):
    z0 = np.c_[x, temb(t)]
    a1 = np.maximum(z0 @ p["W1"] + p["b1"], 0)
    a2 = np.maximum(a1 @ p["W2"] + p["b2"], 0)
    return a2 @ p["W3"] + p["b3"], (z0, a1, a2)

def grads(p, cache, dout):
    z0, a1, a2 = cache
    g = {"W3": a2.T @ dout, "b3": dout.sum(0)}
    d2 = (dout @ p["W3"].T) * (a2 > 0)
    g["W2"], g["b2"] = a1.T @ d2, d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (a1 > 0)
    g["W1"], g["b1"] = z0.T @ d1, d1.sum(0)
    return g

def train(pair_sampler, steps=4000, batch=256, lr=2e-3):
    p = init()
    m = {k: np.zeros_like(v) for k, v in p.items()}
    s = {k: np.zeros_like(v) for k, v in p.items()}
    for i in range(1, steps + 1):
        x0, x1 = pair_sampler(batch)
        t = rng.uniform(0, 1, batch)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1     # straight-line path
        target = x1 - x0                                 # its constant velocity
        v, cache = forward(p, xt, t)
        g = grads(p, cache, 2 * (v - target) / batch)    # gradient of the squared error
        lr_i = lr * 0.5 * (1 + np.cos(np.pi * i / steps))  # cosine decay
        for k in p:                                      # Adam
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            p[k] -= lr_i * (m[k] / (1 - 0.9 ** i)) / (np.sqrt(s[k] / (1 - 0.999 ** i)) + 1e-8)
    return p

def sample(p, x0, n_steps):  # Euler integration of dx/dt = v_theta(x, t), t: 0 -> 1
    x, dt = x0.copy(), 1.0 / n_steps
    for k in range(n_steps):
        x = x + dt * forward(p, x, np.full(len(x), k * dt))[0]
    return x

def near_mode(x):  # share of points within 0.5 of a ring centre
    d = np.linalg.norm(x[:, None, :] - centers[None], axis=2).min(1)
    return (d < 0.5).mean()

# 1) flow matching on independent (noise, data) pairs
p1 = train(lambda n: (rng.normal(size=(n, 2)), sample_data(n)))
z = rng.normal(size=(2000, 2))
print("flow matching: Euler steps -> share of samples near a mode")
for n in (1, 2, 4, 8, 32, 100):
    print(f"  {n:3d} steps: {near_mode(sample(p1, z, n)):.3f}")
x_fm = sample(p1, z, 100)
counts = np.bincount(np.linalg.norm(x_fm[:, None] - centers[None], axis=2).argmin(1), minlength=8)
print("  per-mode share (100 steps):", np.round(counts / counts.sum(), 3))

# 2) reflow: retrain on (noise, model sample) pairs; the coupling no longer crosses
pool0 = rng.normal(size=(20000, 2))
pool1 = sample(p1, pool0, 100)
def coupled(n):
    idx = rng.integers(len(pool0), size=n)
    return pool0[idx], pool1[idx]
p2 = train(coupled)
print("after one reflow")
for n in (1, 2, 4, 8, 32):
    print(f"  {n:3d} steps: {near_mode(sample(p2, z, n)):.3f}")

def straightness(p, x0, n=100):  # |x1 - x0|^2 / path energy; 1.0 = perfectly straight
    x, dt, energy = x0.copy(), 1.0 / n, 0.0
    for k in range(n):
        v = forward(p, x, np.full(len(x), k * dt))[0]
        energy += (v ** 2).sum(1).mean() * dt
        x = x + dt * v
    return ((x - x0) ** 2).sum(1).mean() / energy

print(f"straightness: flow matching {straightness(p1, z):.3f}, reflow {straightness(p2, z):.3f}")
assert near_mode(sample(p1, z, 100)) > 0.8 and near_mode(sample(p2, z, 1)) > 0.5
```

One run (seed 0, single-threaded BLAS; other BLAS builds can differ in the
last digits): with 100 Euler steps 90.9% of samples land within 0.5 of a mode
and every mode gets 11.5–13.2% of them (ideal 12.5%). The first model is
useless at 1 step (0.0%) and needs about 8 steps to pass 80%. After one reflow
the 1-step share jumps to 70.2% and the straightness ratio goes from 0.60 to
1.00, at the cost of a slightly lower ceiling with many steps (84.7% at 32
steps), because the second model inherits the first one's errors. That is the
trade production systems manage with reflow plus distillation.

Same idea with real tools:

<!-- skip-check: needs torch, diffusers and a multi-GB model download -->
```python
# Same call as in image-generation-models; the step count follows the model card.
# Fewer steps than the card recommends = faster but worse, unless the model is distilled.
import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                          torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
print(type(pipe.scheduler).__name__)   # the ODE sampler the pipeline was configured with
pipe("a red bicycle leaning on a brick wall", height=1024, width=1024,
     guidance_scale=1.0, num_inference_steps=4).images[0].save("bike.png")
```

To train your own: `pip install flow_matching` (Meta's PyTorch library from the
"Flow Matching Guide and Code"; its repo badge says CC BY-NC 4.0, so check
before commercial use) or `pip install torchcfm` (MIT, includes OT-CFM and
other couplings).

## Choosing / trade-offs
- **Flow matching vs DDPM (ε-prediction).** Flow matching: one regression target,
  no β schedule, deterministic ODE sampling that solvers can take in large
  steps. DDPM: ancestral sampling with noise, many well-studied schedules and
  samplers. For a new model, flow matching with a linear path is the common
  default; quality differences mostly come from timestep weighting, data and
  architecture.
- **Step count vs quality.** More Euler steps approach the true ODE solution;
  higher-order solvers (Heun) cost 2 network calls per step. Distilled or
  reflowed models sample in 1–4 steps but can lose diversity or detail.
- **Reflow vs distillation.** Reflow keeps the same objective and straightens
  paths but needs a large generated pair set and inherits the teacher's errors;
  distillation (consistency, adversarial) targets few-step quality directly.
- **Independent vs optimal-transport coupling.** Pairing noise and data within
  a minibatch by optimal transport (OT-CFM, `torchcfm`) gives straighter paths
  from the start at the cost of a per-batch assignment (see
  [[sinkhorn-optimal-transport-from-scratch]]); it helps more in low dimension
  than for large images.

## Gotchas
- Time direction conventions differ: some papers and codebases (SD3, many
  diffusers schedulers) put data at `t = 0` and noise at `t = 1` and integrate
  backwards. Mixing conventions flips the sign of the velocity and gives garbage.
- Sampling with fewer steps than the model was tuned for degrades quality
  quickly unless the model is distilled; read the model card's step count.
- The "shift" / resolution-dependent timestep schedule is part of the model's
  recipe; changing resolution without the matching shift washes out detail.
- Classifier-free guidance on velocity behaves like guidance on ε, but useful
  guidance scales differ per model; do not copy values across models.
- A low training loss does not mean good samples: the loss floor is set by the
  crossing paths, not by zero. Judge by samples or a distribution metric.
- Reflow pairs must come from the ODE sampler with enough steps; pairs from a
  1-step sampler teach the next model the first model's errors.

## Related
- [[diffusion-models]] - the production overview; flow matching is the training
  objective of many models it covers.
- [[ddpm-diffusion-from-scratch-numpy]] - the same 2-D toy setup with DDPM
  ε-prediction and ancestral sampling, to compare loop for loop.
- [[normalizing-flows-and-energy-based-models]] - flow matching trains a
  continuous normalizing flow without simulating it.
- [[image-generation-models]] - FLUX.2 and Stable Diffusion 3.5, flow-matching
  models in practice.
- [[video-generation-models]] - video generators built on latent flow matching.
- [[sinkhorn-optimal-transport-from-scratch]] - the OT coupling used by OT-CFM.
- [[knowledge-distillation-and-compression]] - distillation, the usual step
  after reflow for few-step models.
- [[diffusion-language-models]] - discrete-data counterpart.

## References
- Lipman et al., Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- Liu, Gong, Liu, Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow: https://arxiv.org/abs/2209.03003
- Albergo and Vanden-Eijnden, Building Normalizing Flows with Stochastic Interpolants: https://arxiv.org/abs/2209.15571
- Esser et al., Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3): https://arxiv.org/abs/2403.03206
- Lipman et al., Flow Matching Guide and Code: https://arxiv.org/abs/2412.06264 and https://github.com/facebookresearch/flow_matching
- Tong et al., torchcfm / conditional flow matching: https://github.com/atong01/conditional-flow-matching
- FLUX.2 README ("flow matching transformer"): https://github.com/black-forest-labs/flux2
- Wan 2.1 README ("Flow Matching framework"): https://github.com/Wan-Video/Wan2.1
