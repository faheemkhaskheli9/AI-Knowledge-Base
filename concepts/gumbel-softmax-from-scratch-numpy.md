---
title: Gumbel-softmax from scratch in NumPy (Gumbel-max trick, temperature, straight-through, bias vs variance)
category: concepts
tags: [gumbel-softmax, concrete-distribution, gumbel-max-trick, reparameterization-trick, straight-through-estimator, reinforce, discrete-latent-variables, gradient-estimation, temperature, numpy, from-scratch]
use_cases:
  - "backpropagate through a discrete choice (a category, a token, a routing decision) in a neural network"
  - "choose between Gumbel-softmax and REINFORCE for a discrete latent variable or a sampled action"
  - "pick the temperature and hard/soft mode for torch.nn.functional.gumbel_softmax"
  - "sample from a categorical distribution with argmax over perturbed logits"
status: stable
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/1611.01144
  - https://arxiv.org/abs/1611.00712
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html
---

# Gumbel-softmax from scratch in NumPy (Gumbel-max trick, temperature, straight-through, bias vs variance)

## Summary
Sampling a category is not differentiable, so a network that makes a discrete choice (a VAE with categorical latents, a hard attention or routing decision, a generator that emits tokens) gets no gradient through it. Gumbel-softmax (Jang et al. 2016; the same idea is the "Concrete" distribution of Maddison et al. 2016) replaces the sample with `softmax((logits + Gumbel noise) / τ)`: a point on the simplex that approaches a one-hot sample as `τ → 0` and is differentiable in the logits. This file checks the Gumbel-max trick it rests on, then measures the gradient it gives against the exact gradient and against REINFORCE. The result is the trade-off in one table: low temperature means little bias and high variance, high temperature the reverse.

## Key concepts
- **Gumbel-max trick.** If `g_i = −log(−log u_i)` with `u_i ~ U(0, 1)`, then `argmax_i (θ_i + g_i)` is an exact sample from `softmax(θ)`. The randomness moves into noise that does not depend on `θ`, as in the Gaussian reparameterisation trick of a VAE.
- **Relaxation.** Replace the `argmax` with `softmax((θ + g)/τ)`. The output `y` is differentiable, with `∂y/∂θ = (diag(y) − y yᵀ)/τ`.
- **Temperature τ.** Small τ: `y` is nearly one-hot, close to a real sample, but the gradient is large and noisy. Large τ: `y` is smooth and near uniform, the gradient is stable but points at the wrong objective. τ is often annealed from about 1 down to 0.1–0.5 during training.
- **Straight-through (ST) Gumbel-softmax.** Use the one-hot `argmax` in the forward pass and the soft `y`'s gradient in the backward pass (`hard=True` in PyTorch, implemented as `y_hard − y.detach() + y`). Downstream code sees real discrete values. The gradient is the same biased surrogate.
- **REINFORCE (score function).** `f(z) ∇ log p(z)` is unbiased for any `f`, even non-differentiable, but high variance. A baseline subtracted from `f(z)` reduces the variance and keeps it unbiased. See [[reinforce-policy-gradient-from-scratch]].

## When to use / scenarios
- Differentiable discrete latent variables: categorical VAEs, discrete VQ-style codes trained without a codebook loss, learned discrete masks or feature selection.
- Discrete decisions inside a network trained end to end: hard attention, routing (see [[mixture-of-experts-from-scratch-numpy]] for top-k routing, which takes a different route), neural architecture search (DARTS-style relaxations), learned data augmentation policies.
- Sampling from a categorical with only `argmax` and uniform noise: the Gumbel-max trick also gives Gumbel-top-k, sampling k items without replacement in one pass.
- Not when the downstream reward is non-differentiable (a BLEU score, an environment, a human rating): there is no `∂f/∂y` to use, so use REINFORCE or PPO ([[ppo-from-scratch]]).
- Not for text generation with a frozen tokenizer and long sequences: relaxed tokens fed into an embedding are mixtures the model never saw in training, and errors compound per step.

## Setup & code
NumPy only, runs in under a second. The objective is `E[c · z]` for a one-hot sample `z ~ softmax(θ)`, a linear cost per category, chosen because its exact gradient `p ⊙ (c − p·c)` is known in closed form. Each estimator is averaged over 200,000 samples to measure its bias (distance of the mean from the exact gradient) and its variance per sample.

```python
import numpy as np

rng = np.random.default_rng(0)
K = 5
theta = np.array([1.0, 0.5, 0.0, -0.5, -1.0])   # logits
c = np.array([3.0, 1.0, 0.0, 2.0, 5.0])         # cost of each category; minimise E[c . z]


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def gumbel(shape):
    u = rng.random(shape)
    return -np.log(-np.log(u + 1e-20) + 1e-20)


p = softmax(theta)

# 1) Gumbel-max trick: argmax(theta + g) is an exact sample from softmax(theta)
n = 200_000
idx = np.argmax(theta + gumbel((n, K)), axis=1)
freq = np.bincount(idx, minlength=K) / n
print("softmax p     ", np.round(p, 4))
print("Gumbel-max freq", np.round(freq, 4))

# 2) Gradient of E[c . z] w.r.t. theta. Exact: p * (c - p.c)
true = p * (c - p @ c)


def reinforce(n):
    z = np.argmax(theta + gumbel((n, K)), axis=1)
    onehot = np.eye(K)[z]
    return c[z][:, None] * (onehot - p)            # f(z) * d log p(z) / d theta


def reinforce_baseline(n):
    z = np.argmax(theta + gumbel((n, K)), axis=1)
    onehot = np.eye(K)[z]
    b = p @ c                                       # ideal constant baseline (in practice: running mean)
    return (c[z] - b)[:, None] * (onehot - p)


def gumbel_softmax(n, tau):
    y = softmax((theta + gumbel((n, K))) / tau)     # relaxed sample on the simplex
    return y * (c - (y @ c)[:, None]) / tau         # d(c . y)/d theta, by the softmax Jacobian


def summary(name, g):
    m = g.mean(0)
    bias = np.linalg.norm(m - true) / np.linalg.norm(true)
    var = g.var(0).sum()
    cos = m @ true / np.linalg.norm(m) / np.linalg.norm(true)
    print(f"{name:24s} rel.bias {bias:5.3f}  cos {cos:6.3f}  total var/sample {var:8.3f}")


print("true grad     ", np.round(true, 3))
N = 200_000
summary("REINFORCE", reinforce(N))
summary("REINFORCE + baseline", reinforce_baseline(N))
for tau in (0.1, 0.5, 1.0, 2.0, 5.0):
    summary(f"Gumbel-softmax tau={tau}", gumbel_softmax(N, tau))

# 3) Temperature vs. how one-hot the relaxed samples are
for tau in (0.1, 0.5, 1.0, 5.0):
    y = softmax((theta + gumbel((20_000, K))) / tau)
    print(f"tau={tau}: mean max(y) {y.max(1).mean():.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
softmax p      [0.4287 0.26   0.1577 0.0956 0.058 ]
Gumbel-max freq [0.4278 0.2599 0.1568 0.0963 0.0592]
true grad      [ 0.417 -0.267 -0.32  -0.003  0.172]
REINFORCE                rel.bias 0.007  cos  1.000  total var/sample    3.587
REINFORCE + baseline     rel.bias 0.004  cos  1.000  total var/sample    1.232
Gumbel-softmax tau=0.1   rel.bias 0.008  cos  1.000  total var/sample    5.473
Gumbel-softmax tau=0.5   rel.bias 0.127  cos  0.997  total var/sample    0.616
Gumbel-softmax tau=1.0   rel.bias 0.333  cos  0.978  total var/sample    0.146
Gumbel-softmax tau=2.0   rel.bias 0.586  cos  0.925  total var/sample    0.021
Gumbel-softmax tau=5.0   rel.bias 0.816  cos  0.848  total var/sample    0.001
tau=0.1: mean max(y) 0.951
tau=0.5: mean max(y) 0.765
tau=1.0: mean max(y) 0.589
tau=5.0: mean max(y) 0.282
```

How to read it:
- The Gumbel-max frequencies match `softmax(θ)` to sampling noise (±0.001 at 200k draws). The trick is exact.
- REINFORCE is unbiased (its residual 0.007 is Monte Carlo noise) and a baseline cuts its variance about threefold, from 3.6 to 1.2.
- Gumbel-softmax bias grows steadily with τ, from 0.008 at τ = 0.1 to 0.82 at τ = 5. Its gradient is the gradient of the *relaxed* objective `E[c · y_τ]`, which only equals the real one as τ → 0.
- Its variance moves the other way: 5.5 at τ = 0.1 (worse than plain REINFORCE), 0.15 at τ = 1, about 0.001 at τ = 5. At τ = 0.5 it has 2× lower variance than REINFORCE with a baseline and still points the right way (cosine 0.997). That middle region is where it is useful.
- Even at τ = 5 the cosine is 0.85: the biased gradient still points roughly downhill, which is why high-temperature relaxations can train at all.

## Choosing / trade-offs
- **Gumbel-softmax vs REINFORCE.** Gumbel-softmax needs a differentiable downstream `f` and accepts bias for lower variance. REINFORCE works with any `f` and is unbiased, but needs a baseline (and often many samples) to be usable. With a non-differentiable reward, REINFORCE is the only option of the two.
- **Temperature.** Start around τ = 1 and anneal towards 0.1–0.5. The variance rises steeply near zero (9× from τ = 0.5 to τ = 0.1 above), so very low fixed temperatures train worse, not better.
- **Soft vs hard (straight-through).** Use `hard=True` when downstream code needs real one-hot values (an index lookup, a discrete action, consistency between training and inference). Use soft samples when a mixture is acceptable, e.g. a weighted sum of embeddings. For a linear downstream loss, as here, ST and soft give identical gradients; they differ only in the forward value.
- **Alternatives.** REINFORCE with a learned baseline or leave-one-out baselines (RLOO), control variates that combine both (REBAR, RELAX), or straight-through without Gumbel noise for deterministic `argmax` (as in VQ-VAE codebooks).

## Gotchas
- Clip `u` away from 0 and 1 before the double log, or `log(0)` produces `inf` and NaN gradients. PyTorch samples the noise as `-Exponential(1).log()`, which avoids this.
- Evaluate with real discrete samples (or `argmax`), not with the relaxed `y`. A model can learn to exploit soft mixtures that never occur at inference, which shows up as a gap between training and evaluation metrics.
- Low τ with half-precision training overflows: logits divided by 0.1 grow 10×. Compute the softmax in float32.
- The bias is towards the relaxed objective, not random. If the downstream function behaves very differently on mixtures than on one-hot inputs, a high-τ gradient can consistently push the wrong way. Check against REINFORCE on a small batch.
- Gumbel noise in the forward pass makes training stochastic. Fix it (or use `argmax`) when checking gradients with finite differences.

## Related
- [[reinforce-policy-gradient-from-scratch]] - the unbiased score-function estimator used as the baseline here.
- [[vae-from-scratch-numpy]] - the Gaussian reparameterisation trick that Gumbel-softmax extends to categories.
- [[mixture-of-experts-from-scratch-numpy]] - discrete routing, done with top-k and auxiliary losses instead.
- [[decoding-strategies-from-scratch-numpy]] - temperature on logits at inference time, the same knob.
- [[backpropagation-and-autograd]] - why sampling and `argmax` pass no gradient.

## References
- Jang, Gu & Poole (2016), "Categorical Reparameterization with Gumbel-Softmax": https://arxiv.org/abs/1611.01144
- Maddison, Mnih & Teh (2016), "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables": https://arxiv.org/abs/1611.00712
- PyTorch, `torch.nn.functional.gumbel_softmax`: https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html
