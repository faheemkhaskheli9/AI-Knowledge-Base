---
title: DDPM diffusion model from scratch in NumPy (noise schedule, epsilon-prediction, ancestral sampling)
category: concepts
tags: [diffusion-models, ddpm, denoising, noise-schedule, epsilon-prediction, generative-models, score-matching, mlp, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a denoising diffusion probabilistic model (DDPM) end to end in NumPy"
  - "train a noise-prediction network and sample new points from pure noise"
  - "check that a diffusion model covers all modes of a toy distribution"
  - "explain the forward process, the training loss and the sampling loop in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2006.11239
  - https://arxiv.org/abs/1503.03585
  - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
---

# DDPM diffusion model from scratch in NumPy (noise schedule, epsilon-prediction, ancestral sampling)

## Summary
A diffusion model learns to generate data by learning to undo noise. The fixed forward process adds a little Gaussian noise over `T` steps until data become pure noise. A network is trained to predict the noise that was added to a noisy sample at a random step. Sampling then starts from pure noise and repeatedly removes the predicted noise. The NumPy version below does the whole thing on a 2-D mixture of four Gaussians: a 2-layer MLP with a sinusoidal time embedding, hand-written backprop and Adam, `T = 100` steps. After 8,000 training steps, 97.6% of generated points land within 0.5 of a true mode (real data: 99.7%), and the four modes get 24–26% of samples each. That even split is the property GANs often lack, where samples collapse onto a few modes ([[gan-from-scratch-numpy]]). The same sampling loop without the trained network blows up to points 21 units away.

## Key concepts
- **Forward process.** `q(x_t | x_{t−1}) = N(√(1−β_t) x_{t−1}, β_t I)` with a fixed schedule `β_1 … β_T`. Nothing is learned here.
- **Closed form.** With `ᾱ_t = Π (1 − β_s)`: `x_t = √ᾱ_t x_0 + √(1 − ᾱ_t) ε`, `ε ~ N(0, I)`. Training jumps straight to any step without simulating the chain.
- **ε-prediction.** A network `ε_θ(x_t, t)` predicts the noise `ε`. The "simple" DDPM loss is just `‖ε − ε_θ(x_t, t)‖²` at uniformly sampled `t`.
- **Time conditioning.** One network serves every noise level; it is told `t` through a sinusoidal embedding (as in transformers).
- **Reverse step (ancestral sampling).** `x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t)) / √α_t + √β_t z`, with no noise `z` on the last step.
- **Score view.** `ε_θ / √(1 − ᾱ_t)` estimates `−∇ log p(x_t)`. Diffusion is denoising score matching over many noise levels, and the reverse process follows the score toward high-density regions.
- **Schedule end point.** `ᾱ_T` must be close to 0 so `x_T` really is `N(0, I)`, the distribution sampling starts from.

## When to use / scenarios
- Learning: the full training and sampling loop behind Stable Diffusion, DALL·E-style image models, audio and video diffusion, in about 80 lines with no framework.
- Interviews: "what does the network predict", "why sample `t` uniformly", "why do diffusion models not mode-collapse", "why is sampling slow and how do DDIM/distillation speed it up", "diffusion vs GAN vs VAE".
- Toy experiments before scaling: test a schedule, parameterisation (ε vs x_0 vs v) or guidance trick on 2-D data where you can see the result.
- Not for: real image or audio generation from scratch in NumPy (use `diffusers` with pretrained models, see [[diffusion-models]]), real-time generation without step reduction, or discrete data without a discrete-diffusion variant.

## Setup & code
`pip install numpy`. Trains and samples in about 15 seconds on CPU.

```python
import numpy as np

rng = np.random.default_rng(0)

# data: 2-D mixture of 4 Gaussians on a ring
centers = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]], float)
def sample_data(n):
    return centers[rng.integers(4, size=n)] + rng.normal(0, 0.15, (n, 2))

# forward (noising) process: linear beta schedule, closed form q(x_t | x_0)
T = 100
betas = np.linspace(1e-4, 0.1, T)
alphas = 1 - betas
abar = np.cumprod(alphas)

# epsilon-prediction MLP: input [x_t, time embedding] -> predicted noise
def temb(t):
    f = np.arange(1, 9)
    return np.c_[np.sin(np.outer(t / T, f) * np.pi), np.cos(np.outer(t / T, f) * np.pi)]

D_in, Hd = 2 + 16, 128
params = {"W1": rng.normal(0, np.sqrt(2 / D_in), (D_in, Hd)), "b1": np.zeros(Hd),
          "W2": rng.normal(0, np.sqrt(2 / Hd), (Hd, Hd)), "b2": np.zeros(Hd),
          "W3": rng.normal(0, np.sqrt(1 / Hd), (Hd, 2)), "b3": np.zeros(2)}

def forward(p, x):
    h1 = np.maximum(0, x @ p["W1"] + p["b1"])
    h2 = np.maximum(0, h1 @ p["W2"] + p["b2"])
    return h2 @ p["W3"] + p["b3"], (x, h1, h2)

def backward(p, cache, dout):
    x, h1, h2 = cache
    g = {"W3": h2.T @ dout, "b3": dout.sum(0)}
    dh2 = dout @ p["W3"].T * (h2 > 0)
    g["W2"], g["b2"] = h1.T @ dh2, dh2.sum(0)
    dh1 = dh2 @ p["W2"].T * (h1 > 0)
    g["W1"], g["b1"] = x.T @ dh1, dh1.sum(0)
    return g

# Adam
m = {k: np.zeros_like(v) for k, v in params.items()}; v_ = {k: np.zeros_like(v) for k, v in params.items()}
lr, b1, b2, B = 1e-3, 0.9, 0.999, 256
for step in range(1, 8001):
    x0 = sample_data(B)
    t = rng.integers(0, T, B)
    eps = rng.normal(size=(B, 2))
    xt = np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps   # jump straight to step t
    pred, cache = forward(params, np.c_[xt, temb(t)])
    loss = ((pred - eps) ** 2).mean()                                            # simple DDPM loss
    g = backward(params, cache, 2 * (pred - eps) / pred.size)
    for k in params:
        m[k] = b1 * m[k] + (1 - b1) * g[k]; v_[k] = b2 * v_[k] + (1 - b2) * g[k] ** 2
        params[k] -= lr * (m[k] / (1 - b1**step)) / (np.sqrt(v_[k] / (1 - b2**step)) + 1e-8)
    if step in (1, 1000, 4000, 8000):
        print(f"step {step:5d} | noise-prediction MSE {loss:.3f}")

def sample(n, use_model=True):
    x = rng.normal(size=(n, 2))                                                  # start from pure noise
    for t in range(T - 1, -1, -1):
        eps_hat = forward(params, np.c_[x, temb(np.full(n, t))])[0] if use_model else 0
        x = (x - betas[t] / np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(alphas[t])
        if t > 0:
            x += np.sqrt(betas[t]) * rng.normal(size=(n, 2))                     # sigma_t^2 = beta_t
    return x

def report(name, x):
    d = np.linalg.norm(x[:, None] - centers[None], axis=-1)
    near = d.min(1)
    counts = np.bincount(d.argmin(1), minlength=4) / len(x)
    print(f"{name:22s}: within 0.5 of a mode {np.mean(near < 0.5):.3f} | mean dist to nearest mode {near.mean():.3f} | "
          f"mode shares {np.round(counts, 2).tolist()}")

print(f"abar[T-1] = {abar[-1]:.4f} (x_T is ~pure noise)")
report("real data", sample_data(4000))
report("pure Gaussian noise", rng.normal(size=(4000, 2)))
report("reverse w/o model", sample(4000, use_model=False))
report("DDPM samples", sample(4000))
```

Output (numpy 2.5):
```
step     1 | noise-prediction MSE 1.831
step  1000 | noise-prediction MSE 0.328
step  4000 | noise-prediction MSE 0.303
step  8000 | noise-prediction MSE 0.261
abar[T-1] = 0.0056 (x_T is ~pure noise)
real data             : within 0.5 of a mode 0.997 | mean dist to nearest mode 0.189 | mode shares [0.24, 0.25, 0.25, 0.26]
pure Gaussian noise   : within 0.5 of a mode 0.080 | mean dist to nearest mode 1.141 | mode shares [0.25, 0.26, 0.25, 0.25]
reverse w/o model     : within 0.5 of a mode 0.001 | mean dist to nearest mode 21.512 | mode shares [0.26, 0.26, 0.24, 0.24]
DDPM samples          : within 0.5 of a mode 0.976 | mean dist to nearest mode 0.214 | mode shares [0.25, 0.26, 0.24, 0.25]
```

The schedule ends at `ᾱ_T = 0.0056`, so only 7% (`√ᾱ_T`) of the original signal survives to step `T` and starting the sampler from `N(0, I)` is justified. The noise-prediction loss falls from 1.8 to 0.26. It never reaches 0, because at high noise levels the noise cannot be told apart from the data. The samples are what count. 97.6% of generated points fall within 0.5 of one of the four modes, close to real data (99.7%) and far from the noise they started as (8%). The mean distance to the nearest mode, 0.21, is near the real data's 0.19. The mode shares are 24–26%, so all four modes are covered evenly. The "reverse w/o model" row runs the same update with the predicted noise set to 0. Dividing by `√α_t` at each of 100 steps inflates the points to 21 units out. The trained network is what turns that loop into a generator.

## Choosing / trade-offs
- **Schedule.** Linear `β` (as here, and the original DDPM) works; the cosine schedule (Improved DDPM) destroys information more evenly and is the common default for images. Check that `ᾱ_T ≈ 0` either way.
- **Parameterisation.** ε-prediction (here) is the classic; v-prediction (`v = √ᾱ ε − √(1−ᾱ) x_0`) is more stable at high noise and is used in many newer models; x_0-prediction is the third option. Flow matching / rectified flow trains a velocity field instead and is the basis of several recent image models.
- **Sampler and step count.** Ancestral DDPM needs all `T` steps. DDIM, DPM-Solver and similar ODE solvers reuse the same trained network with 10–50 steps; distillation and consistency models go to 1–4.
- **Network.** An MLP is enough for 2-D points. Images need a U-Net or a diffusion transformer (DiT), with time (and text) conditioning injected into every block.
- **Guidance.** Classifier-free guidance (train with and without the condition, then extrapolate between the two predictions) is how text-to-image models trade diversity for prompt adherence.
- **Latent diffusion.** Diffuse in an autoencoder's latent space rather than in pixels, as Stable Diffusion does, to cut compute a lot ([[vae-from-scratch-numpy]]).
- **Diffusion vs GAN vs VAE.** Diffusion gives the best quality and mode coverage with stable training, at the cost of many network calls per sample. GANs sample in one pass but are unstable and drop modes. VAEs are fast and stable but blurrier.

## Gotchas
- `ᾱ_T` must be near 0. If the schedule leaves real signal at step `T` (a first draft of this code had `ᾱ_T = 0.078`), training and sampling start from different distributions and samples come out biased.
- Index the schedule consistently: 0-based arrays vs the 1-based `t` in the paper is the most common off-by-one. Use the same `t` for the embedding in training and sampling.
- Do not add noise on the final step (`t = 0`). Doing so leaves samples blurred by `√β_1`.
- The training loss alone says little about sample quality. Always sample and measure: mode coverage here, FID for images.
- Normalise data to roughly unit scale (images to [−1, 1]). The forward process assumes data of about the same scale as the noise.
- Without the time embedding, one network has to denoise every noise level the same way, and it fails.
- Sampling cost is `T` network calls per sample. Budget for it, or use a fast sampler.

## Related
- [[diffusion-models]] - libraries (`diffusers`), pretrained models, guidance and serving.
- [[gan-from-scratch-numpy]] - the adversarial alternative, and its mode collapse for comparison.
- [[vae-from-scratch-numpy]] - the latent space used by latent diffusion, and the ELBO view of DDPM.
- [[neural-network-from-scratch-numpy]] - the MLP forward and backward pass reused here.
- [[optimizers-from-scratch-numpy]] - the Adam update used in the training loop.
- [[self-attention-from-scratch-numpy]] - the sinusoidal embedding idea, and the building block of diffusion transformers.

## References
- Ho, Jain and Abbeel (2020), "Denoising Diffusion Probabilistic Models", NeurIPS: https://arxiv.org/abs/2006.11239
- Sohl-Dickstein et al. (2015), "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", ICML: https://arxiv.org/abs/1503.03585
- Weng, "What are Diffusion Models?" (2021): https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
