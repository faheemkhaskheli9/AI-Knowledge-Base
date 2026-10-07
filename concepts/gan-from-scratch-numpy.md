---
title: GAN from scratch in NumPy (generator, discriminator, minimax vs non-saturating loss)
category: concepts
tags: [gan, generative-adversarial-network, generator, discriminator, minimax, non-saturating-loss, mode-collapse, adam, backpropagation, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a GAN's alternating generator and discriminator updates by hand"
  - "see how the generator's gradient flows through the discriminator"
  - "understand why the non-saturating generator loss is used in practice"
  - "explain GAN training instability and mode collapse in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1406.2661
  - https://arxiv.org/abs/1511.06434
  - https://arxiv.org/abs/1701.00160
---

# GAN from scratch in NumPy (generator, discriminator, minimax vs non-saturating loss)

## Summary
A generative adversarial network (GAN) trains two networks against each other. A generator `G` maps random noise to fake samples, and a discriminator `D` outputs the probability that a sample is real. `D` is trained to tell real from fake; `G` is trained to fool `D`, using gradients that flow back through `D` into `G`. At the equilibrium `G` produces the data distribution and `D` outputs 0.5 everywhere. The NumPy version below learns a two-peaked 1-D distribution with small MLPs and a hand-written backward pass that passes a gradient check. It shows why the original minimax generator loss gives `G` a weak signal when `D` is winning.

## Key concepts
- **Value function.** `min_G max_D E_x[log D(x)] + E_z[log(1 − D(G(z)))]`. For a fixed `G` the best `D` is `p_data / (p_data + p_G)`, and the game then minimises the Jensen-Shannon divergence between the two distributions.
- **Discriminator step.** Binary cross-entropy with real = 1 and fake = 0. With logits `s`, the loss is `softplus(−s_real) + softplus(s_fake)`, with gradients `−σ(−s)` and `σ(s)`.
- **Generator step.** Hold `D` fixed, compute the loss on `D(G(z))`, backprop through `D` to its input, then on into `G`. Only `G`'s weights are updated.
- **Minimax vs non-saturating.** Minimax `G` minimises `log(1 − D(G(z)))`, whose gradient with respect to the logit is `−σ(s)`. It is tiny when `D` confidently rejects fakes, which is exactly when `G` most needs to learn. The non-saturating loss `−log D(G(z))` has gradient `−σ(−s)`, which is large in that case. Both have the same fixed point.
- **Alternation.** One `D` step then one `G` step per batch. Adam with `β₁ = 0.5` and a learning rate of 2e-4 (the DCGAN settings) damps the oscillation between the players.
- **Mode collapse.** `G` maps many `z` to a few outputs that currently fool `D`, dropping other modes. A two-peaked target makes this visible: check the share of samples on each side.

## When to use / scenarios
- Learning: the smallest adversarial training loop, and the clearest example of backpropagating through one network into another.
- Interviews: "write the GAN losses", "why non-saturating", "what is mode collapse", "how do you evaluate a GAN".
- Practical GANs still matter for fast one-step generation (StyleGAN-style faces, super-resolution, image-to-image translation) and as the adversarial loss inside other models ([[generative-adversarial-networks]]).
- Not for: new image or audio generation projects where quality matters most (diffusion models are now the default, [[diffusion-models]]); likelihoods or an encoder ([[vae-from-scratch-numpy]], [[normalizing-flows-and-energy-based-models]]); tabular synthetic data without careful evaluation ([[data-labeling-and-synthetic-data]]).

## Setup & code
`pip install numpy`. Trains two GANs for 10,000 steps each in about 10 seconds on CPU.

```python
import numpy as np

rng = np.random.default_rng(0)


def init(sizes):
    return [[rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)] for a, b in zip(sizes, sizes[1:])]


def forward(params, x):
    """MLP with leaky-ReLU hidden layers and a linear output. Returns output and cache for backward."""
    acts = [x]
    for i, (W, b) in enumerate(params):
        z = acts[-1] @ W + b
        acts.append(z if i == len(params) - 1 else np.where(z > 0, z, 0.2 * z))
    return acts[-1], acts


def backward(params, acts, g):
    """Gradients of params given dLoss/dOutput; also returns dLoss/dInput."""
    grads = []
    for i in range(len(params) - 1, -1, -1):
        W, _ = params[i]
        if i < len(params) - 1:
            g = g * np.where(acts[i + 1] > 0, 1, 0.2)                # leaky-ReLU derivative
        grads.append([acts[i].T @ g, g.sum(0)])
        g = g @ W.T
    return grads[::-1], g


class Adam:
    def __init__(self, params, lr, b1=0.5, b2=0.999):
        self.p, self.lr, self.b1, self.b2, self.t = params, lr, b1, b2, 0
        self.m = [[np.zeros_like(a) for a in l] for l in params]
        self.v = [[np.zeros_like(a) for a in l] for l in params]

    def step(self, grads):
        self.t += 1
        for l, gl, ml, vl in zip(self.p, grads, self.m, self.v):
            for j in range(2):
                ml[j] = self.b1 * ml[j] + (1 - self.b1) * gl[j]
                vl[j] = self.b2 * vl[j] + (1 - self.b2) * gl[j] ** 2
                mh, vh = ml[j] / (1 - self.b1 ** self.t), vl[j] / (1 - self.b2 ** self.t)
                l[j] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


sigmoid = lambda s: 1 / (1 + np.exp(-s))
softplus = lambda s: np.logaddexp(0, s)                              # -log sigmoid(-s), stable


def real_data(n):
    """Target: mixture of two Gaussians at -2 and +2, std 0.5."""
    return (rng.choice([-2.0, 2.0], n) + 0.5 * rng.normal(size=n))[:, None]


def train(G_loss, steps=10000, n=128):
    G, D = init([4, 32, 32, 1]), init([1, 32, 32, 1])
    oG, oD = Adam(G, 2e-4), Adam(D, 2e-4)                         # DCGAN settings: lr 2e-4, beta1 0.5
    for _ in range(steps):
        # discriminator: maximise log D(real) + log(1 - D(fake)); logit loss = softplus(-s_real) + softplus(s_fake)
        fake, _ = forward(G, rng.normal(size=(n, 4)))
        s_r, a_r = forward(D, real_data(n))
        s_f, a_f = forward(D, fake)
        gr, _ = backward(D, a_r, -sigmoid(-s_r) / n)
        gf, _ = backward(D, a_f, sigmoid(s_f) / n)
        oD.step([[a + b for a, b in zip(x, y)] for x, y in zip(gr, gf)])
        # generator: backprop through D into G
        fake, a_g = forward(G, rng.normal(size=(n, 4)))
        s_f, a_f = forward(D, fake)
        if G_loss == "non-saturating":                             # minimise -log D(G(z))
            dS = -sigmoid(-s_f) / n
        else:                                                        # minimax: minimise log(1 - D(G(z)))
            dS = -sigmoid(s_f) / n
        _, dX = backward(D, a_f, dS)                                 # D's grads discarded: only G updates
        gG, _ = backward(G, a_g, dX)
        oG.step(gG)
    return G, D


def report(name, G, D):
    x = forward(G, rng.normal(size=(20000, 4)))[0][:, 0]
    r = real_data(20000)[:, 0]
    h_r = np.histogram(r, bins=40, range=(-4, 4))[0] / len(r)
    h_g = np.histogram(x, bins=40, range=(-4, 4))[0] / len(x)
    d_mean = sigmoid(forward(D, x[:, None])[0]).mean()
    print(f"{name:15s} mean {x.mean():+.2f} std {x.std():.2f} | share x>0 {(x > 0).mean():.2f} | "
          f"hist overlap {np.minimum(h_r, h_g).sum():.2f} | D(fake) {d_mean:.2f}")


# gradient check of the hand-written backward pass on a tiny net
P = init([3, 5, 1]); x0 = rng.normal(size=(4, 3))
out, acts = forward(P, x0)
grads, dx = backward(P, acts, np.ones_like(out))
eps, W = 1e-6, P[0][0]
W[1, 2] += eps; up = forward(P, x0)[0].sum(); W[1, 2] -= 2 * eps; dn = forward(P, x0)[0].sum(); W[1, 2] += eps
print(f"grad check dW[1,2]: analytic {grads[0][0][1, 2]:.6f} | numeric {(up - dn) / (2 * eps):.6f}")

r = real_data(20000)[:, 0]
print(f"{'real data':15s} mean {r.mean():+.2f} std {r.std():.2f} | share x>0 {(r > 0).mean():.2f}")
report("before training", init([4, 32, 32, 1]), init([1, 32, 32, 1]))

# why the non-saturating loss: train D alone against a fresh G, then compare G's gradient signal
G0, D0 = init([4, 32, 32, 1]), init([1, 32, 32, 1]); oD0 = Adam(D0, 1e-3)
for _ in range(300):
    s_r, a_r = forward(D0, real_data(128)); s_f, a_f = forward(D0, forward(G0, rng.normal(size=(128, 4)))[0])
    gr, _ = backward(D0, a_r, -sigmoid(-s_r) / 128); gf, _ = backward(D0, a_f, sigmoid(s_f) / 128)
    oD0.step([[a + b for a, b in zip(x, y)] for x, y in zip(gr, gf)])
s0 = forward(D0, forward(G0, rng.normal(size=(5000, 4)))[0])[0]
print(f"strong D: mean D(fake) {sigmoid(s0).mean():.3f} | |dLoss/dlogit| minimax {sigmoid(s0).mean():.3f} "
      f"vs non-saturating {sigmoid(-s0).mean():.3f}")
for loss in ["non-saturating", "minimax"]:
    report(loss, *train(loss))
```

Output (numpy 2.5):
```
grad check dW[1,2]: analytic -0.376819 | numeric -0.376819
real data       mean +0.00 std 2.06 | share x>0 0.50
before training mean +0.68 std 1.87 | share x>0 0.67 | hist overlap 0.49 | D(fake) 0.39
strong D: mean D(fake) 0.215 | |dLoss/dlogit| minimax 0.215 vs non-saturating 0.785
non-saturating  mean -0.03 std 2.07 | share x>0 0.50 | hist overlap 0.85 | D(fake) 0.51
minimax         mean -0.09 std 1.98 | share x>0 0.49 | hist overlap 0.84 | D(fake) 0.49
```

The hand-written backward pass matches a finite-difference gradient. The untrained generator overlaps the target histogram by only 0.49, and two-thirds of its samples sit on one side. After a discriminator trains for 300 steps against a frozen generator, it rates fakes 0.215 on average. At that point the minimax loss passes the generator a logit gradient of 0.215, while the non-saturating loss passes 0.785, 3.7 times more, and the gap widens as `D` gets more confident. Both trained generators split their samples 50/50 between the two peaks (no mode collapse), match the target's mean and standard deviation, and overlap its histogram by 0.84-0.85. `D` ends near 0.5 on fakes, the equilibrium value. In this 1-D problem `D` never wins decisively, so the minimax version still trains; the advantage of the non-saturating loss shows up in high-dimensional data, where `D` separates real from fake easily early on.

## Choosing / trade-offs
- **Loss.** Non-saturating BCE is the standard baseline. WGAN-GP (Wasserstein loss with a gradient penalty) and hinge loss give more stable training and a more meaningful loss curve for images.
- **GAN vs diffusion vs VAE.** GANs sample in one forward pass and give sharp outputs, but training is unstable and modes can drop. Diffusion models train stably and cover modes better but sample slowly ([[diffusion-models]]). VAEs are stable and give an encoder but blurrier samples ([[vae-from-scratch-numpy]]).
- **Balance.** If `D` wins too fast, `G` gets no useful signal; if `G` wins, `D` stops teaching. Common fixes: equal learning rates with `β₁ = 0.5`, spectral normalisation in `D`, label smoothing, two time-scale updates (TTUR).
- **Evaluation.** There is no likelihood. Use FID or precision/recall for images, and for low-dimensional data compare histograms or moments as here.

## Gotchas
- Learning rate matters a lot: the same code with Adam at 1e-3 instead of 2e-4 gave the non-saturating GAN a histogram overlap of only 0.68 and a standard deviation of 2.61, because the two players overshoot each other.
- Do not update `D` during the generator step. Compute `D`'s input gradient, then discard `D`'s weight gradients.
- Sample fresh noise for the generator step; reusing the batch that `D` just trained on biases the update.
- The loss values do not show progress (a falling `G` loss can mean `D` got worse). Track sample quality and a diversity measure such as the per-mode share.
- Mode collapse can be partial and silent. Count modes or check coverage on held-out data, not just a few good-looking samples.
- BatchNorm in `D` lets samples within a batch leak information about each other; DCGAN uses separate real and fake batches for this reason.

## Related
- [[generative-adversarial-networks]] - GAN variants, PyTorch code and evaluation.
- [[vae-from-scratch-numpy]] - the likelihood-bound alternative with an encoder.
- [[diffusion-models]] - the current default for image generation.
- [[neural-network-from-scratch-numpy]] - the MLP forward and backward pass reused here.
- [[optimizers-from-scratch-numpy]] - the Adam update.
- [[loss-functions]] - binary cross-entropy and the softplus form used for logits.

## References
- Goodfellow et al. (2014), "Generative Adversarial Networks": https://arxiv.org/abs/1406.2661
- Radford, Metz and Chintala (2015), "Unsupervised Representation Learning with Deep Convolutional GANs" (DCGAN): https://arxiv.org/abs/1511.06434
- Goodfellow (2016), "NIPS 2016 Tutorial: Generative Adversarial Networks": https://arxiv.org/abs/1701.00160
