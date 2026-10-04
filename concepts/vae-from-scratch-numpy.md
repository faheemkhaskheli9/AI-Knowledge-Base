---
title: Variational autoencoder from scratch in NumPy (ELBO, reparameterisation trick, KL term, sampling)
category: concepts
tags: [vae, variational-autoencoder, elbo, reparameterization-trick, kl-divergence, latent-variable-model, generative-models, posterior-collapse, backpropagation, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a VAE forward and backward pass by hand, including the reparameterisation trick"
  - "understand the ELBO as reconstruction loss plus a KL penalty"
  - "generate new samples by decoding draws from the prior"
  - "diagnose inactive latent dimensions (posterior collapse)"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1312.6114
  - https://arxiv.org/abs/1906.02691
  - https://www.deeplearningbook.org/contents/generative_models.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
---

# Variational autoencoder from scratch in NumPy (ELBO, reparameterisation trick, KL term, sampling)

## Summary
A variational autoencoder (VAE) is an autoencoder whose code is a probability distribution. The encoder outputs a mean `μ` and log-variance `log σ²` for each latent dimension, a code `z` is sampled from `N(μ, σ²)`, and the decoder reconstructs the input from `z`. Training minimises the negative evidence lower bound (ELBO): reconstruction loss plus the KL divergence from the code distribution to the prior `N(0, I)`. The KL term keeps the code space smooth and centred, so decoding `z ~ N(0, I)` produces new data. The NumPy version below passes a gradient check through the sampling step, trains on 8×8 digits, generates samples a digit classifier recognises, and shows 3 of 8 latent dimensions switching off.

## Key concepts
- **Encoder `q(z|x)`.** A network outputs `μ(x)` and `log σ²(x)`. Predicting the log-variance keeps `σ` positive without constraints.
- **Reparameterisation trick.** Write `z = μ + σ ⊙ ε` with `ε ~ N(0, I)` drawn outside the network. The randomness is now an input, so gradients flow to `μ` and `σ` by ordinary backprop: `∂z/∂μ = 1`, `∂z/∂log σ² = ½ σ ε`.
- **Decoder `p(x|z)`.** Here a Bernoulli per pixel, so the reconstruction term is binary cross-entropy with logits (pixels in `[0, 1]`). A Gaussian decoder gives MSE instead.
- **KL term, closed form.** `KL(N(μ, σ²) ‖ N(0, 1)) = −½ Σ (1 + log σ² − μ² − σ²)`. Its gradients are `μ` and `½(σ² − 1)`.
- **ELBO.** `log p(x) ≥ E_q[log p(x|z)] − KL(q(z|x) ‖ p(z))`. Minimise the negative: reconstruction + KL, in nats per example.
- **Posterior collapse / inactive units.** A latent dimension whose KL is about 0 has `q(z|x) = N(0, 1)` for every input: it carries no information and the decoder ignores it.
- **β-VAE.** Weight the KL by `β`: `β > 1` gives smoother, more disentangled codes and blurrier reconstructions; `β < 1` the reverse.

## When to use / scenarios
- Learning: the smallest working generative model with a latent space, and the root of latent diffusion and many anomaly detectors ([[diffusion-models]]).
- Interviews: "what is the reparameterisation trick", "derive the KL term", "why are VAE samples blurry", "what is posterior collapse".
- Anomaly detection where a likelihood-like score (the ELBO) is wanted rather than raw reconstruction error ([[anomaly-detection]]).
- Learning a smooth, low-dimensional latent space for interpolation, data augmentation or semi-supervised learning.
- Not for: sharp, high-fidelity image generation (diffusion models or GANs, [[generative-adversarial-networks]]); exact likelihoods ([[normalizing-flows-and-energy-based-models]]); pure compression where a plain autoencoder or PCA is enough ([[autoencoder-from-scratch-numpy]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in under 10 seconds on CPU.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0                                              # pixels in [0, 1], used as Bernoulli targets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
D, H, K = 64, 128, 8                                      # input, hidden, latent sizes


def init(rng):
    g = lambda a, b: rng.normal(0, np.sqrt(1 / a), (a, b))
    return {"W1": g(D, H), "b1": np.zeros(H), "Wmu": g(H, K), "bmu": np.zeros(K), "Wlv": g(H, K),
            "blv": np.zeros(K), "W2": g(K, H), "b2": np.zeros(H), "W3": g(H, D), "b3": np.zeros(D)}


def forward(P, x, eps):
    h1 = np.tanh(x @ P["W1"] + P["b1"])
    mu, lv = h1 @ P["Wmu"] + P["bmu"], h1 @ P["Wlv"] + P["blv"]
    z = mu + np.exp(0.5 * lv) * eps                       # reparameterisation trick
    h2 = np.tanh(z @ P["W2"] + P["b2"])
    logits = h2 @ P["W3"] + P["b3"]
    rec = np.sum(np.logaddexp(0, logits) - x * logits, 1)  # Bernoulli NLL = BCE with logits, per image
    kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv), 1)   # KL(q(z|x) || N(0, I)), closed form
    return dict(h1=h1, mu=mu, lv=lv, z=z, h2=h2, logits=logits, rec=rec, kl=kl, loss=(rec + kl).mean())


def backward(P, x, eps, c):
    B = len(x)
    G = {}
    dlog = (1 / (1 + np.exp(-c["logits"])) - x) / B
    G["W3"], G["b3"] = c["h2"].T @ dlog, dlog.sum(0)
    da2 = (dlog @ P["W3"].T) * (1 - c["h2"] ** 2)
    G["W2"], G["b2"] = c["z"].T @ da2, da2.sum(0)
    dz = da2 @ P["W2"].T
    dmu = dz + c["mu"] / B                                # reconstruction path + KL path
    dlv = dz * eps * 0.5 * np.exp(0.5 * c["lv"]) + 0.5 * (np.exp(c["lv"]) - 1) / B
    G["Wmu"], G["bmu"] = c["h1"].T @ dmu, dmu.sum(0)
    G["Wlv"], G["blv"] = c["h1"].T @ dlv, dlv.sum(0)
    da1 = (dmu @ P["Wmu"].T + dlv @ P["Wlv"].T) * (1 - c["h1"] ** 2)
    G["W1"], G["b1"] = x.T @ da1, da1.sum(0)
    return G


# Gradient check: fix eps so the loss is deterministic
rng = np.random.default_rng(1)
P = init(rng); x = Xtr[:8]; eps = rng.normal(size=(8, K))
G = backward(P, x, eps, forward(P, x, eps))
for name, idx in [("W1", (5, 3)), ("Wlv", (2, 1)), ("W3", (4, 7))]:
    P[name][idx] += 1e-6; up = forward(P, x, eps)["loss"]
    P[name][idx] -= 2e-6; dn = forward(P, x, eps)["loss"]
    P[name][idx] += 1e-6
    print(f"grad check {name}: analytic {G[name][idx]:.6e}, numeric {(up - dn) / 2e-6:.6e}")

# Train with Adam
rng = np.random.default_rng(0)
P = init(rng)
m = {k: np.zeros_like(v) for k, v in P.items()}; v = {k: np.zeros_like(a) for k, a in P.items()}
t = 0
for epoch in range(300):
    for idx in np.array_split(rng.permutation(len(Xtr)), len(Xtr) // 64):
        eps = rng.normal(size=(len(idx), K))
        G = backward(P, Xtr[idx], eps, forward(P, Xtr[idx], eps))
        t += 1
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * G[k]; v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
            P[k] -= 1e-3 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)

c = forward(P, Xte, rng.normal(size=(len(Xte), K)))
print(f"test -ELBO {c['loss']:.2f} nats/image = reconstruction {c['rec'].mean():.2f} + KL {c['kl'].mean():.2f}")
bce = lambda x, p: -np.sum(x * np.log(np.clip(p, 1e-9, 1)) + (1 - x) * np.log(np.clip(1 - p, 1e-9, 1)), 1).mean()
print(f"reference: mean-image model {bce(Xte, Xtr.mean(0)):.2f} | perfect reconstruction (grey pixels are not 0/1) {bce(Xte, Xte):.2f}")

# Latent space: the mean codes separate digit classes without labels
mu_tr, mu_te = forward(P, Xtr, np.zeros((len(Xtr), K)))["mu"], c["mu"]
print(f"digit accuracy from the {K}-D code (logistic regression): {LogisticRegression(max_iter=2000).fit(mu_tr, ytr).score(mu_te, yte):.3f}")
kl_dim = (-0.5 * (1 + c["lv"] - c["mu"] ** 2 - np.exp(c["lv"]))).mean(0)
print("KL per latent dim (nats):", kl_dim.round(2), f"-> {np.sum(kl_dim > 0.05)} of {K} dims active")

# Generation: decode z ~ N(0, I) and ask a digit classifier how digit-like the samples are
clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
z = rng.normal(size=(500, K))
gen = 1 / (1 + np.exp(-(np.tanh(z @ P["W2"] + P["b2"]) @ P["W3"] + P["b3"])))
noise = rng.uniform(0, 1, size=(500, D))
print(f"classifier confidence: real test digits {clf.predict_proba(Xte).max(1).mean():.3f} | "
      f"VAE samples {clf.predict_proba(gen).max(1).mean():.3f} | uniform noise {clf.predict_proba(noise).max(1).mean():.3f}")
print("predicted classes of 500 samples:", np.bincount(clf.predict(gen), minlength=10))
```

Output (numpy 2.5, scikit-learn 1.9):
```
grad check W1: analytic -1.540217e-02, numeric -1.540216e-02
grad check Wlv: analytic 1.156042e-02, numeric 1.156042e-02
grad check W3: analytic 8.347315e-02, numeric 8.347315e-02
test -ELBO 24.49 nats/image = reconstruction 21.02 + KL 3.47
reference: mean-image model 27.19 | perfect reconstruction (grey pixels are not 0/1) 13.80
digit accuracy from the 8-D code (logistic regression): 0.911
KL per latent dim (nats): [0.19 0.95 0.83 0.02 0.01 0.   0.77 0.72] -> 5 of 8 dims active
classifier confidence: real test digits 0.887 | VAE samples 0.714 | uniform noise 0.584
predicted classes of 500 samples: [37 45 38 52 64 60 47 50 47 60]
```

The hand-written gradients, including the path through the sampled `z` into the log-variance weights (`Wlv`), match central finite differences. On held-out digits the negative ELBO is 24.49 nats per image: 21.02 for reconstruction and 3.47 for KL. Read the reconstruction number against its floor: the digit pixels are grey levels, not 0/1, so even a perfect reconstruction costs 13.80 nats, and a model that always outputs the mean image costs 27.19. The 8-number mean code, learned without labels, classifies digits at 91.1% with logistic regression. Only 5 of the 8 latent dimensions are used; dimensions 4-6 have KL near 0, meaning the KL penalty pushed them to the prior because the decoder gained nothing from them. Decoding 500 draws from `N(0, I)` gives samples a pixel classifier assigns to all ten digits with mean confidence 0.714, between real digits (0.887) and uniform noise (0.584): recognisable but blurry, the usual VAE trade-off.

## Choosing / trade-offs
- **VAE vs plain autoencoder.** Use a plain autoencoder for compression and reconstruction-error anomaly scores; use a VAE when you need to sample, interpolate, or want a regularised latent space.
- **VAE vs GAN vs diffusion.** VAEs are stable to train and give an encoder and a likelihood bound, but samples are blurrier. GANs give sharp samples with unstable training and no encoder; diffusion models give the best samples at higher sampling cost, often in a VAE's latent space.
- **Latent size.** Too small and reconstructions blur; too large and extra dimensions collapse (as here), which is harmless but wasted. Count active units to size it.
- **Decoder likelihood.** Bernoulli (BCE) for data in `[0, 1]`, Gaussian (MSE with a learned or fixed variance) for real values. The choice sets the balance with the KL term.

## Gotchas
- Sum the reconstruction loss over pixels and the KL over latent dimensions, then average over the batch. Averaging over pixels shrinks the reconstruction term by 64× and the KL then dominates, collapsing every dimension.
- Clamp or initialise the log-variance head near 0; `exp(log σ²)` overflows if it grows large early in training.
- At generation time decode `z ~ N(0, I)`; at encoding/feature time use `μ`, not a noisy sample.
- Posterior collapse is worse with powerful decoders (autoregressive, large RNNs). Remedies: KL annealing (warm `β` from 0 to 1), free bits, or a weaker decoder.
- The negative ELBO is an upper bound on `−log p(x)`, not the likelihood itself. Use importance-weighted bounds when comparing models by likelihood.
- Bernoulli likelihood on non-binary pixels is a common shortcut; its loss has a non-zero floor (13.80 nats here), so report the gap to the floor, not the raw number.

## Related
- [[autoencoder-from-scratch-numpy]] - the deterministic autoencoder this extends.
- [[autoencoders-and-self-supervised-learning]] - VAEs and autoencoders in PyTorch.
- [[gmm-em-from-scratch]] - a latent-variable model trained with EM instead of a variational bound.
- [[diffusion-models]] - generative models that often run in a VAE's latent space.
- [[generative-adversarial-networks]] - the adversarial alternative for sampling.
- [[optimizers-from-scratch-numpy]] - the Adam update used here.

## References
- Kingma and Welling (2013), "Auto-Encoding Variational Bayes": https://arxiv.org/abs/1312.6114
- Kingma and Welling (2019), "An Introduction to Variational Autoencoders": https://arxiv.org/abs/1906.02691
- Goodfellow, Bengio, Courville, Deep Learning, chapter 20 (Deep generative models): https://www.deeplearningbook.org/contents/generative_models.html
- scikit-learn `load_digits`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
