---
title: Autoencoder from scratch in NumPy (encoder, bottleneck, decoder, backprop, linear autoencoder equals PCA)
category: concepts
tags: [autoencoder, bottleneck, representation-learning, dimensionality-reduction, reconstruction-error, pca, backpropagation, adam, numpy, from-scratch, deep-learning-basics, unsupervised-learning]
use_cases:
  - "implement an autoencoder forward and backward pass from scratch"
  - "show that a linear autoencoder learns the PCA subspace"
  - "compress data to a small code and use it as features"
  - "understand reconstruction error as the basis of autoencoder anomaly detection"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.science.org/doi/10.1126/science.1127647
  - https://www.deeplearningbook.org/contents/autoencoders.html
  - https://doi.org/10.1016/0893-6080(89)90014-2
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
---

# Autoencoder from scratch in NumPy (encoder, bottleneck, decoder, backprop, linear autoencoder equals PCA)

## Summary
An autoencoder is a neural network trained to output its own input through a narrow middle layer. The encoder maps `x` to a short code `h`, the decoder maps `h` back to `x̂`, and the loss is the reconstruction error `‖x − x̂‖²`. Because the code is smaller than the input, the network has to keep only the structure that matters. A linear autoencoder with MSE loss learns exactly the PCA subspace; adding non-linear layers lets it reconstruct better from the same code size. The NumPy version below passes a gradient check, matches PCA's error and subspace when linear, and beats PCA by 31% in reconstruction error with `tanh` layers.

## Key concepts
- **Encoder, bottleneck, decoder.** Here `64 → 32 → 8 → 32 → 64` on 8×8 digit images. The 8-unit layer is the code. Its output layer is linear because the targets are centred pixels.
- **Reconstruction loss.** Mean squared error between input and output. Its gradient with respect to the output is `2(x̂ − x)/N`, and backprop through the layers is the same as for any MLP ([[neural-network-from-scratch-numpy]]).
- **Linear AE = PCA.** With linear layers and MSE, the optimal decoder spans the top-`k` principal subspace (Baldi and Hornik, 1989). The weights are not the principal components themselves, only a basis for the same subspace, so compare subspaces with principal angles, not weights.
- **Non-linear AE.** `tanh` hidden layers let the code describe a curved manifold, so the same 8 numbers reconstruct better than 8 principal components.
- **Undercomplete vs regularised.** The bottleneck alone prevents copying the input. Overcomplete autoencoders need another constraint instead: noise on the input (denoising), a sparsity penalty, or a prior on the code (VAE).
- **Uses of the trained parts.** The encoder gives compact features; the reconstruction error gives an anomaly score; the decoder (in a VAE) gives a generator.

## When to use / scenarios
- Learning: the simplest unsupervised neural network, and a clean link between deep learning and PCA ([[pca-from-scratch]]).
- Interviews: "what does an autoencoder learn", "how does it relate to PCA", "how would you use one for anomaly detection".
- Anomaly detection on sensor, log or transaction data: train on normal data, flag high reconstruction error ([[autoencoders-and-self-supervised-learning]], [[anomaly-detection]]).
- Compression to features for a downstream model when data has non-linear structure and labels are scarce.
- Not for: production image or text representation learning, where contrastive or masked pretraining and pretrained encoders work better ([[embeddings]]). Not for: data where PCA already explains most of the variance; use PCA.

## Setup & code
`pip install numpy scikit-learn`. Runs in about 4 seconds on CPU.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X = X / 16.0                                              # pixels in [0, 1]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
mean = Xtr.mean(0)


def init(sizes, rng):
    return [(rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def forward(params, x, act):
    """Returns every layer's output. act[i] is 'tanh' or 'linear' for layer i."""
    outs = [x]
    for (W, b), a in zip(params, act):
        z = outs[-1] @ W + b
        outs.append(np.tanh(z) if a == "tanh" else z)
    return outs


def backward(params, outs, act, dout):
    grads = []
    for i in reversed(range(len(params))):
        if act[i] == "tanh":
            dout = dout * (1 - outs[i + 1] ** 2)
        grads.append((outs[i].T @ dout, dout.sum(0)))
        dout = dout @ params[i][0].T
    return grads[::-1]


def train(sizes, act, epochs=300, lr=1e-2, batch=64, seed=0):
    rng = np.random.default_rng(seed)
    params = init(sizes, rng)
    flat = [a for Wb in params for a in Wb]               # W0, b0, W1, b1, ... updated in place
    m, v = [np.zeros_like(a) for a in flat], [np.zeros_like(a) for a in flat]
    t = 0
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Xtr)), len(Xtr) // batch):
            x = Xtr[idx] - mean
            outs = forward(params, x, act)
            dout = 2 * (outs[-1] - x) / x.size           # d(MSE)/d(reconstruction)
            grads = [g for gWb in backward(params, outs, act, dout) for g in gWb]
            t += 1
            for a, g, mi, vi in zip(flat, grads, m, v):   # Adam
                mi[:] = 0.9 * mi + 0.1 * g
                vi[:] = 0.999 * vi + 0.001 * g * g
                a -= lr * (mi / (1 - 0.9 ** t)) / (np.sqrt(vi / (1 - 0.999 ** t)) + 1e-8)
    return params


def mse(params, act, X_):
    x = X_ - mean
    return ((forward(params, x, act)[-1] - x) ** 2).mean()


# Gradient check on a tiny net (finite differences, float64)
rng = np.random.default_rng(1)
p = init([64, 8, 64], rng); act = ["tanh", "linear"]
x = Xtr[:5] - mean
outs = forward(p, x, act)
gW = backward(p, outs, act, 2 * (outs[-1] - x) / x.size)[0][0]
eps, (i, j) = 1e-6, (3, 2)
W = p[0][0]
W[i, j] += eps; up = ((forward(p, x, act)[-1] - x) ** 2).mean()
W[i, j] -= 2 * eps; dn = ((forward(p, x, act)[-1] - x) ** 2).mean()
W[i, j] += eps
print(f"grad check: analytic {gW[i, j]:.6e}, numeric {(up - dn) / (2 * eps):.6e}")

k = 8
pca = PCA(k).fit(Xtr)
pca_mse = ((pca.inverse_transform(pca.transform(Xte)) - Xte) ** 2).mean()
lin = train([64, k, 64], ["linear", "linear"])
deep_act = ["tanh", "linear", "tanh", "linear"]
deep = train([64, 32, k, 32, 64], deep_act)
print(f"test MSE, {k}-dim code: PCA {pca_mse:.4f} | linear AE {mse(lin, ['linear'] * 2, Xte):.4f} | "
      f"tanh AE {mse(deep, deep_act, Xte):.4f} | predict-the-mean {((Xte - mean) ** 2).mean():.4f}")

# Linear AE spans the PCA subspace: principal angles between the decoder rows and PCA components
Q1 = np.linalg.qr(lin[1][0].T)[0]
Q2 = np.linalg.qr(pca.components_.T)[0]
print("cos of principal angles (1 = same subspace):", np.linalg.svd(Q1.T @ Q2, compute_uv=False).round(3))

# The code as features: logistic regression on 8 numbers vs 64 pixels
code = lambda X_: forward(deep, X_ - mean, deep_act)[2]
acc = lambda A, B: LogisticRegression(max_iter=2000).fit(A, ytr).score(B, yte)
print(f"digit accuracy: raw 64 pixels {acc(Xtr, Xte):.3f} | 8-dim AE code {acc(code(Xtr), code(Xte)):.3f} | "
      f"8-dim PCA {acc(pca.transform(Xtr), pca.transform(Xte)):.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
grad check: analytic -8.902299e-04, numeric -8.902299e-04
test MSE, 8-dim code: PCA 0.0249 | linear AE 0.0250 | tanh AE 0.0171 | predict-the-mean 0.0731
cos of principal angles (1 = same subspace): [1. 1. 1. 1. 1. 1. 1. 1.]
digit accuracy: raw 64 pixels 0.961 | 8-dim AE code 0.920 | 8-dim PCA 0.878
```

The hand-written gradient agrees with a central finite difference to 7 significant digits. The linear autoencoder reaches PCA's test error (0.0250 vs 0.0249), and all 8 principal angles between its decoder and the top 8 principal components have cosine 1.000: it found the same subspace by gradient descent. The `tanh` autoencoder, with the same 8-number code, cuts test error to 0.0171, 31% below PCA and 77% below predicting the mean image. Its code is also the better feature set: logistic regression on the 8 code values classifies digits at 92.0%, against 87.8% on 8 PCA components and 96.1% on all 64 pixels. None of these numbers used a label during autoencoder training.

## Choosing / trade-offs
- **Autoencoder vs PCA.** PCA is closed-form, deterministic, fast and has no hyperparameters; try it first. Use an autoencoder when the data lies on a curved manifold and PCA's reconstruction error at a useful code size is too high.
- **Code size.** Too small and reconstructions blur; too large and the model can copy its input and learns weak features. Sweep it and watch validation reconstruction error.
- **Plain vs denoising vs variational.** Denoising AEs (corrupt input, reconstruct clean) learn more robust features. VAEs add a KL term so the code space is smooth and can be sampled for generation ([[normalizing-flows-and-energy-based-models]], [[diffusion-models]]).
- **NumPy vs PyTorch.** Write it by hand once to understand it; use PyTorch with convolutional or recurrent layers for real data ([[autoencoders-and-self-supervised-learning]]).

## Gotchas
- Centre (or standardise) inputs and match the output activation to the target range: linear output for centred data, sigmoid for pixels in `[0, 1]`.
- Threshold anomaly scores on reconstruction error of held-out normal data, not training data, which the model has partly memorised.
- A low reconstruction error does not mean useful features. Check the code on a downstream task, as above.
- Comparing a linear AE's weights to PCA components directly fails; any invertible mix of the components is an equally good solution. Compare subspaces or reconstruction error.
- Autoencoders can reconstruct anomalies well too if they are close to the training manifold or the bottleneck is too wide. Validate on known anomalies when you have any.
- Deep autoencoders with `tanh` or sigmoid saturate under poor initialisation; use scaled initialisation and Adam ([[weight-initialization]], [[optimizers-from-scratch-numpy]]).

## Related
- [[autoencoders-and-self-supervised-learning]] - autoencoders in PyTorch, anomaly detection and contrastive pretraining.
- [[pca-from-scratch]] - the linear method a linear autoencoder reproduces.
- [[neural-network-from-scratch-numpy]] - the MLP forward and backward pass used here.
- [[optimizers-from-scratch-numpy]] - the Adam update used in training.
- [[dimensionality-reduction]] - PCA, UMAP, t-SNE and autoencoders compared.
- [[anomaly-detection]] - reconstruction error as one of many anomaly scores.

## References
- Hinton and Salakhutdinov (2006), "Reducing the Dimensionality of Data with Neural Networks", Science: https://www.science.org/doi/10.1126/science.1127647
- Goodfellow, Bengio, Courville, Deep Learning, chapter 14 (Autoencoders): https://www.deeplearningbook.org/contents/autoencoders.html
- Baldi and Hornik (1989), "Neural networks and principal component analysis: Learning from examples without local minima", Neural Networks: https://doi.org/10.1016/0893-6080(89)90014-2
- scikit-learn `load_digits`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
