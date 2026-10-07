---
title: Neural network from scratch in NumPy (forward pass, backprop, training loop)
category: concepts
tags: [deep-learning, neural-network, mlp, backpropagation, softmax, cross-entropy, relu, gradient-check, numpy, from-scratch]
use_cases:
  - "understand exactly what a framework does when it trains a neural network"
  - "implement and train a small MLP without PyTorch or TensorFlow for teaching or an interview"
  - "derive the backward pass of softmax plus cross-entropy and ReLU by hand"
  - "verify hand-written gradients with a numerical gradient check"
status: stable
last_verified: 2026-10-04
sources:
  - https://cs231n.github.io/neural-networks-case-study/
  - https://cs231n.github.io/optimization-2/
  - https://www.deeplearningbook.org/contents/mlp.html
---

# Neural network from scratch in NumPy (forward pass, backprop, training loop)

## Summary
A two-layer neural network is about 40 lines of NumPy: matrix multiplies and a ReLU for the forward pass, a stable softmax with cross-entropy for the loss, the chain rule written out by hand for the backward pass, and a mini-batch gradient-descent loop. Writing one makes every piece a framework hides concrete: shapes, the `p − y` gradient, why initialisation matters, and how to check gradients. Use PyTorch for real work ([[pytorch-basics]]). Use this to understand it.

## Key concepts
- **Shapes.** Inputs `X` are (n, d). Layer 1 has `W1` (d, h) and `b1` (h). Layer 2 has `W2` (h, c) and `b2` (c). Every gradient has the same shape as its parameter, which is the quickest check that a backward pass is wired correctly.
- **Forward pass.** `z1 = XW1 + b1`, `a1 = ReLU(z1)`, `z2 = a1W2 + b2`, `p = softmax(z2)`. Keep `z1` and `a1`, because backward needs them.
- **Stable softmax.** Subtract the row max before `exp`. It does not change the result and prevents overflow.
- **Softmax + cross-entropy gradient.** For loss `−mean log p[y]`, the gradient with respect to the logits is `(p − onehot(y)) / n`. The two derivatives collapse into that one simple expression, which is why frameworks fuse them.
- **Backward pass (chain rule, right to left).** `dW2 = a1ᵀ dz2`, `db2 = Σ dz2`, `da1 = dz2 W2ᵀ`, `dz1 = da1 ⊙ [z1 > 0]`, `dW1 = Xᵀ dz1`, `db1 = Σ dz1`.
- **Initialisation.** Random, scaled by fan-in (He: `std = √(2/fan_in)` for ReLU). All-zero weights make every hidden unit identical forever ([[weight-initialization]]).
- **Gradient check.** Compare each analytic gradient with `(L(w+ε) − L(w−ε)) / 2ε`. The two should agree to several digits before any training is trusted.

## When to use / scenarios
- Learning deep learning or teaching it: build this once, then move to autograd.
- Interviews and exams that ask for backprop by hand.
- Tiny embedded or dependency-free settings where a small, already-trained MLP must run with only NumPy (inference only).
- Not for anything beyond toy size. There is no GPU, no autograd and no optimiser library, so use [[pytorch-basics]] or [[keras-and-tensorflow]].

## Setup & code
```bash
pip install numpy scikit-learn   # scikit-learn only for the toy dataset
```
```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.2, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
rng = np.random.default_rng(0)

# 2 -> 16 (ReLU) -> 2 (softmax). He init for the ReLU layer.
H, C = 16, 2
W1 = rng.normal(0, np.sqrt(2 / 2), (2, H)); b1 = np.zeros(H)
W2 = rng.normal(0, np.sqrt(1 / H), (H, C)); b2 = np.zeros(C)


def forward(X):
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ W2 + b2
    z2 = z2 - z2.max(1, keepdims=True)            # stable softmax
    p = np.exp(z2) / np.exp(z2).sum(1, keepdims=True)
    return z1, a1, p


def loss_and_grads(X, y):
    z1, a1, p = forward(X)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    dz2 = p.copy(); dz2[np.arange(n), y] -= 1; dz2 /= n   # softmax + CE gradient
    dW2 = a1.T @ dz2; db2 = dz2.sum(0)
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)                                   # ReLU gradient
    dW1 = X.T @ dz1; db1 = dz1.sum(0)
    return loss, (dW1, db1, dW2, db2)


# Gradient check on one weight before training: analytic vs numerical.
_, g = loss_and_grads(X_tr[:50], y_tr[:50])
eps = 1e-5
W1[0, 0] += eps; lp, _ = loss_and_grads(X_tr[:50], y_tr[:50])
W1[0, 0] -= 2 * eps; lm, _ = loss_and_grads(X_tr[:50], y_tr[:50])
W1[0, 0] += eps
print(f"grad check dW1[0,0]: analytic={g[0][0, 0]:.6f} numeric={(lp - lm) / (2 * eps):.6f}")

lr, batch = 0.1, 32
for epoch in range(1, 201):
    order = rng.permutation(len(y_tr))
    for s in range(0, len(order), batch):
        i = order[s:s + batch]
        loss, (dW1, db1, dW2, db2) = loss_and_grads(X_tr[i], y_tr[i])
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    if epoch in (1, 50, 200):
        full, _ = loss_and_grads(X_tr, y_tr)
        acc = (forward(X_te)[2].argmax(1) == y_te).mean()
        print(f"epoch {epoch:3d} train loss={full:.4f} test acc={acc:.3f}")
```
Output (NumPy 2.5, scikit-learn 1.9):
```
grad check dW1[0,0]: analytic=-0.036624 numeric=-0.036624
epoch   1 train loss=0.3497 test acc=0.845
epoch  50 train loss=0.1462 test acc=0.945
epoch 200 train loss=0.0880 test acc=0.950
```
A logistic regression on the same two moons is limited to a straight boundary. One hidden layer of 16 ReLUs bends it and reaches 95% test accuracy.

## Choosing / trade-offs
- **From scratch vs autograd.** Hand-written backprop is fine for 2 to 3 layer types. Beyond that, the bookkeeping is where bugs live, and autograd ([[backpropagation-and-autograd]]) does it for any graph.
- **Width vs depth.** One wide hidden layer can approximate any smooth function in theory. In practice, deeper networks learn hierarchical features with fewer units, but need good initialisation, normalisation and residual connections to train ([[residual-and-skip-connections]]).
- **Plain SGD here, Adam in practice.** Swapping the update line for momentum or Adam is a small change ([[optimizers]]). The forward and backward passes are unchanged.

## Gotchas
- **Shape slips that broadcast silently.** `b1` with shape (h, 1) instead of (h,) broadcasts into an (n, h) bias without an error. Assert gradient shapes match parameter shapes.
- **Forgetting the 1/n.** If the loss averages over the batch but the gradient does not, the effective learning rate scales with batch size.
- **`log(0)` from a saturated softmax.** Add a tiny epsilon, or better, compute log-softmax directly (`z − logsumexp(z)`).
- **Gradient check in float32 or with ReLU kinks.** Use float64 and avoid points where `z1` is exactly 0. A small mismatch there is expected, a large one everywhere is a bug.
- **Mutating arrays in place.** `dz2 = p` (without `.copy()`) and then `dz2[...] -= 1` corrupts `p`. NumPy views bite in backward passes.
- **Evaluating on training data only.** Check held-out accuracy as above. A from-scratch net overfits just like a framework one ([[bias-variance-and-learning-curves]]).

## Related
- [[neural-network-fundamentals]] - the same ideas expressed in PyTorch.
- [[backpropagation-and-autograd]] - how frameworks automate the backward pass written by hand here.
- [[gradient-descent]] - the update rule and batch-size choices used in the training loop.
- [[activation-functions]] - ReLU and alternatives, and their gradients.
- [[loss-functions]] - cross-entropy, log-softmax and numerical stability.
- [[weight-initialization]] - why the He scaling above matters.

## References
- Stanford CS231n, "Putting it together: minimal neural network case study": https://cs231n.github.io/neural-networks-case-study/
- Stanford CS231n, backpropagation intuitions: https://cs231n.github.io/optimization-2/
- Goodfellow, Bengio, Courville, *Deep Learning*, chapter 6 (deep feedforward networks): https://www.deeplearningbook.org/contents/mlp.html
