---
title: Softmax regression from scratch (multinomial logistic regression and cross-entropy in NumPy)
category: ml
tags: [softmax, softmax-regression, multinomial-logistic-regression, cross-entropy, one-hot, log-sum-exp, gradient-check, multiclass, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement multiclass logistic regression (softmax regression) from scratch"
  - "derive and check the gradient of softmax cross-entropy"
  - "make softmax numerically stable and understand why it overflows"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - https://www.deeplearningbook.org/contents/numerical.html
---

# Softmax regression from scratch (multinomial logistic regression and cross-entropy in NumPy)

## Summary
Softmax regression extends logistic regression from two classes to `k`: a linear layer produces one score (logit) per class, softmax turns the scores into probabilities, and the model is trained by minimising cross-entropy. The gradient of that loss with respect to the logits is simply `predicted probabilities − one-hot labels`, which makes the whole trainer about 15 lines of NumPy. It is also exactly the output layer of every neural-network classifier. On the 10-class digits dataset the from-scratch model reaches 0.974 test accuracy and agrees with scikit-learn's `LogisticRegression` on 99.8% of test images.

## Key concepts
- **Logits.** `z = xW + b`, one score per class, shape `(n, k)`. `W` has one column of weights per class.
- **Softmax.** `p_j = e^{z_j} / Σ_c e^{z_c}`. Outputs are positive and sum to 1. Adding the same constant to every logit does not change the result, which is what makes the stable version possible.
- **Cross-entropy.** `L = −(1/n) Σ log p_{i, yᵢ}`: the negative log-probability of the true class. It is the negative log-likelihood of a categorical distribution ([[maximum-likelihood-and-map-estimation]], [[loss-functions]]).
- **The gradient.** `∂L/∂z = (P − Y)/n` where `Y` is one-hot. Then `∂L/∂W = Xᵀ(P − Y)/n` and `∂L/∂b = Σ(P − Y)/n`. The Jacobian of softmax and the derivative of the log cancel into this simple form.
- **Numerical stability.** `e^{1000}` overflows to `inf`, and `inf/inf` is `nan`. Subtract the row maximum from the logits before `exp` (the log-sum-exp trick); the result is mathematically identical.
- **Over-parameterisation.** Adding a constant vector to every column of `W` leaves predictions unchanged, so without regularisation the weights have no unique solution. A small L2 penalty fixes that and keeps weights finite on separable data.
- **Relation to binary logistic regression.** With `k = 2`, softmax of `(z₁, z₂)` equals sigmoid of `z₁ − z₂` ([[logistic-regression-from-scratch]]).

## When to use / scenarios
- Learning: the last layer and loss of every classification network, in isolation.
- Interviews: "derive the softmax cross-entropy gradient", "why subtract the max", "softmax vs one-vs-rest".
- A fast, calibrated-ish linear baseline for multiclass problems with standardised features or embeddings ([[multiclass-classification-strategies]]).
- Not for: non-linear boundaries on raw features (use a tree ensemble or an MLP), or multi-label problems where several classes can be true at once (use independent sigmoids, see [[multi-label-and-multi-task-learning]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in a few seconds.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def softmax(z):
    z = z - z.max(1, keepdims=True)          # stability: exp of large logits overflows
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_softmax(X, y, k, lam=1e-3, lr=0.5, epochs=3000):
    n, d = X.shape
    W, b = np.zeros((d, k)), np.zeros(k)
    Y = np.eye(k)[y]                          # one-hot targets, shape (n, k)
    for _ in range(epochs):
        P = softmax(X @ W + b)
        G = (P - Y) / n                       # gradient of mean cross-entropy wrt logits
        W -= lr * (X.T @ G + lam * W)
        b -= lr * G.sum(0)
    return W, b


def cross_entropy(X, y, W, b):
    P = softmax(X @ W + b)
    return -np.log(P[np.arange(len(y)), y]).mean()


X, y = load_digits(return_X_y=True)          # 1797 8x8 images, 10 classes
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

W, b = fit_softmax(Xtr, ytr, k=10)
ours = (Xte @ W + b).argmax(1)
n = len(Xtr)
sk = LogisticRegression(C=1 / (1e-3 * n), max_iter=5000).fit(Xtr, ytr)   # same L2 strength
print(f"test acc ours={np.mean(ours == yte):.3f} sklearn={sk.score(Xte, yte):.3f} "
      f"agree={np.mean(ours == sk.predict(Xte)):.3f}")
print(f"train CE ours={cross_entropy(Xtr, ytr, W, b):.4f} "
      f"sklearn={cross_entropy(Xtr, ytr, sk.coef_.T, sk.intercept_):.4f}")
print(f"probs sum to 1: {np.allclose(softmax(Xte @ W + b).sum(1), 1)}")

# gradient check on a few weights
rng = np.random.default_rng(0)
Wc, bc = rng.normal(0, 0.1, (64, 10)), np.zeros(10)
G = (softmax(Xtr @ Wc + bc) - np.eye(10)[ytr]) / n
analytic = Xtr.T @ G
eps, errs = 1e-5, []
for i, j in [(10, 3), (30, 7), (50, 0)]:
    Wp, Wm = Wc.copy(), Wc.copy()
    Wp[i, j] += eps
    Wm[i, j] -= eps
    num = (cross_entropy(Xtr, ytr, Wp, bc) - cross_entropy(Xtr, ytr, Wm, bc)) / (2 * eps)
    errs.append(abs(num - analytic[i, j]))
print(f"grad check max abs err: {max(errs):.1e}")

# naive softmax overflows
z = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    print("naive:", np.exp(z) / np.exp(z).sum(), "stable:", softmax(z).round(4))
```

Output (numpy 2.5, scikit-learn 1.9):
```
test acc ours=0.974 sklearn=0.972 agree=0.998
train CE ours=0.0371 sklearn=0.0363
probs sum to 1: True
grad check max abs err: 3.0e-11
naive: [[nan nan nan]] stable: [[0.09   0.2447 0.6652]]
```

scikit-learn minimises `Σ CE + ‖W‖²/(2C)`, so `C = 1/(λn)` gives the same objective as the mean-loss form used here. After 3000 full-batch steps the from-scratch model is close to scikit-learn's L-BFGS optimum (training cross-entropy 0.0371 vs 0.0363) and they disagree on one test image out of 540. With 500 steps the gap was larger (0.0569 vs 0.0363): plain gradient descent gets near the optimum quickly and then crawls. The analytic gradient matches central finite differences to `3e-11`, and the naive softmax turns logits of about 1000 into `nan` while the max-shifted version is fine.

## Choosing / trade-offs
- **Softmax (multinomial) vs one-vs-rest.** Softmax fits one joint model whose probabilities sum to 1 and is the default in scikit-learn for multiclass `LogisticRegression`. One-vs-rest trains `k` independent binary models; it parallelises easily but its scores are not a probability distribution ([[multiclass-classification-strategies]]).
- **Solver.** Full-batch gradient descent is fine for learning. L-BFGS or Newton methods converge in far fewer steps on small data; mini-batch SGD is the only option at scale or inside a neural network ([[optimizers]]).
- **Regularisation strength.** Tune `C` (or `λ`) on a log grid with cross-validation. Weak regularisation on separable data drives weights large and probabilities to 0 or 1.
- **Temperature.** Dividing logits by `T > 1` softens the distribution; it is used for calibration ([[probability-calibration]]), distillation ([[knowledge-distillation-and-compression]]) and sampling ([[decoding-and-sampling]]).

## Gotchas
- Computing `log(softmax(z))` in two steps underflows to `log(0) = −inf` for very negative logits. Use a fused `log_softmax` (`z − logsumexp(z)`), as PyTorch's `F.cross_entropy` does internally.
- PyTorch's `nn.CrossEntropyLoss` expects raw logits, not probabilities. Applying softmax first trains a model, just a worse one, and is a very common bug.
- Integer labels must be `0 … k−1`. Labels like `1 … k` or strings must be mapped first, or `np.eye(k)[y]` indexes the wrong row (or fails).
- Standardise the features. Unscaled pixel values (0 to 16 here) make the loss surface badly conditioned and gradient descent needs a much smaller step ([[feature-scaling-and-normalization]]).
- Do not regularise the bias, and check whether a library's penalty is on the sum or the mean of the loss before copying a `C` value.
- Softmax probabilities from a linear model are reasonable but not guaranteed calibrated, and those from deep networks are usually over-confident. Check a reliability diagram before using them as risk scores.

## Related
- [[logistic-regression-from-scratch]] - the two-class special case with sigmoid and log-loss.
- [[multiclass-classification-strategies]] - softmax vs one-vs-rest vs one-vs-one.
- [[loss-functions]] - cross-entropy next to the other training losses.
- [[neural-network-from-scratch-numpy]] - where this layer sits on top of hidden layers.
- [[information-theory-for-ml]] - cross-entropy and KL divergence as information measures.
- [[probability-calibration]] - making the output probabilities trustworthy.

## References
- scikit-learn, Logistic regression (multinomial case): https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- scikit-learn LogisticRegression API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Goodfellow, Bengio and Courville, Deep Learning, ch. 4 "Numerical Computation" (softmax overflow): https://www.deeplearningbook.org/contents/numerical.html
