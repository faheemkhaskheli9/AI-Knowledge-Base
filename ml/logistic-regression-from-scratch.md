---
title: Logistic regression from scratch (sigmoid, log-loss, gradient descent in NumPy)
category: ml
tags: [logistic-regression, sigmoid, log-loss, cross-entropy, gradient-descent, gradient-check, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement logistic regression from scratch for learning or an interview"
  - "understand where the log-loss gradient (p - y) * x comes from"
  - "check a hand-written gradient against finite differences"
  - "match a from-scratch model to scikit-learn's LogisticRegression and its C parameter"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
  - https://web.stanford.edu/~jurafsky/slp3/5.pdf
---

# Logistic regression from scratch (sigmoid, log-loss, gradient descent in NumPy)

## Summary
Logistic regression models the probability of the positive class as `sigmoid(w·x + b)` and learns `w, b` by minimising the log-loss (binary cross-entropy). The loss is convex, and its gradient has the very simple form `Xᵀ(p − y) / n`, so about 15 lines of NumPy reproduce scikit-learn's results. Writing it once explains the pieces every neural network reuses: a linear layer, a squashing output, a likelihood-based loss, gradient descent and a gradient check.

## Key concepts
- **Model.** `z = w·x + b` is the log-odds, `p = σ(z) = 1 / (1 + e^(−z))` is the probability. The decision boundary `p = 0.5` is the hyperplane `z = 0`, so the model is linear ([[perceptron-and-linear-separability]]).
- **Loss.** Mean log-loss `L = −mean(y·log p + (1 − y)·log(1 − p))` is the negative log-likelihood of Bernoulli labels ([[maximum-likelihood-and-map-estimation]]). It punishes confident wrong answers heavily.
- **Gradient.** Because `σ'(z) = σ(z)(1 − σ(z))` cancels against the log, `∂L/∂z = p − y`. Hence `∂L/∂w = Xᵀ(p − y) / n` and `∂L/∂b = mean(p − y)`. The same `p − y` appears for softmax + cross-entropy.
- **Convexity.** Log-loss is convex in `w`, so gradient descent with a small enough step reaches the global minimum. There is no initialisation lottery as in neural nets, and zeros are fine.
- **Regularisation.** Adding `(λ/2)‖w‖²` (L2, MAP with a Gaussian prior) adds `λ·w` to the gradient. scikit-learn's `C` is an inverse strength: its objective `C·Σ loss + ½‖w‖²` matches `λ = 1 / (C·n)` on the mean loss.
- **Separable data.** Without regularisation, perfectly separable classes push `‖w‖` to infinity, because larger weights always lower the loss. L2 keeps the solution finite.
- **Multi-class.** Softmax regression replaces σ with softmax over K logits and the same cross-entropy ([[multiclass-classification-strategies]]).

## When to use / scenarios
- Learning: the bridge from linear regression to neural networks ([[neural-network-from-scratch-numpy]]).
- Interviews and exams: deriving `p − y`, explaining `C`, and spotting the separable-data blow-up.
- Custom variants that libraries do not offer directly: a special per-sample loss weight, a custom penalty, or an online update in an embedded system.
- Production: do NOT ship the from-scratch version. Use `sklearn.linear_model.LogisticRegression` (lbfgs, liblinear, saga), which handles convergence, multi-class, class weights and sparse input ([[linear-models]]).

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in about a second.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # clip avoids overflow warnings


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logreg(X, y, lr=0.1, epochs=2000, l2=0.0):
    """Batch gradient descent on mean log-loss + (l2/2)*||w||^2. y in {0, 1}."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for epoch in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y                      # dLoss/dz for sigmoid + log-loss
        w -= lr * (X.T @ err / n + l2 * w)
        b -= lr * err.mean()             # bias is not regularised
        if epoch % 500 == 0:
            print(f"epoch {epoch:4d}  loss {log_loss(y, p):.4f}")
    return w, b


X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)               # gradient descent needs scaled inputs
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

C = 1.0
w, b = fit_logreg(X_tr, y_tr, l2=1.0 / (C * len(y_tr)))   # same objective as sklearn's C
acc = ((sigmoid(X_te @ w + b) >= 0.5) == y_te).mean()
print(f"from-scratch test acc: {acc:.3f}")

# Gradient check: analytic gradient vs central finite differences on 3 weights
def loss_at(w_):
    return log_loss(y_tr, sigmoid(X_tr @ w_ + b))
g_analytic = X_tr.T @ (sigmoid(X_tr @ w + b) - y_tr) / len(y_tr)
eps = 1e-6
g_num = np.array([(loss_at(w + eps * e) - loss_at(w - eps * e)) / (2 * eps) for e in np.eye(len(w))[:3]])
print("grad check max abs diff:", f"{np.abs(g_analytic[:3] - g_num).max():.2e}")

sk = LogisticRegression(C=C, max_iter=5000).fit(X_tr, y_tr)
print(f"sklearn test acc:      {sk.score(X_te, y_te):.3f}")
print("weight correlation with sklearn:", f"{np.corrcoef(w, sk.coef_[0])[0, 1]:.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
epoch    0  loss 0.6931
epoch  500  loss 0.0639
epoch 1000  loss 0.0568
epoch 1500  loss 0.0540
from-scratch test acc: 0.965
grad check max abs diff: 5.42e-11
sklearn test acc:      0.958
weight correlation with sklearn: 0.978
```

The starting loss is `ln 2 = 0.6931`, as it must be with zero weights (`p = 0.5` everywhere). The weights are close to scikit-learn's but not identical, because 2,000 fixed-step iterations have not fully converged where lbfgs has.

## Choosing / trade-offs
- **Batch GD vs SGD / mini-batch.** Full-batch is simplest and deterministic. Mini-batch SGD scales to data that does not fit in memory and supports online updates (`SGDClassifier(loss="log_loss")`) ([[batch-size-and-gradient-noise]]).
- **First- vs second-order.** Plain GD needs a tuned learning rate and many steps. Newton / IRLS and lbfgs use curvature and converge in tens of iterations, which is why libraries use them.
- **L2 vs L1.** L2 shrinks all weights smoothly. L1 sets some to exactly zero (feature selection) but is not differentiable at 0, so it needs a proximal step or the `saga` / `liblinear` solvers.
- **Threshold.** 0.5 is the default, not a law. Choose the threshold from costs or a precision/recall target on validation data ([[model-evaluation-and-metrics]]).

## Gotchas
- Unscaled features make fixed-step gradient descent crawl or diverge ([[feature-scaling-and-normalization]]).
- `np.exp(-z)` overflows for large negative `z`. Clip `z`, or use `scipy.special.expit`; compute the loss from logits with `np.logaddexp(0, -z)` for full stability.
- `log(0)` gives `-inf` and NaN gradients. Clip probabilities in the loss.
- Labels must be `{0, 1}` for this loss. The perceptron's `{−1, +1}` convention silently breaks it.
- Regularising the bias shifts the base rate. Libraries leave the intercept unpenalised (except `liblinear`).
- Matching scikit-learn needs the same objective: its `C` multiplies the *summed* loss, so the equivalent L2 on the mean loss is `1 / (C·n)`, not `1 / C`.
- A gradient check is only meaningful in float64 with `eps` around `1e-6`. In float32 the finite differences are dominated by rounding.

## Related
- [[linear-models]] - the production logistic regression, regularisation and interpretation.
- [[perceptron-and-linear-separability]] - the same neuron with a step function and no probabilities.
- [[neural-network-from-scratch-numpy]] - adds a hidden layer and backprop to this exact code.
- [[gradient-descent]] - step sizes, momentum and convergence in general.
- [[loss-functions]] - cross-entropy alongside the other losses.
- [[probability-calibration]] - when the predicted probabilities need to be trusted.

## References
- scikit-learn, Logistic regression (objective and solvers): https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- scikit-learn LogisticRegression API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Jurafsky and Martin, Speech and Language Processing, ch. 5 Logistic Regression (gradient derivation): https://web.stanford.edu/~jurafsky/slp3/5.pdf
