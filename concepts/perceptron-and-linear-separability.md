---
title: Perceptron and linear separability (why neural networks need hidden layers)
category: concepts
tags: [perceptron, linear-separability, xor, decision-boundary, mlp, hidden-layer, numpy, scikit-learn, deep-learning-basics]
use_cases:
  - "explain what a single artificial neuron can and cannot learn"
  - "implement the perceptron learning rule from scratch for teaching or an interview"
  - "decide whether a linear model is enough or the problem needs a hidden layer"
  - "show why XOR breaks a linear classifier and how an MLP fixes it"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#perceptron
  - https://scikit-learn.org/stable/modules/neural_networks_supervised.html
  - https://www.deeplearningbook.org/contents/mlp.html
---

# Perceptron and linear separability (why neural networks need hidden layers)

## Summary
The perceptron (Rosenblatt, 1958) is the simplest neural network: one neuron computing `sign(w·x + b)`, trained by nudging the weights toward each misclassified example. It finds a separating hyperplane whenever one exists, and it never converges when one does not. XOR is the classic case. That limit is the reason multi-layer networks exist: a hidden layer with a non-linear activation builds new features in which the classes become linearly separable.

## Key concepts
- **Neuron.** Weighted sum plus bias, then an activation. The perceptron uses a step function. Logistic regression is the same neuron with a sigmoid and a log-loss, so it gives probabilities and a smooth gradient.
- **Decision boundary.** `w·x + b = 0` is a hyperplane (a line in 2-D). A single neuron can only draw one.
- **Learning rule.** For each sample with label `y ∈ {−1, +1}`: if `y(w·x + b) ≤ 0` then `w ← w + lr·y·x` and `b ← b + lr·y`. Correct predictions change nothing.
- **Convergence theorem.** If the data are linearly separable with margin γ and `‖x‖ ≤ R`, the perceptron makes at most `(R/γ)²` mistakes, whatever the order of the data.
- **Linear separability.** AND, OR and NOT are separable. XOR is not: no single line splits `{(0,1),(1,0)}` from `{(0,0),(1,1)}`.
- **Hidden layer.** A layer of non-linear units maps the input into a new space. For XOR, adding the feature `x1·x2` (what a hidden unit can learn) makes the classes separable. Stacking linear layers without activations adds nothing: the product of linear maps is still linear.
- **Universal approximation.** One hidden layer that is wide enough can approximate any continuous function on a compact set. Depth makes many such functions far cheaper to represent ([[neural-network-fundamentals]]).

## When to use / scenarios
- Teaching and interviews: the shortest path from "linear model" to "why deep learning".
- A quick separability check: if a linear model (perceptron, logistic regression, linear SVM) reaches near-zero training error, the classes are close to linearly separable and a deep model may be unnecessary.
- Online, mistake-driven learning on huge sparse data (text, click logs). The averaged perceptron is still a strong, cheap baseline there.
- Do NOT use the plain perceptron as a production classifier: it gives no probabilities, it stops at the first separating line (no margin) and it oscillates on non-separable data. Use logistic regression or a linear SVM ([[linear-models]], [[svm-knn-naive-bayes]]), or an MLP when the boundary is non-linear.

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in about a second.

```python
import numpy as np


def train_perceptron(X, y, epochs=20, lr=1.0):
    """Rosenblatt perceptron. y in {-1, +1}. Returns weights, bias, mistakes per epoch."""
    w, b = np.zeros(X.shape[1]), 0.0
    history = []
    for _ in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:  # misclassified (or on the boundary)
                w += lr * yi * xi
                b += lr * yi
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:
            break
    return w, b, history


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([-1, -1, -1, 1])
y_xor = np.array([-1, 1, 1, -1])

w, b, hist = train_perceptron(X, y_and)
print("AND mistakes/epoch:", hist, "-> preds", np.sign(X @ w + b).astype(int))
w, b, hist = train_perceptron(X, y_xor)
print("XOR mistakes/epoch:", hist[:6], "... never reaches 0")

# XOR is solvable once a hidden layer adds a non-linear feature: x1 AND x2
X_aug = np.hstack([X, (X[:, :1] * X[:, 1:])])
w, b, hist = train_perceptron(X_aug, y_xor)
print("XOR + x1*x2 feature:", hist, "-> preds", np.sign(X_aug @ w + b).astype(int))

# Same thing learned end-to-end: tiny MLP in scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

print("sklearn Perceptron on XOR acc:", Perceptron().fit(X, y_xor).score(X, y_xor))
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                    random_state=0, max_iter=1000).fit(X, y_xor)
print("MLP(4 hidden) on XOR acc:", mlp.score(X, y_xor))
```

Output (numpy 2.5, scikit-learn 1.9):
```
AND mistakes/epoch: [2, 3, 3, 2, 2, 3, 2, 1, 0] -> preds [-1 -1 -1  1]
XOR mistakes/epoch: [4, 4, 4, 4, 4, 4] ... never reaches 0
XOR + x1*x2 feature: [4, 4, 4, 4, 3, 1, 2, 3, 1, 2, 1, 0] -> preds [-1  1  1 -1]
sklearn Perceptron on XOR acc: 0.5
MLP(4 hidden) on XOR acc: 1.0
```

## Choosing / trade-offs
- **Perceptron vs logistic regression.** Same boundary family. Logistic regression optimises a smooth loss, so it converges on non-separable data, gives calibrated-ish probabilities and supports L1/L2 regularisation. Prefer it for anything beyond a demo.
- **Hand-made features vs hidden layers.** A known interaction (`x1·x2`, polynomial terms, kernels) keeps the model linear and interpretable. A hidden layer learns the features but needs more data and tuning.
- **Averaged / voted perceptron.** Averaging the weights over all steps gives a much more stable classifier than the final weights, at no extra cost (`sklearn.linear_model.SGDClassifier(loss="perceptron", average=True)`).

## Gotchas
- Labels must be `{−1, +1}` for the update rule as written. With `{0, 1}` labels the update never pushes the weights down.
- On non-separable data the final weights depend on the order of the last few samples. Shuffle every epoch and keep the averaged weights, or cap the epochs.
- Without a bias term the boundary must pass through the origin, so even AND can fail.
- The convergence bound depends on the margin. Separable data with a tiny margin can still need a very large number of updates.
- A "deep" network built from linear layers with no activation in between is still a single linear model and cannot learn XOR.
- An MLP on XOR can get stuck with an unlucky seed and few hidden units. Use a few more units or a different seed before concluding anything.

## Related
- [[neural-network-fundamentals]] - layers, activations and universal approximation in full.
- [[linear-models]] - logistic regression and linear SVMs, the production versions of one neuron.
- [[activation-functions]] - the non-linearity that makes hidden layers useful.
- [[neural-network-from-scratch-numpy]] - the next step: a full MLP with backprop in NumPy.
- [[kernel-methods-and-density-estimation]] - the other way to get non-linear boundaries from a linear model.

## References
- scikit-learn Perceptron: https://scikit-learn.org/stable/modules/linear_model.html#perceptron
- scikit-learn MLP: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- Goodfellow, Bengio, Courville, Deep Learning, ch. 6 (XOR example): https://www.deeplearningbook.org/contents/mlp.html
