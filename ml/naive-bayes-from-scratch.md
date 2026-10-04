---
title: Naive Bayes from scratch (Gaussian and multinomial, log-space, Laplace smoothing)
category: ml
tags: [naive-bayes, gaussian-naive-bayes, multinomial-naive-bayes, bayes-theorem, laplace-smoothing, log-sum-exp, text-classification, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement Gaussian and multinomial naive Bayes from scratch"
  - "understand Bayes' rule, the independence assumption and why to work in log space"
  - "see what Laplace (additive) smoothing does to a bag-of-words classifier"
  - "build a fast text classification baseline and match scikit-learn"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/naive_bayes.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
  - https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
---

# Naive Bayes from scratch (Gaussian and multinomial, log-space, Laplace smoothing)

## Summary
Naive Bayes classifies with Bayes' rule, `p(c | x) ∝ p(c) · p(x | c)`, and the "naive" assumption that features are independent given the class, so `p(x | c)` becomes a product of per-feature terms. Fitting is just counting (means and variances for Gaussian NB, word counts for multinomial NB), and prediction is a sum of logs. Both variants take about 15 lines of NumPy and match scikit-learn to float precision.

## Key concepts
- **Bayes' rule.** `p(c | x) = p(c) p(x | c) / p(x)`. The denominator is the same for every class, so the class with the largest `log p(c) + log p(x | c)` (the joint log-likelihood) wins.
- **Conditional independence.** `p(x | c) = Πⱼ p(xⱼ | c)`. Almost never true, yet the argmax is often still right. The probabilities themselves are usually overconfident ([[probability-calibration]]).
- **Gaussian NB.** Each feature in each class is a 1-D normal with its own mean and variance. `log p(xⱼ | c) = −½ [log(2πσ²) + (xⱼ − μ)² / σ²]`.
- **Multinomial NB.** For counts (bag of words). `p(word | c)` is the class's share of that word among all its word occurrences, and a document's score is `Σ count(word) · log p(word | c)`. That is a single sparse matrix product.
- **Log space.** A product of thousands of probabilities underflows to 0. Sum logs instead, and turn scores into probabilities with log-sum-exp: `p = exp(s − logsumexp(s))` ([[information-theory-for-ml]]).
- **Laplace / additive smoothing.** Add `α` to every count. Without it, one word unseen in a class sets `p = 0`, `log p = −∞`, and vetoes that class whatever the rest of the document says. Smoothing is a MAP estimate with a Dirichlet prior ([[maximum-likelihood-and-map-estimation]]).
- **Variance smoothing.** Gaussian NB adds a small epsilon (scikit-learn: `1e-9 × max feature variance`) so a constant feature does not divide by zero.

## When to use / scenarios
- Learning: the shortest path from Bayes' rule to a working classifier, and the classic reason for doing probability in log space.
- Interviews: deriving the decision rule, explaining smoothing and why "naive" still works.
- Text baselines: spam filtering, topic and intent classification on bag-of-words. Trains in milliseconds on hundreds of thousands of documents ([[nlp-classic-tasks]]).
- Very small datasets, where its strong assumption acts as regularisation and beats more flexible models.
- Not for: calibrated probabilities without recalibration, strongly correlated features (duplicated evidence is counted twice), or as a final model when a linear model or fine-tuned transformer is affordable ([[svm-knn-naive-bayes]]).

## Setup & code
`pip install numpy scipy scikit-learn`. The 20 Newsgroups subset (about 14 MB) downloads on first run; afterwards it runs in a few seconds.

```python
import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import fetch_20newsgroups, load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB


class GaussianNaiveBayes:
    def fit(self, X, y, var_smoothing=1e-9):
        self.classes = np.unique(y)
        self.log_prior = np.log(np.bincount(y) / len(y))
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])
        self.var = np.array([X[y == c].var(0) for c in self.classes])
        self.var += var_smoothing * X.var(0).max()          # same epsilon as sklearn
        return self

    def joint_log_lik(self, X):                             # (n, C)
        ll = -0.5 * (np.log(2 * np.pi * self.var)[None]
                     + (X[:, None] - self.mu[None]) ** 2 / self.var[None]).sum(-1)
        return ll + self.log_prior

    def predict_proba(self, X):
        jll = self.joint_log_lik(X)
        return np.exp(jll - logsumexp(jll, axis=1, keepdims=True))


class MultinomialNaiveBayes:
    def fit(self, X, y, alpha=1.0):                         # X: (n, vocab) counts
        C = y.max() + 1
        self.log_prior = np.log(np.bincount(y) / len(y))
        counts = np.vstack([np.asarray(X[y == c].sum(0)).ravel() for c in range(C)])
        counts = counts + alpha                             # Laplace smoothing
        self.log_lik = np.log(counts / counts.sum(1, keepdims=True))
        return self

    def predict(self, X):
        return np.asarray(X @ self.log_lik.T + self.log_prior).argmax(1)


X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
g = GaussianNaiveBayes().fit(Xtr, ytr)
p_sk = GaussianNB().fit(Xtr, ytr).predict_proba(Xte)
print("Gaussian NB acc:", (g.predict_proba(Xte).argmax(1) == yte).mean().round(3),
      "| max |proba - sklearn|:", f"{np.abs(g.predict_proba(Xte) - p_sk).max():.1e}")

cats = ["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"]
tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
te = fetch_20newsgroups(subset="test", categories=cats, remove=("headers", "footers", "quotes"))
vec = CountVectorizer(min_df=2, stop_words="english")
Xtr, Xte = vec.fit_transform(tr.data), vec.transform(te.data)
m = MultinomialNaiveBayes().fit(Xtr, tr.target)
ours = m.predict(Xte)
sk = MultinomialNB(alpha=1.0).fit(Xtr, tr.target).predict(Xte)
print(f"Multinomial NB acc: {(ours == te.target).mean():.3f}  "
      f"agreement with sklearn: {(ours == sk).mean():.3f}  vocab={Xtr.shape[1]}")
for a in (1e-10, 0.01, 0.1, 1.0):
    acc = (MultinomialNaiveBayes().fit(Xtr, tr.target, a).predict(Xte) == te.target).mean()
    print(f"alpha={a:g}: acc={acc:.3f}")
```

Output (numpy 2.5, scipy 1.18, scikit-learn 1.9):
```
Gaussian NB acc: 0.978 | max |proba - sklearn|: 6.7e-16
Multinomial NB acc: 0.859  agreement with sklearn: 1.000  vocab=12544
alpha=1e-10: acc=0.831
alpha=0.01: acc=0.864
alpha=0.1: acc=0.862
alpha=1: acc=0.859
```

Gaussian NB matches scikit-learn's probabilities to within `7e-16`. On four 20 Newsgroups topics with quotes and headers stripped, the multinomial version reaches 0.859 accuracy and predicts the same class as scikit-learn for every document. With almost no smoothing (`α = 1e-10`) accuracy drops to 0.831, because words unseen in a class veto it. A smaller `α` than the default 1.0 is slightly better here; tune it by cross-validation.

## Choosing / trade-offs
- **Which variant.** Gaussian for continuous features, multinomial for counts or TF-IDF, Bernoulli for binary presence/absence (short texts), Complement NB for imbalanced text classes. Categorical NB for discrete non-count features.
- **Smoothing α.** `α = 1` (Laplace) is the textbook default. With large vocabularies `α` in 0.01-0.5 is often better. It is the main hyperparameter ([[hyperparameter-tuning]]).
- **NB vs logistic regression.** NB is generative and converges with fewer examples. Logistic regression is discriminative, handles correlated features, and usually wins once you have a few thousand labelled examples ([[logistic-regression-from-scratch]]).
- **Priors.** Learned class priors reflect training frequencies. If deployment class balance differs, set the priors explicitly (scikit-learn `class_prior`).

## Gotchas
- Never multiply raw probabilities. Per-word probabilities are around `1e-4` in a large vocabulary, so a document of a few hundred words drops below the smallest float64 (about `1e-308`) and every class scores 0. Always sum logs.
- Fit the vectoriser on the training split only. Building the vocabulary from test documents leaks information ([[data-leakage-and-validation-splits]]).
- `X[y == c].sum(0)` on a SciPy sparse matrix returns an integer `np.matrix`. Convert with `np.asarray(...).ravel()` and add `α` out of place (`counts + alpha`), or the in-place add fails to cast float into int.
- Predicted probabilities pile up near 0 and 1. Do not threshold or rank on them as if calibrated. Use `CalibratedClassifierCV` when probabilities matter ([[probability-calibration]]).
- Duplicated or highly correlated features are counted as independent evidence, which inflates confidence. Remove near-duplicates in feature engineering ([[feature-selection]]).
- On 20 Newsgroups, keep `remove=("headers", "footers", "quotes")`. Otherwise the model learns sender names, signatures and quoted replies instead of topics, and the score overstates real-world accuracy (scikit-learn's dataset docs make the same warning).

## Related
- [[svm-knn-naive-bayes]] - naive Bayes next to SVMs and k-NN, with scikit-learn usage.
- [[maximum-likelihood-and-map-estimation]] - counting is MLE, smoothing is MAP with a Dirichlet prior.
- [[nlp-classic-tasks]] - bag-of-words text classification pipelines.
- [[probability-calibration]] - fixing naive Bayes' overconfident probabilities.
- [[logistic-regression-from-scratch]] - the discriminative counterpart.
- [[knn-from-scratch]] - another simple from-scratch baseline classifier.

## References
- scikit-learn, Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- scikit-learn MultinomialNB API: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
- Manning, Raghavan and Schütze, Introduction to Information Retrieval, Naive Bayes text classification: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
