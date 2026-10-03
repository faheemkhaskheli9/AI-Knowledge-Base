---
title: SVM, k-nearest neighbours and Naive Bayes
category: ml
tags: [svm, svc, kernel, knn, nearest-neighbors, naive-bayes, text-classification, scikit-learn, baseline]
use_cases:
  - "classify a small dataset (hundreds to a few thousand rows) with a strong non-linear model"
  - "build a spam or ticket-topic classifier that trains in seconds on word counts"
  - "predict a label from the most similar past cases and show those cases as the explanation"
  - "pick a quick baseline before reaching for gradient boosting or deep learning"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/svm.html
  - https://scikit-learn.org/stable/modules/neighbors.html
  - https://scikit-learn.org/stable/modules/naive_bayes.html
---

# SVM, k-nearest neighbours and Naive Bayes

## Summary
Three classic classifiers that still earn their place. A **support vector machine (SVM)** finds the widest-margin boundary between classes and, with a kernel, bends that boundary to fit non-linear data; it is very strong on small, clean, scaled datasets. **k-nearest neighbours (kNN)** has no training step: it labels a point by majority vote of its closest training points, so its "explanation" is the neighbours themselves. **Naive Bayes** multiplies per-feature probabilities under an independence assumption; it is crude but trains in one pass and is a fast, solid baseline for text.

## Key concepts
- **Margin and support vectors.** A linear SVM picks the boundary with the largest gap to the nearest points (the support vectors); only those points define the model.
- **`C` (soft margin).** Small `C` tolerates misclassified training points for a wider margin (simpler); large `C` fits the training data tightly (risk of overfitting).
- **Kernel trick.** `kernel="rbf"` (default) compares points by similarity `exp(-gamma·|x-x'|²)`, giving curved boundaries without building features. `gamma` sets how local each point's influence is: large = wiggly boundary.
- **`LinearSVC` vs `SVC`.** `LinearSVC` scales to hundreds of thousands of rows and sparse text; kernel `SVC` is roughly quadratic-to-cubic in rows, practical up to ~10k-50k.
- **kNN.** Choose `k` (odd for binary; larger = smoother) and a distance (Euclidean by default; cosine for embeddings). `weights="distance"` lets closer neighbours count more.
- **Curse of dimensionality.** In many dimensions all points look equally far apart, so kNN degrades on raw high-dimensional data; reduce first ([[dimensionality-reduction]]) or use learned [[embeddings]].
- **Naive Bayes variants.** `MultinomialNB` / `ComplementNB` for word counts, `BernoulliNB` for binary present/absent features, `GaussianNB` for continuous features.
- **All three need scaled or count-like inputs.** SVM and kNN are distance-based, so unscaled features dominate; Naive Bayes needs non-negative counts for the multinomial variants.

## When to use / scenarios
- **SVM:** small biomedical or sensor datasets (gene expression, spectroscopy, a few hundred labelled signals) where boosting overfits; text classification with `LinearSVC` on TF-IDF.
- **kNN:** "find similar cases" products (similar patients, similar products, case-based support answers), where showing the neighbours is the explanation; quick baseline on low-dimensional data; classification on top of good embeddings.
- **Naive Bayes:** spam filtering, routing support tickets or emails by topic, any first text baseline; streaming updates with `partial_fit`.
- **Not when:** large tabular data with mixed feature types (use [[gradient-boosting-tabular]]); images, audio or raw sequences (use deep learning); millions of rows with kernel SVM (use `LinearSVC` or boosting). For nearest-neighbour search at scale use a vector index ([[vector-databases]]), not `KNeighborsClassifier`.

## Setup & code
```bash
pip install scikit-learn
```
```python
from sklearn.datasets import fetch_20newsgroups, make_moons
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# Non-linear 2D data: a curved boundary is needed.
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
for name, clf in [("linear SVM", LinearSVC()),
                  ("RBF SVM", SVC(kernel="rbf", C=1.0, gamma="scale")),
                  ("kNN k=15", KNeighborsClassifier(n_neighbors=15))]:
    model = make_pipeline(StandardScaler(), clf)
    print(f"{name}: {cross_val_score(model, X, y, cv=5).mean():.3f}")

# Text: Naive Bayes vs linear SVM on TF-IDF (downloads ~14 MB once).
cats = ["sci.med", "sci.space", "rec.autos", "comp.graphics"]
news = fetch_20newsgroups(subset="all", categories=cats,
                          remove=("headers", "footers", "quotes"))
for name, clf in [("ComplementNB", ComplementNB()), ("LinearSVC", LinearSVC())]:
    model = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), clf)
    print(f"{name}: {cross_val_score(model, news.data, news.target, cv=5).mean():.3f}")
```
On the moons data the linear SVM scores about 0.88 and the RBF SVM and kNN about 0.93; on the four newsgroups both text models land near 0.90 (Naive Bayes slightly ahead here, and faster to train).

## Choosing / trade-offs
- **SVM vs boosting.** SVM often wins on small (< ~5k rows), dense, all-numeric, well-scaled data; boosting wins on larger tabular data with categories, missing values and mixed scales, and needs no scaling.
- **Tuning SVM.** Search `C` and `gamma` on a log grid (`1e-3 … 1e3`) together; they interact. See [[hyperparameter-tuning]].
- **Probabilities.** `SVC(probability=True)` runs an extra internal CV (slow, sometimes inconsistent with `predict`); prefer `CalibratedClassifierCV` or use `decision_function` scores for ranking.
- **kNN cost.** Training is free but each prediction searches the training set; fine for thousands of rows, use approximate nearest-neighbour libraries (FAISS, a vector DB) beyond that.
- **Naive Bayes vs logistic regression on text.** NB is better with very few labels; logistic regression / `LinearSVC` overtake it as data grows. NB probabilities are badly calibrated (pushed to 0 and 1).

## Gotchas
- Forgetting `StandardScaler` before SVM or kNN: the feature with the largest units decides everything.
- Kernel `SVC` on 100k+ rows can run for hours; check `n_samples` before choosing it.
- `LinearSVC` may emit convergence warnings on unscaled data; scale or raise `max_iter`.
- kNN with `k=1` memorises noise (training accuracy 100%); choose `k` by cross-validation.
- `MultinomialNB` fails on negative inputs (e.g. after `StandardScaler`); feed it counts or TF-IDF only.
- Naive Bayes "probabilities" should not be shown to users as confidence without calibration ([[model-evaluation-and-metrics]]).

## Related
- [[linear-models]] - logistic regression, the usual alternative to linear SVM and NB.
- [[classic-ml-scikit-learn]] - pipelines and the wider scikit-learn workflow.
- [[feature-engineering]] - scaling and text vectorisation these models depend on.
- [[nlp-classic-tasks]] - text classification beyond bag-of-words baselines.
- [[embeddings]] - kNN on embeddings is semantic similarity search.
- [[vector-databases]] - nearest-neighbour search at production scale.

## References
- scikit-learn SVM guide: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn nearest neighbours: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
