---
title: Linear discriminant analysis from scratch (shared-covariance Gaussian classifier and Fisher projection)
category: ml
tags: [lda, linear-discriminant-analysis, fisher-discriminant, generative-classifier, dimensionality-reduction, shrinkage, covariance, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement linear discriminant analysis from scratch and match scikit-learn"
  - "project labelled data to K-1 dimensions that best separate the classes"
  - "explain the difference between LDA, PCA, QDA, naive Bayes and logistic regression"
  - "fix LDA when there are more features than samples (shrinkage)"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/lda_qda.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
  - https://hastie.su.domains/ElemStatLearn/
  - https://doi.org/10.1111/j.1469-1809.1936.tb02137.x
---

# Linear discriminant analysis from scratch (shared-covariance Gaussian classifier and Fisher projection)

## Summary
Linear discriminant analysis (LDA) models each class as a Gaussian with its own mean and one covariance matrix shared by all classes, then classifies with Bayes' rule. Because the covariance is shared, the quadratic terms cancel and the decision boundaries are linear. The same matrices give Fisher's projection: the `K−1` directions that maximise between-class variance relative to within-class variance, a supervised alternative to PCA. The NumPy version below matches scikit-learn's probabilities to 4e-16, separates wine classes 1.5× better than PCA in 2-D, and shows why shrinkage is needed when features outnumber samples.

## Key concepts
- **Generative model.** `p(x | y=k) = N(μ_k, Σ)`, `p(y=k) = π_k`. Fit by class means, class frequencies and the pooled within-class covariance `Σ = Σ_k Σ_{i∈k} (x_i − μ_k)(x_i − μ_k)ᵀ / n`.
- **Linear discriminant.** `δ_k(x) = xᵀΣ⁻¹μ_k − ½ μ_kᵀΣ⁻¹μ_k + log π_k`. Predict the arg-max; a softmax over `δ` gives the posterior probabilities.
- **Fisher's criterion.** Find `w` maximising `wᵀS_B w / wᵀS_W w`. The solutions are the top eigenvectors of `S_W⁻¹S_B`; `S_B` has rank at most `K−1`, so there are at most `K−1` useful directions.
- **LDA vs logistic regression.** Same linear form of the boundary. LDA estimates it through Gaussian assumptions (more efficient if they hold, sensitive to outliers); logistic regression fits it directly ([[logistic-regression-from-scratch]], [[softmax-regression-from-scratch]]).
- **QDA** gives each class its own covariance: quadratic boundaries, many more parameters. **Gaussian naive Bayes** is QDA with diagonal covariances ([[naive-bayes-from-scratch]]).
- **Shrinkage.** `Σ̂ = (1−α)Σ + α·(tr Σ/d)·I` keeps the covariance invertible and well-conditioned when `d` is large relative to `n`; Ledoit-Wolf picks `α` analytically.

## When to use / scenarios
- Learning: the cleanest generative classifier, and the bridge between classification and supervised dimensionality reduction.
- Interviews: "LDA vs PCA", "LDA vs logistic regression", "why at most K−1 components".
- Small tabular datasets with roughly Gaussian, similarly spread classes (lab measurements, spectra, sensor summaries): strong, fast, no tuning.
- Supervised 2-D or 3-D visualisation of labelled data, or a preprocessing step before k-NN ([[knn-from-scratch]]).
- High-dimensional, few-sample problems (EEG, genomics, spectroscopy) with shrinkage.
- Not for: classes with very different covariances (try QDA), multi-modal classes, heavy outliers, or large datasets where a discriminative model will win.

## Setup & code
`pip install numpy scikit-learn`. Runs in under a second.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LDA:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n, d = X.shape
        self.means = np.array([X[y == k].mean(0) for k in self.classes])
        self.priors = np.array([(y == k).mean() for k in self.classes])
        Xc = X - self.means[np.searchsorted(self.classes, y)]
        self.Sw = Xc.T @ Xc / n                           # pooled within-class covariance (sklearn uses /n too)
        self.Sw_inv = np.linalg.inv(self.Sw)
        # Fisher directions: eigenvectors of Sw^-1 Sb, at most K-1 non-zero eigenvalues
        mu = X.mean(0)
        Sb = sum(np.sum(y == k) * np.outer(m - mu, m - mu) for k, m in zip(self.classes, self.means)) / n
        evals, evecs = np.linalg.eig(self.Sw_inv @ Sb)
        order = np.argsort(evals.real)[::-1][: len(self.classes) - 1]
        self.W, self.evals = evecs[:, order].real, evals.real[order]
        return self

    def decision(self, X):
        # linear discriminant: x^T S^-1 mu_k - 0.5 mu_k^T S^-1 mu_k + log pi_k
        A = self.means @ self.Sw_inv
        return X @ A.T - 0.5 * np.sum(A * self.means, 1) + np.log(self.priors)

    def predict(self, X):
        return self.classes[self.decision(X).argmax(1)]

    def predict_proba(self, X):
        z = self.decision(X); z -= z.max(1, keepdims=True)
        p = np.exp(z); return p / p.sum(1, keepdims=True)

    def transform(self, X):
        return (X - X.mean(0)) @ self.W


X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

ours = LDA().fit(Xtr, ytr)
sk = LinearDiscriminantAnalysis(solver="lsqr").fit(Xtr, ytr)
print(f"test accuracy: ours {(ours.predict(Xte) == yte).mean():.3f} | sklearn {sk.score(Xte, yte):.3f}")
print(f"max |prob diff| vs sklearn: {np.abs(ours.predict_proba(Xte) - sk.predict_proba(Xte)).max():.2e}")
print("Fisher eigenvalues (K-1 = 2 directions):", ours.evals.round(3))

# 13-D -> 2-D: LDA uses labels, PCA does not. Class separation = between / within variance ratio.
def fisher_ratio(Z, y):
    m = Z.mean(0)
    sb = sum(np.sum(y == k) * np.sum((Z[y == k].mean(0) - m) ** 2) for k in np.unique(y))
    sw = sum(np.sum((Z[y == k] - Z[y == k].mean(0)) ** 2) for k in np.unique(y))
    return sb / sw
print(f"2-D class separation (between/within): LDA {fisher_ratio(ours.transform(Xte), yte):.2f} | "
      f"PCA {fisher_ratio(PCA(2).fit(Xtr).transform(Xte), yte):.2f}")

# small-n, high-d: Sw becomes singular -> shrinkage
rng = np.random.default_rng(0)
Xs = rng.normal(size=(40, 100)); ys = np.r_[np.zeros(20), np.ones(20)]; Xs[ys == 1, :5] += 1.0
Xv = rng.normal(size=(1000, 100)); yv = np.r_[np.zeros(500), np.ones(500)]; Xv[yv == 1, :5] += 1.0
print(f"rank of Sw with n=40, d=100: {np.linalg.matrix_rank(np.cov(Xs.T))}")
for name, m in [("lsqr, no shrinkage", LinearDiscriminantAnalysis(solver="lsqr")),
                ("lsqr, shrinkage='auto'", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]:
    print(f"{name}: test acc {m.fit(Xs, ys).score(Xv, yv):.3f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
test accuracy: ours 1.000 | sklearn 1.000
max |prob diff| vs sklearn: 4.44e-16
Fisher eigenvalues (K-1 = 2 directions): [8.262 4.226]
2-D class separation (between/within): LDA 5.33 | PCA 3.46
rank of Sw with n=40, d=100: 39
lsqr, no shrinkage: test acc 0.606
lsqr, shrinkage='auto': test acc 0.747
```

On the 3-class wine data (13 features, 124 training rows) both implementations classify all 54 test wines correctly, and their posterior probabilities agree to floating-point precision, so the discriminant formula and the `/n` pooled covariance are exactly what scikit-learn computes. There are exactly `K−1 = 2` non-zero Fisher eigenvalues. Projected to 2-D, LDA's classes have a between/within variance ratio of 5.33 against 3.46 for PCA's first two components: PCA keeps directions of largest total variance, LDA keeps directions that separate the labels. With 40 samples and 100 features the within-class covariance has rank 39 and cannot be inverted reliably; plain LDA reaches 60.6% on a large test set, Ledoit-Wolf shrinkage 74.7%.

## Choosing / trade-offs
- **LDA vs logistic regression.** With few samples and roughly Gaussian classes LDA is often better; with many samples, outliers or non-Gaussian features logistic regression is the safer default.
- **LDA vs QDA.** QDA needs `K·d(d+1)/2` covariance parameters; use it only when classes clearly differ in spread and you have the data. Regularised discriminant analysis interpolates between the two.
- **LDA vs PCA for reduction.** LDA when you have labels and want class separation (max `K−1` dims); PCA when unsupervised or you need more dimensions ([[pca-from-scratch]], [[dimensionality-reduction]]).
- **Solvers in scikit-learn.** `svd` (default) avoids forming the covariance and supports `transform` but not shrinkage; `lsqr` supports shrinkage but not `transform`; `eigen` supports both.

## Gotchas
- `S_W⁻¹S_B` is not symmetric, so `np.linalg.eig` can return tiny imaginary parts; take `.real`, or solve the generalised symmetric problem with `scipy.linalg.eigh(S_B, S_W)`.
- Fisher directions are defined up to scale and sign; compare projections by class separation, not by coordinates.
- `n_components` cannot exceed `min(K−1, d)`; asking for more raises an error in scikit-learn.
- When `d ≥ n − K`, `S_W` is singular. Use shrinkage, PCA first, or a regularised model, never a raw `inv`.
- Priors default to class frequencies. If deployment class balance differs, pass `priors=` or adjust the `log π_k` term ([[probability-calibration]]).
- Standardising does not change LDA's predictions (it is affine-invariant) but does make the projection weights interpretable.

## Related
- [[naive-bayes-from-scratch]] - the diagonal-covariance generative classifier.
- [[logistic-regression-from-scratch]] - the discriminative classifier with the same linear boundary.
- [[gmm-em-from-scratch]] - Gaussian class-conditionals without labels.
- [[pca-from-scratch]] - the unsupervised projection LDA is compared with.
- [[dimensionality-reduction]] - PCA, LDA, t-SNE and UMAP compared.
- [[linear-models]] - the wider family of linear classifiers and regressors.

## References
- scikit-learn user guide, linear and quadratic discriminant analysis: https://scikit-learn.org/stable/modules/lda_qda.html
- scikit-learn `LinearDiscriminantAnalysis`: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- Hastie, Tibshirani and Friedman, The Elements of Statistical Learning, section 4.3: https://hastie.su.domains/ElemStatLearn/
- Fisher (1936), "The use of multiple measurements in taxonomic problems", Annals of Eugenics: https://doi.org/10.1111/j.1469-1809.1936.tb02137.x
