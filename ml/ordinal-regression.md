---
title: Ordinal regression (ordered classes, ratings, severity grades)
category: ml
tags: [ordinal-regression, ordered-logit, proportional-odds, cumulative-link, coral, ratings, severity, pytorch, statsmodels]
use_cases:
  - "predict a 1-5 star rating or a satisfaction score"
  - "grade disease severity, damage level or risk tier (low/medium/high)"
  - "classify where predicting 'severe' for 'mild' is worse than predicting 'moderate'"
  - "estimate age group, education level or credit grade"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html
  - https://arxiv.org/abs/1901.07884
  - https://arxiv.org/abs/2111.08851
---

# Ordinal regression (ordered classes, ratings, severity grades)

## Summary
Ordinal regression predicts a label from a set of **ordered** categories (1-5 stars, mild < moderate < severe), where the gaps between levels are not known to be equal. Treating it as plain classification throws away the order. Treating it as regression assumes equal spacing. The standard approach is a **cumulative link model**: one latent score per row and K-1 increasing thresholds that cut the score into K levels. Train it with K-1 binary "is y > k?" targets, and evaluate it with order-aware metrics (MAE, quadratic weighted kappa).

## Key concepts
- **Cumulative probabilities.** Model P(y > k | x) = σ(f(x) - θ_k) for k = 0..K-2, with θ_0 < θ_1 < ... . The predicted level is the number of thresholds the score passes. The class probability is P(y = k) = P(y > k-1) - P(y > k).
- **Proportional odds (ordered logit).** f(x) is linear and shared across thresholds, so a feature moves the whole distribution up or down by the same log-odds. The probit link gives ordered probit.
- **Rank consistency.** Thresholds must be ordered so the cumulative probabilities are monotone. CORAL shares the weights and learns separate biases. CORN uses conditional training to allow a richer network while staying consistent.
- **Extended binary targets.** y=2 with K=5 becomes [1, 1, 0, 0]. Each output is a binary cross-entropy target. Any classifier or neural net can be made ordinal this way.
- **Metrics.** Accuracy ignores how far off a prediction is. Use MAE on the level index, off-by-≥2 rate, and quadratic weighted kappa (QWK, `cohen_kappa_score(weights="quadratic")`), the standard metric for graded scoring.

## When to use / scenarios
- Healthcare: disease stage, pain scale, diabetic retinopathy grade (0-4, QWK was the Kaggle metric).
- Insurance and engineering: damage severity, defect grade, risk tier.
- Product and HR: review stars, NPS buckets, performance ratings, essay scoring.
- Credit: rating grades (AAA..D), delinquency buckets.
- Vision: age estimation in age bands, image quality score.
- NOT needed when the levels are numerous and roughly evenly spaced (0-100 score): plain regression with rounding is simpler.
- NOT ordinal when the order is not real (product categories, colors): use [[multiclass-classification-strategies]].

## Setup & code
```bash
pip install torch numpy
# classic statistical version with p-values: pip install statsmodels
# -> statsmodels.miscmodels.ordinal_model.OrderedModel(y, X, distr="logit")
```
A cumulative-link (CORAL-style) head in PyTorch vs a plain softmax head on 5 ordered levels:
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
K = 5                                   # ratings 0..4
X = torch.randn(3000, 8)
latent = X @ torch.linspace(1, -1, 8) + 0.5 * torch.randn(3000)
y = torch.bucketize(latent, torch.tensor([-2.0, -0.7, 0.7, 2.0]))   # ordered labels 0..4
X_tr, y_tr, X_te, y_te = X[:2400], y[:2400], X[2400:], y[2400:]


class CumulativeLink(nn.Module):
    """One score per row plus K-1 ordered thresholds (proportional-odds / CORAL-style)."""
    def __init__(self, d, k):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=False)
        self.first = nn.Parameter(torch.tensor(-1.0))
        self.gaps = nn.Parameter(torch.zeros(k - 2))      # softplus keeps thresholds increasing

    def thresholds(self):
        return torch.cat([self.first.view(1), self.first + torch.cumsum(F.softplus(self.gaps), 0)])

    def forward(self, x):                                 # logits of P(y > k) for k=0..K-2
        return self.score(x) - self.thresholds()


def ordinal_targets(y, k):                                # y=2 -> [1, 1, 0, 0]
    return (y.unsqueeze(1) > torch.arange(k - 1)).float()


def predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(1)


def train(model, loss_fn, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X_tr)).backward()
        opt.step()
    return model


ord_model = train(CumulativeLink(8, K),
                  lambda out: F.binary_cross_entropy_with_logits(out, ordinal_targets(y_tr, K)))
ce_model = train(nn.Linear(8, K), lambda out: F.cross_entropy(out, y_tr))

with torch.no_grad():
    for name, pred in [("cumulative link", predict(ord_model(X_te))),
                       ("plain softmax  ", ce_model(X_te).argmax(1))]:
        acc = (pred == y_te).float().mean().item()
        mae = (pred - y_te).abs().float().mean().item()
        off2 = ((pred - y_te).abs() >= 2).float().mean().item()
        print(f"{name}: acc={acc:.3f}  MAE={mae:.3f}  off-by-2+={off2:.3f}")
    print("thresholds:", np.round(ord_model.thresholds().numpy(), 2))
```
Output (torch 2.13.0, CPU):
```
cumulative link: acc=0.732  MAE=0.268  off-by-2+=0.000
plain softmax  : acc=0.723  MAE=0.277  off-by-2+=0.000
thresholds: [-6.41 -2.21  2.25  6.44]
```
On this clean, linear toy data both heads do about as well. The ordinal head uses 8 + 4 parameters instead of 8x5 + 5, and its thresholds come out ordered by construction. The gap widens with less data per level, rare extreme levels, or a deep network that can overfit a softmax. For deep models, the `coral-pytorch` package provides CORAL/CORN layers and losses.

## Choosing / trade-offs
- **Ordered logit (statsmodels `OrderedModel`)** for small tabular data when you need coefficients and p-values. It assumes proportional odds; test it, and relax it (partial proportional odds) if one feature affects the low and high levels differently.
- **CORAL/CORN head on a neural net** for images, text or large data. CORN is more flexible; CORAL is simpler and more constrained.
- **Regression + rounding/threshold tuning.** Strong baseline for gradient boosting: fit a regressor on the level index, then tune the K-1 cut points on validation to maximize QWK (common in Kaggle solutions).
- **Plain multiclass** when you have lots of data per level and only care about exact accuracy. It can produce non-monotone probabilities (high P(mild) and P(severe), low P(moderate)).
- **Expected value vs mode.** Predict the median (count of P(y>k) > 0.5) for MAE. Use argmax of class probabilities for accuracy. Use the expected value for a continuous score. Pick the one that matches the metric.

## Gotchas
- Plain softmax with accuracy as the metric hides the costly errors. Always report MAE or QWK and the confusion matrix. Errors far from the diagonal are the ones users notice.
- Unordered thresholds (independent biases trained freely) give inconsistent cumulative probabilities: P(y>3) > P(y>2). Parameterize them as a cumulative sum of positive gaps, as above.
- A bias in the score layer plus free thresholds is not identifiable. Drop one (the example uses `bias=False`).
- Rating scales are noisy and rater-dependent. Two annotators often differ by one level, so inter-rater QWK is the ceiling to compare against ([[label-noise-and-data-cleaning]]).
- Extreme levels are usually rare (few 1-star reviews, few critical cases). Stratify splits and check per-level recall ([[imbalanced-data]]).
- When the scale changes (4 levels become 5, a new grading guideline), old labels need mapping or retraining. Thresholds do not transfer.

## Related
- [[multiclass-classification-strategies]] - the unordered case.
- [[generalized-linear-models]] - ordered logit is a GLM with a cumulative link.
- [[loss-functions]] - cross-entropy and BCE used by the extended-binary trick.
- [[model-evaluation-and-metrics]] - MAE, confusion matrices, kappa.
- [[learning-to-rank]] - ordering items rather than assigning a level to each.
- [[imbalanced-data]] - rare extreme levels.

## References
- statsmodels, Ordinal regression (`OrderedModel`): https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html
- Cao, Mirjalili & Raschka, "Rank consistent ordinal regression for neural networks" (CORAL): https://arxiv.org/abs/1901.07884
- Shi, Cao & Raschka, "Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities" (CORN): https://arxiv.org/abs/2111.08851
- McCullagh, "Regression Models for Ordinal Data", JRSS-B 1980.
- scikit-learn, `cohen_kappa_score` (quadratic weights): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
