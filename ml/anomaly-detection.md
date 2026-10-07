---
title: Anomaly detection
category: ml
tags: [anomaly-detection, outlier, isolation-forest, autoencoder, fraud, predictive-maintenance, pyod]
use_cases:
  - "flag fraudulent or unusual transactions with few or no labels"
  - "detect equipment failure early from sensor streams in a factory"
  - "monitor application metrics and alert on abnormal behaviour"
  - "find defective items from images of only good samples"
  - "detect data drift or corrupted records in an ML pipeline"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/outlier_detection.html
  - https://pyod.readthedocs.io/en/latest/
---

# Anomaly detection

## Summary
Anomaly (outlier) detection finds rare observations that deviate from normal behaviour, typically when labelled anomalies are scarce or absent. Choose by data type: tabular (Isolation Forest, LOF), time series (forecast residuals, seasonal decomposition), images (reconstruction or feature-distance methods), text/logs (embeddings plus distance). If you do have enough labelled anomalies, a supervised classifier usually wins.

## Key concepts
- Settings: supervised (labels for both), semi-supervised/novelty (train on normal only), unsupervised (unlabelled, assume anomalies are rare).
- Point, contextual (abnormal only in context, e.g. 30C in winter) and collective anomalies (a pattern over a window).
- Scores, not labels: models output an anomaly score; the threshold encodes cost of false alarms vs misses and must be chosen with domain input.
- Isolation Forest: random splits isolate anomalies in fewer steps; fast, little tuning.
- LOF / kNN distance: density-based, good for local anomalies; scales poorly.
- One-class SVM: sensitive to scaling and kernel, slow on big data.
- Autoencoders: high reconstruction error implies anomaly; for sensors, images, logs.
- Statistical baselines: z-score/MAD, rolling quantiles; surprisingly strong and explainable.
- Evaluation: PR-AUC, precision@k, alert rate per day; ROC-AUC flatters rare-event problems.

## When to use / scenarios
- Payments: unusual transaction patterns as a first-line filter alongside rules and supervised fraud models.
- Manufacturing/IoT: vibration and temperature drift for predictive maintenance; visual surface defects (PatchCore-style, via the anomalib library).
- IT/Security: unusual login, traffic or log patterns (defensive monitoring).
- Healthcare ops/finance: audit sampling of unusual claims or invoices.
- NOT for: anything where "anomaly" must be defined precisely by rules (use rules), or when labelled examples abound (use [[gradient-boosting-tabular]]).

## Setup & code
```bash
pip install scikit-learn
```
```python
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(0)
normal = rng.normal(0, 1, size=(1000, 3))
outliers = rng.uniform(-6, 6, size=(10, 3))
X = np.vstack([normal, outliers])

iso = IsolationForest(n_estimators=200, contamination="auto", random_state=0).fit(X)
scores = -iso.score_samples(X)            # higher = more anomalous
flag = scores > np.quantile(scores, 0.99) # threshold by alert budget
print(flag.sum(), "flagged")
```
PyOD (`pip install pyod`) offers 40+ detectors behind one API for comparison.

## Choosing / trade-offs
- Start with Isolation Forest or robust z-scores; add complexity only if recall on known incidents is poor.
- Time series: forecast then flag large residuals (see [[time-series-forecasting]]); handles seasonality that raw thresholds cannot.
- Images with only good samples: pretrained-feature methods are strong baselines; autoencoders often reconstruct defects too well.
- Alert budget: set the threshold to a tolerable alerts-per-day, then measure precision with analyst review.
- Explainability: attach top contributing features (SHAP or per-feature z-scores) so analysts can act.

## Gotchas
- Contamination parameter is a guess; do not treat it as truth.
- Training data containing anomalies teaches the model they are normal; clean or use robust methods.
- Concept drift: "normal" changes (seasonality, new product), so retrain and monitor alert rate.
- Unscaled features dominate distance-based methods.
- No labels means no accuracy number; build a small reviewed set early to validate.
- Class-imbalanced metrics: report precision at the review capacity you actually have.

## Related
- [[classic-ml-scikit-learn]] - preprocessing and the estimator API.
- [[time-series-forecasting]] - residual-based detection.
- [[gradient-boosting-tabular]] - supervised alternative when labels exist.
- [[image-classification]] - visual inspection pipelines.
- [[data-labeling-and-synthetic-data]] - building a validation set for anomalies.
- [[isolation-forest-from-scratch]] - Isolation Forest built by hand: random isolation trees and path-length scores.

## References
- https://scikit-learn.org/stable/modules/outlier_detection.html
- https://pyod.readthedocs.io/en/latest/
