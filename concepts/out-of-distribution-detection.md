---
title: Out-of-distribution detection
category: concepts
tags: [ood-detection, out-of-distribution, novelty-detection, open-set, softmax-confidence, energy-score, mahalanobis, selective-prediction, uncertainty, pytorch]
use_cases:
  - "make a classifier say 'I don't know' on inputs unlike its training data"
  - "flag unknown defect types, new fraud patterns or unseen document types for human review"
  - "detect when production inputs drift away from the training distribution"
  - "reject garbage inputs (wrong camera, blank image, other language) before they reach the model"
  - "add a confidence gate to an automated decision pipeline"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1610.02136
  - https://arxiv.org/abs/2010.03759
  - https://arxiv.org/abs/1807.03888
  - https://arxiv.org/abs/2110.11334
  - https://github.com/Jingkang50/OpenOOD
---

# Out-of-distribution detection

## Summary
A classifier trained on K classes will still assign one of them, often confidently, to an input from none of them: a cat photo given to a defect classifier, a scanned receipt given to an invoice model. Out-of-distribution (OOD) detection adds a score that says how unlike the training data an input is, so the system can abstain or route it to a human. Simple post-hoc scores on an already-trained model (max softmax probability, energy, Mahalanobis or k-NN distance in feature space) are strong baselines; they need only in-distribution data to fit and a small labelled OOD set to choose the threshold.

## Key concepts
- **Kinds of shift.** *Semantic* shift (new classes: open-set recognition) vs *covariate* shift (same classes, different conditions: new camera, season). OOD detection usually targets semantic shift; covariate shift is often better handled by monitoring ([[online-learning-and-concept-drift]]) and robustness ([[adversarial-examples-and-robustness]]).
- **Post-hoc scores** (no retraining):
  - **MSP** (Hendrycks & Gimpel 2016): max softmax probability; low = OOD. The baseline everything is compared with.
  - **Energy** (Liu 2020): `logsumexp(logits)`; uses the logits' scale, which softmax throws away. Usually better than MSP, same cost.
  - **Max logit**, **ODIN** (temperature scaling + input perturbation).
  - **Feature-distance**: Mahalanobis distance to class means in the penultimate layer (shared covariance), or **k-NN** distance to training embeddings. Often the strongest for far-OOD and works with any embedding model.
  - **ReAct / ASH**: clip or prune penultimate activations before computing the score; cheap boosts.
- **Training-time methods.** Outlier exposure (add a generic OOD dataset with a uniform-label loss), contrastive/self-supervised pretraining (better features → better distance scores), deep ensembles and MC dropout (disagreement as uncertainty).
- **Density models** (normalizing flows, VAEs) seem natural but are known to give *higher* likelihood to some OOD data (e.g. SVHN under a CIFAR model); prefer classifier/feature-based scores ([[normalizing-flows-and-energy-based-models]]).
- **Metrics.** AUROC (ID vs OOD separation), **FPR@95TPR** (share of OOD accepted when 95% of ID is kept), AUPR. Report near-OOD (similar classes) and far-OOD (very different data) separately.
- **Selective prediction.** The production form: accept if score ≥ threshold, otherwise abstain/escalate; report coverage (share accepted) vs accuracy on accepted ([[conformal-prediction-and-uncertainty]] gives coverage guarantees for the in-distribution part).

## When to use / scenarios
- Manufacturing inspection: a model trained on known defect types must flag a never-seen defect or a misplaced part rather than call it "scratch" ([[manufacturing-iot]]).
- Medical imaging and triage: reject images from a different modality, body part or scanner; route uncertain cases to a clinician ([[healthcare]]).
- Document processing: an invoice classifier receiving contracts, photos or blank pages; route to manual queue ([[document-processing]]).
- Fraud and security: new attack patterns look unlike known classes; combine with [[anomaly-detection]].
- Customer-support intent classifiers: detect "none of the known intents" and fall back to a human or an LLM.
- NOT a replacement for drift monitoring on aggregate statistics, and NOT for detecting adversarially crafted inputs (attackers can craft confident OOD inputs).
- If you have no classifier at all, only "normal" data, you need anomaly detection instead ([[anomaly-detection]]).

## Setup & code
```bash
pip install torch scikit-learn
```
Train on digits 0-5, treat 6-9 as unseen (near-OOD) and random noise as far-OOD, and compare MSP, energy and Mahalanobis scores by AUROC and FPR@95TPR:
```python
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score

torch.manual_seed(0); rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
known = y < 6
idx = torch.randperm(int(known.sum()))
Xk, yk = X[known][idx], y[known][idx]
Xtr, ytr, Xid = Xk[:700], yk[:700], Xk[700:]      # in-distribution train / test
Xnear = X[~known]                                   # unseen digits 6-9
Xfar = torch.tensor(rng.uniform(0, 1, (300, 64)), dtype=torch.float32)

feat = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
head = nn.Linear(64, 6)
opt = torch.optim.Adam([*feat.parameters(), *head.parameters()], lr=1e-3)
for _ in range(100):
    opt.zero_grad(); nn.functional.cross_entropy(head(feat(Xtr)), ytr).backward(); opt.step()

with torch.no_grad():
    ftr = feat(Xtr)
    means = torch.stack([ftr[ytr == c].mean(0) for c in range(6)])
    cov = torch.cov((ftr - means[ytr]).T) + 1e-3 * torch.eye(64)
    prec = torch.linalg.inv(cov)

    def scores(x):  # higher = more in-distribution
        f = feat(x); logits = head(f)
        d = f[:, None, :] - means[None]              # (N, classes, D)
        maha = torch.einsum("ncd,de,nce->nc", d, prec, d).min(1).values
        return {"MSP": logits.softmax(1).max(1).values,
                "energy": torch.logsumexp(logits, 1),
                "mahalanobis": -maha}

    s_id = scores(Xid)
    print("ID accuracy", (head(feat(Xid)).argmax(1) == yk[700:]).float().mean().item())
    for ood_name, Xo in [("near (digits 6-9)", Xnear), ("far (noise)", Xfar)]:
        s_ood = scores(Xo)
        for m in s_id:
            a, b = s_id[m].numpy(), s_ood[m].numpy()
            auroc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
            thr = np.percentile(a, 5)                # keep 95% of ID
            fpr = (b >= thr).mean()
            print(f"{ood_name:18s} {m:12s} AUROC {auroc:.3f}  FPR@95TPR {fpr:.3f}")
```
On this toy run no score wins everywhere: energy beats MSP on both sets and is best on the near-OOD digits, while Mahalanobis is near-perfect on the far-OOD noise but weak on near-OOD. That is the usual pattern, and the reason to evaluate several scores. Note the threshold is picked from ID data alone (keep 95% of it); the OOD set is used only to measure. For image models at scale, the OpenOOD benchmark/library implements most published scores.

## Choosing / trade-offs
- **Start post-hoc.** MSP and energy are one line on any classifier; Mahalanobis/k-NN need stored training features (memory, plus a vector index like FAISS for k-NN at scale). Try all on a validation OOD set; which wins depends on the model and the shift.
- **Better features beat cleverer scores.** Large pretrained backbones (CLIP, DINOv2, sentence-transformers) plus k-NN distance are a strong default for open-set problems.
- **Near vs far OOD.** Far OOD (noise, other domains) is easy for most scores; near OOD (similar new classes) is hard and is where feature-distance and better pretraining pay off.
- **Threshold = business decision.** Set it from the acceptable abstention rate on ID data or the cost of a missed OOD; revisit as traffic changes.
- **Outlier exposure** helps when you can collect realistic "other" data, but risks overfitting to that particular OOD set.

## Gotchas
- Softmax confidence is not calibrated probability of correctness, and ReLU networks can be arbitrarily confident far from the data; never treat 0.99 as "safe" without measuring.
- Temperature scaling improves calibration on ID data but does little for OOD detection by itself.
- Evaluating on one convenient OOD dataset overstates performance; test several, including near-OOD from your real domain.
- Feature statistics (means, covariance) must come from the same preprocessing and model checkpoint used in production; recompute after every retrain.
- Batch-norm in train mode or test-time augmentation changes the scores; compute in `eval()`/`no_grad`.
- A score that flags 20% of production traffic is a drift signal, not just a set of outliers; investigate and retrain ([[mlops-lifecycle]]).

## Related
- [[anomaly-detection]] - when you only have "normal" data and no classifier.
- [[conformal-prediction-and-uncertainty]] - calibrated prediction sets and coverage for in-distribution inputs.
- [[adversarial-examples-and-robustness]] - deliberate attacks vs natural shift.
- [[online-learning-and-concept-drift]] - monitoring and adapting to distribution change over time.
- [[metric-learning-and-few-shot]] - embedding spaces where distance-based OOD scores work well.
- [[model-evaluation-and-metrics]] - AUROC, PR-AUC and threshold selection.

## References
- Hendrycks and Gimpel, A Baseline for Detecting Misclassified and Out-of-Distribution Examples: https://arxiv.org/abs/1610.02136
- Liu et al., Energy-based Out-of-distribution Detection: https://arxiv.org/abs/2010.03759
- Lee et al., A Simple Unified Framework for Detecting OOD Samples (Mahalanobis): https://arxiv.org/abs/1807.03888
- Yang et al., Generalized Out-of-Distribution Detection: A Survey: https://arxiv.org/abs/2110.11334
- OpenOOD benchmark: https://github.com/Jingkang50/OpenOOD
