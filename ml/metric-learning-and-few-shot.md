---
title: Metric learning and few-shot classification
category: ml
tags: [metric-learning, few-shot, siamese, triplet-loss, contrastive-loss, prototypical-networks, arcface, face-verification, re-identification]
use_cases:
  - "recognize faces, products or people across cameras from one or a few reference images"
  - "add new classes without retraining the classifier"
  - "verify whether two items (signatures, documents, users) are the same"
  - "search for visually or semantically similar items by embedding distance"
  - "classify with 1-5 labelled examples per class"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1503.03832
  - https://arxiv.org/abs/1703.05175
  - https://arxiv.org/abs/1801.07698
  - https://kevinmusgrave.github.io/pytorch-metric-learning/
---

# Metric learning and few-shot classification

## Summary
Metric learning trains an embedding where distance means similarity: items of
the same class sit close together, different classes far apart. Classification
then becomes nearest-neighbour search against stored examples, so new classes
are added by storing a few embeddings, not by retraining. It underlies face
verification, person/vehicle re-identification, product matching and most
few-shot classification.

## Key concepts
- Siamese setup: one shared encoder applied to two (or three) inputs, compared
  by distance (usually cosine on L2-normalized embeddings).
- Pair/contrastive loss: pull positive pairs together, push negatives beyond
  a margin.
- Triplet loss: `max(0, d(a, p) - d(a, n) + margin)` for anchor, positive,
  negative. Needs good negatives.
- Mining: random triplets become easy quickly; semi-hard or batch-hard mining
  picks informative ones. Losses with many negatives per batch (InfoNCE,
  multi-similarity) reduce the need.
- Classification-style losses: ArcFace/CosFace train a softmax with an angular
  margin; usually the strongest and simplest choice for face and re-ID with
  many identities.
- Verification vs identification: verification thresholds one distance
  ("same person?"); identification ranks a gallery ("who is this?").
- Few-shot episode (N-way K-shot): N classes, K labelled "support" examples
  each, classify "query" examples. Prototypical networks average the support
  embeddings into one prototype per class and pick the nearest.
- Open-set: queries may belong to no known class; reject when the nearest
  distance exceeds a threshold.

## When to use / scenarios
- Security/retail: face verification, person and vehicle re-ID across cameras
  ([[face-and-pose]], [[video-analytics]]).
- E-commerce: duplicate listings, "visually similar" products, matching a
  shopper's photo to a catalogue ([[ecommerce-retail]]).
- Manufacturing: a defect type with five examples; new SKUs added weekly
  ([[manufacturing-iot]]).
- Finance/identity: signature or document verification.
- Text: intent classification with few examples via sentence embeddings
  (SetFit fine-tunes a sentence-transformer contrastively).
- NOT for: a fixed label set with plenty of labels per class; a standard
  classifier is simpler and more accurate. Also try off-the-shelf embeddings
  (CLIP/DINOv2 for images, sentence-transformers for text) with a nearest-
  prototype or logistic-regression head before training your own
  ([[embeddings]], [[image-classification]]).

## Setup & code
Triplet training on digits 0-6 with 70% of their rows; digits 7-9 are never
seen. Then nearest-prototype few-shot episodes on held-out rows.
```bash
pip install torch scikit-learn
```
```python
import torch
from torch import nn
from sklearn.datasets import load_digits

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
train = (y < 7) & (torch.rand(len(y)) < 0.7)   # 70% of digits 0-6; digits 7-9 never seen

emb = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 32))
loss_fn = nn.TripletMarginLoss(margin=0.2)
opt = torch.optim.Adam(emb.parameters(), lr=1e-3)
Xs, ys = X[train], y[train]
same = ys[:, None] == ys[None, :]
for step in range(300):
    a = torch.randint(len(Xs), (64,))
    pos = torch.multinomial(same[a].float(), 1).squeeze(1)      # same class as anchor
    neg = torch.multinomial((~same[a]).float(), 1).squeeze(1)   # different class
    f = lambda idx: nn.functional.normalize(emb(Xs[idx]), dim=1)
    loss = loss_fn(f(a), f(pos), f(neg))
    opt.zero_grad(); loss.backward(); opt.step()

def few_shot(feats, classes, k_shot, episodes=200):
    """N-way k-shot episodes on held-out rows: label each query by its nearest class prototype."""
    g = torch.Generator().manual_seed(1)
    F, Y = feats[~train], y[~train]
    acc = []
    for _ in range(episodes):
        protos, queries, labels = [], [], []
        for c_i, c in enumerate(classes):
            idx = torch.nonzero(Y == c).flatten()
            idx = idx[torch.randperm(len(idx), generator=g)]
            protos.append(F[idx[:k_shot]].mean(0))
            queries.append(F[idx[k_shot:k_shot + 10]]); labels += [c_i] * 10
        d = torch.cdist(torch.cat(queries), torch.stack(protos))
        acc.append((d.argmin(1) == torch.tensor(labels)).float().mean())
    return torch.stack(acc).mean().item()

with torch.no_grad():
    learned = nn.functional.normalize(emb(X), dim=1)
for name, classes in [("seen 0-6 (new rows)", [0, 1, 2, 3, 4, 5, 6]), ("unseen 7-9", [7, 8, 9])]:
    for k in (1, 5):
        print(f"{name} {len(classes)}-way {k}-shot: raw pixels {few_shot(X, classes, k):.3f}"
              f"  triplet embedding {few_shot(learned, classes, k):.3f}")
```
Output (torch 2.13.0 CPU), mean episode accuracy:

| Episode | Raw pixels | Triplet embedding |
|---|---|---|
| Seen classes 0-6, new rows, 7-way 1-shot | 0.745 | 0.969 |
| Seen classes 0-6, new rows, 7-way 5-shot | 0.903 | 0.985 |
| Unseen classes 7-9, 3-way 1-shot | 0.751 | 0.620 |
| Unseen classes 7-9, 3-way 5-shot | 0.900 | 0.781 |

The embedding is excellent for new examples of the classes it was trained
on and worse than raw pixels for classes it never saw. Seven training classes
teach features that separate those seven, not digits in general. Changing
margin, steps and width did not reverse this. Generalizing to new classes is
what face and re-ID models get from training on thousands of identities; with
few training classes, start from a large pretrained encoder instead.

For production training, pytorch-metric-learning provides losses (triplet,
ArcFace, multi-similarity), miners and accuracy calculators; for search over
large galleries use a vector index ([[vector-databases]]).

## Choosing / trade-offs
- Many identities, a few images each (faces, re-ID): ArcFace/CosFace-style
  classification loss, then use the embedding.
- Pairs are all you have (match / no match labels): contrastive or triplet loss.
- Few classes and few labels: pretrained embeddings + nearest prototype or
  logistic regression; fine-tune only if that fails
  ([[semi-supervised-and-active-learning]] to label more efficiently).
- Gallery size: brute-force cosine up to ~100k embeddings; ANN index beyond.
- Threshold for verification: pick it on a validation set at the false-accept
  rate the application tolerates (see [[model-evaluation-and-metrics]]).

## Gotchas
- Evaluating on classes seen in training overstates few-shot accuracy; split
  by class, not by row, when the real use is new classes.
- Leakage across splits: the same person/product in train and test under
  different IDs.
- Random triplets collapse to easy cases and training stalls; use mining or
  in-batch negatives.
- Unnormalized embeddings let the loss cheat by scaling; L2-normalize and use
  cosine distance.
- One global threshold can work badly across demographics or lighting
  conditions; check error rates per subgroup, especially for face biometrics,
  which are also legally regulated in many jurisdictions
  ([[ai-security-privacy-compliance]]).
- Changing the encoder invalidates every stored embedding; re-embed the gallery.

## Related
- [[embeddings]] - pretrained embedding models and similarity search.
- [[autoencoders-and-self-supervised-learning]] - contrastive pretraining without labels.
- [[face-and-pose]] - face recognition pipelines built on this.
- [[semi-supervised-and-active-learning]] - other routes when labels are scarce.
- [[vector-databases]] - storing and searching galleries.
- [[recommender-systems]] - two-tower retrieval uses the same contrastive idea.

## References
- Schroff et al., FaceNet (triplet loss): https://arxiv.org/abs/1503.03832
- Snell et al., Prototypical Networks for Few-shot Learning: https://arxiv.org/abs/1703.05175
- Deng et al., ArcFace: https://arxiv.org/abs/1801.07698
- pytorch-metric-learning: https://kevinmusgrave.github.io/pytorch-metric-learning/
