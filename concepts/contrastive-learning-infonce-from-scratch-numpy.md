---
title: Contrastive learning with InfoNCE from scratch in NumPy (positive pairs, in-batch negatives, temperature)
category: concepts
tags: [contrastive-learning, infonce, nt-xent, simclr, clip, self-supervised-learning, representation-learning, temperature, in-batch-negatives, augmentation, embeddings, backpropagation, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement the InfoNCE (NT-Xent / CLIP) loss and its gradient in NumPy"
  - "learn an embedding from unlabelled data with positive pairs, then evaluate it with k-NN"
  - "understand how augmentations define what a contrastive model learns to ignore"
  - "explain temperature, in-batch negatives and the log N bound in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1807.03748
  - https://arxiv.org/abs/2002.05709
  - https://arxiv.org/abs/2103.00020
  - https://arxiv.org/abs/2005.10242
---

# Contrastive learning with InfoNCE from scratch in NumPy (positive pairs, in-batch negatives, temperature)

## Summary
Contrastive learning trains an encoder so that two views of the same example (a positive pair) map close together while the other examples in the batch (negatives) map apart. No labels are needed. InfoNCE turns this into a classification problem: among N candidates, find the true partner, using cross-entropy over cosine similarities divided by a temperature τ. SimCLR (two augmentations of one image), CLIP (an image and its caption) and most text-embedding models train with this loss. Below, a linear encoder is trained on data whose class signal sits in 4 of 32 dimensions, buried under high-variance nuisance noise. k-NN accuracy rises from 0.15 on the raw inputs to 0.93 on the learned 16-dimensional embedding, without the encoder ever seeing a label. Only 0 to 3% of the embedding's variance still comes from the nuisance dimensions, against 98% for a random projection. The augmentations decide what counts as nuisance.

## Key concepts
- **Loss.** With L2-normalised embeddings `a_i, b_i` and logits `s_ij = a_iᵀ b_j / τ`, InfoNCE is `−log softmax(s_i·)_i`, the cross-entropy with the diagonal as the target. The symmetric (CLIP) version averages the a→b and b→a directions. SimCLR's NT-Xent also uses the other view within the same side as negatives.
- **In-batch negatives.** Every other example in the batch serves as a negative, so N pairs give N − 1 negatives each at no extra encoding cost.
- **Mutual-information bound.** `I(a; b) ≥ log N − L_InfoNCE`. The loss of a random encoder is about `log N`, so loss values cannot be compared across batch sizes.
- **Temperature.** A small τ sharpens the softmax and concentrates the gradient on the hardest negatives. A large τ treats all negatives alike. τ also caps how confident the model can be: the logits are bounded by `±1/τ`.
- **Alignment and uniformity.** The loss pulls positives together (alignment) and spreads all embeddings over the hypersphere (uniformity). Without negatives, the trivial solution maps everything to one point (collapse).
- **Augmentations define invariance.** Whatever changes between the two views is what the encoder learns to ignore. Here the views resample the nuisance dimensions and keep the class signal, so the encoder learns to drop the nuisance.

## When to use / scenarios
- Learning: self-supervision as classification, and how a loss with no labels can still recover class structure.
- Interviews: "how does CLIP train", "why does SimCLR need large batches", "what does temperature do", "how do you prevent representation collapse".
- Practice: pretraining an image or audio encoder on unlabelled data before fine-tuning on few labels; training text or code embeddings for retrieval and RAG from (query, passage) pairs; multimodal retrieval (image-text, audio-text); and deduplication or near-duplicate search.
- Not for: tasks with plenty of labels for the exact target (plain supervised training is simpler); domains where you cannot construct meaningful positive pairs; or tiny batches with no memory bank (few negatives, a weak signal). Non-contrastive methods (BYOL, SimSiam, DINO) and masked modelling (MAE) avoid explicit negatives.

## Setup & code
`pip install numpy`. Runs in about 11 seconds on CPU. The encoder is one linear map `W` (32 → 16), trained with Adam.

```python
import numpy as np


def info_nce(za, zb, tau):
    """Symmetric InfoNCE (CLIP / SimCLR-style). Row i of za and zb is a positive pair; the rest of the batch are negatives.
    Returns the loss and its gradients with respect to the (unnormalised) za and zb."""
    na, nb = np.linalg.norm(za, axis=1, keepdims=True), np.linalg.norm(zb, axis=1, keepdims=True)
    a, b = za / na, zb / nb
    logits = a @ b.T / tau
    N = len(a)

    def ce(l):                                    # cross-entropy with the diagonal as the target class
        l = l - l.max(1, keepdims=True)
        p = np.exp(l) / np.exp(l).sum(1, keepdims=True)
        return -np.mean(np.log(p[np.arange(N), np.arange(N)])), p

    la, pa = ce(logits)                           # a_i must pick b_i among all b
    lb, pb = ce(logits.T)                         # b_i must pick a_i among all a
    eye = np.eye(N)
    dlogits = ((pa - eye) + (pb - eye).T) / (2 * N)
    da = dlogits @ b / tau
    db = dlogits.T @ a / tau
    # back through the L2 normalisation: d(z/|z|) = (I - u u^T) / |z|
    dza = (da - a * np.sum(da * a, 1, keepdims=True)) / na
    dzb = (db - b * np.sum(db * b, 1, keepdims=True)) / nb
    return (la + lb) / 2, dza, dzb


def make_data(n, rng, d_sig=4, d_noise=28):
    """Latent class signal in 4 dims buried under 28 dims of high-variance nuisance noise."""
    centers = rng.normal(0, 1, (10, d_sig)) * 1.5
    y = rng.integers(0, 10, n)
    sig = centers[y] + 0.3 * rng.normal(size=(n, d_sig))
    noise = 3.0 * rng.normal(size=(n, d_noise))
    M = np.linalg.qr(rng.normal(size=(d_sig + d_noise,) * 2))[0]       # hide the split with a rotation
    return np.hstack([sig, noise]) @ M, y, M


def augment(x, M, rng, d_sig=4):
    """Views keep the signal and resample the nuisance dims: what the augmentations say is irrelevant."""
    raw = x @ M.T
    raw[:, d_sig:] = 3.0 * rng.normal(size=raw[:, d_sig:].shape)
    raw[:, :d_sig] += 0.1 * rng.normal(size=(len(x), d_sig))
    return raw @ M


def knn_acc(Ztr, ytr, Zte, yte, k=5):
    Ztr = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True)
    Zte = Zte / np.linalg.norm(Zte, axis=1, keepdims=True)
    nn = np.argsort(-(Zte @ Ztr.T), axis=1)[:, :k]
    pred = np.array([np.bincount(r, minlength=10).argmax() for r in ytr[nn]])
    return np.mean(pred == yte)


rng = np.random.default_rng(0)
X, y, M = make_data(3000, rng)
Xtr, ytr, Xte, yte = X[:2000], y[:2000], X[2000:], y[2000:]

# gradient check
za, zb = rng.normal(size=(5, 8)), rng.normal(size=(5, 8))
_, ga, _ = info_nce(za, zb, 0.2)
num = np.zeros_like(za)
for i in np.ndindex(za.shape):
    za[i] += 1e-6; lp = info_nce(za, zb, 0.2)[0]; za[i] -= 2e-6; lm = info_nce(za, zb, 0.2)[0]; za[i] += 1e-6
    num[i] = (lp - lm) / 2e-6
print(f"InfoNCE gradient check, max abs diff: {np.max(np.abs(ga - num)):.1e}")


def nuisance_share(W):
    """Fraction of the embedding's variance that comes from the nuisance dims rather than the signal."""
    raw = Xtr @ M.T
    sig, noise = raw.copy(), raw.copy()
    sig[:, 4:], noise[:, :4] = 0, 0
    vs, vn = (sig @ M @ W).var(0).sum(), (noise @ M @ W).var(0).sum()
    return vn / (vs + vn)


print(f"k-NN accuracy on raw inputs:             {knn_acc(Xtr, ytr, Xte, yte):.3f}")
W0 = rng.normal(0, 1 / np.sqrt(32), (32, 16))
print(f"k-NN accuracy, random linear projection: {knn_acc(Xtr @ W0, ytr, Xte @ W0, yte):.3f}; "
      f"nuisance share of output variance {nuisance_share(W0):.0%}")

for tau, batch in [(0.1, 256), (0.5, 256), (0.1, 16)]:
    W = W0.copy()
    m, v = np.zeros_like(W), np.zeros_like(W)
    for t in range(1, 1501):
        xb = Xtr[rng.integers(0, 2000, batch)]
        va, vb = augment(xb, M, rng), augment(xb, M, rng)
        loss, dza, dzb = info_nce(va @ W, vb @ W, tau)
        g = va.T @ dza + vb.T @ dzb
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2
        W -= 0.01 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    print(f"tau={tau:<4} batch={batch:<4} final loss {loss:.3f} (log N = {np.log(batch):.2f}); "
          f"k-NN acc {knn_acc(Xtr @ W, ytr, Xte @ W, yte):.3f}; nuisance share of output variance "
          f"{nuisance_share(W):.0%}")
```

Output (numpy 2.5):
```
InfoNCE gradient check, max abs diff: 5.5e-10
k-NN accuracy on raw inputs:             0.151
k-NN accuracy, random linear projection: 0.150; nuisance share of output variance 98%
tau=0.1  batch=256  final loss 2.805 (log N = 5.55); k-NN acc 0.933; nuisance share of output variance 0%
tau=0.5  batch=256  final loss 4.166 (log N = 5.55); k-NN acc 0.930; nuisance share of output variance 0%
tau=0.1  batch=16   final loss 0.859 (log N = 2.77); k-NN acc 0.935; nuisance share of output variance 3%
```

The analytic gradient, including the backward pass through the L2 normalisation, matches finite differences to 5.5e-10. On the raw inputs, k-NN reaches 0.151 (chance is 0.10), because 28 nuisance dimensions with variance 9 swamp the 4 signal dimensions. A random projection keeps that mix (98% of the output variance is nuisance) and gets 0.150. After training on augmented pairs alone, the nuisance share falls to 0 to 3% and k-NN accuracy reaches 0.93. The encoder learned to drop what the augmentation changes, and the class structure is what remained.

The final loss values are not comparable across settings: 0.859 with a batch of 16 sits under `log 16 = 2.77`, and 2.805 with 256 sits under `log 256 = 5.55`. On this easy, linear toy, temperature (0.1 vs 0.5) and batch size (16 vs 256) barely change accuracy. With real deep encoders and real images they matter a lot. The SimCLR paper reports that larger batches and a tuned temperature give clearly better linear-probe accuracy.

## Choosing / trade-offs
- **Batch size and negatives.** More negatives tighten the bound and help, especially early in training. Large batches cost memory, so the alternatives are a memory bank or queue (MoCo), gradient caching, or a loss that needs fewer negatives (SigLIP's sigmoid loss).
- **Temperature.** Typical values are 0.05 to 0.2 for embeddings. CLIP learns τ as a parameter (clipped at a maximum logit scale). Too low causes instability and over-fitting to false negatives; too high gives a weak, uniform signal.
- **Hard negatives.** Mining similar-but-wrong examples (BM25 negatives in retrieval) gives sharper embeddings, but mined "negatives" are often actually positives. Filter them or down-weight them.
- **Projection head.** SimCLR computes the loss on `g(h)` from a small MLP head and uses `h` for downstream tasks. The head absorbs the invariances, and `h` keeps more information.
- **Contrastive vs non-contrastive vs masked.** Contrastive methods need negatives and good augmentations. BYOL and DINO need careful collapse prevention (stop-gradient, EMA teacher). Masked autoencoding (MAE) scales well for vision transformers. For text and multimodal retrieval, contrastive training is the standard.

## Gotchas
- Normalise embeddings before the dot product. Raw dot products let the model inflate norms instead of learning directions.
- False negatives: two different examples of the same class in a batch are pushed apart. With few classes and large batches this hurts. Supervised contrastive loss (SupCon) or deduplication helps.
- Weak augmentations give shortcut features (colour histograms, crop position). Strong ones can destroy the signal. The augmentation set is the main hyperparameter.
- In a distributed setup, gather embeddings across GPUs before computing the loss, or each GPU only sees its local negatives.
- Evaluate with a frozen-encoder probe (k-NN or linear) on labelled data. The contrastive loss going down does not show the representation is useful.
- Watch for dimensional collapse: embeddings that use only a few directions. Check the singular values of the embedding matrix.

## Related
- [[autoencoders-and-self-supervised-learning]] - the wider family of self-supervised methods (masked, non-contrastive, autoencoding).
- [[embeddings]] - using trained embeddings for search, clustering and RAG.
- [[word2vec-skip-gram-from-scratch-numpy]] - negative sampling, an earlier contrastive objective for word vectors.
- [[knn-from-scratch]] - the k-NN classifier used here as the evaluation probe.
- [[information-theory-for-ml]] - mutual information and cross-entropy behind the log N bound.

## References
- van den Oord, Li and Vinyals (2018), "Representation learning with contrastive predictive coding" (InfoNCE): https://arxiv.org/abs/1807.03748
- Chen et al. (2020), "A simple framework for contrastive learning of visual representations" (SimCLR): https://arxiv.org/abs/2002.05709
- Radford et al. (2021), "Learning transferable visual models from natural language supervision" (CLIP): https://arxiv.org/abs/2103.00020
- Wang and Isola (2020), "Understanding contrastive representation learning through alignment and uniformity on the hypersphere": https://arxiv.org/abs/2005.10242
