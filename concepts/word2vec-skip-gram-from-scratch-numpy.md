---
title: word2vec skip-gram with negative sampling from scratch (NumPy)
category: concepts
tags: [word2vec, skip-gram, negative-sampling, word-embeddings, embeddings, sgd, sigmoid, numpy, nlp, from-scratch, deep-learning-basics]
use_cases:
  - "implement word2vec skip-gram with negative sampling from scratch"
  - "understand how an embedding table is learned from co-occurrence alone"
  - "derive the negative-sampling gradients and the sparse embedding update"
  - "see why the unigram^0.75 noise distribution and small batches matter"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1301.3781
  - https://arxiv.org/abs/1310.4546
  - https://web.stanford.edu/class/cs224n/
---

# word2vec skip-gram with negative sampling from scratch (NumPy)

## Summary
Skip-gram learns a vector for each word by training it to tell real (center, context) pairs from random ones. Each word has two vectors: an input vector `v` when it is the center and an output vector `u` when it is the context. Negative sampling replaces the full-vocabulary softmax with `k` binary logistic regressions, so one update touches only `k + 2` rows of two tables. About 40 lines of NumPy on a toy corpus learn embeddings where `cat` is nearest to `dog`, `horse`, `cow` and `car` to `truck`, `train`, `bus`. This is the training side of [[word-embeddings-word2vec]], which covers using gensim and pretrained vectors.

## Key concepts
- **Training pairs.** Slide a window (here ±2) over each sentence; every (center, neighbour) pair is a positive example. Words that share neighbours get pushed toward similar vectors (distributional hypothesis).
- **Two tables.** `Win` (center) and `Wout` (context), both `(V, D)`. The final embedding is usually `Win`; some setups average the two.
- **Negative-sampling loss.** For center `c`, true context `o` and `k` sampled noise words `n`: `L = −log σ(u_o·v_c) − Σ log σ(−u_n·v_c)`. Push the true pair's dot product up, the noise pairs' down.
- **Gradients.** With `s_p = σ(u_o·v_c)` and `s_n = σ(u_n·v_c)`: `∂L/∂u_o = (s_p − 1)·v_c`, `∂L/∂u_n = s_n·v_c`, `∂L/∂v_c = (s_p − 1)·u_o + Σ s_n·u_n`. These are logistic-regression gradients ([[logistic-regression-from-scratch]]).
- **Noise distribution.** Sample negatives from unigram counts raised to 0.75. The power flattens the distribution so rare words get sampled more often than their raw frequency.
- **Sparse updates.** Only the rows for `c`, `o` and the `k` negatives change. `np.add.at` accumulates updates when the same row appears more than once in a batch.
- **Output vs input init.** `Win` starts with small random values and `Wout` at zero, as in the original C code. Both at zero would never break symmetry.

## When to use / scenarios
- Learning: the smallest model that shows how an embedding table is learned, the same table every transformer starts with ([[embeddings]]).
- Interviews: the negative-sampling objective, its gradients, and why it avoids the `O(V)` softmax.
- Contrastive learning intuition: positive pair vs sampled negatives is the same idea behind CLIP and sentence-embedding training ([[autoencoders-and-self-supervised-learning]]).
- Real use: train with gensim, not this loop. Its C implementation, subsampling, learning-rate decay and multi-threading handle millions of tokens ([[word-embeddings-word2vec]]).
- Not for: modern semantic search or RAG; contextual sentence embeddings from a transformer encoder beat static word vectors there.

## Setup & code
`pip install numpy`. Runs in about a second on CPU.

```python
import numpy as np

# Toy corpus: animals and vehicles appear in different contexts.
rng = np.random.default_rng(0)
animals, vehicles = ["cat", "dog", "horse", "cow"], ["car", "bus", "truck", "train"]
a_ctx, v_ctx = ["eats", "sleeps", "runs", "barn", "fur"], ["drives", "road", "fuel", "wheels", "parks"]
sents = []
for _ in range(2000):
    if rng.random() < 0.5:
        sents.append(["the", rng.choice(animals), rng.choice(a_ctx), rng.choice(a_ctx)])
    else:
        sents.append(["the", rng.choice(vehicles), rng.choice(v_ctx), rng.choice(v_ctx)])

vocab = sorted({str(w) for s in sents for w in s})
w2i = {w: i for i, w in enumerate(vocab)}
ids = [[w2i[w] for w in s] for s in sents]

# (center, context) pairs within window 2
pairs = np.array([(s[i], s[j]) for s in ids for i in range(len(s))
                  for j in range(max(0, i - 2), min(len(s), i + 3)) if j != i])
counts = np.bincount(pairs[:, 1], minlength=len(vocab)).astype(float)
noise = counts ** 0.75; noise /= noise.sum()                   # unigram^0.75 noise distribution

V, D, NEG, lr, B = len(vocab), 16, 5, 0.025, 16
Win = (rng.random((V, D)) - 0.5) / D                            # center ("input") vectors
Wout = np.zeros((V, D))                                         # context ("output") vectors
sig = lambda x: 1 / (1 + np.exp(-x))

for epoch in range(5):
    rng.shuffle(pairs)
    total = 0.0
    for s in range(0, len(pairs), B):
        c, o = pairs[s:s + B, 0], pairs[s:s + B, 1]
        neg = rng.choice(V, size=(len(c), NEG), p=noise)
        vc = Win[c]                                             # (B, D)
        uo, un = Wout[o], Wout[neg]                             # (B, D), (B, NEG, D)
        sp = sig((vc * uo).sum(1))                              # positive score
        sn = sig(np.einsum("bd,bkd->bk", vc, un))               # negative scores
        total += -(np.log(sp + 1e-9).sum() + np.log(1 - sn + 1e-9).sum())
        gp, gn = (sp - 1)[:, None], sn[:, :, None]              # dL/dscore
        dvc = gp * uo + (gn * un).sum(1)
        np.add.at(Wout, o, -lr * gp * vc)
        np.add.at(Wout, neg, -lr * gn * vc[:, None, :])
        np.add.at(Win, c, -lr * dvc)
    print(f"epoch {epoch + 1}: loss/pair={total / len(pairs):.3f}")

E = Win / np.linalg.norm(Win, axis=1, keepdims=True)
for w in ("cat", "car", "road"):
    sims = E @ E[w2i[w]]
    top = [vocab[i] for i in np.argsort(-sims)[1:4]]
    print(f"nearest to {w!r}: {top}")
cos = lambda a, b: float(E[w2i[a]] @ E[w2i[b]])
print(f"cos(cat, dog)={cos('cat', 'dog'):.2f}  cos(cat, car)={cos('cat', 'car'):.2f}")
```

Output (numpy 2.5):
```
epoch 1: loss/pair=2.528
epoch 2: loss/pair=2.231
epoch 3: loss/pair=2.225
epoch 4: loss/pair=2.229
epoch 5: loss/pair=2.228
nearest to 'cat': ['dog', 'horse', 'cow']
nearest to 'car': ['truck', 'train', 'bus']
nearest to 'road': ['parks', 'drives', 'fuel']
cos(cat, dog)=1.00  cos(cat, car)=0.34
```

The model never sees a label, only co-occurrence, yet each animal's nearest neighbours are the other animals and each vehicle's are the other vehicles. Context words cluster the same way (`road` with `parks`, `drives`, `fuel`). `cos(cat, dog)` rounds to 1.00 because in this synthetic corpus the two words have identical context distributions; on real text, near-synonyms land around 0.6-0.8. The loss flattens at about 2.2 per pair after two epochs: with 5 noise samples drawn from a 19-word vocabulary, some "negatives" are real context words, so the loss cannot reach zero.

## Choosing / trade-offs
- **Skip-gram vs CBOW.** Skip-gram predicts each context word from the center and does better on rare words. CBOW averages the context to predict the center and trains faster on large corpora.
- **Negative sampling vs hierarchical softmax.** Negative sampling (k = 5-20 for small data, 2-5 for large) is simpler and the usual default. Hierarchical softmax uses a Huffman tree for an `O(log V)` exact softmax and favours rare words.
- **Dimension.** 16 is plenty for 19 words. 100-300 is the usual range for real vocabularies; past that, gains are small.
- **Window size.** Small windows (2-5) capture syntactic similarity (words that can replace each other); larger windows (5-10) capture topical relatedness.
- **Batch size.** Original word2vec updates per pair (batch 1). Larger NumPy batches are faster but sum many updates into frequent rows at once.

## Gotchas
- With `np.add.at` and a large batch, a frequent word like `the` receives hundreds of summed updates in one step. The first version of this code used batch 256 and lr 0.05 and diverged (loss rising to ~70 per pair, `exp` overflow). Batch 16 at lr 0.025 trains cleanly. Scale lr down as batch size grows, or update per pair.
- Fancy-index assignment `Wout[o] -= g` silently drops repeated indices (last write wins). Use `np.add.at`.
- Initialise `Wout` at zero but `Win` random. If both are zero, every gradient is zero and nothing learns.
- Real corpora need subsampling of frequent words (drop `the`, `of` with probability `1 − √(t/f)`) and a decaying learning rate; without them, stopwords dominate training and embeddings drift late in training.
- A noise sample can equal the true context word. It is usually left in; the effect is small with a large vocabulary but visible on a toy one like this.
- Cosine similarity, not raw dot product, for nearest neighbours. Frequent words have longer vectors and dominate dot products ([[distance-metrics-and-similarity]]).

## Related
- [[word-embeddings-word2vec]] - word2vec, GloVe and fastText with gensim, and when to use static vectors.
- [[embeddings]] - modern embedding models and how they are used for search and RAG.
- [[logistic-regression-from-scratch]] - the per-pair objective is binary logistic regression.
- [[self-attention-from-scratch-numpy]] - what a transformer does with the embedding table next.
- [[tokenization]] - how text becomes the ids an embedding table indexes.
- [[distance-metrics-and-similarity]] - cosine similarity and nearest-neighbour search.

## References
- Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space": https://arxiv.org/abs/1301.3781
- Mikolov et al. (2013), "Distributed Representations of Words and Phrases and their Compositionality" (negative sampling, subsampling): https://arxiv.org/abs/1310.4546
- Stanford CS224n, Natural Language Processing with Deep Learning: https://web.stanford.edu/class/cs224n/
