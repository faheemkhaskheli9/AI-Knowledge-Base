---
title: Linear-chain CRF from scratch (forward-backward, Viterbi, BIO sequence tagging)
category: ml
tags: [crf, conditional-random-field, sequence-labeling, ner, bio-tagging, viterbi, forward-backward, structured-prediction, numpy, from-scratch]
use_cases:
  - "tag tokens in a sequence (NER, part-of-speech, slot filling) with valid label transitions"
  - "add a CRF layer on top of a BiLSTM or transformer token classifier"
  - "understand why per-token classifiers produce invalid BIO sequences"
  - "implement the forward algorithm, marginals and Viterbi decoding for a chain model"
status: stable
last_verified: 2026-10-04
sources:
  - https://repository.upenn.edu/cis_papers/159/
  - https://arxiv.org/abs/1011.4088
  - https://sklearn-crfsuite.readthedocs.io/
  - https://pytorch-crf.readthedocs.io/
---

# Linear-chain CRF from scratch (forward-backward, Viterbi, BIO sequence tagging)

## Summary
A linear-chain conditional random field (CRF) scores a whole label sequence as the sum of per-token emission scores and label-to-label transition scores, and normalises over all possible sequences. Training maximises the log-likelihood of the gold sequence. The gradient is "observed feature counts minus expected counts", with the expectations computed by forward-backward. Decoding uses Viterbi to find the best whole sequence. This file implements all three in NumPy, checks them against brute-force enumeration, and trains a BIO tagger on ambiguous tokens. A per-token softmax with the same features reaches 0.773 token accuracy and outputs 651 invalid `O → I` transitions. The CRF reaches 0.911 with none, because it learns that `O → I` is forbidden.

## Key concepts
- **Score of a sequence.** `s(x, y) = start[y_1] + Σ_t E[t, y_t] + Σ_t A[y_{t-1}, y_t]`. `E` comes from features (here a weight per word and tag, in practice a BiLSTM or transformer). `A` is a `T × T` transition matrix.
- **Normalisation.** `P(y | x) = exp(s(x, y)) / Z(x)`, where `Z` sums over all `T^L` sequences. The forward algorithm computes `log Z` in `O(L T²)` with a log-sum-exp recursion.
- **Discriminative.** Unlike an HMM ([[hmm-from-scratch]]), a CRF models `P(y | x)` directly, so emission features can look at any part of the input (neighbouring words, capitalisation, embeddings) without independence assumptions.
- **Gradient = observed − expected.** For each feature, add its count in the gold sequence and subtract its expected count under the model. Node marginals `P(y_t = j | x)` and edge marginals `P(y_{t-1} = i, y_t = j | x)` come from forward (`α`) and backward (`β`) messages.
- **Viterbi decoding** replaces log-sum-exp with max and keeps back-pointers. It returns the single highest-scoring sequence, which can never use a transition the model scores as very unlikely.

## When to use / scenarios
- Named entity recognition, slot filling in chatbots, part-of-speech tagging, address and invoice field parsing, gene or protein sequence labelling, segmenting sensor streams into activities.
- On top of a neural encoder (BiLSTM-CRF, BERT-CRF) when label consistency matters and the training set is small or labels are structured (BIO / BIOES spans).
- Classic feature-based CRFs (`sklearn-crfsuite`) for small data, CPU-only deployment or interpretable models.
- Not usually needed with a large fine-tuned transformer and plenty of data: a plain token-classification head is often within a fraction of a point, and constrained decoding (masking invalid transitions at inference) fixes most invalid spans without training a CRF.
- Not for labels that do not depend on neighbours (independent per-token classification), or for generation tasks, where an LLM extracting JSON may be simpler, see [[nlp-classic-tasks]].

## Setup & code
NumPy only, runs in about 2 seconds. The toy task has spans of BIO tags. Word ids are noisy: `O` words are drawn from ids 0-14, `B` from 10-24, `I` from 15-29, so a single word is often ambiguous between tags, and the sequence structure has to resolve it.

```python
import numpy as np

rng = np.random.default_rng(0)
TAGS = ["O", "B", "I"]                      # BIO tagging: I may only follow B or I
V, T = 30, 3                                # vocabulary size, tags


def lse(a, axis):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def make_data(n):
    """Spans of 1-4 tokens. Words are noisy: a B word and an I word share half the vocabulary,
    so the tag of a single token is ambiguous and the sequence structure has to resolve it."""
    data = []
    for _ in range(n):
        tags = []
        while len(tags) < 20:
            tags += [0] * rng.integers(1, 4) + [1] + [2] * rng.integers(0, 4)
        tags = np.array(tags[:20])
        ranges = {0: (0, 15), 1: (10, 25), 2: (15, 30)}
        words = np.array([rng.integers(*ranges[t]) for t in tags])
        data.append((words, tags))
    return data


def forward(E, A, start):
    """log Z via the forward algorithm. E: (L, T) emission scores, A: (T, T) transitions."""
    alpha = start + E[0]
    alphas = [alpha]
    for t in range(1, len(E)):
        alpha = lse(alpha[:, None] + A, 0) + E[t]
        alphas.append(alpha)
    return np.array(alphas), lse(alpha, 0)


def backward(E, A):
    beta = np.zeros(T)
    betas = [beta]
    for t in range(len(E) - 1, 0, -1):
        beta = lse(A + (E[t] + beta)[None], 1)
        betas.append(beta)
    return np.array(betas[::-1])


def viterbi(E, A, start):
    score, back = start + E[0], []
    for t in range(1, len(E)):
        s = score[:, None] + A
        back.append(s.argmax(0))
        score = s.max(0) + E[t]
    path = [score.argmax()]
    for b in back[::-1]:
        path.append(b[path[-1]])
    return np.array(path[::-1])


def train_crf(data, epochs=15, lr=0.1, l2=1e-3):
    W, A, start = np.zeros((V, T)), np.zeros((T, T)), np.zeros(T)
    for _ in range(epochs):
        for words, tags in data:
            E = W[words]
            alphas, logZ = forward(E, A, start)
            betas = backward(E, A)
            node = np.exp(alphas + betas - logZ)                       # P(y_t = j | x)
            edge = np.exp(alphas[:-1, :, None] + A + (E[1:] + betas[1:])[:, None] - logZ)
            # gradient of the log-likelihood = observed counts - expected counts
            gW = np.zeros_like(W)
            np.add.at(gW, (words, tags), 1)
            np.add.at(gW, words, -node)
            gA = -edge.sum(0)
            np.add.at(gA, (tags[:-1], tags[1:]), 1)
            gs = -node[0]
            gs[tags[0]] += 1
            W += lr * (gW - l2 * W); A += lr * (gA - l2 * A); start += lr * (gs - l2 * start)
    return W, A, start


def train_softmax(data, epochs=15, lr=0.1, l2=1e-3):
    """Per-token classifier with the same emission features and no transitions."""
    W = np.zeros((V, T))
    for _ in range(epochs):
        for words, tags in data:
            p = np.exp(W[words] - lse(W[words], 1)[:, None])
            g = -p
            g[np.arange(len(tags)), tags] += 1
            gW = np.zeros_like(W)
            np.add.at(gW, words, g)
            W += lr * (gW - l2 * W)
    return W


def evaluate(name, predict, data):
    acc, bad = [], 0
    for words, tags in data:
        y = predict(words)
        acc.append(np.mean(y == tags))
        bad += np.sum((y[1:] == 2) & (y[:-1] == 0)) + (y[0] == 2)     # O -> I or sentence-initial I
    print(f"{name:16s} token acc {np.mean(acc):.3f}  invalid O->I transitions {bad}")


train, test = make_data(300), make_data(200)
Ws = train_softmax(train)
evaluate("per-token softmax", lambda w: Ws[w].argmax(1), test)
W, A, start = train_crf(train)
evaluate("linear-chain CRF", lambda w: viterbi(W[w], A, start), test)
print("learned transition scores (rows = from, cols = to):")
print("     " + "   ".join(f"{t:>4s}" for t in TAGS))
for i, t in enumerate(TAGS):
    print(f"{t:>3s} " + " ".join(f"{v:6.2f}" for v in A[i]))
```

Output (Python 3.14, NumPy 2.5):
```
per-token softmax token acc 0.773  invalid O->I transitions 651
linear-chain CRF token acc 0.911  invalid O->I transitions 0
learned transition scores (rows = from, cols = to):
        O      B      I
  O   1.98   2.41  -4.49
  B   1.58  -3.75   2.47
  I   1.60  -3.94   2.14
```

A separate check enumerated all 3⁵ sequences of a random 5-token problem: `log Z`, the node marginals and the Viterbi path matched brute force exactly. On the tagging task, the CRF gains 14 points of token accuracy with the same emission features. The learned transitions explain why. `O → I` scores −4.49, so Viterbi never produces an `I` right after an `O`. `B → B` and `I → B` are also strongly negative, because in this data every span is followed by at least one `O`. An ambiguous word between `B` and `I` is resolved by what came before it, which a per-token classifier cannot see.

## Choosing / trade-offs
- **CRF layer vs plain softmax head.** A CRF helps most with small data, weak encoders and strict span structure. With a large pretrained transformer and enough data, the gain is usually small and the CRF adds training cost (`O(L T²)` per sequence) and a dependency.
- **Constrained decoding** is the cheap middle ground: train a softmax head, then run Viterbi at inference with `−inf` on invalid transitions. It guarantees valid spans without CRF training, but it does not learn soft transition preferences.
- **Tag scheme.** BIO is simplest. BIOES (with explicit end and single tags) gives the CRF more structure to learn and often helps span F1 slightly.
- **Libraries.** `sklearn-crfsuite` (feature dicts, L-BFGS, fast on CPU) for classic CRFs. `pytorch-crf` or `torchcrf` for a CRF layer on a neural encoder. spaCy and Hugging Face token classification for transformer taggers without a CRF.
- **Features (classic CRF).** Word identity, lowercase form, suffixes, shape (`Xxxx`, `dd-dd`), neighbouring words and gazetteer hits. Feature quality matters more than the optimiser.
- **Higher-order or semi-Markov CRFs** model longer label dependencies or whole segments, at a higher cost. Rarely worth it over a strong encoder plus a first-order CRF.

## Gotchas
- Compute `Z` in log space. Multiplying probabilities over a 100-token sentence underflows to zero.
- Evaluate NER with span-level F1 (`seqeval`), not token accuracy. A span with one wrong token is a full miss.
- Subword tokenizers split words into pieces. Label only the first subword (mask the rest) or the CRF learns transitions between subword pieces that do not exist at word level.
- Viterbi gives the best sequence, not the best tag per position. For per-token confidence, use the forward-backward marginals.
- Padding: mask padded positions out of both the score and `Z`. Leaving them in adds fake transitions and corrupts the gradient.
- Initialise transitions at zero (or with `−inf` masks for known-invalid moves). Random large transitions can dominate weak early emission scores.

## Related
- [[hmm-from-scratch]] - the generative chain model with the same forward and Viterbi recursions.
- [[hidden-markov-models-and-kalman-filters]] - sequence models over hidden states.
- [[probabilistic-graphical-models]] - CRFs as undirected graphical models.
- [[nlp-classic-tasks]] - NER, POS tagging and when to use a tagger vs an LLM.
- [[softmax-regression-from-scratch]] - the per-token baseline.
- [[ctc-loss-from-scratch-numpy]] - another sum over all alignments computed with a forward recursion.

## References
- Lafferty, McCallum and Pereira (2001), "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data", ICML: https://repository.upenn.edu/cis_papers/159/
- Sutton and McCallum (2010), "An Introduction to Conditional Random Fields": https://arxiv.org/abs/1011.4088
- sklearn-crfsuite documentation: https://sklearn-crfsuite.readthedocs.io/
- pytorch-crf documentation: https://pytorch-crf.readthedocs.io/
