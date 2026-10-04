---
title: Character-level MLP language model from scratch in NumPy (embeddings, n-gram baselines, perplexity, sampling)
category: concepts
tags: [language-model, character-level, n-gram, mlp, embedding-layer, bengio-2003, cross-entropy, perplexity, sampling, backpropagation, numpy, from-scratch, deep-learning-basics, nlp]
use_cases:
  - "build a neural language model from scratch before moving to RNNs and transformers"
  - "implement an embedding layer and its scatter-add backward pass"
  - "compare a neural language model with count-based n-gram baselines using perplexity"
  - "sample text from a trained language model"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.jmlr.org/papers/v3/bengio03a.html
  - https://web.stanford.edu/~jurafsky/slp3/3.pdf
  - https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
  - https://docs.python.org/3/library/inspect.html#inspect.getdoc
---

# Character-level MLP language model from scratch in NumPy (embeddings, n-gram baselines, perplexity, sampling)

## Summary
A language model gives a probability to the next token given the ones before it. Count-based n-gram models look up how often each context was followed by each token; they are exact on contexts they have seen and fail on contexts they have not. Bengio et al. (2003) replaced the lookup table with a neural network: embed each context token as a learned vector, concatenate, apply a hidden layer, and predict the next token with a softmax. Similar contexts then share statistics through their embeddings. The NumPy version below trains such a model on characters of Python standard-library docstrings, passes a gradient check through the embedding layer, and compares it with smoothed n-grams. On this small corpus a 5-gram still wins, but at the same 8-character context the MLP is far better than the 9-gram, which runs out of data.

## Key concepts
- **Autoregressive factorisation.** `p(c_1 … c_n) = Π p(c_t | c_{<t})`; a fixed-window model approximates the history by the last `T` characters.
- **Embedding layer.** A matrix `C` of shape `(V, d)`; looking up a token is row indexing `C[ids]`. Its gradient is a scatter-add: every occurrence of a character in the batch adds into that row (`np.add.at`, not fancy-index assignment, which drops repeats).
- **MLP head.** `h = tanh([C[c_{t−T}], …, C[c_{t−1}]] W1 + b1)`, `p = softmax(h W2 + b2)`. The cross-entropy gradient at the logits is `p − onehot(y)`, as in [[softmax-regression-from-scratch]].
- **Loss and perplexity.** Mean negative log-likelihood in nats per character; perplexity `= exp(NLL)` is the effective number of equally likely next characters. A uniform model over 28 characters has perplexity 28.
- **n-gram with add-α smoothing.** `p(c | ctx) = (count(ctx, c) + α) / (count(ctx) + αV)`. Longer `n` sees more context but more contexts are unseen, so their predictions fall back to near-uniform.
- **Sampling.** Feed the model's own sampled character back in as context; temperature and top-k change the sharpness ([[decoding-and-sampling]]).

## When to use / scenarios
- Learning: the step between word2vec ([[word2vec-skip-gram-from-scratch-numpy]]) and RNN/transformer language models ([[rnn-from-scratch-numpy]], [[transformer-block-from-scratch-numpy]]). Same embedding + softmax machinery, fixed window instead of recurrence or attention.
- Interviews: "what is perplexity", "why do n-grams fail on long contexts", "how is an embedding layer's gradient computed".
- Small, fast next-character models for autocomplete of codes, IDs or names on a device where a transformer is too large.
- Baselines: always report an n-gram number next to a neural language model; it is often closer than expected.
- Not for: modern text generation, where pretrained transformers with subword tokenisation dominate ([[tokenization]]).

## Setup & code
`pip install numpy`. The corpus is built from docstrings of the installed Python standard library, so no download is needed; the exact text and therefore the numbers depend on the Python version. Runs in about 1 minute on CPU.

```python
import collections, importlib, inspect, re, sys
import numpy as np

# Corpus: docstrings from the Python standard library (no download needed)
names = ("argparse ast asyncio base64 bisect calendar codecs collections concurrent.futures configparser contextlib copy "
         "csv dataclasses datetime decimal difflib email enum fractions functools heapq http.client inspect ipaddress "
         "itertools json logging multiprocessing operator os pathlib pickle pprint random re shutil socket sqlite3 ssl "
         "statistics string subprocess tarfile tempfile textwrap threading typing unittest urllib.request uuid zipfile").split()
mods = [importlib.import_module(n) for n in names]
docs = []
for mod in mods:
    for _, obj in [(None, mod)] + inspect.getmembers(mod):
        d = inspect.getdoc(obj) if (obj is mod or getattr(obj, "__module__", None) == mod.__name__) else None
        if d: docs.append(d)
docs = list(dict.fromkeys(docs))                          # dedupe, then shuffle so val is not one module
docs = [docs[i] for i in np.random.default_rng(0).permutation(len(docs))]
text = re.sub(r"[^a-z .\n]", "", " ".join(docs).lower())  # keep letters, space, dot
text = re.sub(r"\s+", " ", text)
chars = sorted(set(text)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in text])
split = int(0.9 * len(data)); train, val = data[:split], data[split:]
print(f"python {sys.version.split()[0]}: {len(text):,} chars, vocab {V}")

T = 8                                                     # context length (characters)
def windows(d):
    return np.stack([d[i: len(d) - T + i] for i in range(T)], 1), d[T:]
Xtr, ytr = windows(train); Xva, yva = windows(val)

# Count baselines: n-gram with add-alpha smoothing, evaluated by mean NLL (nats per char)
def ngram_nll(n, alpha=0.1):
    tr, va = text[:split], text[split:]
    cnt = collections.Counter(tr[i: i + n] for i in range(len(tr) - n + 1))
    ctx = collections.Counter(tr[i: i + n - 1] for i in range(len(tr) - n + 2))
    return -np.mean([np.log((cnt[va[i - n + 1: i + 1]] + alpha) / (ctx[va[i - n + 1: i]] + alpha * V))
                     for i in range(n - 1, len(va))])

# MLP language model (Bengio et al. 2003): embed T chars, concat, tanh hidden layer, softmax over V
E_DIM, H = 16, 256
def init(rng):
    return {"C": rng.normal(0, 1, (V, E_DIM)), "W1": rng.normal(0, np.sqrt(1 / (T * E_DIM)), (T * E_DIM, H)),
            "b1": np.zeros(H), "W2": rng.normal(0, 0.01, (H, V)), "b2": np.zeros(V)}

def forward(P, X, y):
    e = P["C"][X].reshape(len(X), -1)                     # embedding lookup = row indexing
    h = np.tanh(e @ P["W1"] + P["b1"])
    z = h @ P["W2"] + P["b2"]; z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y]).mean(), (e, h, p)

def backward(P, X, y, cache):
    e, h, p = cache
    dz = p.copy(); dz[np.arange(len(y)), y] -= 1; dz /= len(y)
    G = {"W2": h.T @ dz, "b2": dz.sum(0)}
    da = (dz @ P["W2"].T) * (1 - h ** 2)
    G["W1"], G["b1"] = e.T @ da, da.sum(0)
    G["C"] = np.zeros_like(P["C"])
    np.add.at(G["C"], X, (da @ P["W1"].T).reshape(len(X), T, E_DIM))   # scatter-add: repeated chars accumulate
    return G

rng = np.random.default_rng(1)
P = init(rng); P["W2"] = rng.normal(0, 0.3, (H, V))      # non-tiny weights for a meaningful check
Xb, yb = Xtr[:32], ytr[:32]
G = backward(P, Xb, yb, forward(P, Xb, yb)[1])
r, cidx = int(Xb[0, 0]), 3
P["C"][r, cidx] += 1e-6; up = forward(P, Xb, yb)[0]
P["C"][r, cidx] -= 2e-6; dn = forward(P, Xb, yb)[0]
P["C"][r, cidx] += 1e-6
print(f"grad check embedding: analytic {G['C'][r, cidx]:.6e}, numeric {(up - dn) / 2e-6:.6e}")

rng = np.random.default_rng(0)
P = init(rng)
m = {k: np.zeros_like(a) for k, a in P.items()}; v = {k: np.zeros_like(a) for k, a in P.items()}
for t in range(1, 20001):
    idx = rng.integers(0, len(Xtr), 128)
    loss, cache = forward(P, Xtr[idx], ytr[idx])
    G = backward(P, Xtr[idx], ytr[idx], cache)
    lr = 3e-3 if t < 15000 else 1e-3
    for k in P:
        m[k] = 0.9 * m[k] + 0.1 * G[k]; v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
        P[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)

unigram = -np.mean(np.log((np.bincount(train, minlength=V)[val] + 0.1) / (len(train) + 0.1 * V)))
rows = [("uniform", np.log(V)), ("unigram", unigram), ("bigram", ngram_nll(2)), ("trigram", ngram_nll(3)),
        ("5-gram", ngram_nll(5)), ("9-gram (same context as MLP)", ngram_nll(9)),
        ("MLP train", forward(P, Xtr[: len(Xva)], ytr[: len(Xva)])[0]), ("MLP val", forward(P, Xva, yva)[0])]
for name, nll in rows:
    print(f"{name:30s} {nll:.3f} nats/char  perplexity {np.exp(nll):5.2f}")

# Sample by feeding predictions back in
rng = np.random.default_rng(42)
prompt = "returns the "
ctx = [ix[c] for c in prompt]; out = []
for _ in range(200):
    p = forward(P, np.array([ctx[-T:]]), np.array([0]))[1][2][0]
    ctx.append(rng.choice(V, p=p)); out.append(chars[ctx[-1]])
print("sample: " + prompt + "".join(out))
```

Output (Python 3.14.6, numpy 2.5):
```
python 3.14.6: 290,299 chars, vocab 28
grad check embedding: analytic -1.284186e-02, numeric -1.284186e-02
uniform                        3.332 nats/char  perplexity 28.00
unigram                        2.897 nats/char  perplexity 18.11
bigram                         2.404 nats/char  perplexity 11.06
trigram                        1.852 nats/char  perplexity  6.38
5-gram                         1.343 nats/char  perplexity  3.83
9-gram (same context as MLP)   2.093 nats/char  perplexity  8.11
MLP train                      1.209 nats/char  perplexity  3.35
MLP val                        1.445 nats/char  perplexity  4.24
sample: returns the specilsol. using argument od not be following metadata stract method in the input she encoding in attrieself the patterns alg elatart. ... . is flag arguments set applict. it closekproring cs. share f
```

The embedding gradient, accumulated with `np.add.at`, matches a finite difference exactly. Each step of context helps the count models up to a point: perplexity falls from 28 (uniform) to 18.1 (unigram), 11.1 (bigram), 6.4 (trigram) and 3.83 (5-gram), then rises to 8.11 for the 9-gram, because most 8-character validation contexts never appear in 261k training characters and their predictions collapse towards uniform. The MLP reads the same 8 characters and reaches 4.24, half the 9-gram's perplexity, because its embeddings and hidden layer generalise across contexts that were never seen verbatim. It does not beat the 5-gram on this corpus: docstrings repeat many phrases, which suits exact counting, and 290k characters is small for a neural model (train 3.35 vs val 4.24 shows it already overfits). The sample has English-like words ("returns", "argument", "metadata", "method", "encoding", "patterns") and punctuation, with misspellings typical of a character-level model with a short window.

## Choosing / trade-offs
- **n-gram vs neural.** n-grams are instant to train, exact on seen contexts and strong on small or repetitive corpora; tune `n` and use interpolation or Kneser-Ney smoothing before concluding a neural model is needed. Neural models win as data and context length grow.
- **Fixed window vs RNN vs transformer.** The MLP's parameters grow with the window `T`; an RNN shares weights across positions with unlimited (but fading) memory; a transformer attends to the whole context in parallel and is the default today.
- **Characters vs subwords.** Characters give a tiny vocabulary and no unknown tokens, but sequences are long and spelling must be learned. Subword tokenisers (BPE) are the standard for real text.
- **Model size.** With little data, a bigger hidden layer overfits faster; add weight decay or dropout, and stop on validation loss ([[regularization-in-deep-learning]]).

## Gotchas
- `G[X] += grad` silently drops repeated indices; use `np.add.at` (or `torch.index_add_`, which `nn.Embedding` does for you).
- Subtract the row max before `exp` in the softmax; logits overflow otherwise.
- Compare models on the same held-out text, the same vocabulary and the same unit (nats or bits per character, not per word). Perplexities on different tokenisations are not comparable.
- Splitting a concatenated corpus by position can put whole modules or authors only in validation; shuffle documents before splitting (done above) unless you want to measure that shift.
- Add-α with a fixed α penalises long n-grams; the 9-gram number above is pessimistic compared with backoff or Kneser-Ney smoothing.
- Sampling with temperature 1 shows what the model learned; lower temperature hides errors by repeating frequent phrases.

## Related
- [[word2vec-skip-gram-from-scratch-numpy]] - embeddings learned from context, without a language-model head.
- [[softmax-regression-from-scratch]] - the softmax and cross-entropy gradient used at the output.
- [[rnn-from-scratch-numpy]] - a recurrent character model with unbounded context.
- [[transformer-block-from-scratch-numpy]] - the attention block that replaced fixed windows.
- [[tokenization]] - characters vs BPE and other subword vocabularies.
- [[decoding-and-sampling]] - temperature, top-k and top-p sampling.

## References
- Bengio, Ducharme, Vincent and Jauvin (2003), "A Neural Probabilistic Language Model", JMLR: https://www.jmlr.org/papers/v3/bengio03a.html
- Jurafsky and Martin, Speech and Language Processing, chapter 3 (N-gram language models): https://web.stanford.edu/~jurafsky/slp3/3.pdf
- NumPy `ufunc.at` (unbuffered scatter-add): https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
- Python `inspect.getdoc`: https://docs.python.org/3/library/inspect.html#inspect.getdoc
