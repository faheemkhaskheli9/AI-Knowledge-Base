---
title: Kneser-Ney n-gram language model from scratch (smoothing, continuation counts, perplexity)
category: concepts
tags: [language-model, n-gram, kneser-ney, smoothing, absolute-discounting, perplexity, nlp, python, from-scratch, llm-basics, ml-basics]
use_cases:
  - "implement an interpolated Kneser-Ney trigram language model in plain Python"
  - "compare MLE, add-one, absolute discounting and Kneser-Ney by test perplexity"
  - "explain continuation counts and why Kneser-Ney beats plain absolute discounting"
  - "build a cheap n-gram baseline before training a neural language model"
status: draft
last_verified: 2026-10-04
sources:
  - https://web.stanford.edu/~jurafsky/slp3/
  - https://dash.harvard.edu/handle/1/25104739
  - https://doi.org/10.1109/ICASSP.1995.479394
  - https://kheafield.com/code/kenlm/
---

# Kneser-Ney n-gram language model from scratch (smoothing, continuation counts, perplexity)

## Summary
An n-gram language model predicts the next word from the previous n−1 words using counts. The raw count ratio (maximum likelihood) gives zero probability to every word sequence not seen in training, so on new text its perplexity is infinite. **Smoothing** moves some probability mass from seen to unseen events. **Kneser-Ney** does this best among classic methods. It subtracts a fixed discount D from every count and gives the freed mass to a lower-order model, and its lower-order model counts in how many different contexts a word appears, rather than how often. Below, trigram models are trained on 90% of *Alice's Adventures in Wonderland* and scored on the last 10%. MLE assigns zero probability to 2,262 of 3,006 test tokens. Add-one smoothing gives perplexity 1,624. Interpolated absolute discounting gives 195, and Kneser-Ney 172, with the same discounts and the only change being continuation counts.

## Key concepts
- **Perplexity.** `exp(−mean log p(w_i | context))` on held-out text: the effective number of equally likely choices per token. Lower is better, and a single zero probability makes it infinite.
- **Add-one (Laplace).** Add 1 to every count. With a vocabulary of 2,424 words and contexts seen only a few times, the added counts swamp the real ones, so it gives far too much mass to unseen words.
- **Absolute discounting.** Subtract D (between 0 and 1) from every seen count and give the freed mass `D·(number of distinct followers)/count(context)` to a lower-order distribution. A good D comes from count-of-counts: `D = n1 / (n1 + 2·n2)`, where n1 and n2 are the numbers of n-grams seen once and twice.
- **Continuation counts.** The lower-order model is used only when the higher-order context did not predict the word well, so it should answer "how likely is w to appear after a new context?" Kneser-Ney therefore uses `N1+(• w)`, the number of distinct words that precede w. The classic example is "Francisco": frequent, but almost only after "San", so it should get a small continuation probability.
- **Interpolation.** Every level mixes its discounted counts with the level below, recursively: trigram → bigram (continuation counts) → unigram (continuation counts). The top level uses raw counts. Each level sums to 1, which the code checks.

## When to use / scenarios
- Learning: the bridge from counting to neural language models, and the clearest illustration of smoothing and perplexity ([[char-mlp-language-model-from-scratch-numpy]], [[tokenization]]).
- Interviews: "what is perplexity", "why does add-one smoothing do badly", "explain Kneser-Ney", "why not just use the unigram frequency for backoff".
- Practice: very fast baselines and features (ASR and OCR rescoring, spelling correction, keyboard suggestions), on-device models with tiny memory and CPU budgets, data filtering (CCNet-style perplexity filtering of web text with KenLM before LLM pretraining: [[pretraining-and-scaling-laws]]), and language identification.
- Not for: anything needing long-range context or fluent generation. A trigram sees two words; neural language models beat it by a wide margin given enough data.

## Setup & code
Standard library only. Downloads Project Gutenberg's *Alice's Adventures in Wonderland* (about 170 KB) and runs in about 2 seconds.

```python
import math
import re
import urllib.request
from collections import Counter

URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"            # Alice in Wonderland
raw = urllib.request.urlopen(URL).read().decode("utf-8")
raw = raw[raw.index("CHAPTER I."):raw.index("*** END OF THE PROJECT GUTENBERG")]
sents = [["<s>", "<s>"] + re.findall(r"[a-z']+", s.lower()) + ["</s>"]
         for s in re.split(r"[.!?]+", raw) if re.search(r"[a-z]", s.lower())]
cut = int(0.9 * len(sents))
train, test = sents[:cut], sents[cut:]
vocab = {w for s in train for w in s} | {"<unk>"}
unk = lambda s: [w if w in vocab else "<unk>" for w in s]
test = [unk(s) for s in test]
V = len(vocab) - 1                                    # predictable tokens (no <s>)

uni, bi, tri = Counter(), Counter(), Counter()
for s in train:
    for i in range(2, len(s)):
        uni[s[i]] += 1
        bi[s[i - 1], s[i]] += 1
        tri[s[i - 2], s[i - 1], s[i]] += 1


def ngram_tables(counts):
    """Context totals and number of distinct followers per context."""
    tot, types = Counter(), Counter()
    for (*ctx, w), c in counts.items():
        tot[tuple(ctx)] += c
        types[tuple(ctx)] += 1
    return tot, types


bi_tot, bi_types = ngram_tables(bi)
tri_tot, tri_types = ngram_tables(tri)

# Continuation counts: in how many distinct contexts does a word / bigram appear?
cont_uni = Counter(w for (_, w) in bi)                          # N1+(. w)
cont_bi = Counter((v, w) for (_, v, w) in tri)                  # N1+(. v w)
cont_bi_tot, cont_bi_types = Counter(), Counter()
for (v, w), c in cont_bi.items():
    cont_bi_tot[v] += c
    cont_bi_types[v] += 1
n_bigram_types = len(bi)


def discounts(counts):
    n1 = sum(1 for c in counts.values() if c == 1)
    n2 = sum(1 for c in counts.values() if c == 2)
    return n1 / (n1 + 2 * n2)


D2, D3, D2c = discounts(bi), discounts(tri), discounts(cont_bi)


def p_kn(w, v, u):
    """Interpolated Kneser-Ney trigram probability P(w | u v)."""
    p = (cont_uni[w] + 0.5) / (n_bigram_types + 0.5 * V)        # continuation unigram, +0.5 so <unk> > 0
    if cont_bi_tot[v]:                                           # middle level uses continuation counts
        p = (max(cont_bi[v, w] - D2c, 0) + D2c * cont_bi_types[v] * p) / cont_bi_tot[v]
    if tri_tot[u, v]:                                            # top level uses raw counts
        p = (max(tri[u, v, w] - D3, 0) + D3 * tri_types[u, v] * p) / tri_tot[u, v]
    return p


def p_abs(w, v, u):
    """Same interpolation, but the lower orders use ordinary counts (absolute discounting)."""
    p = (uni[w] + 0.5) / (sum(uni.values()) + 0.5 * V)
    if bi_tot[(v,)]:
        p = (max(bi[v, w] - D2, 0) + D2 * bi_types[(v,)] * p) / bi_tot[(v,)]
    if tri_tot[u, v]:
        p = (max(tri[u, v, w] - D3, 0) + D3 * tri_types[u, v] * p) / tri_tot[u, v]
    return p


def p_mle(w, v, u):
    return tri[u, v, w] / tri_tot[u, v] if tri_tot[u, v] else 0.0


def p_add1(w, v, u):
    return (tri[u, v, w] + 1) / (tri_tot[u, v] + V)


def perplexity(p):
    ll, n, zeros = 0.0, 0, 0
    for s in test:
        for i in range(2, len(s)):
            q = p(s[i], s[i - 1], s[i - 2])
            n += 1
            if q == 0:
                zeros += 1
            else:
                ll += math.log(q)
    return (math.inf if zeros else math.exp(-ll / n)), zeros, n


# sanity check: KN distribution sums to 1 over the vocabulary for a seen and an unseen context
words = sorted(vocab - {"<s>"})
for ctx in [("the", "white"), ("purple", "zebra")]:
    print(f"sum_w P_KN(w | {' '.join(ctx)}) = {sum(p_kn(w, ctx[1], ctx[0]) for w in words):.6f}")

print(f"train sentences {len(train)}, test {len(test)}, vocab {V}, D2={D2:.3f}, D2c={D2c:.3f}, D3={D3:.3f}")
print("model                         test perplexity  zero-prob tokens")
for name, p in [("trigram MLE", p_mle), ("trigram add-one", p_add1),
                ("trigram absolute discounting", p_abs), ("trigram Kneser-Ney", p_kn)]:
    ppl, zeros, n = perplexity(p)
    print(f"{name:<29} {ppl:>15.1f}  {zeros:>6} / {n}")
```

Output (Python 3.14):
```
sum_w P_KN(w | the white) = 1.000000
sum_w P_KN(w | purple zebra) = 1.000000
train sentences 1487, test 166, vocab 2424, D2=0.743, D2c=0.788, D3=0.862
model                         test perplexity  zero-prob tokens
trigram MLE                               inf    2262 / 3006
trigram add-one                        1623.7       0 / 3006
trigram absolute discounting            195.1       0 / 3006
trigram Kneser-Ney                      171.9       0 / 3006
```

75% of the test trigrams never occurred in training, which is normal for a training set of about 26,000 tokens, so almost all of the probability comes from the lower orders. That is why the lower-order model matters so much: the only difference between the last two rows is ordinary counts vs continuation counts below the trigram level, and it is worth 12% in perplexity. The large discounts (0.74-0.86) also show how sparse the data is: most n-grams are seen once. The sum-to-1 check over a seen context ("the white") and an unseen one ("purple zebra") confirms every level is a proper distribution, which is the first thing to test in any smoothing implementation.

## Choosing / trade-offs
- **Modified Kneser-Ney** (Chen and Goodman) uses three discounts per order (for counts 1, 2 and 3+) and is the standard choice. It is what KenLM and SRILM build by default.
- **Backoff vs interpolation.** Backoff uses the lower order only for unseen n-grams; interpolation always mixes it in. Interpolated KN is usually slightly better and simpler to normalize.
- **Order.** Trigrams suit small corpora like this one. With billions of tokens, 4- and 5-grams help, and KenLM stores them compactly with quantized probabilities.
- **Stupid backoff** (Google, for web-scale data) skips normalization and uses a fixed 0.4 multiplier. It is not a probability distribution, but at trillions of tokens it works about as well and is much cheaper to build.
- **Neural LMs.** A small LSTM or transformer beats KN by a large margin once you have more than a few million tokens. Keep KN as the baseline, for perplexity filtering at scale, and where latency and memory budgets are tiny.

## Gotchas
- Close the vocabulary before scoring. Map rare training words and unknown test words to `<unk>`, and compare perplexities only between models with the same vocabulary and tokenization; perplexity is not comparable across tokenizers.
- Count sentence boundaries: pad with `<s>` and predict `</s>`, otherwise the model never learns how sentences start and end.
- Check that each context's distribution sums to 1. Off-by-one errors in the "distinct followers" count or in the vocabulary size are silent and make perplexity look better than it is.
- The discount must be below every count it is subtracted from (D < 1 here). With D ≥ 1, counts of 1 drop to zero and the distribution breaks.
- Keep the test split separate when choosing D and the vocabulary. Estimating D from the test set leaks information.
- Splitting text by sentence order (as here) is a harder, more honest test than a random sentence split, because the last chapters reuse fewer phrases.

## Related
- [[char-mlp-language-model-from-scratch-numpy]] - the neural successor: learned embeddings instead of counts.
- [[tokenization]] - vocabulary choice, which perplexity depends on.
- [[bpe-tokenizer-from-scratch]] - subword units that remove most out-of-vocabulary problems.
- [[information-theory-for-ml]] - cross-entropy and perplexity.
- [[nlp-classic-tasks]] - classical NLP pipelines where n-gram models still appear.
- [[pretraining-and-scaling-laws]] - perplexity filtering of pretraining data.

## References
- Jurafsky and Martin, "Speech and Language Processing" (3rd ed. draft), chapter on n-gram language models: https://web.stanford.edu/~jurafsky/slp3/
- Chen and Goodman (1998), "An Empirical Study of Smoothing Techniques for Language Modeling", Harvard TR-10-98: https://dash.harvard.edu/handle/1/25104739
- Kneser and Ney (1995), "Improved backing-off for M-gram language modeling", ICASSP: https://doi.org/10.1109/ICASSP.1995.479394
- KenLM language model toolkit: https://kheafield.com/code/kenlm/
