---
title: Decoding strategies from scratch (greedy, beam search, temperature, top-k, top-p)
category: concepts
tags: [decoding, sampling, greedy-decoding, beam-search, temperature, top-k, top-p, nucleus-sampling, language-model, numpy, from-scratch]
use_cases:
  - "implement greedy, beam search, temperature, top-k and top-p (nucleus) sampling over a language model's logits"
  - "see why greedy decoding is not the most likely sequence and what beam search fixes"
  - "understand what temperature, top_k and top_p actually do to the next-token distribution"
  - "explain decoding parameters and their trade-offs in an LLM interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1904.09751
  - https://arxiv.org/abs/1805.04833
  - https://huggingface.co/docs/transformers/generation_strategies
---

# Decoding strategies from scratch (greedy, beam search, temperature, top-k, top-p)

## Summary
A language model only gives a probability distribution over the next token. The decoding strategy decides which token to emit, and it changes the output as much as the model does. Below, a word-level bigram model trained on nine short sentences stands in for an LLM, so every probability can be checked. Greedy decoding from "the" gives "the cat ." with log-probability −3.16, while a beam of 2 finds "the mat ." at −1.53, a sequence about 5x more likely. Sampling 200 continuations shows the diversity/quality trade-off. At temperature 0.5 there are 68 distinct sequences and 5% contain a word pair never seen in training. At 1.5 there are 155 distinct sequences and 76% contain one. Top-k = 3 and top-p = 0.9 cut the next-token candidates after "the" from 14 to 4 and 5, and lower the nonsense rate to 41-43%, compared with 50% for plain sampling.

## Key concepts
- **Greedy.** Take `argmax p(x_t | x_<t)` at each step. Deterministic and cheap, but locally optimal only: a high-probability first word can lead into low-probability continuations. On real LLMs it also tends to loop.
- **Beam search.** Keep the `k` best partial sequences by total log-probability. Expand each by every token, keep the top `k`, and stop beams at EOS. Without length normalization (`score / length^α`) it prefers short outputs, since every token adds a negative log-probability.
- **Temperature.** Divide the logits by `T` before the softmax. `T < 1` sharpens the distribution (T → 0 is greedy), `T > 1` flattens it. It reshapes the distribution without removing any token.
- **Top-k.** Keep only the `k` highest-probability tokens and renormalize. The cut-off is fixed, so it is too tight when the model is unsure and too loose when it is confident.
- **Top-p (nucleus).** Keep the smallest set of tokens whose probability adds up to at least `p`. The candidate set adapts to the model's confidence: one token when it is sure, many when it is not.
- **Order.** Libraries usually apply temperature first, then top-k, then top-p, then sample. Combining them is common (for example `temperature=0.7, top_p=0.9`).

## When to use / scenarios
- Learning: shows that "the model's answer" depends on the decoder, and why sampled outputs vary from run to run.
- Interviews: "greedy vs beam vs sampling", "what does temperature do mathematically", "top-k vs top-p", "why does beam search give bland text".
- Practice: setting generation parameters for an API or a local model. Use greedy or low temperature for extraction, classification, code and structured output. Use beam search for translation and summarization with seq2seq models. Use temperature with top-p for chat and creative writing, and sample several outputs for best-of-n with a verifier or for self-consistency voting ([[decoding-and-sampling]]).
- Not for: forcing a format. Use constrained decoding or structured outputs, which mask invalid tokens at each step ([[structured-output]]), rather than hoping low temperature gives valid JSON.

## Setup & code
NumPy only. The bigram model is a toy, but the decoding functions take a vector of log-probabilities exactly as they would from an LLM.

```python
import numpy as np

text = """the cat sat on the mat . the dog sat on the log . a cat saw a dog .
the dog saw the cat on the mat . a bird sat on a log . the bird saw a cat .
the cat ran to the log . a dog ran to the mat . the cat slept on the mat ."""
words = text.split()
vocab = sorted(set(words))
ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
EOS = ix["."]

# bigram "language model": P(next | prev), add-0.1 smoothing
counts = np.full((V, V), 0.1)
for a, b in zip(words, words[1:]):
    counts[ix[a], ix[b]] += 1
logP = np.log(counts / counts.sum(1, keepdims=True))


def seq_logp(seq):
    return sum(logP[a, b] for a, b in zip(seq, seq[1:]))


def show(seq):
    return " ".join(vocab[t] for t in seq)


def greedy(start, max_len=12):
    seq = [start]
    while len(seq) < max_len and seq[-1] != EOS:
        seq.append(int(logP[seq[-1]].argmax()))
    return seq


def beam(start, k, max_len=12):
    beams = [([start], 0.0)]
    done = []
    while beams:
        cand = [(s + [t], lp + logP[s[-1], t]) for s, lp in beams for t in range(V)]
        cand.sort(key=lambda c: -c[1])
        beams = []
        for s, lp in cand[:k]:
            (done if s[-1] == EOS or len(s) >= max_len else beams).append((s, lp))
    # length-normalised score so short hypotheses do not always win
    return max(done, key=lambda c: c[1] / (len(c[0]) - 1))[0]


def filter_logits(logits, temperature=1.0, top_k=None, top_p=None):
    z = logits / temperature
    if top_k is not None:
        z = np.where(z < np.sort(z)[-top_k], -np.inf, z)
    p = np.exp(z - z.max())
    p /= p.sum()
    if top_p is not None:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: np.searchsorted(cum, top_p) + 1]   # smallest set with mass >= top_p
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return p


def sample(start, rng, max_len=12, **kw):
    seq = [start]
    while len(seq) < max_len and seq[-1] != EOS:
        seq.append(int(rng.choice(V, p=filter_logits(logP[seq[-1]], **kw))))
    return seq


start = ix["the"]
print(f"vocab {V} words")
g = greedy(start)
print(f"greedy           logp {seq_logp(g):6.2f}  | {show(g)}")
for k in [2, 5]:
    b = beam(start, k)
    print(f"beam k={k}         logp {seq_logp(b):6.2f}  | {show(b)}")

p = filter_logits(logP[ix["the"]])
print("\nP(next | 'the'), top 5:", ", ".join(f"{vocab[i]} {p[i]:.2f}" for i in np.argsort(-p)[:5]))
for name, kw in [("T=0.5", dict(temperature=0.5)), ("T=1.5", dict(temperature=1.5)),
                 ("top_k=3", dict(top_k=3)), ("top_p=0.9", dict(top_p=0.9))]:
    q = filter_logits(logP[ix["the"]], **kw)
    print(f"  {name:<9} non-zero tokens {np.sum(q > 1e-12):>2}  max prob {q.max():.2f}")

print("\nsampling, 200 sequences each:  distinct  mean logp/token  hit a 'nonsense' bigram")
rare = counts < 1                       # bigram never seen in the corpus
for name, kw in [("T=1.0", {}), ("T=0.5", dict(temperature=0.5)), ("T=1.5", dict(temperature=1.5)),
                 ("top_k=3", dict(top_k=3)), ("top_p=0.9", dict(top_p=0.9))]:
    rng = np.random.default_rng(0)
    seqs = [sample(start, rng, **kw) for _ in range(200)]
    lp = np.mean([seq_logp(s) / (len(s) - 1) for s in seqs])
    bad = np.mean([any(rare[a, b] for a, b in zip(s, s[1:])) for s in seqs])
    print(f"  {name:<10} {len({tuple(s) for s in seqs}):>17}  {lp:>15.2f}  {bad:>22.0%}")
rng = np.random.default_rng(1)
print("\nexample, top_p=0.9:", show(sample(start, rng, top_p=0.9)))
```

Output (Python 3.14, NumPy 2.5):
```
vocab 14 words
greedy           logp  -3.16  | the cat .
beam k=2         logp  -1.53  | the mat .
beam k=5         logp  -1.53  | the mat .

P(next | 'the'), top 5: cat 0.28, mat 0.28, dog 0.15, log 0.15, bird 0.08
  T=0.5     non-zero tokens 14  max prob 0.38
  T=1.5     non-zero tokens 14  max prob 0.22
  top_k=3   non-zero tokens  4  max prob 0.33
  top_p=0.9 non-zero tokens  5  max prob 0.30

sampling, 200 sequences each:  distinct  mean logp/token  hit a 'nonsense' bigram
  T=1.0                    120            -1.51                     50%
  T=0.5                     68            -1.05                      5%
  T=1.5                    155            -2.13                     76%
  top_k=3                  105            -1.36                     41%
  top_p=0.9                117            -1.37                     43%

example, top_p=0.9: the dog saw a log .
```

After "the", "cat" and "mat" tie at 0.28, and greedy takes "cat" because it comes first in the vocabulary. But "cat" is followed by many different words in training, so "cat ." is unlikely. "mat" is almost always followed by ".", so beam search, which scores the whole sequence, prefers "the mat .". This is the general failure of greedy decoding: it never looks ahead. Temperature reshapes the distribution but keeps all 14 tokens, so even at T = 0.5 a smoothed, never-seen word pair is still possible (5% of samples). Top-k and top-p instead cut the tail outright. Top-k = 3 keeps 4 tokens here because "dog" and "log" tie at the cut-off. The nonsense rate does not reach zero with top-k or top-p, because of smoothing. After a word seen with only one or two followers (such as "sat" or "ran"), every unseen word gets the same small smoothed probability. Those words all tie at the top-k cut-off, so this implementation keeps all of them, and their combined mass can also fall inside the top-p nucleus. The "nonsense bigram" column stands in for incoherent text in an LLM. The tail of the distribution holds many individually unlikely tokens whose combined mass is large, and sampling from it is what produces derailed text (Holtzman et al.).

## Choosing / trade-offs
- **Deterministic tasks** (extraction, classification, code, tool arguments): greedy or `temperature=0`. Some APIs remain slightly non-deterministic even at 0 because of batching and floating-point order, so do not rely on exact reproducibility.
- **Open-ended text** (chat, writing): `temperature` around 0.7-1.0 with `top_p` around 0.9-0.95. Change one parameter at a time; both control diversity.
- **Beam search**: good for tasks with one correct output (translation, ASR, short summaries). For open-ended LLM generation it gives bland, repetitive text, and it multiplies compute and KV cache by the beam width ([[kv-cache-from-scratch-numpy]]).
- **Top-k vs top-p**: top-p adapts to confidence and is the usual default. Top-k is a simple safety cap on very flat distributions, and some stacks apply both. Min-p (keep tokens with `p ≥ min_p × p_max`) is a newer adaptive alternative in local inference tools.
- **Repetition controls**: repetition and frequency penalties reduce loops but also penalize legitimately repeated tokens (variable names in code, names in prose).

## Gotchas
- Beam search without length normalization favors short outputs. With normalization, `α` is another parameter to tune.
- Top-p can keep a single token when the model is confident, so "top_p=0.9" is not "more random" by itself.
- Temperature applies to logits, not probabilities. Dividing probabilities by T and renormalizing gives a different, wrong distribution.
- Ties in `argmax` are broken by index, as "cat" vs "mat" shows. Small changes to a prompt can flip a tie and change the whole greedy continuation.
- Reasoning models and some hosted APIs fix or ignore temperature and top_p. Check the provider docs before tuning parameters that are silently ignored.
- Sampling parameters do not fix a bad model or prompt. If every sample is wrong, the distribution is wrong, not the decoder.
- Evaluate generation with the same decoding settings you deploy. Accuracy at `temperature=0` says little about behavior at 0.8.

## Related
- [[decoding-and-sampling]] - decoding parameters in Hugging Face transformers and hosted APIs.
- [[char-mlp-language-model-from-scratch-numpy]] - a trained neural LM to plug these decoders into.
- [[kv-cache-from-scratch-numpy]] - the inference loop that produces each step's logits.
- [[structured-output]] - constrained decoding for guaranteed formats.
- [[hallucination-and-grounding]] - why low temperature alone does not prevent made-up facts.
- [[sequence-to-sequence-and-ctc]] - beam search in translation and speech recognition.

## References
- Holtzman et al. (2019), "The Curious Case of Neural Text Degeneration" (nucleus / top-p sampling): https://arxiv.org/abs/1904.09751
- Fan, Lewis and Dauphin (2018), "Hierarchical Neural Story Generation" (top-k sampling): https://arxiv.org/abs/1805.04833
- Hugging Face transformers, generation strategies: https://huggingface.co/docs/transformers/generation_strategies
