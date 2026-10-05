---
title: BLEU and ROUGE from scratch (clipped n-gram precision, brevity penalty, LCS, smoothing, where they fail)
category: llm-apps
tags: [bleu, rouge, rouge-l, n-gram-overlap, machine-translation, summarization, llm-evaluation, reference-based-metrics, sacrebleu, from-scratch]
use_cases:
  - "score translations or summaries against reference texts with a cheap, reproducible metric"
  - "understand why an LLM's good paraphrase gets a low BLEU or ROUGE score"
  - "choose between corpus BLEU, sentence BLEU, ROUGE-1/2/L and a learned or LLM-judge metric"
  - "report BLEU numbers that are comparable with published ones"
status: draft
last_verified: 2026-10-05
sources:
  - https://aclanthology.org/P02-1040/
  - https://aclanthology.org/W04-1013/
  - https://aclanthology.org/W18-6319/
---

# BLEU and ROUGE from scratch (clipped n-gram precision, brevity penalty, LCS, smoothing, where they fail)

## Summary
BLEU (Papineni et al. 2002) and ROUGE (Lin 2004) score generated text by counting word n-grams it shares with one or more human references. BLEU is precision-oriented (how much of the output appears in a reference), with clipping and a brevity penalty, and was built for machine translation. ROUGE is recall-oriented (how much of the reference the output covers), including ROUGE-L based on the longest common subsequence, and was built for summarisation. Both are cheap, deterministic and still widely reported, but they reward surface overlap, not meaning. This file implements both in plain Python and shows the cases where they mislead.

## Key concepts
- **Clipped n-gram precision.** For each n-gram in the output, count matches up to the maximum number of times it appears in any single reference. Without clipping, "the the the the" would score perfect unigram precision.
- **Geometric mean over n = 1..4.** BLEU multiplies the four precisions (geometric mean). If any order has zero matches, BLEU is 0, which is common for short sentences.
- **Brevity penalty.** Precision alone rewards short outputs, so BLEU multiplies by `exp(1 − r/c)` when the output length `c` is shorter than the reference length `r`.
- **Corpus-level.** BLEU pools n-gram counts over the whole test set before dividing. Corpus BLEU is not the mean of sentence BLEU scores.
- **Smoothing.** Sentence-level BLEU needs smoothing (e.g. add one to the numerator and denominator for n ≥ 2, "BLEU+1", Lin & Och 2004) to avoid zeros; the value then depends on the smoothing method.
- **ROUGE-N and ROUGE-L.** ROUGE-N counts overlapping n-grams and reports precision, recall or F1. ROUGE-L uses the longest common subsequence, which rewards words in the right order without requiring them to be contiguous.
- **Multiple references.** Both metrics improve with more references, because each valid wording has a chance of being in one of them.

## When to use / scenarios
- Machine translation regression tests: a fixed test set, a fixed tokenisation, compare model versions with sacreBLEU (or chrF, which is often more robust).
- Summarisation, headline or caption generation where references exist and outputs stay close to source wording: ROUGE-1/2/L as a sanity metric next to human or LLM-judge scores.
- Extractive or near-extractive tasks (copying spans, templated outputs, code or SQL with a canonical form) where surface overlap does track quality.
- CI checks that catch a broken pipeline (empty outputs, wrong language, truncation) cheaply.
- Not for open-ended LLM answers, chat, reasoning or anything with many valid wordings: use task-specific checks, embedding or learned metrics (BERTScore, COMET for MT), or an LLM judge calibrated against humans ([[llm-evaluation]]).
- Not for comparing systems across papers unless the exact tokenisation and settings are reported (that is what sacreBLEU's signature is for).

## Setup & code
Standard library only. One reference, six hypotheses that each break the metrics in a different way.

```python
import math
from collections import Counter


def ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def corpus_bleu(hyps, refs_list, max_n=4, smooth=False):
    """BLEU (Papineni et al. 2002): clipped n-gram precisions pooled over the corpus, geometric mean, brevity penalty."""
    match, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, refs in zip(hyps, refs_list):
        h = hyp.split()
        rs = [r.split() for r in refs]
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]       # closest reference length
        for n in range(1, max_n + 1):
            hc = ngrams(h, n)
            max_ref = Counter()
            for r in rs:
                max_ref |= ngrams(r, n)                                       # max count of each n-gram in any ref
            match[n - 1] += sum(min(c, max_ref[g]) for g, c in hc.items())   # clipping
            total[n - 1] += max(len(h) - n + 1, 0)
    if smooth:                                                                # add-one on n >= 2 (Lin & Och 2004 style)
        match = [match[0]] + [m + 1 for m in match[1:]]
        total = [total[0]] + [t + 1 for t in total[1:]]
    if min(match) == 0:
        return 0.0
    log_p = sum(math.log(m / t) for m, t in zip(match, total)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100 * bp * math.exp(log_p)


def lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], prev + 1 if x == y else max(dp[j], dp[j - 1])
    return dp[-1]


def rouge(hyp, ref):
    """ROUGE-1/2 (n-gram overlap) and ROUGE-L (longest common subsequence), each as F1."""
    h, r = hyp.lower().split(), ref.lower().split()
    out = {}
    for n in (1, 2):
        overlap = sum((ngrams(h, n) & ngrams(r, n)).values())
        p, rec = overlap / max(len(h) - n + 1, 1), overlap / max(len(r) - n + 1, 1)
        out[f"R{n}"] = 0 if overlap == 0 else 2 * p * rec / (p + rec)
    l = lcs(h, r)
    out["RL"] = 0 if l == 0 else 2 * (l / len(h)) * (l / len(r)) / (l / len(h) + l / len(r))
    return out


ref = "the cat sat on the mat near the door"
cases = {
    "exact copy": ref,
    "one word changed": "the cat sat on the rug near the door",
    "paraphrase": "a cat was sitting on the doormat",
    "shuffled words": "door the near mat the on sat cat the",
    "truncated": "the cat sat",
    "the the the": "the the the the the the the the the",
}
print(f"{'hypothesis':18s} {'BLEU':>6s} {'BLEU+1':>7s} {'R-1':>5s} {'R-2':>5s} {'R-L':>5s}")
for name, hyp in cases.items():
    r = rouge(hyp, ref)
    print(f"{name:18s} {corpus_bleu([hyp], [[ref]]):6.1f} {corpus_bleu([hyp], [[ref]], smooth=True):7.1f} "
          f"{r['R1']:5.2f} {r['R2']:5.2f} {r['RL']:5.2f}")

# A second reference rescues the paraphrase; corpus BLEU != mean of sentence BLEU
refs2 = [ref, "a cat was sitting on the mat by the door"]
print(f"\nparaphrase with 2 references: BLEU {corpus_bleu([cases['paraphrase']], [refs2]):.1f}")
hyps = [cases["one word changed"], cases["truncated"]]
sent = [corpus_bleu([h], [[ref]], smooth=True) for h in hyps]
print(f"corpus BLEU of 2 sentences {corpus_bleu(hyps, [[ref], [ref]]):.1f} vs mean sentence BLEU+1 {sum(sent) / 2:.1f}")
```

Output (Python 3.14):
```
hypothesis           BLEU  BLEU+1   R-1   R-2   R-L
exact copy          100.0   100.0  1.00  1.00  1.00
one word changed     59.7    65.6  0.89  0.75  0.89
paraphrase            0.0    19.0  0.38  0.14  0.38
shuffled words        0.0    21.1  1.00  0.00  0.33
truncated             0.0    13.5  0.50  0.40  0.50
the the the           0.0    16.0  0.33  0.00  0.33

paraphrase with 2 references: BLEU 60.8
corpus BLEU of 2 sentences 37.9 vs mean sentence BLEU+1 39.6
```

How to read it:
- A correct paraphrase ("a cat was sitting on the doormat") gets BLEU 0 and ROUGE-1 0.38, lower than the meaningless shuffled sentence on ROUGE-1 (1.00) and on smoothed BLEU (21.1 vs 19.0). Overlap metrics measure wording, not meaning.
- ROUGE-1 alone is blind to order: the shuffled words score a perfect 1.00. ROUGE-2 (0.00) and ROUGE-L (0.33) catch it, which is why papers report all three.
- Clipping works: nine copies of "the" get unigram credit only for the three "the"s in the reference (ROUGE-1 0.33), and BLEU is 0.
- One changed word costs 40 BLEU points (100 → 59.7), because it breaks 1 unigram, 2 bigrams, 3 trigrams and 4 four-grams. Small edits have big effects at sentence level.
- Adding a second reference that happens to use the paraphrase's wording lifts its BLEU from 0 to 60.8: scores depend as much on the references as on the system.
- Unsmoothed sentence BLEU is 0 for every non-trivial case above; smoothing gives numbers, but they depend on the smoothing method. Corpus BLEU pools counts, so it differs from averaging sentence scores (37.9 vs 39.6 here).

## Choosing / trade-offs
- **Use the standard tools for reported numbers.** `sacrebleu` (BLEU, chrF, TER with a version signature) and `rouge-score` (Google's ROUGE with optional stemming) or Hugging Face `evaluate` wrappers. Tokenisation alone moves BLEU by several points, so a from-scratch number is not comparable with a published one.
- **BLEU vs chrF.** chrF uses character n-grams and recall, so it gives partial credit for morphology and compounds; it correlates better with humans for many languages and is less zero-prone at sentence level.
- **ROUGE variant.** ROUGE-1/2 for content coverage, ROUGE-L (or ROUGE-Lsum on sentence-split summaries) for order. Report F1 unless the task specifically wants recall (e.g. summaries with a length cap).
- **Overlap vs learned vs judge.** Overlap: free, deterministic, weak on paraphrase. Learned metrics (BERTScore, COMET, BLEURT): better correlation, need a model and are not comparable across versions. LLM judges: flexible, costly, have position and length biases; calibrate against human labels ([[llm-evaluation]]).
- **System-level only.** Even where BLEU tracks quality across systems, differences under about 1–2 points on one test set are within noise. Use paired bootstrap resampling (sacreBLEU supports it) before claiming an improvement.

## Gotchas
- Never compare BLEU numbers computed with different tokenisation, lowercasing or reference sets; report sacreBLEU's signature.
- Sentence-level BLEU is noisy and needs smoothing; do not use it to rank individual outputs or as an RL reward without care.
- ROUGE on LLM summaries rewards verbosity and copying the source; a short, abstractive summary that is better can score lower.
- Languages without spaces (Chinese, Japanese, Thai) need a tokeniser (sacreBLEU has `zh`, `ja-mecab` options) or character-level metrics; whitespace splitting gives nonsense.
- `Counter | Counter` is an element-wise max (used for multi-reference clipping), while `&` is a min (used for overlap). Mixing them up silently changes the metric.
- A model fine-tuned to maximise BLEU or ROUGE learns the metric's blind spots; keep a human or judge-based check alongside.

## Related
- [[llm-evaluation]] - where overlap metrics sit next to task checks and LLM judges.
- [[elo-and-bradley-terry-leaderboard-from-scratch]] - pairwise preference evaluation when references do not exist.
- [[decoding-strategies-from-scratch-numpy]] - decoding choices that change these scores.
- [[mmr-diversity-reranking-from-scratch]] - another n-gram/similarity tool used around generation.

## References
- Papineni et al. (2002), "BLEU: a Method for Automatic Evaluation of Machine Translation", ACL: https://aclanthology.org/P02-1040/
- Lin (2004), "ROUGE: A Package for Automatic Evaluation of Summaries": https://aclanthology.org/W04-1013/
- Lin & Och (2004), "ORANGE: a Method for Evaluating Automatic Evaluation Metrics for Machine Translation" (smoothed sentence BLEU): https://aclanthology.org/C04-1072/
- Post (2018), "A Call for Clarity in Reporting BLEU Scores" (sacreBLEU): https://aclanthology.org/W18-6319/
