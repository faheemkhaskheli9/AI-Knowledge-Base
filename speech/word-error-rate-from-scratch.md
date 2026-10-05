---
title: Word error rate (WER) and CER from scratch (Levenshtein alignment, S/D/I counts, normalisation)
category: speech
tags: [wer, cer, word-error-rate, levenshtein, edit-distance, asr-evaluation, text-normalization, jiwer, from-scratch]
use_cases:
  - "measure the accuracy of a speech-to-text model on my own recordings"
  - "compare two ASR vendors or Whisper sizes fairly on the same test set"
  - "understand why two tools report different WER for the same transcripts"
status: draft
last_verified: 2026-10-05
sources:
  - https://github.com/jitsi/jiwer
  - https://doi.org/10.1145/321796.321811
  - https://arxiv.org/abs/2212.04356
---

# Word error rate (WER) and CER from scratch (Levenshtein alignment, S/D/I counts, normalisation)

## Summary
Word error rate is the standard accuracy metric for speech recognition. Align the hypothesis transcript to the reference with the minimum number of word edits, count substitutions (S), deletions (D) and insertions (I), and divide by the number of reference words: `WER = (S + D + I) / N`. Character error rate (CER) does the same on characters and is the norm for languages without spaces (Chinese, Japanese, Thai). This file implements the alignment with a backtrace, a text normaliser, corpus-level WER and CER. It cross-checks against `jiwer` and shows how normalisation and averaging change the number by large amounts.

## Key concepts
- **Edit distance.** Dynamic programming over a `(n+1) x (m+1)` table: `d[i][j] = min(d[i-1][j-1] + [ref_i != hyp_j], d[i-1][j] + 1, d[i][j-1] + 1)`. The backtrace through the table labels each step as a match, S, D or I (Wagner & Fischer 1974).
- **WER is not a percentage of words wrong.** Insertions are unbounded, so WER can exceed 100%. An empty hypothesis scores exactly 100% (all deletions).
- **Corpus WER vs mean per-utterance WER.** The standard is total edits over total reference words across the test set. Averaging per-utterance WERs weights a two-word command as much as a 40-word sentence.
- **Normalisation.** Case, punctuation, contractions, numbers ("ten" vs "10"), spelling variants and fillers all count as errors unless you normalise both sides the same way first. Whisper's paper reports WER after its own English normaliser for this reason (Radford et al. 2022).
- **CER.** The same alignment over characters (here including spaces). It is less sensitive to word splitting ("new york" vs "newyork" costs 2 word errors but 1 character).

## When to use / scenarios
- Picking or monitoring an ASR model ([[speech-to-text]]): score each candidate on 50–200 of your own recordings with human reference transcripts. Public leaderboard WERs rarely transfer to your accents, microphones and vocabulary.
- Regression tests for a voice pipeline: track corpus WER and the S/D/I split after every model or VAD change ([[voice-activity-detection-from-scratch]]). Deletions jumping usually means audio was cut, insertions jumping means hallucinated text in silence.
- Low-resource and character languages: report CER, or both.
- Not for: meaning. "Book a table for two" vs "book a table for 2" costs one error, and "don't" vs "do" costs one as well. For downstream intent or entity accuracy, measure the task (slot accuracy, entity F1) or use a semantic metric ([[bleu-and-rouge-from-scratch]] covers n-gram overlap for generated text).

## Setup & code
Standard library only for the metric. `jiwer` (`pip install jiwer`) is used only as an optional cross-check.

```python
import re
import unicodedata


def align(ref, hyp):
    """Levenshtein alignment of two token lists. Returns (S, D, I, ops) with ops as (op, ref_tok, hyp_tok)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i][j] = min(sub, d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m                                   # backtrace: prefer match/sub, then del, then ins
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j - 1])); j -= 1
    ops.reverse()
    S = sum(o == "S" for o, _, _ in ops)
    D = sum(o == "D" for o, _, _ in ops)
    I = sum(o == "I" for o, _, _ in ops)
    return S, D, I, ops


def normalize(text):
    """Minimal English normaliser: NFKC, lowercase, drop punctuation except in-word apostrophes, squeeze spaces."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = re.sub(r"[^\w\s']|(?<!\w)'|'(?!\w)", " ", text)
    return text.split()


def wer(refs, hyps, norm=normalize):
    """Corpus WER = (S + D + I) / N summed over all utterances (not the mean of per-utterance WERs)."""
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        r, h = norm(r), norm(h)
        s, d, i, _ = align(r, h)
        S, D, I, N = S + s, D + d, I + i, N + len(r)
    return (S + D + I) / N, S, D, I, N


def cer(refs, hyps):
    chars = lambda t: list(" ".join(normalize(t)))
    return wer(refs, hyps, norm=chars)[0]


refs = [
    "Turn on the kitchen lights.",
    "What's the weather in New York tomorrow?",
    "Set a timer for ten minutes",
    "Play the song I liked yesterday",
    "Call mom",
]
hyps = [
    "turn on the kitchen light",
    "whats the weather in newyork tomorrow",
    "set a timer for 10 minutes",
    "play the song i like yesterday please",
    "call mom mom uh call mom",
]

S, D, I, ops = align(normalize(refs[1]), normalize(hyps[1]))
print("alignment:", " ".join(t if o == "=" else f"[{o}:{t}>{h}]" for o, t, h in ops))
print(f"utt 2: S={S} D={D} I={I} N={len(normalize(refs[1]))}")

w, S, D, I, N = wer(refs, hyps)
print(f"corpus WER {w:.3f}  (S={S} D={D} I={I} N={N})")
per_utt = [wer([r], [h])[0] for r, h in zip(refs, hyps)]
print("per-utterance WER:", " ".join(f"{x:.2f}" for x in per_utt))
print(f"mean of per-utterance WER {sum(per_utt) / len(per_utt):.3f}  <- weights 'Call mom' like a long sentence")
print(f"CER {cer(refs, hyps):.3f}")
raw = lambda t: t.split()
print(f"WER without normalisation {wer(refs, hyps, norm=raw)[0]:.3f}")

# Sanity checks
assert wer(["a b c"], ["a b c"])[0] == 0
assert wer(["a b"], ["x y z w"])[0] == 2.0          # WER is not capped at 100%
assert wer(["a b c"], [""])[0] == 1.0                # empty hypothesis = all deletions

try:
    import jiwer
    lower = [" ".join(normalize(t)) for t in refs], [" ".join(normalize(t)) for t in hyps]
    print(f"jiwer on the same normalised text: {jiwer.wer(*lower):.3f}")
except ImportError:
    pass
```

Output (Python 3.14):
```
alignment: [S:what's>whats] the weather in [D:new>] [S:york>newyork] tomorrow
utt 2: S=2 D=1 I=0 N=7
corpus WER 0.423  (S=5 D=1 I=5 N=26)
per-utterance WER: 0.20 0.43 0.17 0.33 2.00
mean of per-utterance WER 0.626  <- weights 'Call mom' like a long sentence
CER 0.229
WER without normalisation 0.654
jiwer on the same normalised text: 0.423
```

How to read it:
- The alignment shows where the errors are: one missing apostrophe and one merged word ("newyork") cost 3 edits in a 7-word sentence. A tie in the table can be broken different ways (here `new>` deleted and `york>newyork` substituted), but every valid alignment has the same total cost.
- "Call mom" → "call mom mom uh call mom" has 4 insertions on 2 reference words: 200% WER for that utterance. Repetition loops like this one are a known failure of seq2seq ASR on short or silent clips.
- Corpus WER is 42.3%. The mean of per-utterance WERs is 62.6% because the 2-word command counts as much as the 7-word sentence. Always say which one you report.
- Skipping normalisation raises WER from 42.3% to 65.4% for identical transcripts, purely from case and punctuation. Most "vendor A beats vendor B" claims that differ by a few points are this effect.
- "ten" vs "10" is still a substitution here. A full normaliser (number words, British/American spelling, fillers like "uh") would remove it, so decide and document what you normalise.
- The from-scratch result matches `jiwer` exactly on the same normalised text.

## Choosing / trade-offs
- **Library vs own code.** Use `jiwer` (or `evaluate`'s `wer`, which wraps it) in practice; it is fast and gives alignments. The value of owning the code is knowing what is being counted. The normaliser matters more than the library.
- **Normaliser choice.** For English, Whisper's `EnglishTextNormalizer` (in the `openai-whisper` package) is a reasonable shared standard: it standardises numbers, spellings and contractions. For a product, keep a custom normaliser that matches how you use the text (e.g. keep case if you display it).
- **WER vs CER.** WER for space-delimited languages and when words are the unit users care about. CER for character languages, agglutinative languages (Turkish, Finnish), and OCR. Report both when comparing across languages.
- **WER vs task metrics.** WER treats every word equally. If the transcript feeds an LLM or intent parser, a small WER difference may not matter. Measure end-task accuracy too, and track entity/keyword error rate for names, numbers and domain terms.

## Gotchas
- Normalise reference and hypothesis with the same function. Normalising only one side inflates WER.
- Different references give different numbers: verbatim transcripts (with "um", restarts) vs clean transcripts. Decide which style your references use and normalise fillers to match.
- WER of a few test clips is noisy. With 20 utterances, one bad clip can swing WER by several points. Use hundreds of utterances and report a bootstrap confidence interval over utterances.
- Segmentation mismatches inflate errors. If the ASR splits audio differently from your reference segments, concatenate per recording before aligning.
- The pure-Python DP is O(n·m). That is fine for utterances, but aligning hour-long transcripts at once needs the C-backed `jiwer`/`rapidfuzz` or chunking.

## Related
- [[speech-to-text]] - choosing and running ASR models; WER is how you compare them.
- [[voice-activity-detection-from-scratch]] - bad VAD shows up as deletions (cut speech) or insertions (text in silence).
- [[bleu-and-rouge-from-scratch]] - n-gram overlap metrics for generated text, the MT/summarisation counterpart.
- [[dynamic-time-warping-from-scratch]] - the same DP-alignment idea on continuous sequences.
- [[ctc-loss-from-scratch-numpy]] - how ASR models are trained to produce the transcripts being scored.

## References
- jiwer (WER/CER/MER/WIL implementation): https://github.com/jitsi/jiwer
- Wagner & Fischer (1974), "The String-to-String Correction Problem", JACM: https://doi.org/10.1145/321796.321811
- Radford et al. (2022), "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper; see its text standardisation appendix): https://arxiv.org/abs/2212.04356
