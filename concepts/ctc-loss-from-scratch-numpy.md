---
title: CTC loss from scratch in NumPy (forward-backward, blanks, gradient, greedy decoding)
category: concepts
tags: [ctc, connectionist-temporal-classification, speech-recognition, ocr, sequence-alignment, forward-backward, dynamic-programming, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement the CTC loss and its gradient with the forward-backward algorithm in NumPy"
  - "verify a CTC implementation against brute-force enumeration and finite differences"
  - "see why repeated labels need a blank between them and how greedy CTC decoding collapses paths"
  - "explain CTC for speech or handwriting recognition in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.cs.toronto.edu/~graves/icml_2006.pdf
  - https://distill.pub/2017/ctc/
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
---

# CTC loss from scratch in NumPy (forward-backward, blanks, gradient, greedy decoding)

## Summary
Connectionist Temporal Classification (CTC) trains a model that outputs one label distribution per input frame (audio frames, image columns) to produce a shorter label sequence, without frame-level alignments. It adds a **blank** symbol and defines a many-to-one collapse: merge repeated symbols, then delete blanks, so `33-11--11444 → 3 1 1 4`. The loss is `−log` of the total probability of all frame paths that collapse to the target. A forward-backward dynamic program over the blank-extended target computes it in `O(T·S)` time. Below, a log-space implementation matches brute-force enumeration of all 3⁶ paths to six decimals. Its analytic gradient (`softmax − posterior occupancy`) matches finite differences to 1e-10. Gradient descent on free logits for one 12-frame "utterance" gets the greedy decode right after one step and takes the NLL from 10.3 to 0.03.

## Key concepts
- **Extended target.** `l' = (blank, l₁, blank, l₂, …, blank)`, of length `S = 2|l| + 1`. Every valid path walks through `l'` left to right. At each frame it stays in place, moves one step, or skips one step over a blank, but only between two *different* labels.
- **Why the blank.** Without it, `1 1` could not be told apart from `1`, because repeats are merged. A repeated label must have a blank between its copies, which is why the skip transition is forbidden when `l'_s = l'_{s−2}`. The blank also lets the model emit "nothing" on silent or in-between frames.
- **Forward variable.** `α_t(s)` is the total probability of all path prefixes up to frame `t` that end at position `s` of `l'`. `p(l|x) = α_T(S) + α_T(S−1)`, because a path may end on the last label or on the trailing blank.
- **Gradient.** With the backward variable `β`, the posterior occupancy is `γ_t(k) = Σ_{s: l'_s = k} α_t(s) β_t(s) / (p · y_t(k))` (β includes frame t's emission, hence the division). Then `∂(−log p)/∂z_t(k) = y_t(k) − γ_t(k)`: softmax minus the expected alignment, the same form as cross-entropy against a soft target.
- **Conditional independence.** CTC assumes the outputs at different frames are independent given the input. It has no internal language model, so decoding with an external LM or beam search adds that knowledge back.

## When to use / scenarios
- Learning: a classic dynamic program inside a loss function, and the clearest example of marginalizing over latent alignments ([[sequence-to-sequence-and-ctc]], [[hmm-from-scratch]]).
- Interviews: "how does CTC handle unaligned labels", "why the blank", "CTC vs attention seq2seq vs RNN-T", "how do you decode CTC".
- Practice: speech recognition encoders (wav2vec 2.0 and Conformer-CTC fine-tuning), handwriting and scene-text OCR (CRNN), keyword spotting, sign language and gesture recognition, and as an auxiliary loss for attention encoder-decoders ([[speech-to-text]], [[ocr]]).
- Not for: outputs longer than the input (CTC needs `T ≥ |l| + number of repeats`); tasks where output order differs from input order, such as translation (use attention seq2seq); or streaming ASR that needs label dependencies (use RNN-Transducer).

## Setup & code
NumPy and the stdlib. Runs in a few seconds.

```python
import itertools
import numpy as np

rng = np.random.default_rng(0)
BLANK = 0


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def ctc_loss_and_grad(logits, target):
    """CTC negative log-likelihood and its gradient w.r.t. logits (T x V), log-space."""
    T = len(logits)
    lp = log_softmax(logits)
    ext = [BLANK]
    for c in target:
        ext += [c, BLANK]                                 # l' = blank, c1, blank, c2, ..., blank
    S = len(ext)
    alpha = np.full((T, S), -np.inf)
    alpha[0, 0] = lp[0, ext[0]]
    if S > 1:
        alpha[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [alpha[t - 1, s]]
            if s >= 1:
                terms.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2]:
                terms.append(alpha[t - 1, s - 2])         # skip a blank between different labels
            alpha[t, s] = np.logaddexp.reduce(terms) + lp[t, ext[s]]
    beta = np.full((T, S), -np.inf)
    beta[T - 1, S - 1] = lp[T - 1, ext[S - 1]]
    if S > 1:
        beta[T - 1, S - 2] = lp[T - 1, ext[S - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            terms = [beta[t + 1, s]]
            if s + 1 < S:
                terms.append(beta[t + 1, s + 1])
            if s + 2 < S and ext[s] != BLANK and ext[s] != ext[s + 2]:
                terms.append(beta[t + 1, s + 2])
            beta[t, s] = np.logaddexp.reduce(terms) + lp[t, ext[s]]
    log_p = np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2]) if S > 1 else alpha[T - 1, 0]
    # posterior occupancy per (t, label): gamma = alpha*beta / (p * y_t(k))
    ab = alpha + beta
    occ = np.full_like(lp, -np.inf)
    for s in range(S):
        occ[:, ext[s]] = np.logaddexp(occ[:, ext[s]], ab[:, s])
    grad = np.exp(lp) - np.exp(occ - lp - log_p)          # d(-log p)/d logits
    return -log_p, grad


def collapse(path):
    out, prev = [], None
    for k in path:
        if k != prev and k != BLANK:
            out.append(int(k))
        prev = k
    return out


def brute_force_nll(logits, target):
    p = np.exp(log_softmax(logits))
    total = 0.0
    for path in itertools.product(range(logits.shape[1]), repeat=len(logits)):
        if collapse(path) == list(target):
            total += np.prod(p[np.arange(len(path)), path])
    return -np.log(total)


# 1) check the forward algorithm against enumerating every alignment
for target in ([1, 2], [1, 1], [2, 1, 2]):
    logits = rng.normal(size=(6, 3))
    nll, _ = ctc_loss_and_grad(logits, target)
    print(f"target {target}: forward-backward NLL {nll:.6f}  brute force over 3^6 paths {brute_force_nll(logits, target):.6f}")

# 2) check the analytic gradient with finite differences
logits = rng.normal(size=(8, 4))
target = [1, 3, 3]
_, g = ctc_loss_and_grad(logits, target)
num = np.zeros_like(logits)
for i, j in np.ndindex(*logits.shape):
    d = np.zeros_like(logits)
    d[i, j] = 1e-5
    num[i, j] = (ctc_loss_and_grad(logits + d, target)[0] - ctc_loss_and_grad(logits - d, target)[0]) / 2e-5
print(f"\nmax |analytic - numeric| gradient: {np.abs(g - num).max():.2e}")

# 3) learn free logits for one utterance by gradient descent, then greedy-decode
T, V = 12, 5
target = [3, 1, 1, 4]                                    # repeated label needs a blank in between
logits = rng.normal(scale=0.1, size=(T, V))
for step in range(301):
    nll, g = ctc_loss_and_grad(logits, target)
    if step in (0, 1, 5, 50, 300):
        best_path = logits.argmax(axis=1)
        print(f"step {step:>3}: NLL {nll:7.4f}  best path {''.join('-' if k == 0 else str(k) for k in best_path)}"
              f"  -> {collapse(best_path)}")
    logits -= 1.0 * g
```

Output (Python 3.14, NumPy 2.5):
```
target [1, 2]: forward-backward NLL 3.086137  brute force over 3^6 paths 3.086137
target [1, 1]: forward-backward NLL 2.857436  brute force over 3^6 paths 2.857436
target [2, 1, 2]: forward-backward NLL 1.731289  brute force over 3^6 paths 1.731289

max |analytic - numeric| gradient: 1.06e-10
step   0: NLL 10.2638  best path 3-4-13--12--  -> [3, 4, 1, 3, 1, 2]
step   1: NLL  7.9773  best path 33--11--144-  -> [3, 1, 1, 4]
step   5: NLL  3.5621  best path 33-11--11444  -> [3, 1, 1, 4]
step  50: NLL  0.2295  best path 33-11--11444  -> [3, 1, 1, 4]
step 300: NLL  0.0327  best path 33-11--11444  -> [3, 1, 1, 4]
```

The brute-force check is the real test. Summing over all 729 paths for 6 frames and 3 symbols gives the same number as the `O(T·S)` recursion, including the `[1, 1]` target where the skip rule matters. The learned best path `33-11--11444` shows what CTC alignments look like. Each label occupies a run of frames, and the two `1`s are kept apart by a blank (`11--11`), because without it they would merge into a single `1`. After one gradient step the greedy decode is already correct. The rest of training sharpens the path distribution (NLL → 0.03) without changing the argmax path. Real CTC models end up "peaky", with most frames assigned to blank and labels emitted in short spikes.

## Choosing / trade-offs
- **CTC vs attention encoder-decoder.** CTC is non-autoregressive, fast, streaming-friendly and monotonic by construction, but it has no output-side context. Attention seq2seq models label dependencies but can hallucinate or skip on long inputs. Hybrid CTC/attention training (an auxiliary CTC loss) is a common best-of-both setup.
- **CTC vs RNN-Transducer.** RNN-T adds a prediction network over previous labels, removing the independence assumption while staying streamable. It is the standard for on-device ASR, but more expensive to train.
- **Decoding.** Greedy (argmax per frame, then collapse) is fast and often close to the best. Prefix beam search with an n-gram or neural LM gives large gains on words that need context. Note that the most likely *path* is not the most likely *label sequence*: beam search sums over the paths of each prefix.
- **Libraries.** Use `torch.nn.CTCLoss` (cuDNN-accelerated, batched, with `zero_infinity`), or `tf.nn.ctc_loss` / `optax.ctc_loss`. Decoders: `torchaudio`'s CTC decoder or `pyctcdecode` with KenLM.

## Gotchas
- Work in log space. `α` underflows to zero within tens of frames in probability space. Use `logaddexp`, or the original paper's per-frame rescaling.
- Infeasible targets give an infinite loss. You need `T ≥ |l| + (number of adjacent repeats in l)`. Check the downsampling of the encoder (for example 4× subsampling in speech) against the longest transcript, or set `zero_infinity=True` and log those samples.
- The blank index must agree everywhere (PyTorch defaults to `blank=0`). Shifting the vocabulary by one when adding the blank is a common off-by-one bug.
- PyTorch's `CTCLoss` expects **log-probabilities** shaped `(T, N, C)`, not raw logits, and `reduction='mean'` divides by target lengths. Compare losses across setups carefully.
- A model stuck emitting only blanks (loss plateau, empty transcripts) is a typical early-training phase. Lower the learning rate, warm it up, or check that inputs are normalized and target lengths are right.
- Peaky CTC outputs make frame-level timestamps approximate. Use forced alignment (Viterbi over the same lattice) if you need word timings.

## Related
- [[sequence-to-sequence-and-ctc]] - CTC vs seq2seq vs transducers, and when to use each.
- [[speech-to-text]] - ASR models and services built on these losses.
- [[ocr]] - CRNN + CTC for text recognition.
- [[hmm-from-scratch]] - the same forward-backward idea over hidden states.
- [[decoding-strategies-from-scratch-numpy]] - greedy and beam decoding for language models.
- [[lstm-from-scratch-numpy]] - a typical CTC encoder.

## References
- Graves, Fernández, Gomez and Schmidhuber (2006), "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", ICML: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Hannun (2017), "Sequence Modeling with CTC", Distill: https://distill.pub/2017/ctc/
- PyTorch `torch.nn.CTCLoss`: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
