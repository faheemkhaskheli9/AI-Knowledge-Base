---
title: Sequence-to-sequence models and CTC
category: concepts
tags: [seq2seq, encoder-decoder, attention, teacher-forcing, beam-search, ctc, alignment, lstm, gru, pytorch, sequence-modeling]
use_cases:
  - "map an input sequence to an output sequence of a different length (translation, summarisation, transliteration)"
  - "train speech or handwriting recognition without frame-level alignments (CTC)"
  - "build a small custom OCR or keyword-spotting model for a fixed domain"
  - "understand teacher forcing, exposure bias and beam search before using a transformer"
  - "convert one structured sequence into another (code normalisation, address parsing, g2p)"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1409.3215
  - https://arxiv.org/abs/1409.0473
  - https://www.cs.toronto.edu/~graves/icml_2006.pdf
  - https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
  - https://distill.pub/2017/ctc/
---

# Sequence-to-sequence models and CTC

## Summary
Sequence-to-sequence (seq2seq) problems turn an input sequence into an output sequence whose length and alignment differ: translation, summarisation, speech-to-text, OCR. Two families solve them. **Encoder-decoder** models read the whole input and generate the output token by token (autoregressively), using attention to look back at the input; this is the design of Bahdanau-attention RNNs and of the original Transformer, T5 and Whisper. **CTC** (connectionist temporal classification) instead emits one label (or "blank") per input frame and sums over all alignments, so it trains without knowing which frame produced which character; it suits monotonic tasks like speech and line OCR where output is shorter than input.

## Key concepts
- **Encoder-decoder.** Encoder turns input `x_1..x_T` into hidden states; decoder predicts `y_t` from its own previous outputs plus a context computed from the encoder states. Output ends at an end-of-sequence token.
- **Attention (Bahdanau 2014).** At each decoder step, score every encoder state, softmax the scores, take the weighted sum as context. Removes the fixed-vector bottleneck of the first seq2seq (Sutskever 2014) and is the direct ancestor of transformer cross-attention ([[transformers-and-attention]]).
- **Teacher forcing.** During training feed the *ground-truth* previous token to the decoder, not its own prediction. Fast and stable, but causes **exposure bias**: at inference the model sees its own mistakes, which it never saw in training. Mitigations: scheduled sampling, sequence-level training, or simply larger models/data.
- **Decoding.** Greedy, beam search (keep top-k partial outputs), sampling; see [[decoding-and-sampling]]. Beam search with a length penalty is standard for translation/ASR.
- **CTC.** Network outputs a distribution over `vocab + blank` at every input frame. A path like `hh-e-ll-ll-oo` collapses (merge repeats, then remove blanks) to `hello`. CTC loss = −log of the sum of probabilities of all paths that collapse to the target, computed by dynamic programming. Assumes monotonic alignment and conditional independence between frames.
- **CTC constraints.** Input length must be ≥ target length plus the number of repeated adjacent characters (a blank is needed between `l` and `l`). Decoding: greedy best path, or beam search with a language model (e.g. KenLM) for big gains.
- **Hybrids.** Modern ASR often trains joint CTC + attention, or uses RNN-Transducer (RNN-T) which keeps CTC's monotonic alignment but conditions on previous outputs; it is the streaming on-device ASR standard.

## When to use / scenarios
- Translation, summarisation, question generation, data-to-text: encoder-decoder; in practice fine-tune a pretrained T5/BART/mT5 or use an LLM ([[fine-tuning-and-peft]]).
- Speech recognition on custom vocabularies or low-resource languages: CTC fine-tuning of wav2vec 2.0-style encoders, or Whisper (encoder-decoder) ([[speech-to-text]]).
- Line-level OCR (licence plates, meter readings, receipts, handwritten forms): CNN+BiLSTM+CTC (CRNN) is small, fast and trains on (image, text) pairs with no character boxes ([[ocr]]).
- Grapheme-to-phoneme, transliteration, normalising addresses or product codes: small seq2seq models train on a CPU/GPU in minutes.
- Gesture and sign recognition from sensor/pose sequences: CTC over frame features.
- NOT for fixed-length labelling (one tag per token, e.g. NER): use sequence labelling (token classification) instead, simpler and more accurate.
- NOT CTC when output order differs from input order (translation) or output can be longer than input.

## Setup & code
```bash
pip install torch
```
A tiny CTC model learns to read digit sequences from noisy one-hot "frames" where each digit spans a random number of frames - no alignment is given, only the target string:
```python
import torch
from torch import nn

torch.manual_seed(0)
BLANK, N_DIGITS = 0, 10          # class 0 = blank, classes 1..10 = digits 0..9

def make_batch(bs=32, max_len=5):
    xs, ys, x_lens, y_lens = [], [], [], []
    for _ in range(bs):
        digits = torch.randint(0, N_DIGITS, (torch.randint(2, max_len + 1, ()).item(),))
        frames = []
        for d in digits:  # each digit lasts 2-4 frames, then a short gap
            frames += [d.item() + 1] * torch.randint(2, 5, ()).item() + [BLANK]
        x = nn.functional.one_hot(torch.tensor(frames), N_DIGITS + 1).float()
        xs.append(x + 0.3 * torch.randn_like(x))  # noise
        ys.append(digits + 1); x_lens.append(len(frames)); y_lens.append(len(digits))
    x = nn.utils.rnn.pad_sequence(xs)                       # (T, B, F)
    return x, torch.cat(ys), torch.tensor(x_lens), torch.tensor(y_lens)

class CTCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(N_DIGITS + 1, 64, bidirectional=True)
        self.out = nn.Linear(128, N_DIGITS + 1)
    def forward(self, x):
        return self.out(self.rnn(x)[0]).log_softmax(-1)      # CTCLoss wants log-probs (T, B, C)

model, ctc = CTCModel(), nn.CTCLoss(blank=BLANK, zero_infinity=True)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(600):
    x, y, xl, yl = make_batch()
    loss = ctc(model(x), y, xl, yl)
    opt.zero_grad(); loss.backward(); opt.step()
print("final CTC loss", round(loss.item(), 3))

def greedy_decode(log_probs):  # merge repeats, drop blanks
    best = log_probs.argmax(-1).tolist()
    out, prev = [], None
    for c in best:
        if c != prev and c != BLANK:
            out.append(c - 1)
        prev = c
    return out

x, y, xl, yl = make_batch(bs=1)
print("target ", (y - 1).tolist())
print("decoded", greedy_decode(model(x)[: xl[0], 0]))
```
For encoder-decoder work, start from a pretrained checkpoint (`transformers` `AutoModelForSeq2SeqLM`, e.g. T5) rather than an RNN from scratch.

## Choosing / trade-offs
- **CTC vs attention encoder-decoder.** CTC: non-autoregressive, fast, streaming-friendly, robust alignment, but no output-to-output dependency (needs an external LM for spelling/grammar). Encoder-decoder: models output dependencies, handles reordering and length changes, best accuracy, but slower (token-by-token), can hallucinate or loop on long or silent inputs.
- **RNN vs transformer seq2seq.** Transformers win on accuracy and parallel training; small GRU/LSTM models still fit microcontrollers and tiny datasets ([[cnn-and-rnn-architectures]]).
- **From scratch vs pretrained.** From scratch only for narrow domains with synthetic or plentiful data (plates, meters, g2p). Everything open-domain: fine-tune pretrained.
- **Beam width.** 4-10 gives most of the gain; larger beams can *reduce* quality (shorter, generic outputs) without a length penalty.

## Gotchas
- `nn.CTCLoss` needs **log-probabilities** shaped `(T, N, C)` (time first), integer targets without blanks, and real per-sample input lengths; passing softmax probabilities or the padded length for every sample gives wrong loss silently.
- Infinite CTC loss means an input is too short for its target; `zero_infinity=True` hides those samples, so also check your downsampling (CNN strides shrink T).
- The blank index must not collide with a real label; keep it at 0 and shift labels by +1.
- Teacher-forced training loss looks great while free-running generation is poor: always evaluate with real decoding (BLEU, chrF, WER/CER), not training loss.
- Forgetting to shift decoder inputs right (start token) lets the decoder see the token it must predict; the loss collapses to near zero and inference is garbage.
- Padding tokens must be masked in attention and ignored in the loss (`ignore_index`).
- Encoder-decoder ASR/translation can hallucinate fluent text on silence or noise; CTC does not, which is a reason to prefer it in some production systems.

## Related
- [[cnn-and-rnn-architectures]] - LSTM/GRU and CNN encoders used in seq2seq/CTC models.
- [[transformers-and-attention]] - attention generalised; the encoder-decoder transformer.
- [[decoding-and-sampling]] - greedy, beam search and sampling at inference.
- [[speech-to-text]] - CTC, RNN-T and Whisper-style ASR in practice.
- [[ocr]] - CRNN+CTC line recognisers and modern OCR models.
- [[loss-functions]] - masking and reduction for sequence losses.

## References
- Sutskever et al., Sequence to Sequence Learning with Neural Networks: https://arxiv.org/abs/1409.3215
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/abs/1409.0473
- Graves et al., Connectionist Temporal Classification (ICML 2006): https://www.cs.toronto.edu/~graves/icml_2006.pdf
- PyTorch `CTCLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
- Hannun, Sequence Modeling with CTC (Distill): https://distill.pub/2017/ctc/
