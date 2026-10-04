---
title: Log-mel spectrogram and MFCC from scratch (STFT, mel filterbank, DCT)
category: speech
tags: [mfcc, mel-spectrogram, log-mel, stft, filterbank, dct, audio-features, speech-recognition, numpy, from-scratch]
use_cases:
  - "turn raw audio into features for a speech, keyword-spotting or sound classifier"
  - "choose between log-mel spectrograms and MFCCs as model input"
  - "understand what Whisper-style models see instead of the waveform"
  - "debug audio features that differ between training and on-device inference"
status: draft
last_verified: 2026-10-04
sources:
  - https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
  - https://pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
  - https://github.com/openai/whisper
---

# Log-mel spectrogram and MFCC from scratch (STFT, mel filterbank, DCT)

## Summary
Most speech and audio models do not read the waveform directly. They read a log-mel spectrogram: short-time Fourier transform (STFT) power, pooled into triangular bands equally spaced on the perceptual mel scale, then logged. MFCCs (mel-frequency cepstral coefficients) add one more step, a DCT over the mel bands, which decorrelates them and keeps the coarse spectral shape in about 13 numbers per frame. This file builds both in NumPy (25 ms windows, 10 ms hop, 40 mel bands, 13 MFCCs), checks the DCT against `scipy.fft.dct`, shows that the loudest mel band tracks a chirp's frequency, and shows that volume changes only coefficient `c0`.

## Key concepts
- **Framing and windowing.** Speech is roughly stationary over 20-30 ms. Cut the signal into overlapping frames (25 ms every 10 ms is standard for 16 kHz speech) and multiply each by a Hann or Hamming window to reduce spectral leakage.
- **STFT power.** The FFT of each frame gives `n_fft/2 + 1` frequency bins. Squared magnitude is the power spectrum.
- **Mel scale.** `mel = 2595 log10(1 + f / 700)` (HTK form). It is roughly linear below 1 kHz and logarithmic above, like human pitch perception. Filters equally spaced in mel are narrow at low frequency and wide at high frequency.
- **Log compression.** Taking the log turns multiplicative gain into an additive offset and compresses dynamic range. A log-mel spectrogram is the standard input to CNN, Conformer and Transformer audio models.
- **DCT → cepstrum.** The DCT of the log-mel vector separates the slowly varying spectral envelope (vocal tract shape, low coefficients) from fine structure (pitch harmonics, high coefficients). Keeping the first ~13 coefficients keeps the envelope. `c0` is the overall log energy.

## When to use / scenarios
- Input features for keyword spotting, speaker or sound-event classifiers on small devices where a compact 13-39 dimensional vector per frame matters.
- Classical pipelines: GMM-HMM ASR, [[dynamic-time-warping-from-scratch]] template matching, speaker verification baselines, gradient-boosted models on summary statistics of MFCCs.
- Log-mel (not MFCC) for neural models: CNNs and Transformers learn their own decorrelation, and the DCT throws away information they could use. Whisper-style ASR uses log-mel input, see [[speech-to-text]].
- Not when using self-supervised encoders that take the raw waveform (wav2vec 2.0, HuBERT): feed them the audio at the sample rate they were trained on.
- Not for music pitch tasks: mel bands blur exact pitch, use a constant-Q transform or chroma features.

## Setup & code
NumPy only. The test signal is 0.5 s of a 300 Hz tone followed by a chirp from 300 Hz to 4000 Hz, at 16 kHz.

```python
import numpy as np

sr = 16000


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)          # HTK formula


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def stft_power(x, n_fft=512, hop=160, win=400):
    """25 ms Hann windows every 10 ms, zero-padded to n_fft; returns (frames, n_fft//2+1) power."""
    x = np.r_[x[0], x[1:] - 0.97 * x[:-1]]        # pre-emphasis
    n = 1 + (len(x) - win) // hop
    frames = np.stack([x[i * hop:i * hop + win] for i in range(n)]) * np.hanning(win)
    return np.abs(np.fft.rfft(frames, n_fft)) ** 2 / n_fft


def mel_filterbank(n_mels=40, n_fft=512, fmin=0, fmax=sr / 2):
    """Triangular filters equally spaced on the mel scale."""
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    lo, ctr, hi = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    return np.clip(np.minimum((freqs - lo) / (ctr - lo), (hi - freqs) / (hi - ctr)), 0, None)


def dct_matrix(n_out, n_in):
    """Orthonormal DCT-II, same as scipy.fft.dct(..., type=2, norm='ortho')."""
    k, n = np.arange(n_out)[:, None], np.arange(n_in)[None]
    m = np.sqrt(2 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    m[0] /= np.sqrt(2)
    return m


def mfcc(x, n_mfcc=13, n_mels=40):
    log_mel = np.log(stft_power(x) @ mel_filterbank(n_mels).T + 1e-10)
    return log_mel, log_mel @ dct_matrix(n_mfcc, n_mels).T


# 1 s test signal: 300 Hz tone, then a 300 -> 4000 Hz chirp, plus a little noise
rng = np.random.default_rng(0)
t = np.arange(sr) / sr
f_inst = np.where(t < 0.5, 300, 300 + (t - 0.5) / 0.5 * 3700)
x = np.sin(2 * np.pi * np.cumsum(f_inst) / sr) + 0.01 * rng.normal(size=sr)

log_mel, cc = mfcc(x)
print("log-mel", log_mel.shape, " mfcc", cc.shape)
centres = mel_to_hz(np.linspace(0, hz_to_mel(sr / 2), 42))[1:-1]
for frame in [10, 30, 60, 80, 97]:
    print(f"t={frame * 0.01:.2f}s  true {f_inst[frame * 160 + 200]:6.0f} Hz  "
          f"loudest mel band centre {centres[log_mel[frame].argmax()]:6.0f} Hz")

# MFCCs barely change with loudness except c0; that is why c0 is often dropped or normalised
_, cc_loud = mfcc(10 * x)
print("max |diff| c0:", np.abs(cc_loud[:, 0] - cc[:, 0]).max().round(2),
      " c1..c12:", np.abs(cc_loud[:, 1:] - cc[:, 1:]).max().round(4))
```

Output (Python 3.14, NumPy 2.5):
```
log-mel (98, 40)  mfcc (98, 13)
t=0.10s  true    300 Hz  loudest mel band centre    312 Hz
t=0.30s  true    300 Hz  loudest mel band centre    312 Hz
t=0.60s  true   1133 Hz  loudest mel band centre   1171 Hz
t=0.80s  true   2612 Hz  loudest mel band centre   2554 Hz
t=0.97s  true   3871 Hz  loudest mel band centre   4005 Hz
max |diff| c0: 29.13  c1..c12: 0.0062
```

One second of audio becomes 98 frames of 40 log-mel values or 13 MFCCs. The loudest band follows the chirp, and the error grows with frequency (12 Hz at 300 Hz, 134 Hz near 4 kHz) because mel bands widen as frequency rises. That is the mel scale doing its job: fine resolution where speech formants and pitch live, coarse resolution above. Scaling the signal by 10 (+20 dB) adds a constant `log(100)` to every log-mel value. The orthonormal DCT puts all of that offset into `c0` (`log(100) · √40 ≈ 29.1`) and leaves `c1..c12` unchanged up to the `1e-10` floor in near-silent bands. The DCT matrix was checked against `scipy.fft.dct(type=2, norm="ortho")`.

## Choosing / trade-offs
- **Log-mel vs MFCC.** Neural models: log-mel, 64-128 bands. Small classical models or tight memory: 13 MFCCs, often with deltas and delta-deltas (39 values per frame).
- **Number of mel bands.** 40 for classic ASR at 16 kHz, 64-128 for neural audio tagging and ASR. More bands keep more detail at a higher input cost.
- **Window and hop.** 25 ms / 10 ms for speech. Longer windows give better frequency resolution for music or environmental sound. Shorter hops give better time resolution for onsets.
- **Normalisation.** Per-utterance mean (and variance) normalisation of each coefficient (CMVN) removes channel effects such as microphone colour. It is essential for MFCC pipelines that mix recording devices.
- **Library.** `librosa.feature.melspectrogram` / `mfcc` for analysis, `torchaudio.transforms.MelSpectrogram` / `MFCC` for GPU training pipelines, `torchaudio.compliance.kaldi.fbank` when you need Kaldi-compatible features.

## Gotchas
- Implementations differ in defaults, and features from different libraries are not interchangeable. librosa uses the Slaney mel formula and area-normalised filters by default, HTK and Kaldi use the formula above, and pre-emphasis, window, padding, power vs magnitude, `log` vs `log10` vs dB all vary. Use the exact same feature code at training and inference time, including on device.
- Sample-rate mismatch: a model trained on 16 kHz features fed 44.1 kHz audio sees every frequency shifted. Resample first.
- `log(0)` on digital silence gives `-inf`. Add a floor (`1e-10` here, `1e-6` or dB clipping elsewhere).
- `n_fft` smaller than the window truncates frames. Keep `n_fft >= win_length`.
- `c0` (or frame energy) dominates distances in MFCC space. Drop it, replace it with log energy, or normalise each coefficient before k-NN, DTW or GMM modelling.
- Centre padding (librosa's `center=True`) changes the frame count and alignment. Matters when matching labels to frames.

## Related
- [[speech-to-text]] - ASR models that consume log-mel features.
- [[ctc-loss-from-scratch-numpy]] - the loss usually trained on top of these frame features.
- [[dynamic-time-warping-from-scratch]] - classic template matching over MFCC sequences.
- [[hmm-from-scratch]] - the GMM-HMM pipelines MFCCs were designed for.
- [[convolution-layer-from-scratch-numpy]] - CNNs that treat a log-mel spectrogram as an image.
- [[voice-agents]] - where feature extraction sits in a real-time speech pipeline.

## References
- librosa `feature.mfcc` and `feature.melspectrogram`: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
- torchaudio `transforms.MFCC`: https://pytorch.org/audio/stable/generated/torchaudio.transforms.MFCC.html
- SciPy `fft.dct`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
- OpenAI Whisper (log-mel spectrogram input): https://github.com/openai/whisper
- Davis and Mermelstein (1980), "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences", IEEE TASSP 28(4).
