---
title: Griffin-Lim phase reconstruction from scratch (STFT/iSTFT, alternating projections, fast Griffin-Lim)
category: speech
tags: [griffin-lim, phase-reconstruction, stft, istft, vocoder, spectrogram-inversion, text-to-speech, audio-synthesis, numpy, from-scratch]
use_cases:
  - "turn a magnitude or mel spectrogram back into audio without a neural vocoder"
  - "listen to what a spectrogram-based model or augmentation actually produces"
  - "understand why TTS systems need a vocoder and what Griffin-Lim artefacts sound like"
  - "choose iterations and momentum for librosa.griffinlim or torchaudio GriffinLim"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/TASSP.1984.1164317
  - https://arxiv.org/abs/1306.5229
  - https://librosa.org/doc/0.11.0/generated/librosa.griffinlim.html
---

# Griffin-Lim phase reconstruction from scratch (STFT/iSTFT, alternating projections, fast Griffin-Lim)

## Summary
A magnitude spectrogram discards phase, and most audio models (TTS acoustic models, spectrogram-domain enhancement, audio augmentation) predict only magnitudes. Griffin-Lim recovers a waveform by alternating two projections: impose the target magnitude while keeping the current phase, then round-trip through iSTFT and STFT to get the nearest spectrogram that a real signal can have. Each round lowers the mismatch. This file implements STFT, iSTFT and Griffin-Lim in NumPy, measures the spectral convergence against iteration count, and shows that the momentum variant ("fast Griffin-Lim", the default in librosa) halves the error for the same compute. It also shows why the waveform never matches the original sample by sample, even when the spectrogram does.

## Key concepts
- **Consistency.** Not every complex array is the STFT of some signal: overlapping frames share samples, so neighbouring frames' phases are tied. `STFT(iSTFT(X))` projects `X` onto the nearest consistent spectrogram.
- **Magnitude projection.** `S · e^{i∠X}` keeps the phase of `X` and replaces its magnitude with the target `S`.
- **Alternating projections.** Repeating the two projections never increases the distance between `|STFT(x)|` and `S`, but it can stall in a local optimum.
- **Fast Griffin-Lim.** Extrapolate along the last step, `X_t − α/(1+α) · X_{t−1}`, before the magnitude projection (Perraudin et al. 2013). With `α ≈ 0.99` it converges much faster at no extra cost per iteration.
- **Spectral convergence.** `‖ |STFT(y)| − S ‖ / ‖S‖`, the standard measure of how well the reconstruction matches the magnitude. Waveform SNR against the original is the wrong metric, because many signals share one magnitude spectrogram.

## When to use / scenarios
- Listening to a spectrogram: debugging a TTS acoustic model before training its vocoder, checking spectrogram augmentations (SpecAugment, pitch shifts), sonifying a model's output.
- Quick baselines and offline jobs on CPU, where audio quality is secondary.
- Not for production TTS or voice agents. Griffin-Lim has an audible "phasey", metallic quality. Neural vocoders (HiFi-GAN, Vocos, BigVGAN) are far better and real-time on GPU. See [[text-to-speech]].
- From a **mel** spectrogram: first invert the mel filterbank (non-negative least squares or pseudo-inverse) to get an approximate linear magnitude, then run Griffin-Lim. The mel step loses detail Griffin-Lim cannot restore. See [[mel-spectrogram-and-mfcc-from-scratch]].

## Setup & code
NumPy only, runs in about 3 s. STFT and iSTFT are written out so the projection is visible. The test signal is a gliding harmonic tone with vibrato and a click train, which stresses both tonal and transient parts.

```python
import numpy as np

sr, n_fft, hop = 16000, 512, 128
win = np.hanning(n_fft + 1)[:-1]                      # periodic Hann: overlap-adds cleanly at hop n_fft/4


def stft(x):
    x = np.pad(x, n_fft // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    return np.fft.rfft(frames * win, axis=1)           # (frames, n_fft/2 + 1)


def istft(X, length):
    frames = np.fft.irfft(X, n=n_fft, axis=1) * win
    n = n_fft + hop * (len(X) - 1)
    y, norm = np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames):                      # weighted overlap-add (least-squares inverse)
        y[i * hop:i * hop + n_fft] += f
        norm[i * hop:i * hop + n_fft] += win ** 2
    y /= np.maximum(norm, 1e-8)
    return y[n_fft // 2:n_fft // 2 + length]


def griffin_lim(S, length, iters=100, momentum=0.0, init="random", seed=0):
    """Find a signal whose STFT magnitude matches S by alternating two projections."""
    rng = np.random.default_rng(seed)
    phase = np.exp(2j * np.pi * rng.random(S.shape)) if init == "random" else np.ones(S.shape)
    prev = 0
    history = {}
    for i in range(1, iters + 1):
        X = stft(istft(S * phase, length))              # project onto consistent spectrograms
        Xa = X - momentum / (1 + momentum) * prev if momentum else X   # fast Griffin-Lim (Perraudin 2013)
        prev = X
        phase = np.exp(1j * np.angle(Xa))               # project onto "magnitude == S", keep phase
        if i in (1, 10, 32, 100, 300):
            y = istft(S * phase, length)
            history[i] = np.linalg.norm(np.abs(stft(y)) - S) / np.linalg.norm(S)
    return istft(S * phase, length), history


# 1 s test signal: a harmonic "voice" with vibrato and a gliding pitch, plus a click train
t = np.arange(sr) / sr
f0 = 140 + 60 * t + 6 * np.sin(2 * np.pi * 5 * t)
ph = 2 * np.pi * np.cumsum(f0) / sr
x = sum(np.sin(k * ph) / k for k in range(1, 15)) * (0.5 + 0.5 * np.sin(np.pi * t))
x[::1600] += 2.0
x /= np.abs(x).max()

assert np.allclose(istft(stft(x), len(x)), x)       # STFT -> iSTFT is exact with the true phase
S = np.abs(stft(x))

for name, kw in [("zero-phase init     ", dict(init="zero")),
                 ("random init         ", dict()),
                 ("random + fast GL .99", dict(momentum=0.99))]:
    y, h = griffin_lim(S, len(x), iters=300, **kw)
    snr = 10 * np.log10(np.sum(x ** 2) / np.sum((x - y) ** 2))
    print(name, "spectral convergence at iter " +
          ", ".join(f"{i}: {v:.3f}" for i, v in h.items()) + f" | SNR vs original {snr:5.1f} dB")
```

Output (Python 3.14, NumPy 2.5):
```
zero-phase init      spectral convergence at iter 1: 0.636, 10: 0.275, 32: 0.180, 100: 0.114, 300: 0.080 | SNR vs original  -2.2 dB
random init          spectral convergence at iter 1: 0.387, 10: 0.247, 32: 0.178, 100: 0.104, 300: 0.078 | SNR vs original  -4.1 dB
random + fast GL .99 spectral convergence at iter 1: 0.387, 10: 0.193, 32: 0.093, 100: 0.054, 300: 0.038 | SNR vs original  -4.0 dB
```

How to read it:
- The `assert` passes: with the true phase, iSTFT inverts STFT exactly. Everything Griffin-Lim gets wrong comes from the missing phase.
- Plain Griffin-Lim reaches a spectral convergence of 0.18 at 32 iterations and 0.08 at 300. The gains slow down sharply.
- Fast Griffin-Lim reaches 0.093 at 32 iterations, about what plain Griffin-Lim needs 100 or more for, and ends at half the error after 300. The momentum is free: same work per iteration.
- The initial phase matters less than the method. Zero phase starts worse and ends level with random phase.
- **SNR vs the original is negative** in every row, though the spectrograms match well. Griffin-Lim finds *a* signal with this magnitude, not *the* signal: a global sign flip, a time-varying phase drift, or shifted harmonics all leave `|STFT|` nearly unchanged. Judge it by spectral convergence and by listening, never by sample-wise error.

## Choosing / trade-offs
- **Iterations.** librosa defaults to `n_iter=32` with `momentum=0.99`. Raise iterations for offline jobs; past a few hundred, gains are small and the remaining artefacts are a local optimum, not a lack of iterations.
- **Overlap.** Spectral convergence is not comparable across STFT settings. Rerunning the script with hop `n_fft/2` gives a *lower* error (0.085 vs 0.093 at 32 fast iterations), because fewer overlapping frames means fewer constraints for the phase to satisfy, not because the audio is better. Pick the hop for the analysis you need (`n_fft/4` is the usual choice), then compare methods at that fixed hop, by ear as well as by the metric.
- **Initialisation.** Random phase is the common default. A better start (phase from a previous frame, or phase-gradient heuristics such as PGHI) reduces the iterations needed and some artefacts.
- **Griffin-Lim vs neural vocoders.** Griffin-Lim needs no training and works for any signal, but sounds metallic. A neural vocoder trained on the target domain (speech, music) sounds natural but must match the exact mel settings it was trained with. Some vocoders (Vocos) predict STFT magnitude and phase directly and run an iSTFT.

## Gotchas
- STFT settings must be identical in analysis and synthesis: `n_fft`, hop, window, centring/padding and normalisation. A mismatch gives a constant-gain error or a time shift that no number of iterations fixes.
- The window and hop must overlap-add correctly (the COLA condition, or the window-squared normalisation used in `istft` above). A periodic Hann window at hop `n_fft/4` is a safe default.
- Spectrograms stored as dB or log-magnitude must be converted back to linear amplitude first (`10^(dB/20)`). Feeding power instead of magnitude (or the reverse) squares or square-roots the spectrum and distorts the sound.
- Clip or normalise the output: the reconstruction can exceed `[-1, 1]` even if the original did not.
- Transients (clicks, plosives) smear across the window length. Shorter windows keep them sharper at the cost of pitch resolution.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the forward transform whose output Griffin-Lim inverts.
- [[text-to-speech]] - neural vocoders and TTS systems that replace Griffin-Lim in production.
- [[speech-to-text]] - models that consume the same spectrogram features.
- [[voice-agents]] - where vocoder latency and quality matter end to end.

## References
- Griffin & Lim (1984), "Signal estimation from modified short-time Fourier transform", IEEE TASSP 32(2): https://doi.org/10.1109/TASSP.1984.1164317
- Perraudin, Balazs & Søndergaard (2013), "A fast Griffin-Lim algorithm": https://arxiv.org/abs/1306.5229
- librosa, `librosa.griffinlim` (0.11.0): https://librosa.org/doc/0.11.0/generated/librosa.griffinlim.html
