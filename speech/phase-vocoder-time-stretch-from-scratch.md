---
title: Phase vocoder time-stretch and pitch-shift from scratch (STFT, phase propagation, overlap-add)
category: speech
tags: [phase-vocoder, time-stretch, pitch-shift, stft, istft, overlap-add, audio-augmentation, librosa, numpy, from-scratch]
use_cases:
  - "speed audio up or slow it down without changing the pitch"
  - "add speed/pitch perturbation augmentation to a speech or audio training set"
  - "understand why naive frame dropping makes audio sound phasey or robotic"
status: draft
last_verified: 2026-10-06
sources:
  - https://doi.org/10.1002/j.1538-7305.1966.tb01706.x
  - https://doi.org/10.1109/89.759041
  - https://librosa.org/doc/main/generated/librosa.effects.time_stretch.html
---

# Phase vocoder time-stretch and pitch-shift from scratch (STFT, phase propagation, overlap-add)

## Summary
A phase vocoder changes the duration of audio without changing its pitch. It works in the STFT domain: step through the analysis frames at a different rate than they are written out, interpolate magnitudes, and rebuild each output frame's phase from the true frequency measured in each bin. If the frames keep their original phases they interfere when overlap-added, which causes the "phasey" beating artefact. Pitch shifting is a time-stretch followed by resampling. This file implements STFT/ISTFT, the phase vocoder, a naive no-phase-fix version, and pitch shift, and measures duration, pitch and artefacts on a test tone.

## Key concepts
- **Expected phase advance.** A sinusoid exactly at bin `k`'s centre frequency advances its phase by `omega_k = 2*pi*k*hop/n_fft` per hop.
- **Instantaneous frequency.** The measured phase difference between frames, minus `omega_k`, wrapped to `[-pi, pi]`, gives the deviation from the bin centre. `omega_k + deviation` is the true per-hop phase advance (Flanagan & Golden 1966).
- **Phase propagation.** The output phase accumulates `omega_k + deviation` once per output frame. Output frames then line up with each other wherever the analysis frames came from.
- **Overlap-add (OLA).** Inverse-FFT each frame, multiply by the synthesis window, add at hop spacing, and divide by the summed squared window. The STFT then inverts exactly.
- **Rate.** `rate > 1` speeds up (shorter), `rate < 1` slows down. Pitch shift by `r`: stretch by `1/r`, then resample by `r`. The duration comes back to the original and every frequency is multiplied by `r`.

## When to use / scenarios
- Speed perturbation for ASR training: 0.9x/1.0x/1.1x copies are a standard Kaldi-era augmentation. Note that Kaldi-style speed perturbation *resamples*, changing pitch too. Use a phase vocoder when you want tempo only ([[speech-to-text]]).
- Playback speed controls (podcasts, lectures, language learning) and aligning audio lengths for dubbing.
- Pitch shift for data augmentation, simple voice effects, or matching key in music.
- Not for: large speech changes beyond about 0.5x–2x, where the classic vocoder smears transients and sounds reverberant. For speech prefer WSOLA/PSOLA (time-domain), or regenerate the audio with a TTS/voice-conversion model ([[text-to-speech]]).

## Setup & code
NumPy only.

```python
import numpy as np


def stft(x, n_fft=1024, hop=256):
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * win
    return np.fft.rfft(frames, axis=1)


def istft(X, n_fft=1024, hop=256):
    win = np.hanning(n_fft)
    frames = np.fft.irfft(X, n=n_fft, axis=1) * win
    out = np.zeros(hop * (len(frames) - 1) + n_fft)
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):                        # weighted overlap-add
        out[i * hop:i * hop + n_fft] += f
        norm[i * hop:i * hop + n_fft] += win ** 2
    return out / np.maximum(norm, 1e-8)


def time_stretch(x, rate, n_fft=1024, hop=256):
    """Phase vocoder: rate > 1 = faster/shorter, rate < 1 = slower/longer. Pitch unchanged."""
    X = stft(x, n_fft, hop)
    omega = 2 * np.pi * hop * np.arange(X.shape[1]) / n_fft   # expected phase advance per hop
    steps = np.arange(0, len(X) - 1, rate)
    phase = np.angle(X[0])
    out = np.empty((len(steps), X.shape[1]), complex)
    for k, t in enumerate(steps):
        i, a = int(t), t - int(t)
        mag = (1 - a) * np.abs(X[i]) + a * np.abs(X[i + 1])  # interpolate magnitude
        out[k] = mag * np.exp(1j * phase)
        dphi = np.angle(X[i + 1]) - np.angle(X[i]) - omega
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))       # wrap to [-pi, pi]
        phase += omega + dphi                                  # true-frequency phase advance
    return istft(out, n_fft, hop)


def naive_stretch(x, rate, n_fft=1024, hop=256):
    """Same frame selection but keep each frame's original phase: the classic 'phasey' artefact."""
    X = stft(x, n_fft, hop)
    return istft(X[np.arange(0, len(X) - 1, rate).astype(int)], n_fft, hop)


def resample(x, rate):
    return np.interp(np.arange(0, len(x) - 1, rate), np.arange(len(x)), x)


def pitch(x, sr):
    """Dominant frequency via a zero-padded FFT peak with parabolic interpolation."""
    seg = x[len(x) // 4: 3 * len(x) // 4] * np.hanning(len(x) // 2)
    S = np.abs(np.fft.rfft(seg, 8 * len(seg)))
    k = np.argmax(S)
    k = k + 0.5 * (S[k - 1] - S[k + 1]) / (S[k - 1] - 2 * S[k] + S[k + 1])
    return k * sr / (8 * len(seg))


def wobble(y, sr):
    """Amplitude-envelope ripple (std/mean of 10 ms RMS) in the steady middle. A clean tone is ~0;
    phase mismatches between frames cancel partially and show up as beating."""
    mid = y[len(y) // 4: 3 * len(y) // 4]
    blk = sr // 100
    rms = np.sqrt((mid[:len(mid) // blk * blk].reshape(-1, blk) ** 2).mean(axis=1))
    return rms.std() / rms.mean()


sr = 16000
t = np.arange(2 * sr) / sr
x = 0.5 * np.sin(2 * np.pi * 330 * t)                     # 330 Hz: 5.28 cycles per hop, not a whole number

print(f"input: {len(x) / sr:.2f}s, {pitch(x, sr):.1f} Hz, envelope ripple {wobble(x, sr):.3f}")
for rate in [0.5, 1.25, 2.0]:
    y = time_stretch(x, rate)
    print(f"phase vocoder rate={rate}: {len(y) / sr:.2f}s, {pitch(y, sr):.1f} Hz, ripple {wobble(y, sr):.3f}")
y = naive_stretch(x, 0.5)
print(f"no phase fix rate=0.5:   {len(y) / sr:.2f}s, ripple {wobble(y, sr):.3f}  <- frames interfere")
y = resample(x, 1.25)
print(f"plain resample 1.25x:    {len(y) / sr:.2f}s, {pitch(y, sr):.1f} Hz  <- pitch moves with speed")

# Pitch shift = stretch by 1/r, then resample by r: same length, pitch x r
r = 2 ** (3 / 12)                                         # +3 semitones
y = resample(time_stretch(x, 1 / r), r)
print(f"pitch shift +3 st:       {len(y) / sr:.2f}s, {pitch(y, sr):.1f} Hz (target {330 * r:.1f})")

rt = istft(stft(x))                                       # STFT -> ISTFT round trip is exact away from the edges
assert np.allclose(rt[1024:-1024], x[1024:len(rt) - 1024], atol=1e-6)
```

Output (Python 3.14, NumPy 2.5):
```
input: 2.00s, 330.0 Hz, envelope ripple 0.016
phase vocoder rate=0.5: 3.92s, 330.0 Hz, ripple 0.016
phase vocoder rate=1.25: 1.60s, 330.0 Hz, ripple 0.016
phase vocoder rate=2.0: 1.02s, 330.0 Hz, ripple 0.016
no phase fix rate=0.5:   3.92s, ripple 0.121  <- frames interfere
plain resample 1.25x:    1.60s, 412.5 Hz  <- pitch moves with speed
pitch shift +3 st:       1.98s, 392.4 Hz (target 392.4)
```

How to read it:
- The phase vocoder changes duration by `1/rate` (2 s becomes 3.92 s, 1.60 s, 1.02 s; the small shortfall is the last partial frame), keeps 330 Hz exactly, and leaves the envelope as flat as the input. The 0.016 baseline is the 10 ms blocks not aligning with the tone's period.
- Repeating frames without fixing the phase gives the same length but 7.5x more envelope ripple. Each repeated frame starts 0.28 cycles out of step with its neighbour, and the overlapping copies partly cancel. On real audio you hear this as phasiness or "underwater" sound.
- The test tone is chosen on purpose. At 440 Hz with hop 256 at 16 kHz, a hop is 7.04 cycles, almost a whole number, and the naive version sounds nearly fine. Test with several frequencies, or with real speech.
- Plain resampling changes speed and pitch together (1.25x faster = 412.5 Hz). The pitch shift lands exactly on the +3-semitone target (392.4 Hz) at the original length.

## Choosing / trade-offs
- **Library.** `librosa.effects.time_stretch` / `pitch_shift` implement this same algorithm. `torchaudio.transforms.TimeStretch` works on complex spectrograms inside a training graph. For production-quality audio, the Rubber Band library (`pyrubberband`) handles transients and formants far better.
- **Phase vocoder vs WSOLA/PSOLA.** The phase vocoder is best for music and polyphonic sound. Time-domain WSOLA/PSOLA usually sounds more natural for single-speaker speech and is cheaper.
- **Speed perturbation for ASR.** Resampling (pitch changes with speed) is the established recipe and adds speaker variety. Tempo-only perturbation via a phase vocoder adds rhythm variety without changing the voice. Both are cheap. Measure with [[word-error-rate-from-scratch]].
- **n_fft / hop.** Larger `n_fft` resolves close harmonics (good for music) but smears transients. A hop of n_fft/4 is the usual overlap. Speech at 16 kHz is typically fine with 512–1024.

## Gotchas
- Transients smear: drums and plosives lose their attack because their energy is spread across frames with propagated phase. Transient detection with a phase reset fixes this (Rubber Band does it).
- "Phasiness" remains even with phase propagation, because neighbouring bins of one partial drift apart. Phase locking (Laroche & Dolson 1999) ties each bin's phase to its spectral peak.
- Pitch shifting moves formants too, so large shifts sound like chipmunks or giants. Formant-preserving shift needs envelope separation ([[lpc-formant-estimation-from-scratch]]).
- The `istft` normalisation must match the analysis and synthesis windows. A wrong OLA normaliser gives amplitude ripple at the hop rate.
- Resampling with `np.interp` is linear and aliases on downsampling. Use `scipy.signal.resample_poly` or `soxr` for real audio.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the same STFT front end, used for features.
- [[griffin-lim-from-scratch]] - reconstructing phase from magnitude only, where the vocoder propagates measured phase.
- [[yin-pitch-detection-from-scratch]] - measuring the pitch that time-stretch keeps and pitch-shift moves.
- [[speech-to-text]] - where speed and tempo perturbation are used as augmentation.
- [[word-error-rate-from-scratch]] - checking that an augmentation actually lowers ASR error.

## References
- Flanagan & Golden (1966), "Phase Vocoder", Bell System Technical Journal: https://doi.org/10.1002/j.1538-7305.1966.tb01706.x
- Laroche & Dolson (1999), "Improved phase vocoder time-scale modification of audio", IEEE TSAP: https://doi.org/10.1109/89.759041
- librosa `effects.time_stretch`: https://librosa.org/doc/main/generated/librosa.effects.time_stretch.html
