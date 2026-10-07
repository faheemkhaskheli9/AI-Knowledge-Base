---
title: Spectral subtraction and Wiener denoising from scratch (STFT, noise estimate, over-subtraction, musical noise)
category: speech
tags: [speech-enhancement, noise-reduction, denoising, spectral-subtraction, wiener-filter, decision-directed, musical-noise, stft, numpy, from-scratch]
use_cases:
  - "remove stationary background noise (fan, hum, hiss, road noise) from speech recordings"
  - "clean audio before speech-to-text or a voice agent without a neural model or GPU"
  - "understand musical noise and how over-subtraction, spectral floors and Wiener gains reduce it"
  - "decide between classical spectral denoising and a learned speech enhancement model"
status: stable
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1109/TASSP.1979.1163209
  - https://doi.org/10.1109/TASSP.1984.1164453
---

# Spectral subtraction and Wiener denoising from scratch (STFT, noise estimate, over-subtraction, musical noise)

## Summary
Spectral subtraction removes additive, slowly varying noise from speech. It estimates the noise power spectrum from a stretch with no speech, subtracts that estimate from every short-time spectrum, and resynthesises the audio with the noisy phase (Boll 1979). It needs no training data and runs in real time on any CPU. Plain subtraction leaves "musical noise": random isolated spectral peaks that survive the subtraction and sound like twinkling tones. This file implements the STFT round trip, power subtraction, over-subtraction with a spectral floor, and a decision-directed Wiener gain (Ephraim & Malah 1984) on synthetic voiced speech with white noise, and measures output SNR, residual noise in silences and a musical-noise proxy.

## Key concepts
- **STFT round trip.** Frame the signal with a window, FFT each frame, modify, inverse FFT, overlap-add. With a periodic Hann window at hop N/4 and weighted overlap-add, an unmodified spectrum reconstructs the signal to machine precision.
- **Additive noise in power.** `|Y|² ≈ |S|² + |N|²` on average, so `|S|² ≈ |Y|² − E|N|²`. The cross term averages out but not in any single bin, which is where musical noise comes from.
- **Gain formulation.** Every method is a real gain `G(t, f)` in [0, 1] applied to the noisy spectrum `Y`. Power subtraction: `G² = max(1 − α·N̂/|Y|², β)`.
- **Over-subtraction (α > 1) and floor (β).** Subtract more than the estimate to kill noise peaks, and never go below a floor so the residual is a smooth low-level noise instead of silence with isolated tones (Berouti et al. 1979).
- **Wiener gain, decision-directed.** `G = ξ/(1+ξ)` with the a priori SNR `ξ` smoothed over time: `ξ_t = 0.98·(previous clean estimate)/N̂ + 0.02·max(|Y|²/N̂ − 1, 0)`. The smoothing is what suppresses musical noise.
- **Noise estimate.** From leading silence (as here), from a voice activity detector, or by tracking spectral minima over a second or two (minimum statistics), which also follows slowly changing noise.

## When to use / scenarios
- Stationary noise in recordings: fans, HVAC, computer hum, tape hiss, steady road or engine noise in call-centre and in-car audio.
- Pre-processing on low-power devices (embedded, telephony, hearing aids) where a neural model does not fit the latency or power budget.
- A quick baseline before reaching for a learned enhancer, and as a component inside classic VoIP stacks.
- Not for non-stationary noise (babble, music, keyboard clicks, a second speaker): the noise estimate cannot follow it. Use a learned speech enhancement or separation model.
- Often not needed before modern ASR. Large ASR models trained on noisy data ([[speech-to-text]]) can lose accuracy on denoised audio because of the artefacts. Measure WER with and without denoising before adding it.

## Setup & code
NumPy only, about a second. The "speech" is four voiced syllables (harmonic stacks with gliding pitch and a formant-like envelope) separated by silence, 3 s at 16 kHz. White noise is added at 10 dB and 0 dB SNR. The first 0.4 s is noise only and gives the noise estimate.

```python
import numpy as np

SR, N, HOP = 16000, 512, 128
rng = np.random.default_rng(0)


def fake_speech(sec=3.0):
    """Voiced 'syllables': harmonic stacks with gliding pitch and a formant-like envelope, separated by silence."""
    t = np.arange(int(sec * SR)) / SR
    x = np.zeros_like(t)
    for start, dur, f0 in ((0.5, 0.35, 140), (1.0, 0.45, 180), (1.7, 0.3, 120), (2.2, 0.5, 160)):
        m = (t >= start) & (t < start + dur)
        tt = t[m] - start
        f = f0 * (1 + 0.15 * np.sin(2 * np.pi * tt / dur))            # pitch glide
        phase = 2 * np.pi * np.cumsum(f) / SR
        env = np.sin(np.pi * tt / dur) ** 2
        for h in range(1, 25):
            if h * f0 < 4000:
                x[m] += env * np.exp(-((h * f0 - 600) / 900) ** 2) * np.sin(h * phase) / h ** 0.3
    return x / np.abs(x).max() * 0.5


def stft(x):
    w = np.hanning(N + 1)[:N]                    # periodic Hann: overlap-adds to a constant at hop N/4
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(x, N), N)[::HOP]
    return np.fft.rfft(frames * w, axis=1)


def istft(X, length):
    w = np.hanning(N + 1)[:N]
    frames = np.fft.irfft(X, n=N, axis=1) * w
    y = np.zeros(len(frames) * HOP + N)
    norm = np.zeros_like(y)
    for i, f in enumerate(frames):
        y[i * HOP:i * HOP + N] += f
        norm[i * HOP:i * HOP + N] += w ** 2
    return (y / np.maximum(norm, 1e-8))[N:N + length]


def denoise(noisy, noise_sec=0.4, alpha=1.0, beta=0.0, mode="power"):
    """Estimate the noise spectrum from the leading noise-only stretch, subtract it, keep the noisy phase."""
    X = stft(noisy)
    P = np.abs(X) ** 2
    noise = P[:int(noise_sec * SR / HOP)].mean(0)                    # mean noise power per frequency bin
    if mode == "power":
        G2 = np.maximum(1 - alpha * noise / np.maximum(P, 1e-12), beta)   # over-subtraction + spectral floor
    else:                                                              # decision-directed Wiener gain
        G2, prev = np.empty_like(P), np.zeros(P.shape[1])
        for i, p in enumerate(P):
            snr_prior = 0.98 * prev / noise + 0.02 * np.maximum(p / noise - 1, 0)
            g = snr_prior / (1 + snr_prior)
            G2[i] = g ** 2
            prev = g ** 2 * p
        G2 = np.maximum(G2, beta)
    return istft(np.sqrt(G2) * X, len(noisy)), G2


def snr(clean, est):
    return 10 * np.log10(np.sum(clean ** 2) / np.sum((clean - est) ** 2))


clean = fake_speech()
print(f"STFT round trip max error: {np.abs(istft(stft(clean), len(clean)) - clean).max():.1e}")
silent = np.ones(len(clean), bool)
silent[int(0.45 * SR):] = np.abs(np.convolve(clean, np.ones(400), "same"))[int(0.45 * SR):] < 1e-6
for in_snr in (10, 0):
    noise = rng.standard_normal(len(clean))
    noise *= np.sqrt(np.sum(clean ** 2) / np.sum(noise ** 2) / 10 ** (in_snr / 10))
    noisy = clean + noise
    print(f"input SNR {snr(clean, noisy):5.1f} dB")
    print("  method                         out SNR  residual in silence  isolated bins in silence")
    for name, kw in (("power subtraction a=1", dict(alpha=1)),
                     ("over-subtraction a=3, floor", dict(alpha=3, beta=0.01)),
                     ("Wiener, decision-directed", dict(mode="wiener", beta=0.01))):
        y, G2 = denoise(noisy, **kw)
        resid = 10 * np.log10(np.sum(y[silent] ** 2) / np.sum(noise[silent] ** 2))
        # musical noise: bins in noise-only frames that pass with gain > 0.5 while their neighbours are cut
        frames = silent[np.minimum(np.arange(len(G2)) * HOP, len(clean) - 1)]
        on = G2[frames] > 0.25
        lonely = on[1:-1, 1:-1] & ~on[:-2, 1:-1] & ~on[2:, 1:-1] & ~on[1:-1, :-2] & ~on[1:-1, 2:]
        print(f"  {name:30s} {snr(clean, y):6.1f}   {resid:8.1f} dB          {lonely.mean():6.2%}")
```

Output (Python 3.14, NumPy 2.5):
```
STFT round trip max error: 2.2e-16
input SNR  10.0 dB
  method                         out SNR  residual in silence  isolated bins in silence
  power subtraction a=1            14.2       -4.9 dB           1.43%
  over-subtraction a=3, floor      18.9      -13.1 dB           0.96%
  Wiener, decision-directed        19.7      -19.8 dB           0.00%
input SNR   0.0 dB
  method                         out SNR  residual in silence  isolated bins in silence
  power subtraction a=1             4.5       -4.9 dB           1.34%
  over-subtraction a=3, floor       9.9      -13.1 dB           0.80%
  Wiener, decision-directed        10.8      -19.8 dB           0.00%
```

How to read it:
- The STFT round trip is exact (2e-16), so every change in the output comes from the gain, not the framework.
- Plain power subtraction gains about 4 dB SNR but leaves the noise in silences only 4.9 dB quieter, and 1.3–1.4% of time-frequency bins in silence pass alone with their neighbours cut. That is the musical noise: scattered short tones on a quiet background, more annoying to listeners than the original hiss.
- Over-subtraction by 3 with a 1% floor gains about 9–10 dB and lowers the residual to −13 dB, but still leaves isolated bins (0.8–1.0%).
- The decision-directed Wiener gain is best on all three numbers: about 10 dB SNR gain, residual noise at the −20 dB floor, and no isolated bins. Smoothing the a priori SNR over time means one random noise peak cannot open the gain for a single frame.
- The SNR gain is about the same at 10 dB and 0 dB input. What changes is speech distortion: at 0 dB input more weak harmonics sit under the noise floor and are removed with it. SNR does not show this; listen, or measure with PESQ, STOI or WER.

## Choosing / trade-offs
- **Gain rule.** Power subtraction is the simplest; over-subtraction with a floor is the usual minimum for anything a person will listen to; a decision-directed Wiener or MMSE log-spectral amplitude gain is the classical best and costs almost nothing more.
- **Floor β.** A floor at −20 to −15 dB leaves a little smooth noise, which masks residual artefacts and sounds more natural than gaps of digital silence. Lower floors give better SNR numbers and worse listening.
- **Noise tracking.** A fixed estimate from leading silence fails as soon as the noise changes. Minimum statistics or VAD-gated updates track it; both lag noise changes by about a second or more.
- **Frame size.** 20–32 ms frames (512 samples at 16 kHz) resolve speech harmonics. Longer frames smear onsets (pre-echo); shorter ones raise the variance of each bin and so the musical noise.
- **Classical vs learned.** Neural enhancers (RNNoise-style hybrids, larger DNN models) handle non-stationary noise and produce fewer artefacts. Classical gains are deterministic, tiny and need no data. In a voice pipeline ([[voice-agents]]) try the ASR on raw audio first; add denoising only if it measurably lowers WER.

## Gotchas
- Speech in the noise estimate: if the "silence" stretch contains speech, its harmonics are subtracted from the whole file. Verify with a VAD, or use minimum statistics.
- Using a symmetric `np.hanning(N)` instead of a periodic window: the overlap-add no longer sums to a constant and every frame boundary gets a small ripple.
- Reporting SNR improvement alone. Aggressive gains raise SNR while removing consonants and weak harmonics; intelligibility (STOI) and WER can drop at the same time.
- Clipping before denoising: clipped samples spread energy across all frequencies, which no spectral gain can remove. Fix gain staging first.
- Denoising audio that goes to an ASR model trained on noisy data can increase WER. The model learned to ignore the noise but not the artefacts.
- Coloured noise (hum with harmonics, road rumble) works only if the estimate is per frequency bin, as here. A single broadband noise level under-subtracts the loud bands and over-subtracts the quiet ones.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the same STFT front end, used for features.
- [[griffin-lim-from-scratch]] - the opposite problem: rebuilding a signal from magnitudes without the phase.
- [[yin-pitch-detection-from-scratch]] - pitch tracking, which also breaks down as SNR falls.
- [[speech-to-text]] - where cleaned audio often ends up, and where to measure the effect.
- [[voice-agents]] - real-time pipelines where latency limits denoising.

## References
- Boll (1979), "Suppression of acoustic noise in speech using spectral subtraction", IEEE Trans. ASSP: https://doi.org/10.1109/TASSP.1979.1163209
- Ephraim & Malah (1984), "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator", IEEE Trans. ASSP: https://doi.org/10.1109/TASSP.1984.1164453
