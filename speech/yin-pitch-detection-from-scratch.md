---
title: YIN pitch (f0) detection from scratch (difference function, CMND, absolute threshold, vs autocorrelation)
category: speech
tags: [yin, pyin, pitch-detection, f0-estimation, fundamental-frequency, autocorrelation, octave-errors, voicing-detection, prosody, librosa, numpy, from-scratch]
use_cases:
  - "estimate the pitch (fundamental frequency) contour of speech or singing"
  - "understand why autocorrelation pitch trackers make octave errors and how YIN avoids them"
  - "choose fmin, fmax, frame length and threshold for librosa.yin or librosa.pyin"
  - "detect voiced vs unvoiced frames for prosody, intonation or singing-practice features"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1121/1.1458024
  - https://doi.org/10.1109/ICASSP.2014.6853678
  - https://librosa.org/doc/0.11.0/generated/librosa.yin.html
---

# YIN pitch (f0) detection from scratch (difference function, CMND, absolute threshold, vs autocorrelation)

## Summary
Pitch (the fundamental frequency, f0) is the rate at which a voiced sound repeats: vocal folds opening and closing, or a string vibrating. A pitch tracker finds that period in each short frame. The classic method, picking the largest autocorrelation peak, makes gross errors (octave jumps) whenever loudness changes within the frame, which is constantly in speech. YIN (de Cheveigné & Kawahara 2002) fixes this with a difference function, a normalisation that removes the bias towards zero lag, and a rule that takes the *first* good dip instead of the deepest. This file implements both trackers in NumPy and measures gross pitch error on synthetic voices with known f0, with and without syllable-like loudness swells and a weak fundamental, at three noise levels.

## Key concepts
- **Difference function.** `d(τ) = Σ (x_j − x_{j+τ})²`. It is 0 at the period of a perfectly periodic signal. Unlike the autocorrelation peak, it compares the two windows sample by sample, so a change in loudness raises it instead of creating a false maximum.
- **Cumulative mean normalised difference (CMND).** `d'(τ) = d(τ) / ((1/τ) Σ_{j≤τ} d(j))`, with `d'(0) = 1`. It removes the dip at τ = 0 and makes the values comparable across signals: about 1 for noise, near 0 for a clean period.
- **Absolute threshold.** Take the first τ where `d'` dips below a threshold (0.1 to 0.15), then walk to the bottom of that dip. A period and its multiples (2T, 3T) all dip; the first is the period. Taking the global minimum instead invites "too low" octave errors.
- **Parabolic interpolation.** Fit a parabola through the dip and its neighbours for a sub-sample period. Without it, at 16 kHz a 300 Hz pitch (lag about 53 samples) is quantised in steps of about 30 cents.
- **Aperiodicity.** `d'` at the chosen dip doubles as a voicing measure: low for voiced frames, near 1 for noise and unvoiced consonants.
- **Gross pitch error (GPE).** The share of voiced frames off by more than a threshold (here 50 cents, half a semitone; 20% is also common). Separate from the fine error on the frames that are right.

## When to use / scenarios
- Prosody and intonation features: question vs statement detection, emotion and stress features, speaker characterisation, language-learning tone feedback (Mandarin, Vietnamese).
- Singing and music: tuning meters, karaoke scoring, melody transcription of monophonic audio, singing-practice apps.
- Speech health and clinical voice analysis: f0 range, jitter, and voicing for voice disorder screening (use validated tools like Praat for clinical numbers).
- Data preparation for TTS and voice conversion: many acoustic models condition on an f0 contour.
- Not for polyphonic music (chords, mixtures): YIN assumes one periodic source. Use multi-pitch models (e.g. Basic Pitch) or source separation first.
- For noisy, real-world audio where accuracy matters, use pYIN (`librosa.pyin`, adds an HMM over pitch candidates and voicing) or a neural tracker such as CREPE. YIN remains the fast, dependency-free baseline.

## Setup & code
NumPy only, runs in about 3 s. A 2-second harmonic "voice" glides from 110 to 311 Hz with 5 Hz vibrato and a strong second harmonic. Two variations make it harder: loudness swells at syllable rate (4 Hz), and a fundamental at 15% of its normal level. White noise is added at 40, 10 and 0 dB SNR (computed over the whole signal). Three trackers are compared on 64 ms frames: the autocorrelation peak, YIN's CMND with the global minimum (no threshold step), and full YIN.

```python
import numpy as np

sr = 16000
frame, hop = 1024, 256                     # 64 ms analysis frame, longest lag searched below = sr / fmin
fmin, fmax = 60, 500
tau_min, tau_max = int(sr / fmax), int(sr / fmin)


def acf_pitch(x):
    """Baseline: lag of the largest autocorrelation peak in the allowed range."""
    w = len(x) - tau_max
    r = np.array([x[:w] @ x[t:t + w] for t in range(tau_max + 1)])
    return sr / (tau_min + np.argmax(r[tau_min:]))


def yin(x, threshold=0.1):
    """de Cheveigne & Kawahara (2002), steps 2-5. Returns (f0 in Hz, aperiodicity)."""
    w = len(x) - tau_max
    # step 2: difference function d(tau) = sum (x_j - x_{j+tau})^2
    d = np.array([np.sum((x[:w] - x[t:t + w]) ** 2) for t in range(tau_max + 1)])
    # step 3: cumulative mean normalised difference, d'(0) = 1
    cmnd = np.ones_like(d)
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(np.cumsum(d[1:]), 1e-12)
    # step 4: absolute threshold -> first dip below it (picks the period, not a multiple of it)
    below = np.nonzero(cmnd[tau_min:] < threshold)[0]
    if len(below):
        t = tau_min + below[0]
        while t + 1 <= tau_max and cmnd[t + 1] < cmnd[t]:
            t += 1                                         # walk down to the bottom of that dip
    else:
        t = tau_min + np.argmin(cmnd[tau_min:])            # no dip: global minimum, flagged by aperiodicity
    # step 5: parabolic interpolation for sub-sample period
    if tau_min < t < tau_max:
        a, b, c = cmnd[t - 1], cmnd[t], cmnd[t + 1]
        t = t + 0.5 * (a - c) / (a - 2 * b + c)
    return sr / t, cmnd[int(round(t))]


def voice(f0_hz, dur=2.0, weak_fundamental=False, syllables=False, rng=None):
    """Harmonic source with a gliding, vibrating f0; optional weak fundamental and syllable-rate swells."""
    t = np.arange(int(dur * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0_hz(t)) / sr
    amps = [0.15 if (k == 1 and weak_fundamental) else 1.0 / k ** 0.7 for k in range(1, 20)]
    amps[1] *= 2.2                                          # strong 2nd harmonic
    x = sum(a * np.sin(k * phase + rng.uniform(0, 2 * np.pi)) for k, a in enumerate(amps, 1))
    if syllables:
        x *= 0.05 + np.sin(np.pi * 4 * t) ** 2              # 4 Hz loudness swells, like syllables
    return x / np.abs(x).max(), f0_hz


rng = np.random.default_rng(0)
contour = lambda t: 110 * 2 ** (t * 0.75) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))   # 110 -> 311 Hz over 2 s


def evaluate(x, f0_fn, method):
    errs, gross = [], 0
    starts = range(0, len(x) - frame, hop)
    for s in starts:
        est = method(x[s:s + frame])
        est = est[0] if isinstance(est, tuple) else est
        ref = f0_fn((s + frame / 2) / sr)
        cents = 1200 * np.log2(est / ref)
        if abs(cents) > 50:                                # > half a semitone: gross pitch error (octave etc.)
            gross += 1
        else:
            errs.append(abs(cents))
    return gross / len(starts), np.mean(errs)


methods = {"ACF peak": acf_pitch,
           "CMND global min": lambda x: yin(x, threshold=-1),   # YIN without step 4
           "YIN": yin}
print("signal               SNR    | " + " | ".join(f"{m}: GPE  fine(c)" for m in methods))
for label, kw in (("steady loudness", {}), ("syllable swells", dict(syllables=True)),
                  ("swells + weak f0", dict(syllables=True, weak_fundamental=True))):
    clean, f0_fn = voice(contour, rng=rng, **kw)
    for snr in (40, 10, 0):
        noise = rng.standard_normal(len(clean))
        noise *= np.sqrt(np.mean(clean ** 2) / np.mean(noise ** 2) / 10 ** (snr / 10))
        x = clean + noise
        cells = [evaluate(x, f0_fn, fn) for fn in methods.values()]
        print(f"{label:18s} {snr:4d} dB | " + " | ".join(
            f"{g:{len(m) + 2}.1%} {e:7.1f}" for m, (g, e) in zip(methods, cells)))

# Voicing: aperiodicity (CMND at the chosen dip) separates voiced frames from noise
clean, _ = voice(contour, rng=rng)
ap_voiced = np.median([yin(clean[s:s + frame])[1] for s in range(0, len(clean) - frame, hop)])
noise_only = rng.standard_normal(sr)
ap_noise = np.median([yin(noise_only[s:s + frame])[1] for s in range(0, sr - frame, hop)])
print(f"median aperiodicity: voiced {ap_voiced:.3f}, white noise {ap_noise:.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
signal               SNR    | ACF peak: GPE  fine(c) | CMND global min: GPE  fine(c) | YIN: GPE  fine(c)
steady loudness      40 dB |       2.5%     5.9 |              2.5%     4.6 |  0.0%     4.5
steady loudness      10 dB |       4.1%     6.3 |              1.7%     4.3 |  1.7%     4.3
steady loudness       0 dB |      19.0%    11.3 |             16.5%     7.7 | 16.5%     7.7
syllable swells      40 dB |      31.4%    10.0 |              0.8%     7.2 |  0.0%     7.1
syllable swells      10 dB |      30.6%    10.1 |              6.6%     8.3 |  5.8%     8.3
syllable swells       0 dB |      45.5%    11.3 |             34.7%     8.6 | 34.7%     8.6
swells + weak f0     40 dB |      28.1%     9.8 |              0.8%     7.1 |  0.0%     7.1
swells + weak f0     10 dB |      28.9%    10.3 |              6.6%     8.3 |  5.8%     8.2
swells + weak f0      0 dB |      47.1%    10.9 |             40.5%     9.3 | 40.5%     9.3
median aperiodicity: voiced 0.024, white noise 0.900
```

How to read it:
- **Loudness changes break autocorrelation.** With syllable swells and almost no noise, the autocorrelation peak is wrong on 31% of frames; YIN on 0%. While the sound gets louder, the later window `x[t:t+w]` has more energy at longer lags, so a multiple of the period outscores the period itself. The difference function does not reward that.
- **The threshold step matters less than the difference function here.** CMND with the global minimum already removes almost all of the autocorrelation's errors; the "first dip below 0.1" rule removes the remaining 0.8–2.5% of octave-too-low picks.
- **A weak fundamental is not a problem for either.** Periodicity analysis finds the repetition period even when the first harmonic is at 15%, as the ear does (the "missing fundamental"). Spectral peak-picking would report the second harmonic.
- **Fine error of 4–9 cents** on correct frames comes mostly from the pitch moving within a 64 ms frame (vibrato plus the glide); a shorter frame lowers it, at the cost of the lowest pitch it can see.
- **0 dB rows are dominated by the quiet frames.** SNR is measured over the whole signal, so in the troughs of the swells the local SNR is far below 0 dB and no tracker can find a pitch. Those frames are where a voicing decision should say "unvoiced": aperiodicity is 0.02 on clean voiced frames and 0.90 on pure noise.

## Choosing / trade-offs
- **fmin, fmax and frame length.** The frame must hold at least two periods of the lowest pitch: `frame ≥ 2·sr/fmin`. Speech: about 60–400 Hz for adults, higher for children; singing: set from the voice type. A wider range costs time and adds chances for octave errors, so set it as tight as the material allows.
- **Threshold.** 0.1–0.15 is typical. Lower makes the first-dip rule stricter (falls back to the global minimum more often); higher accepts shallow, noisy dips that may be harmonics.
- **YIN vs pYIN vs neural.** YIN decides each frame alone and is fast. pYIN keeps several candidates per frame and picks a smooth path with an HMM, which removes isolated octave jumps and gives voicing probabilities; it is several times slower. CREPE-style neural trackers are the most robust in noise and reverberation and need a GPU for real time.
- **Post-processing.** A median filter over 3–5 frames removes isolated octave errors cheaply. Report f0 in cents or semitones relative to a reference, not Hz, when comparing speakers.

## Gotchas
- The O(N·τ_max) loop above is fine for a demo but slow for long files. Compute the difference function with FFT autocorrelations (`d(τ) = r_x(0) + r_{x_τ}(0) − 2r(τ)`), as librosa does.
- `librosa.yin` returns an f0 for every frame, including silence and noise; it has no voicing decision. Use `librosa.pyin` (returns `voiced_flag` and probabilities) or threshold the aperiodicity yourself.
- Very high-pitched voices with a small `tau_min` hit integer-lag quantisation; never skip the parabolic interpolation.
- Clipping, heavy compression and background music add periodic structure at other periods. Check a few contours by ear and on a spectrogram before trusting statistics.
- Averages of f0 in Hz are biased towards high voices; average in a log scale (cents) instead.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - spectral features that blur pitch, the complement to an f0 track.
- [[griffin-lim-from-scratch]] - STFT tools; f0-conditioned vocoders are the production follow-on.
- [[dynamic-time-warping-from-scratch]] - aligning a sung or spoken pitch contour to a reference melody.
- [[text-to-speech]] - acoustic models that predict and condition on f0.
- [[speech-to-text]] - where pitch is mostly discarded, except for tonal languages.

## References
- de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency estimator for speech and music", JASA 111(4): https://doi.org/10.1121/1.1458024
- Mauch & Dixon (2014), "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", ICASSP: https://doi.org/10.1109/ICASSP.2014.6853678
- librosa, `librosa.yin` (0.11.0): https://librosa.org/doc/0.11.0/generated/librosa.yin.html
