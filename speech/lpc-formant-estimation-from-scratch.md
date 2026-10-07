---
title: LPC and formant estimation from scratch (source-filter model, autocorrelation, Levinson-Durbin, pole roots)
category: speech
tags: [lpc, linear-prediction, formants, levinson-durbin, source-filter-model, autocorrelation, pre-emphasis, vowels, praat, numpy, from-scratch]
use_cases:
  - "estimate vowel formants (F1, F2, F3) from a recording for phonetics, speech therapy or accent analysis"
  - "understand linear prediction, the spectral envelope model behind classic speech codecs and vocoders"
  - "choose the LPC order and pre-emphasis, and know when formant tracks are not to be trusted (high pitch, noise)"
status: stable
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1109/PROC.1975.9792
  - https://doi.org/10.1121/1.1906875
---

# LPC and formant estimation from scratch (source-filter model, autocorrelation, Levinson-Durbin, pole roots)

## Summary
In the source-filter model of speech, the vocal folds produce a buzz (a pulse train at the pitch f0) and the vocal tract filters it. The tract's resonances, the formants, decide which vowel you hear: F1 tracks tongue height and F2 tongue backness. Linear predictive coding (LPC) fits an all-pole filter to a short frame by predicting each sample from the previous `p` samples. The filter's poles land on the formants, so their angles give formant frequencies and their radii give bandwidths. This file synthesises three vowels with known formants, solves the LPC equations with the Levinson-Durbin recursion, reads formants from the pole roots, and measures how LPC order, pitch and noise change the error.

## Key concepts
- **Source-filter model.** `speech = source * vocal tract * lip radiation`. The source spectrum falls about 12 dB per octave and lip radiation adds about +6 dB per octave. The tract is close to all-pole for vowels, so an all-pole model `1 / A(z)` fits it well.
- **Linear prediction.** Predict `x[n]` as `-sum(a_k * x[n-k])` for `k = 1..p`. Minimising the squared error over a windowed frame gives the normal equations `R a = -r`, where `R` is the Toeplitz matrix of autocorrelations.
- **Levinson-Durbin.** Solves the Toeplitz system in O(p²) instead of O(p³). It also gives reflection coefficients, which stay in (-1, 1) when the filter is stable; that is what codecs quantise.
- **Pre-emphasis.** `y[n] = x[n] - 0.97 x[n-1]` boosts high frequencies by about 6 dB per octave, so the model spends its poles on formants rather than on the overall spectral tilt.
- **Poles to formants.** Each complex root pair `r * exp(±j*theta)` is a resonance at `f = theta * sr / (2 pi)` with bandwidth `bw = -ln(r) * sr / pi`. Formants have narrow bandwidths (roughly 50–300 Hz). Broad poles model tilt and are discarded.
- **Order rule of thumb.** About one pole pair per kHz of bandwidth plus 2–4 extra poles: `p ≈ sr / 1000 + 2`. At 10 kHz that is 12.

## When to use / scenarios
- Phonetics and sociolinguistics: vowel charts (F1 vs F2) per speaker or accent; this is what Praat's formant tracker does.
- Speech therapy and pronunciation feedback: show a learner where their vowel sits relative to a target ([[speech-to-text]] tells you what was said, formants tell you how).
- Speech codecs and vocoders: LPC is the core of LPC-10, CELP-family codecs and LPCNet. The coefficients describe the spectral envelope compactly.
- Features: LPC coefficients and LPC cepstra are classic ASR and speaker features, now mostly replaced by log-mel ([[mel-spectrogram-and-mfcc-from-scratch]]).
- Voice conversion and synthesis experiments: change formants while keeping pitch, or the reverse ([[text-to-speech]]).

## Setup & code
NumPy only, a few seconds (the IIR filter loop is plain Python). Vowels /a/, /i/, /u/ use the Peterson & Barney (1952) average male formants. A jittered pulse train passes through a source-tilt filter and three formant resonators at 10 kHz. One 25.6 ms frame from the middle of each vowel is analysed.

```python
import numpy as np

sr = 10000                                    # 10 kHz: formants F1-F3 of adult speech sit below 3.5 kHz
rng = np.random.default_rng(0)
VOWELS = {"a": (730, 1090, 2440), "i": (270, 2290, 3010), "u": (300, 870, 2240)}   # Peterson & Barney male means
BW = (60, 90, 120)                             # formant bandwidths in Hz


def resonator(x, f, bw):
    """Two-pole IIR filter: a pole pair at radius exp(-pi*bw/sr), angle 2*pi*f/sr."""
    r, th = np.exp(-np.pi * bw / sr), 2 * np.pi * f / sr
    a1, a2 = -2 * r * np.cos(th), r * r
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = x[n] - a1 * y[n - 1] - a2 * y[n - 2] if n >= 2 else x[n]
    return y


def vowel(formants, f0=120, dur=0.3, snr_db=None):
    n = int(dur * sr)
    t0 = sr / f0 * (1 + 0.01 * rng.standard_normal(int(dur * f0) + 2)).cumsum()   # slight jitter
    src = np.zeros(n)
    src[t0[t0 < n].astype(int)] = 1.0                                               # glottal pulse train
    y = resonator(src, 0, 250)                                     # double pole near DC: -12 dB/oct source tilt
    for f, bw in zip(formants, BW):
        y = resonator(y, f, bw)
    if snr_db is not None:
        y = y + rng.standard_normal(n) * np.sqrt((y ** 2).mean() / 10 ** (snr_db / 10))
    return y


def levinson(r, p):
    """Solve the Toeplitz normal equations for LPC coefficients a (A(z) = 1 + a1 z^-1 + ...) in O(p^2)."""
    a, err = np.zeros(p + 1), r[0]
    a[0] = 1.0
    for i in range(1, p + 1):
        k = -(r[i] + a[1:i] @ r[i - 1:0:-1]) / err          # reflection coefficient
        a[1:i + 1] = a[1:i + 1] + k * a[i - 1::-1][:i]
        err *= 1 - k * k
    return a, err


def lpc_formants(x, order=12, preemph=0.97, n_formants=3):
    x = np.append(x[0], x[1:] - preemph * x[:-1])                     # pre-emphasis flattens the source tilt
    mid = len(x) // 2
    w = x[mid - 128:mid + 128] * np.hamming(256)                       # one 25.6 ms frame
    r = np.correlate(w, w, "full")[len(w) - 1:len(w) + order]
    a, _ = levinson(r, order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]                                  # one of each conjugate pair
    f = np.angle(roots) * sr / (2 * np.pi)
    bw = -np.log(np.abs(roots)) * sr / np.pi
    keep = (f > 90) & (bw < 600)                                       # drop DC/tilt poles and very broad poles
    return np.sort(f[keep])[:n_formants]


# Levinson-Durbin vs a direct solve of the same normal equations
x = vowel(VOWELS["a"])
w = x[1000:1256] * np.hamming(256)
r = np.correlate(w, w, "full")[255:255 + 13]
R = np.array([[r[abs(i - j)] for j in range(12)] for i in range(12)])
assert np.allclose(levinson(r, 12)[0][1:], np.linalg.solve(R, -r[1:13]))
print("Levinson-Durbin matches np.linalg.solve")


def err(v, order=12, f0=120, snr_db=None):
    est = lpc_formants(vowel(VOWELS[v], f0=f0, snr_db=snr_db), order=order)
    if len(est) < 3:
        return f"only {len(est)} formant(s) found: " + " ".join(f"{e:5.0f}" for e in est)
    return " ".join(f"{e:5.0f}" for e in est) + f"  (max err {np.max(np.abs(est - VOWELS[v])):3.0f} Hz)"


for v, F in VOWELS.items():
    print(f"/{v}/ true {F[0]:5d} {F[1]:5d} {F[2]:5d}")
    for order in (6, 10, 14):
        print(f"     order {order:2d}, f0 120: {err(v, order=order)}")
    print(f"     order 12, f0 250: {err(v, order=12, f0=250)}")
    for snr in (20, 10):
        print(f"     order 12, {snr} dB: {err(v, order=12, snr_db=snr)}")
```

Output (Python 3.14, NumPy 2.5):
```
Levinson-Durbin matches np.linalg.solve
/a/ true   730  1090  2440
     order  6, f0 120: only 2 formant(s) found:   701  1032
     order 10, f0 120:   719  1080  2413  (max err  27 Hz)
     order 14, f0 120:   723  1091  2414  (max err  26 Hz)
     order 12, f0 250:   750  1068  2458  (max err  22 Hz)
     order 12, 20 dB:   697   968  2382  (max err 122 Hz)
     order 12, 10 dB: only 2 formant(s) found:   755  4045
/i/ true   270  2290  3010
     order  6, f0 120: only 2 formant(s) found:   236  2749
     order 10, f0 120:   250  2289  3005  (max err  20 Hz)
     order 14, f0 120:   242  2288  3015  (max err  28 Hz)
     order 12, f0 250:   252  2260  2981  (max err  30 Hz)
     order 12, 20 dB:   240  1456  2277  (max err 834 Hz)
     order 12, 10 dB: only 2 formant(s) found:  1438  3852
/u/ true   300   870  2240
     order  6, f0 120: only 2 formant(s) found:   265   808
     order 10, f0 120:   287   848  2376  (max err 136 Hz)
     order 14, f0 120:   275   874  2206  (max err  34 Hz)
     order 12, f0 250:   247   647   889  (max err 1351 Hz)
     order 12, 20 dB:   275  2329  2967  (max err 1459 Hz)
     order 12, 10 dB: only 1 formant(s) found:  2365
```

How to read it:
- Levinson-Durbin returns the same coefficients as a general linear solve, in O(p²).
- On clean low-pitched vowels, order 10–14 recovers F1–F3 within about 20–35 Hz, apart from /u/ at order 10 (F3 off by 136 Hz). /u/ has F1 and F2 close together and a weak F3, which is harder to model with few poles.
- Order 6 is too low. Three formants plus the source tilt need at least 8 poles, so the model merges or drops one and only two formants pass the bandwidth filter.
- High pitch breaks it for close formants. At f0 = 250 Hz (a child's or high female voice), harmonics are 250 Hz apart and the spectrum is sampled too sparsely. For /u/, LPC puts a pole on the first harmonic (247 Hz) and splits F1/F2 into 647 and 889 Hz. /a/ and /i/ still come out within 30 Hz. LPC formants for high voices need care: more frames, closed-phase analysis or a formant tracker with continuity constraints.
- Noise is the bigger problem. At 20 dB SNR (a decent recording) /i/ loses F2 and gets a spurious pole at 1456 Hz; at 10 dB most vowels lose formants entirely. White noise is flat while the speech spectrum falls with frequency, so the higher formants sink into the noise first. The model also spends poles on the noise.

## Choosing / trade-offs
- **LPC vs a tool.** For research-grade formant tracks use Praat (`To Formant (burg)...`, also scriptable from Python via the `parselmouth` package). It uses the Burg method and gives tracks over time. Write your own LPC to learn it, for codec work, or when you need a fast envelope inside a pipeline.
- **Autocorrelation vs covariance vs Burg.** Autocorrelation (this file) always gives a stable filter and is cheapest; windowing biases bandwidths. Covariance and Burg methods are more accurate on short frames; Burg is also guaranteed stable and is Praat's default.
- **Order and sample rate.** Downsample to about 10 kHz for F1–F3 (Praat's default "maximum formant" is 5500 Hz for women and 5000 Hz for men) and use `p ≈ sr/1000 + 2`. Too many poles split a formant or model individual harmonics; too few merge formants.
- **LPC envelope vs mel features.** For ASR and classification, log-mel and learned features replaced LPC long ago. LPC still wins when you need an interpretable resonance model or a compact, invertible envelope for coding.

## Gotchas
- Choose frames inside stable vowels. Formants in consonants, transitions or silence are meaningless, so gate with a voicing or VAD decision first ([[voice-activity-detection-from-scratch]], [[yin-pitch-detection-from-scratch]]).
- Formant labels are by order: "F1" is the lowest kept pole. One spurious or missing pole shifts every label up or down, which is how a 20 dB recording reports F2 = 1456 Hz for /i/. Use bandwidth limits, expected ranges per formant, and continuity across frames.
- Pre-emphasis matters. Without it the tilt eats poles and high formants are underestimated. Use 0.95–0.97, and apply it once (some loaders already do).
- Phone and VoIP audio is band-limited to about 3.4 kHz (narrowband) and compressed by codecs, which distort the envelope. F3 and above are unreliable there.
- Formant values differ by speaker (vocal tract length): children's formants are much higher than adults'. Normalise per speaker (Lobanov z-scores) before comparing vowels across people.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the framing and spectral features that replaced LPC in ASR.
- [[yin-pitch-detection-from-scratch]] - the source side of the source-filter model (f0).
- [[voice-activity-detection-from-scratch]] - pick voiced frames before estimating formants.
- [[text-to-speech]] - vocoders, some of which (LPCNet) use an LPC envelope.
- [[griffin-lim-from-scratch]] - another way to turn a spectral description back into audio.

## References
- Makhoul (1975), "Linear Prediction: A Tutorial Review", Proceedings of the IEEE: https://doi.org/10.1109/PROC.1975.9792
- Peterson & Barney (1952), "Control Methods Used in a Study of the Vowels", JASA (the formant values used here): https://doi.org/10.1121/1.1906875
