---
title: Voice activity detection from scratch (frame energy, adaptive noise floor, zero-crossing rate, hangover)
category: speech
tags: [vad, voice-activity-detection, endpointing, energy, zero-crossing-rate, hangover, silero-vad, webrtc-vad, speech-to-text, numpy, from-scratch]
use_cases:
  - "detect where speech starts and stops in a recording or stream before sending it to ASR"
  - "endpoint user turns in a voice agent (when has the user stopped talking)"
  - "trim silence from audio datasets or skip silent chunks to save transcription cost"
  - "understand why a simple energy VAD fails in noise and what a neural VAD buys"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1002/j.1538-7305.1975.tb02828.x
  - https://github.com/snakers4/silero-vad
---

# Voice activity detection from scratch (frame energy, adaptive noise floor, zero-crossing rate, hangover)

## Summary
Voice activity detection (VAD) labels each short frame of audio as speech or non-speech. It sits in front of almost every speech system: it decides what to send to ASR, when a voice agent's user has finished a turn, and which parts of a dataset to keep. The classic detector compares frame energy with an estimated noise floor, uses the zero-crossing rate to catch quiet unvoiced sounds, and smooths decisions with a minimum duration and a hangover (Rabiner & Sambur 1975 is the classic endpointing version). This file implements it in NumPy on synthetic speech-like audio with exact labels and measures where it breaks as the SNR drops.

## Key concepts
- **Frames.** 20–30 ms windows with a 10 ms hop: short enough that speech is roughly stationary, long enough to measure energy and pitch.
- **Energy against a noise floor.** Speech frames are louder than background. A fixed threshold fails when the recording level or room changes, so estimate the floor from the data (a low percentile of frame energies, or a slowly tracking minimum in streaming) and threshold at floor + margin in dB.
- **Zero-crossing rate (ZCR).** Unvoiced consonants (s, f, sh) are quiet but noise-like with many sign changes per frame. A secondary rule "moderately loud and high ZCR" recovers some of them. Background noise also has a high ZCR, so the rule only helps when the noise is quieter than the fricatives.
- **Minimum duration.** Isolated speech decisions of one or two frames are usually clicks or bumps; drop runs shorter than a minimum.
- **Hangover.** Extend each speech run by a few frames (here 80 ms). Word endings decay into the noise and short pauses inside a sentence should not split it. Hangover trades precision for recall and is what stops ASR cutting off the last phoneme.
- **Frame vs segment metrics.** Frame precision/recall says how many frames are right. For endpointing, segment-level errors (missed start, clipped end, false trigger, latency to detect the end) matter more.

## When to use / scenarios
- Voice agents and IVR: end-of-turn detection, barge-in detection, and not sending silence to ASR or an LLM ([[voice-agents]]).
- Batch transcription: split long recordings into speech chunks before Whisper-style ASR, which reduces cost and hallucinated text on silence ([[speech-to-text]]).
- Dataset preparation: trim leading and trailing silence for TTS training data ([[text-to-speech]]), remove empty clips.
- Diarization front end: only speech frames go to speaker embedding ([[speaker-diarization]]).
- On-device wake-word or low-power front ends where an energy gate runs before a heavier model.
- Not as the only detector in noisy or music-heavy audio (cafes, cars, TV in the background): use a neural VAD such as Silero VAD or the VAD in your ASR/realtime API, and keep the energy gate only as a cheap pre-filter.

## Setup & code
NumPy only, a few seconds. 30 s of synthetic "speech": bursts of voiced harmonics with a moving pitch and syllable-like envelope, each containing one noise-like fricative, separated by silences, with exact sample labels. Slightly coloured noise is added at several SNRs.

```python
import numpy as np

sr, frame, hop = 16000, 400, 160                     # 25 ms frames, 10 ms hop
rng = np.random.default_rng(0)


def synth_utterances(total_s=30.0):
    """Speech-like bursts (voiced harmonics with a pitch glide, plus fricative noise) between silences; per-sample labels."""
    x = np.zeros(int(total_s * sr))
    label = np.zeros(len(x), bool)
    t = 0.5
    while t < total_s - 2:
        dur = rng.uniform(0.3, 1.5)
        n = int(dur * sr)
        tt = np.arange(n) / sr
        f0 = rng.uniform(90, 220) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(1, 4) * tt))
        phase = 2 * np.pi * np.cumsum(f0) / sr
        voiced = sum(np.sin(k * phase) / k for k in range(1, 15))
        env = np.abs(np.sin(np.pi * tt * rng.uniform(2, 6))) ** 0.5        # syllable-like amplitude
        seg = voiced * env * rng.uniform(0.3, 1.0)
        k = int(rng.uniform(0.05, 0.12) * sr)                                 # one unvoiced fricative per burst
        s0 = rng.integers(0, max(n - k, 1))
        seg[s0:s0 + k] = 0.25 * rng.standard_normal(k) * np.hanning(k)
        i = int(t * sr)
        x[i:i + n] += seg
        label[i:i + n] = True
        t += dur + rng.uniform(0.2, 1.5)
    return x, label


def frames(x):
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None] + hop * np.arange(n)[:, None]
    return x[idx] * np.hanning(frame)


def vad(x, margin_db=6.0, hangover=8, min_speech=3, use_zcr=True):
    f = frames(x)
    e_db = 10 * np.log10((f ** 2).mean(1) + 1e-12)
    zcr = (np.abs(np.diff(np.sign(f), axis=1)) > 0).mean(1)
    noise_db = np.percentile(e_db, 10)                       # noise floor: quietest 10% of frames
    speech = e_db > noise_db + margin_db
    if use_zcr:                                              # weak but noisy-looking frames: fricatives
        speech |= (e_db > noise_db + margin_db / 2) & (zcr > 0.3)
    out = speech.copy()
    # drop bursts shorter than min_speech frames (clicks), then extend each speech run by `hangover` frames
    run = 0
    for i, s in enumerate(np.r_[speech, False]):
        if s:
            run += 1
        else:
            if 0 < run < min_speech:
                out[i - run:i] = False
            run = 0
    held = out.copy()
    last = -10 ** 9
    for i, s in enumerate(out):
        if s:
            last = i
        elif i - last <= hangover:
            held[i] = True
    return held


def frame_labels(label):
    n = 1 + (len(label) - frame) // hop
    return np.array([label[i * hop:i * hop + frame].mean() > 0.5 for i in range(n)])


clean, lab = synth_utterances()
y = frame_labels(lab)
print(f"{len(y)} frames, {y.mean():.0%} speech")
p_sig = (clean[lab] ** 2).mean()


def noisy(snr):
    noise = np.convolve(rng.standard_normal(len(clean)), np.ones(4) / 2, "same")   # slightly coloured noise
    return clean + noise * np.sqrt(p_sig / 10 ** (snr / 10) / (noise ** 2).mean())


def score(pred):
    tp = (pred[:len(y)] & y).sum()
    return f"P {tp / max(pred[:len(y)].sum(), 1):.2f} R {tp / y.sum():.2f}"


for snr in (20, 10, 5, 0):
    x = noisy(snr)
    print(f"SNR {snr:2d} dB: " + " | ".join(f"{name} {score(vad(x, **kw))}" for name, kw in [
        ("energy only", dict(hangover=0, min_speech=1, use_zcr=False)),
        ("+ZCR", dict(hangover=0, min_speech=1)),
        ("+ZCR+smoothing", dict())]))
for snr in (5, 0):
    x = noisy(snr)
    print(f"SNR {snr} dB, margin sweep (with smoothing): " + " | ".join(
        f"{m} dB {score(vad(x, margin_db=m))}" for m in (1.5, 3.0, 6.0)))
```

Output (Python 3.14, NumPy 2.5):
```
2998 frames, 53% speech
SNR 20 dB: energy only P 0.99 R 0.98 | +ZCR P 0.99 R 0.98 | +ZCR+smoothing P 0.92 R 1.00
SNR 10 dB: energy only P 1.00 R 0.78 | +ZCR P 1.00 R 0.82 | +ZCR+smoothing P 0.92 R 0.95
SNR  5 dB: energy only P 1.00 R 0.46 | +ZCR P 1.00 R 0.48 | +ZCR+smoothing P 0.95 R 0.64
SNR  0 dB: energy only P 1.00 R 0.11 | +ZCR P 1.00 R 0.11 | +ZCR+smoothing P 0.93 R 0.18
SNR 5 dB, margin sweep (with smoothing): 1.5 dB P 0.91 R 0.99 | 3.0 dB P 0.92 R 0.95 | 6.0 dB P 0.95 R 0.63
SNR 0 dB, margin sweep (with smoothing): 1.5 dB P 0.91 R 0.94 | 3.0 dB P 0.95 R 0.62 | 6.0 dB P 0.96 R 0.19
```

How to read it:
- At 20 dB SNR plain energy thresholding is nearly perfect (P 0.99, R 0.98). Smoothing lifts recall to 1.00 and drops precision to 0.92: the 80 ms hangover deliberately marks frames after each word as speech. Those "false positives" are the tail protection you want for ASR.
- With a fixed 6 dB margin, recall collapses as noise rises: 0.78 at 10 dB, 0.46 at 5 dB, 0.11 at 0 dB. Precision stays high, so the failure mode is clipped and missed speech, not false triggers.
- ZCR adds 2–4 recall points at moderate SNR and nothing at 0 dB, where the noise itself has a high ZCR.
- The margin is the main knob. At 0 dB, lowering it from 6 to 1.5 dB brings recall from 0.19 back to 0.94 with precision 0.91. The right margin depends on the SNR, so a fixed setting tuned on clean audio fails in noise. Adaptive margins (from the gap between the noise and speech energy distributions) or a learned model fix that.
- The synthetic noise here is stationary. Real noise (babble, music, traffic) is not, and that is where energy VADs fail far worse than these numbers suggest.

## Choosing / trade-offs
- **Energy VAD vs WebRTC VAD vs neural VAD.** Energy: zero dependencies, microseconds per frame, fine for clean close-talk audio. WebRTC VAD (GMM on sub-band energies, aggressiveness 0–3): tiny and fast, better than energy, still weak in babble. Silero VAD and similar small neural models: much more robust to non-stationary noise and music, run in real time on CPU, and are what most voice-agent stacks use today.
- **Hangover and minimum silence.** Longer hangover avoids clipping and splitting sentences but adds latency to end-of-turn detection in a voice agent. Typical endpointing waits 300–800 ms of silence; turn-taking models that use the transcript as well as audio do better than any fixed silence timeout.
- **Thresholds.** Favour recall in front of ASR (missed speech cannot be recovered; extra silence only costs compute). Favour precision for wake-up or barge-in, where false triggers interrupt the user.
- **Streaming vs offline.** Offline can use percentiles over the whole file and look ahead. Streaming needs a running noise estimate (minimum tracking with slow upward adaptation) and decides with no future context, so it needs a pre-roll buffer to keep the start of each utterance.

## Gotchas
- A percentile noise floor assumes some frames are silence. A clip that is 95% speech, or a recording that starts mid-sentence in noise, gives a wrong floor; clamp it with an absolute minimum.
- Keep a pre-roll (100–300 ms of audio before the detected start). Detection lags the real onset, and ASR needs the first consonant.
- Normalising loudness per clip before VAD changes the energy scale between clips; set thresholds relative to the floor, not in absolute dB.
- Breaths, lip smacks, keyboard clicks and coughs pass an energy test; minimum duration filters some, a neural VAD filters most.
- Resampling: WebRTC VAD accepts only 8, 16, 32 or 48 kHz and 10/20/30 ms frames; Silero expects 8 or 16 kHz. Feed the wrong rate and the output is garbage without an error.
- Frame-level accuracy can look high while every utterance is clipped by 50 ms at each end. Check segment boundaries on real recordings by listening.

## Related
- [[voice-agents]] - turn detection and barge-in, the main production use of VAD.
- [[speech-to-text]] - chunking long audio and avoiding hallucinations on silence.
- [[spectral-subtraction-denoising-from-scratch]] - uses a VAD-style noise estimate and can run before a VAD.
- [[speaker-diarization]] - speech regions feed the speaker embedding step.
- [[mel-spectrogram-and-mfcc-from-scratch]] - the framing and features neural VADs build on.
- [[yin-pitch-detection-from-scratch]] - voicing decisions from periodicity, a complement to energy.

## References
- Rabiner & Sambur (1975), "An Algorithm for Determining the Endpoints of Isolated Utterances", Bell System Technical Journal: https://doi.org/10.1002/j.1538-7305.1975.tb02828.x
- Silero VAD (pre-trained neural VAD, MIT licence): https://github.com/snakers4/silero-vad
