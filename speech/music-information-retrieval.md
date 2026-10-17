---
title: Music information retrieval (MIR)
category: speech
tags: [mir, music, beat-tracking, tempo, key-detection, source-separation, demucs, librosa, essentia, clap, music-tagging, audio-similarity]
use_cases:
  - "detect the tempo (BPM) and beat positions of songs for a DJ or fitness app"
  - "estimate the musical key and chords of tracks in a catalogue"
  - "split a song into vocals, drums, bass and accompaniment for karaoke or remixing"
  - "auto-tag a music library with genre, mood and instruments"
  - "find similar-sounding tracks or search music by a text description"
  - "transcribe a recorded melody or instrument part to MIDI"
status: draft
last_verified: 2026-10-15
sources:
  - https://librosa.org/doc/latest/index.html
  - https://github.com/librosa/librosa
  - https://github.com/MTG/essentia
  - https://github.com/facebookresearch/demucs
  - https://github.com/adefossez/demucs
  - https://github.com/LAION-AI/CLAP
  - https://github.com/spotify/basic-pitch
  - https://arxiv.org/abs/2211.08553
---

# Music information retrieval (MIR)

## Summary
Music information retrieval pulls structured facts out of music audio. The common tasks are tempo and beat tracking, key and chord estimation, source separation into stems, tagging (genre, mood, instrument), similarity search and transcription to MIDI. Classic DSP features (onset strength, chroma, mel spectrograms) built with librosa or Essentia handle rhythm and tonality well. Neural models handle the harder perceptual tasks: Demucs for separation, CLAP-style audio-text embeddings for tagging and search, and Basic Pitch for transcription.

## Key concepts
- **Onset strength / spectral flux**: frame-to-frame increase in spectral energy. Peaks mark note attacks, and it is the input to most beat trackers.
- **Tempo**: the dominant periodicity of onset strength, found by autocorrelation or a tempogram. Octave errors, where the estimate is half or double the true tempo, are the classic failure.
- **Beat tracking**: placing beat times consistent with the tempo. Dynamic programming over onset strength is the method librosa's `beat_track` uses.
- **Chroma**: energy folded into the 12 pitch classes (C, C#, ..., B). This is the base feature for key and chord estimation.
- **Key estimation**: correlate the average chroma with major and minor key profiles (Krumhansl-Schmuckler) across all 24 rotations and pick the best. Relative major and minor keys are easily confused.
- **Source separation**: estimate stems (vocals, drums, bass, other) from a stereo mix. Hybrid Transformer Demucs (v4, `htdemucs`) reports 9.00 dB SDR on the MUSDB HQ test set. Classic linear methods such as [[fastica-from-scratch]] and [[nmf-from-scratch]] show the idea but cannot separate a real stereo mix.
- **Tagging and similarity**: embed audio into vectors. Either train a classifier on the embeddings (genre, mood) or compare them with cosine similarity. CLAP (contrastive language-audio pretraining) puts audio and text in one space, so it supports zero-shot tags and text-to-music search ([[contrastive-learning-infonce-from-scratch-numpy]]).
- **Transcription (AMT)**: converting audio to notes or MIDI. Spotify's Basic Pitch is a lightweight, instrument-agnostic polyphonic transcriber that works best on one instrument at a time. For monophonic pitch, see [[yin-pitch-detection-from-scratch]].
- **Evaluation**: the `mir_eval` package implements the standard metrics for beats, key, chords, melody and separation. Beat tracking is scored within a tolerance window, and separation is scored in SDR.

## When to use / scenarios
- DJ, fitness and rhythm-game apps: BPM, beat grid and downbeats for syncing, tempo matching or step generation.
- Music catalogues and streaming: auto-tag genre, mood, instrumentation and key for search filters and playlists.
- Karaoke, remixing and practice tools: separate vocals or drums with Demucs, then transpose or time-stretch the result ([[phase-vocoder-time-stretch-from-scratch]]).
- Recommendation: use audio embeddings as content features for cold-start tracks that have no plays yet ([[recommender-systems]]).
- Sync licensing and ad agencies: text-to-music search ("upbeat acoustic guitar, no vocals") over a licensed library using CLAP.
- Musicology and education: chord charts, melody transcription and practice feedback.
- Not for: identifying an exact recording from a noisy clip. That is audio fingerprinting (Shazam-style landmark hashing), a different technique. Also not for generating music, which needs generative audio models. For speech content, use [[speech-to-text]].

## Setup & code
```bash
pip install librosa            # ISC-licensed; tempo, beats, chroma, onset strength
pip install demucs             # stems: demucs --two-stems=vocals song.mp3 (needs torch)
pip install laion-clap         # audio-text embeddings for tagging and search (needs torch)
pip install basic-pitch        # audio -> MIDI transcription
```
The self-contained numpy block below implements the two most useful classic MIR steps from scratch. It estimates tempo by autocorrelating spectral flux, then estimates key by correlating chroma with Krumhansl-Schmuckler profiles. Both run on synthetic audio that has a known answer.
```python
import numpy as np

SR, HOP, NFFT = 22050, 512, 2048
rng = np.random.default_rng(0)

def stft_mag(x):
    win = np.hanning(NFFT)
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.stack([x[i * HOP:i * HOP + NFFT] * win for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1))          # (frames, bins)

def estimate_tempo(x, lo=60, hi=200):
    S = np.log1p(stft_mag(x))
    flux = np.maximum(0, np.diff(S, axis=0)).sum(axis=1)  # onset strength
    flux -= flux.mean()
    ac = np.correlate(flux, flux, mode="full")[len(flux) - 1:]
    fps = SR / HOP
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= lo) & (bpm <= hi)
    best = lags[ok][np.argmax(ac[ok])]
    return 60 * fps / best

def chroma(x):
    M = stft_mag(x).mean(axis=0)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    c = np.zeros(12)
    valid = (freqs > 55) & (freqs < 2000)
    midi = 69 + 12 * np.log2(freqs[valid] / 440.0)
    np.add.at(c, np.round(midi).astype(int) % 12, M[valid] ** 2)
    return c / c.sum()

# Krumhansl-Kessler key profiles, index 0 = tonic
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def estimate_key(x):
    c = chroma(x)
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", MAJOR), ("minor", MINOR)):
            scores.append((np.corrcoef(c, np.roll(prof, tonic))[0, 1], f"{NAMES[tonic]} {mode}"))
    return max(scores)[1]

# Synthetic test 1: click track at 128 BPM with noise
bpm_true, dur = 128.0, 12.0
x = 0.01 * rng.standard_normal(int(SR * dur))
for t in np.arange(0, dur - 0.1, 60 / bpm_true):
    i = int(t * SR)
    x[i:i + 400] += np.exp(-np.arange(400) / 60) * rng.standard_normal(400)
tempo = estimate_tempo(x)
print(f"tempo estimate: {tempo:.1f} BPM (true {bpm_true})")
assert abs(tempo - bpm_true) < 3 or abs(tempo - 2 * bpm_true) < 6 or abs(tempo - bpm_true / 2) < 3

# Synthetic test 2: G major scale and chords (G B D, C E G, D F# A) as harmonic tones
def tone(midi, d=0.5):
    t = np.arange(int(SR * d)) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    return sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) * np.hanning(len(t))

seq = [67, 69, 71, 72, 74, 76, 78, 79]                   # G major scale
chords = [(67, 71, 74), (60, 64, 67), (62, 66, 69), (67, 71, 74)]
y = np.concatenate([tone(m) for m in seq] + [sum(tone(m, 1.0) for m in ch) for ch in chords])
key = estimate_key(y)
print("key estimate:", key)
assert key in ("G major", "E minor")                     # relative minor is an accepted confusion
```
The same steps with librosa on a real file:
<!-- skip-check: needs an audio file on disk -->
```python
import librosa, numpy as np
y, sr = librosa.load("song.mp3")                         # mono, resampled to 22050 Hz
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)   # feed to the key estimator above
print(np.atleast_1d(tempo)[0], beat_times[:8])
```

## Choosing / trade-offs
| Task | Lightweight / classic | Neural / stronger | Watch for |
|---|---|---|---|
| Tempo, beats | librosa `beat_track`, onset autocorrelation | madmom, Essentia, or beat-tracking networks for downbeats and tempo changes | Octave errors; rubato and live recordings |
| Key, chords | Chroma + key profiles | Essentia key extractors, chord-recognition models | Modal and atonal music, key changes |
| Separation | NMF or ICA (toy cases only) | Demucs `htdemucs` (MIT code; repo no longer actively maintained, a fork lives at `adefossez/demucs`) | GPU time, artefacts, rights to the source audio |
| Tagging, similarity | MFCC or mel statistics + [[classic-ml-scikit-learn]] | CLAP or other audio embeddings + a linear probe or vector search ([[vector-databases]]) | Label taxonomy, genre bias |
| Transcription | YIN for monophonic pitch | Basic Pitch (Apache-2.0) | Dense polyphony, drums |

- Licences: librosa is ISC and Basic Pitch is Apache-2.0. Essentia is AGPL-3.0, so check that before embedding it in a closed-source network service. Pretrained model weights carry their own terms.
- On-device vs server: classic features run anywhere in real time. Separation and large embedding models want a GPU for catalogue-scale batches.
- Train your own vs zero-shot: CLAP zero-shot text labels are quick to try. A linear probe on its embeddings, trained with a few hundred labelled tracks per tag, is usually more reliable for a fixed taxonomy.

## Gotchas
- Tempo is ambiguous by a factor of 2. Constrain the BPM range to the genre, or report both candidates.
- Sample rate and mono mixing: librosa resamples to 22050 Hz and mixes to mono by default. Separation models expect stereo at their own rate.
- Compressed, clipped or loudness-maximised masters smear onsets. Live recordings drift in tempo, so a single BPM value is wrong for them.
- Key-profile methods confuse relative major and minor (G major and E minor) and fifth-related keys. Report a confidence or the top-2 keys.
- Separated stems contain bleed and artefacts. They are fine for karaoke, but not a substitute for the original multitrack.
- Copyright: analysing audio you hold is different from redistributing separated stems or training on a catalogue. Check the licence and rights before shipping.
- Tag datasets are skewed toward Western popular music. Evaluate on your own catalogue, especially for regional genres.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the spectral features MIR builds on.
- [[yin-pitch-detection-from-scratch]] - monophonic pitch for melody tasks.
- [[phase-vocoder-time-stretch-from-scratch]] - tempo change without pitch change.
- [[nmf-from-scratch]], [[fastica-from-scratch]] - classic source separation ideas.
- [[dynamic-time-warping-from-scratch]] - aligning performances and cover-song matching.
- [[embeddings]], [[vector-databases]] - audio embedding search.
- [[recommender-systems]] - audio features for cold-start music recommendation.
- [[speech-to-text]] - lyrics and spoken-word content.
- [[speech-translation]], [[wake-word-detection]] - other speech-category tasks in this batch.

## References
- librosa documentation: https://librosa.org/doc/latest/index.html
- Essentia (AGPL-3.0): https://github.com/MTG/essentia
- Demucs v4 (Hybrid Transformer Demucs): https://github.com/facebookresearch/demucs, maintained fork https://github.com/adefossez/demucs; paper https://arxiv.org/abs/2211.08553
- LAION CLAP: https://github.com/LAION-AI/CLAP
- Spotify Basic Pitch: https://github.com/spotify/basic-pitch
- Krumhansl, C. L. (1990). Cognitive Foundations of Musical Pitch. Oxford University Press (key profiles).
- mir_eval: https://github.com/craffel/mir_eval
