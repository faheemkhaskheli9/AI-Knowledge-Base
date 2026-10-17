---
title: Audio classification and sound event detection (AST, CLAP, PANNs, anomalous sound)
category: speech
tags: [audio-classification, sound-event-detection, sed, audio-tagging, ast, clap, panns, beats, yamnet, audioset, esc-50, dcase, anomalous-sound-detection, log-mel]
use_cases:
  - "detect glass breaking, gunshots, alarms or screams in audio from cameras or smart speakers"
  - "classify environmental sounds with no labelled data using text prompts"
  - "find when each sound event starts and ends in long recordings"
  - "detect abnormal machine sounds for predictive maintenance"
  - "tag a large audio library for search and moderation"
status: draft
last_verified: 2026-10-15
sources:
  - https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
  - https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
  - https://huggingface.co/docs/transformers/model_doc/clap
  - https://huggingface.co/laion/clap-htsat-unfused
  - https://ar5iv.labs.arxiv.org/html/1912.10211
  - https://research.google.com/audioset/
  - https://github.com/karolpiczak/ESC-50
  - https://dcase.community/
---

# Audio classification and sound event detection (AST, CLAP, PANNs, anomalous sound)

## Summary
Audio classification assigns labels to a clip ("dog bark", "siren"); audio tagging allows several labels per clip; sound event detection (SED) also finds when each event starts and ends. Most models turn audio into a log-mel spectrogram and apply a CNN or transformer pretrained on AudioSet (over 5,000 hours, 527 classes): PANNs (CNNs), AST (a ViT on spectrogram patches) and BEATs (self-supervised). CLAP learns a joint audio-text embedding, so it classifies sounds zero-shot from text labels and powers audio search. Industrial anomalous sound detection is the unsupervised variant: learn what normal machines sound like and flag deviations.

## Key concepts
- **Features.** Resample to the model's rate (often 16 kHz or 32 kHz), compute a log-mel spectrogram with the exact parameters the checkpoint was trained with ([[mel-spectrogram-and-mfcc-from-scratch]]).
- **Clip vs frame outputs.** Tagging gives one score per class per clip; SED gives scores per frame (for example every 10-40 ms), then thresholds, smooths (median filter) and merges frames into events with onset and offset.
- **Weak vs strong labels.** AudioSet labels say what is in a 10-second clip, not when (weak). Strong labels with timestamps are expensive; weakly supervised SED uses attention or pooling over frames to learn timing from clip labels.
- **Pretrained backbones.** PANNs (CNN14 and others) output 527 AudioSet scores and embeddings; AST is ViT-style on 16x16 spectrogram patches (`MIT/ast-finetuned-audioset-10-10-0.4593` in Transformers); YAMNet (MobileNet, 521 classes) is small enough for phones; BEATs is a strong self-supervised backbone used in DCASE systems.
- **Zero-shot with CLAP.** Audio and text encoders trained contrastively ([[contrastive-learning-infonce-from-scratch-numpy]]); classify by comparing an audio embedding with "This is a sound of {label}." prompts. The Transformers pipeline `zero-shot-audio-classification` wraps it (`laion/clap-htsat-unfused`).
- **Metrics.** Tagging: mAP across classes. SED: segment-based F1 and error rate (fixed time slices) or event-based F1 with an onset collar; DCASE also uses PSDS. Anomaly detection: AUC and partial AUC at low false-positive rates.
- **Benchmarks.** AudioSet; ESC-50 (2,000 five-second clips, 50 classes, 5 official folds); UrbanSound8K (8,732 clips up to 4 s, 10 classes, 10 official folds); DCASE challenge tasks for SED and machine anomalous sound detection.

## When to use / scenarios
- Safety and security: glass break, gunshot, smoke alarm, scream or crash detection on cameras and smart-home devices, with a human or second-stage check before alerting.
- Industrial and IoT: abnormal sounds from pumps, fans, valves and bearings for predictive maintenance ([[manufacturing-iot]], [[anomaly-detection]]).
- Media and content: tagging sound libraries, detecting music vs speech vs silence, moderation (explicit or violent audio), audio search with CLAP embeddings ([[embeddings]]).
- Health and nature: cough or snore detection (regulated if diagnostic), bird and wildlife monitoring with passive recorders.
- Pre-processing for speech pipelines: drop music and noise segments before transcription ([[speech-to-text]], [[voice-activity-detection-from-scratch]]).
- Not for: understanding spoken content (use STT), identifying who is speaking ([[speaker-diarization]]), or detecting deepfake speech, which needs dedicated anti-spoofing models ([[voice-cloning-and-ethics]]).

## Setup & code
```bash
pip install transformers torch librosa
```
<!-- skip-check: needs torch and model downloads -->
```python
import librosa
from transformers import pipeline

audio, sr = librosa.load("clip.wav", sr=16000)                     # AST expects 16 kHz mono

tagger = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
print(tagger(audio, top_k=5))                                      # AudioSet labels with scores

clap = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
audio48, _ = librosa.load("clip.wav", sr=48000)                    # CLAP feature extractor expects 48 kHz
print(clap(audio48, candidate_labels=["glass breaking", "dog barking", "car horn", "silence"]))
```

Self-contained NumPy + scikit-learn demo of the SED pipeline: log-mel features, a frame classifier, median-filter smoothing, event extraction, and event-based F1 with a 200 ms onset collar. A second part scores machine-sound anomalies with a Gaussian model of normal spectra (the classic unsupervised baseline):
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(5)
SR, HOP, NFFT, NMEL = 16000, 320, 1024, 40                  # 20 ms frames
CLASSES = ["background", "siren", "alarm_beep", "knock"]


def mel_filterbank(n_mels=NMEL, n_fft=NFFT, sr=SR):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(0, mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        fb[i, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[i, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


FB = mel_filterbank()


def logmel(x):
    n = 1 + (len(x) - NFFT) // HOP
    frames = np.stack([x[i * HOP:i * HOP + NFFT] for i in range(n)]) * np.hanning(NFFT)
    return np.log(FB @ (np.abs(np.fft.rfft(frames, axis=1)) ** 2).T + 1e-8).T   # (frames, mels)


def event_sound(kind, dur):
    t = np.arange(int(SR * dur)) / SR
    if kind == "siren":
        return 0.5 * np.sin(2 * np.pi * np.cumsum(900 + 400 * np.sin(2 * np.pi * 0.8 * t)) / SR)
    if kind == "alarm_beep":
        return 0.5 * np.sin(2 * np.pi * 3000 * t) * (np.sin(2 * np.pi * 4 * t) > 0)
    burst = np.zeros_like(t)                                         # knock: short decaying broadband bursts
    for s in np.arange(0, dur - 0.1, 0.25):
        i = int(s * SR)
        burst[i:i + 1600] += rng.standard_normal(1600) * np.exp(-np.arange(1600) / 200)
    return 0.8 * burst


def scene(seconds=12.0, n_events=4):
    """Background noise with non-overlapping events; returns audio, frame labels and event list."""
    x = 0.05 * np.cumsum(rng.standard_normal(int(SR * seconds))) / 50 + 0.02 * rng.standard_normal(int(SR * seconds))
    events, t = [], 0.5
    for _ in range(n_events):
        kind, dur = CLASSES[rng.integers(1, 4)], rng.uniform(0.6, 1.5)
        if t + dur > seconds - 0.5:
            break
        i = int(t * SR)
        x[i:i + int(dur * SR)] += event_sound(kind, dur)[: len(x[i:i + int(dur * SR)])]
        events.append((t, t + dur, kind))
        t += dur + rng.uniform(0.5, 1.5)
    feats = logmel(x)
    centres = (np.arange(len(feats)) * HOP + NFFT / 2) / SR
    y = np.zeros(len(feats), int)
    for on, off, kind in events:
        y[(centres >= on) & (centres < off)] = CLASSES.index(kind)
    return feats, y, events, centres


def median_filter(labels, k=9):
    pad = np.pad(labels, k // 2, mode="edge")
    return np.array([np.bincount(pad[i:i + k]).argmax() for i in range(len(labels))])


def to_events(labels, centres):
    out, start = [], None
    for i in range(len(labels) + 1):
        cur = labels[i] if i < len(labels) else 0
        prev = labels[i - 1] if i > 0 else 0
        if cur != prev:
            if prev != 0:
                out.append((centres[start], centres[i - 1], CLASSES[prev]))
            start = i
    return out


def event_f1(pred, ref, collar=0.2):
    matched, tp = set(), 0
    for on, off, k in pred:
        for j, (ron, roff, rk) in enumerate(ref):
            if j not in matched and k == rk and abs(on - ron) <= collar:
                matched.add(j)
                tp += 1
                break
    return 2 * tp / max(len(pred) + len(ref), 1)


train = [scene() for _ in range(15)]
X = np.vstack([f for f, *_ in train])
mu, sd = X.mean(0), X.std(0) + 1e-6
ctx = lambda f: np.hstack([np.roll(f, s, 0) for s in (-2, -1, 0, 1, 2)])   # +-2 frames of context
clf = LogisticRegression(max_iter=2000).fit(ctx((X - mu) / sd), np.concatenate([y for _, y, *_ in train]))

raw_f1, smooth_f1, frame_acc = [], [], []
for _ in range(8):
    f, y, ref, centres = scene()
    p = clf.predict(ctx((f - mu) / sd))
    frame_acc.append((p == y).mean())
    raw_f1.append(event_f1(to_events(p, centres), ref))
    smooth_f1.append(event_f1(to_events(median_filter(p), centres), ref))
print(f"frame accuracy {np.mean(frame_acc):.3f}")
print(f"event F1 (200 ms collar): raw {np.mean(raw_f1):.3f}  -> median-filtered {np.mean(smooth_f1):.3f}")

# unsupervised anomalous machine sound: fit a Gaussian to normal clips, score by Mahalanobis distance
def machine(anomalous=False, seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    x = sum(np.sin(2 * np.pi * h * 50 * t + rng.uniform(0, 6)) / h for h in range(1, 8)) * 0.2
    if anomalous:                                                   # bearing fault: extra tone and periodic clicks
        x += 0.05 * np.sin(2 * np.pi * 2370 * t) + 0.3 * (np.sin(2 * np.pi * 7 * t) > 0.995) * rng.standard_normal(t.size)
    return x + 0.02 * rng.standard_normal(t.size)


def clip_vec(x):
    m = logmel(x)
    return np.concatenate([m.mean(0), m.std(0)])


normal = np.array([clip_vec(machine()) for _ in range(60)])
m0, cov = normal.mean(0), np.cov(normal.T) + 1e-3 * np.eye(normal.shape[1])
inv = np.linalg.inv(cov)
score = lambda v: float((v - m0) @ inv @ (v - m0))
s_norm = np.array([score(clip_vec(machine())) for _ in range(30)])
s_anom = np.array([score(clip_vec(machine(True))) for _ in range(30)])
auc = (s_anom[:, None] > s_norm[None, :]).mean()
print(f"anomalous sound AUC {auc:.3f}")
```
Reading the output: frame accuracy looks high because background dominates, so judge SED on event-level F1; median filtering removes one-frame flickers that otherwise split or invent events. The anomaly part needs no faulty examples for training, which is why it is the standard starting point for machine monitoring.

## Choosing / trade-offs
- **Zero-shot CLAP vs fine-tuned classifier.** CLAP needs no labels and handles new classes by prompt, but accuracy varies with wording and drops on niche sounds; fine-tuning AST, PANNs or BEATs on a few hundred labelled clips per class is usually much more accurate. Use CLAP to bootstrap labels ([[data-labeling-and-synthetic-data]]).
- **Model size vs device.** YAMNet-class models run on phones and microcontroller-class hardware; AST/BEATs need a GPU or a larger edge device ([[edge-on-device]]). Distil a large teacher into a small student for always-on devices ([[knowledge-distillation-and-compression]]).
- **Tagging vs SED.** If you only need "did it happen in this minute", clip tagging is simpler; timestamps (alerts, counting events, syncing with video) need frame-level SED.
- **Supervised vs unsupervised for machines.** Faults are rare and varied: train on normal sound only, then add supervised classes as real faults get labelled.
- **Licences.** Check checkpoint and dataset licences; AudioSet is distributed as YouTube IDs and labels, and some benchmark datasets are non-commercial ([[model-licenses]]).

## Gotchas
- Sample-rate or feature mismatch with the checkpoint (8 kHz telephone audio into a 16 kHz model, different mel settings) silently ruins accuracy.
- Microphone, room and device shift: a model trained on YouTube audio can fail on a ceiling microphone in a factory; collect validation data from the target devices ([[transfer-learning-and-domain-adaptation]]).
- Split by recording session or device, not by overlapping windows of the same recording ([[data-leakage-and-validation-splits]]).
- Rare critical events (gunshot, glass break) need threshold tuning at very low false-alarm rates; report false alarms per hour, not only F1 ([[imbalanced-data]]).
- Overlapping sounds: multi-label sigmoid outputs, not softmax, when events co-occur.
- Privacy: always-on microphones capture speech; process on-device and discard raw audio when only events are needed, and tell people they are being recorded where the law requires it.
- Use the official folds on ESC-50 and UrbanSound8K; random splits overstate accuracy.

## Related
- [[mel-spectrogram-and-mfcc-from-scratch]] - the features every model here consumes.
- [[voice-activity-detection-from-scratch]] - the speech/non-speech special case.
- [[anomaly-detection]] - unsupervised methods for machine sound.
- [[manufacturing-iot]] - predictive maintenance scenario.
- [[embeddings]] - audio embeddings for search and clustering.
- [[contrastive-learning-infonce-from-scratch-numpy]] - the training objective behind CLAP.
- [[edge-on-device]] - running small audio models on devices.
- [[voice-cloning-and-ethics]] - synthetic speech detection, a specialised audio classifier.
- [[speech-to-speech-realtime-models]] - native audio models that also react to non-speech sounds.

## References
- Gong et al., AST in Transformers: https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer
- AST AudioSet checkpoint: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- CLAP in Transformers: https://huggingface.co/docs/transformers/model_doc/clap
- CLAP checkpoint: https://huggingface.co/laion/clap-htsat-unfused
- Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition": https://ar5iv.labs.arxiv.org/html/1912.10211
- AudioSet: https://research.google.com/audioset/
- ESC-50: https://github.com/karolpiczak/ESC-50
- DCASE community and challenges: https://dcase.community/
