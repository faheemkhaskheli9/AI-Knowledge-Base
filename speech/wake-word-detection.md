---
title: Wake word detection (keyword spotting)
category: speech
tags: [wake-word, keyword-spotting, kws, hotword, openwakeword, microwakeword, porcupine, on-device, false-accept, false-reject, voice-assistant]
use_cases:
  - "add a 'hey <name>' wake word to a voice assistant or smart-home device"
  - "detect a few spoken commands on a microcontroller without the cloud"
  - "tune a wake word threshold to trade false activations against missed activations"
  - "train a custom wake word without recording thousands of real samples"
  - "gate an always-listening app so audio only leaves the device after the wake word"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/dscripka/openWakeWord
  - https://github.com/kahrendt/microWakeWord
  - https://github.com/Picovoice/porcupine
  - https://github.com/Picovoice/wake-word-benchmark
  - https://arxiv.org/abs/1804.03209
  - https://arxiv.org/abs/2005.06720
---

# Wake word detection (keyword spotting)

## Summary
Wake word detection, or keyword spotting (KWS), is a tiny always-on classifier that listens to a microphone stream and fires when a specific phrase ("hey jarvis", "alexa") is spoken. Only after it fires does the device start full speech recognition or stream audio to a server. The engineering problem is mostly operating-point tuning: trading **false accepts** (waking by mistake, measured per hour of non-wake audio) against **false rejects** (missing a real wake word, as a percentage of attempts) within a CPU, memory and battery budget. Open options are openWakeWord (Python/ONNX/TFLite on Linux-class devices), microWakeWord (TFLite Micro on ESP32-class microcontrollers) and Picovoice Porcupine (commercial SDK, needs an AccessKey).

## Key concepts
- **Streaming inference**: audio arrives in short frames (openWakeWord uses 80 ms frames of 16 kHz 16-bit PCM). The model emits a score per frame, and a detection is a score above a threshold.
- **False-reject rate (FRR)**: the share of intended wake words that are missed. Measure it on realistic recordings with noise and far-field reverberation, not clean close-talk clips.
- **False-accept rate (FAR)**: unintended activations per hour of real background audio (speech, TV, music). openWakeWord targets below 5% FRR and below 0.5 false accepts per hour, and measures FAR on the Dinner Party Corpus (about 5.5 hours of far-field audio).
- **Operating point**: a single threshold picks one point on the FRR-vs-FA/hour curve. Choose it on a held-out set from the deployment environment.
- **Post-processing**: require N consecutive frames above the threshold (microWakeWord does this), add a refractory period after each detection, and optionally gate on a VAD score. All three cut false accepts cheaply.
- **Two-stage detection**: a cheap always-on detector, then a heavier verifier. The verifier can be a speaker-specific model (openWakeWord "custom verifier models") or server-side ASR that confirms the phrase.
- **Synthetic training data**: openWakeWord and microWakeWord train custom words mostly on TTS-generated positives plus large negative sets (openWakeWord's bundled models used about 30,000 hours of negative audio). Negatives matter as much as positives.
- **Classic baseline**: the Google Speech Commands dataset (35 short words) is the standard small-vocabulary KWS benchmark.
- **Phrase design**: 3-4 syllables with distinct phonemes work better. Short or common words ("hey", "ok") collide with everyday speech.

## When to use / scenarios
- Smart-home and voice assistants: a local wake word before streaming audio to an STT or LLM backend ([[voice-agents]]), for privacy and bandwidth.
- In-car, kiosk and wearable devices: hands-free activation with offline operation.
- Industrial and medical devices: a few fixed commands ("start", "stop") where a cloud round trip is unacceptable ([[manufacturing-iot]]).
- Accessibility: hands-free control for users with limited mobility.
- Not for: open-vocabulary commands (use streaming [[speech-to-text]] after the wake word), speaker authentication (a security task with its own threat model), or long-audio keyword search (run ASR and search the transcript).

## Setup & code
```bash
pip install openwakeword        # code Apache-2.0; bundled pretrained models are CC BY-NC-SA 4.0
```
The self-contained numpy block below shows the core tuning loop for any engine. It takes per-frame scores, applies smoothing, a consecutive-frames rule and a refractory period, then sweeps the threshold to find the lowest false-reject rate that meets a false-accept budget. Synthetic scores stand in for a real model's output.
```python
import numpy as np

rng = np.random.default_rng(0)
FRAME_S = 0.08                                    # 80 ms frames, as in openWakeWord

def detections(scores, thr, n_consec=2, refractory_s=2.0):
    """Indices where n_consec frames in a row exceed thr, with a refractory gap."""
    above = scores > thr
    run, last, out = 0, -10**9, []
    gap = int(refractory_s / FRAME_S)
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= n_consec and i - last > gap:
            out.append(i)
            last = i
    return out

# Positives: 300 clips of 1.5 s; a real utterance gives a bump of high scores mid-clip.
def positive_clip():
    s = rng.beta(1, 12, 19)                       # background frames: low scores
    peak = rng.beta(8, 2)                         # some utterances are weak (far-field, accent)
    s[8:12] = np.clip(peak + 0.05 * rng.standard_normal(4), 0, 1)
    return s

positives = [positive_clip() for _ in range(300)]
# Negatives: 10 hours of continuous non-wake audio with rare confusable spikes.
neg_hours = 10.0
neg = rng.beta(1, 15, int(neg_hours * 3600 / FRAME_S))
spikes = rng.choice(len(neg) - 3, 40, replace=False)
for j in spikes:
    neg[j:j + rng.integers(1, 4)] = rng.uniform(0.3, 0.8)   # 1-3 frame bursts

def evaluate(thr):
    frr = np.mean([len(detections(p, thr)) == 0 for p in positives])
    fa_per_hour = len(detections(neg, thr)) / neg_hours
    return frr, fa_per_hour

FA_BUDGET = 0.5                                   # false accepts per hour
rows = [(t, *evaluate(t)) for t in np.arange(0.30, 0.95, 0.05)]
for t, frr, fa in rows:
    print(f"thr={t:.2f}  FRR={frr:6.1%}  FA/h={fa:5.2f}")
ok = [(frr, t, fa) for t, frr, fa in rows if fa <= FA_BUDGET]
frr, thr, fa = min(ok)
print(f"chosen threshold {thr:.2f}: FRR {frr:.1%}, {fa:.2f} false accepts/hour")
assert fa <= FA_BUDGET and frr < 0.5
```
The sweep shows the trade-off. Each step down in false accepts per hour costs missed wake words, so a threshold only picks a point on the curve. To move the whole curve, you need better training data, noise suppression, a VAD gate or a second-stage verifier.

Running a pretrained openWakeWord model on microphone or file audio:
<!-- skip-check: downloads models and needs 16 kHz audio input -->
```python
import numpy as np, openwakeword
from openwakeword.model import Model

openwakeword.utils.download_models()                       # one-time download of the bundled models
model = Model(wakeword_models=["hey_jarvis"], vad_threshold=0.5)   # Silero VAD gate cuts noise false accepts
frame = np.zeros(1280, dtype=np.int16)                     # 80 ms of 16 kHz 16-bit PCM from your mic stream
scores = model.predict(frame)                              # {"hey_jarvis": 0.0...1.0}
if max(scores.values()) > 0.5:                             # default threshold; tune per deployment
    print("wake word detected")
```
For ESP32-class microcontrollers, use microWakeWord (TFLite Micro models, used by ESPHome). For a commercial SDK with custom keywords on many platforms, `pip install pvporcupine` and create a detector with `pvporcupine.create(access_key=..., keywords=[...])`. Porcupine requires a Picovoice AccessKey.

## Choosing / trade-offs
| Engine | Runs on | Custom words | Licence / cost | Notes |
|---|---|---|---|---|
| openWakeWord | Linux, Raspberry Pi class, desktops (ONNX/TFLite) | Train from synthetic TTS data (Colab notebook) | Code Apache-2.0; bundled models CC BY-NC-SA 4.0 | Multiple wake words share one feature backbone; optional Speex noise suppression and VAD |
| microWakeWord | Microcontrollers (ESP32-S3 class) via TFLite Micro | Train yourself; author says training a good model is still hard | Open source | The openWakeWord author recommends it over openWakeWord for very low-power devices |
| Picovoice Porcupine | Linux, macOS, Windows, Raspberry Pi, Android, iOS, web, MCUs | Custom keywords via Picovoice Console | Commercial, AccessKey required | Polished SDKs; publishes an open wake-word benchmark |
| Platform assistants (Siri, Google, Alexa) | Their own devices | Fixed phrases | Platform terms | Use when you integrate with the platform rather than ship your own word |

- FRR vs FAR: households tolerate a missed wake (users repeat it) better than a device that wakes during a film. Bias toward a low FAR unless missing a command is costly (accessibility, safety).
- Model size vs accuracy: microcontrollers force tiny streaming CNNs. A Pi-class device can run openWakeWord plus a verifier.
- Custom phrase cost: synthetic-data pipelines make a new word cheap to try. Real recordings from target users (accents, children, far-field) still improve FRR the most.
- Privacy: on-device detection means no audio leaves the device before the wake word. Make that an explicit, tested property, and show a visible indicator when streaming.

## Gotchas
- Thresholds tuned on clean or close-mic audio fail in the living room. Evaluate with noise and reverberation at realistic SNRs (openWakeWord uses 5-10 dB plus room impulse responses).
- Measure FAR on hours of the actual background (TV, music, kitchen noise, other languages). A few minutes of negatives tells you nothing about the per-hour rate.
- TV adverts and the device's own TTS can trigger it. Mute or raise the threshold while the device is speaking, and use acoustic echo cancellation.
- Frame size and sample rate must match the model (16 kHz mono 16-bit for openWakeWord). A wrong rate gives silently bad scores, not an error.
- Bundled openWakeWord models are non-commercial (CC BY-NC-SA 4.0). Train your own model for a commercial product.
- Children's voices, strong accents and whispered speech have much higher FRR. Include them in the test set.
- A refractory period hides double triggers but also hides a quick repeat after a miss. Keep it short.
- Report FRR and FA/hour together, at the chosen threshold. Accuracy or AUC alone hides the operating point.

## Related
- [[voice-activity-detection-from-scratch]] - VAD gating and frame-based streaming detection.
- [[mel-spectrogram-and-mfcc-from-scratch]] - the input features of KWS models.
- [[voice-agents]] - what runs after the wake word fires.
- [[speech-to-text]] - open-vocabulary recognition after activation.
- [[edge-on-device]] - TFLite, ONNX and microcontroller deployment.
- [[classification-metrics-and-cross-validation-from-scratch]] - threshold selection and error trade-offs.
- [[imbalanced-data]] - rare positives among hours of negatives.
- [[music-information-retrieval]], [[speech-translation]] - other speech-category tasks in this batch.

## References
- openWakeWord (targets, evaluation method, licences): https://github.com/dscripka/openWakeWord
- microWakeWord: https://github.com/kahrendt/microWakeWord
- Picovoice Porcupine: https://github.com/Picovoice/porcupine; wake word benchmark https://github.com/Picovoice/wake-word-benchmark
- Warden (2018), Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition: https://arxiv.org/abs/1804.03209
- Rybakov et al. (2020), Streaming Keyword Spotting on Mobile Devices: https://arxiv.org/abs/2005.06720
