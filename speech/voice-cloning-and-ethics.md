---
title: Voice cloning and its ethics (zero-shot cloning, consent, watermarking, synthetic speech detection)
category: speech
tags: [voice-cloning, zero-shot-tts, deepfake, consent, watermarking, audioseal, anti-spoofing, asvspoof, eer, f5-tts, elevenlabs, ai-act]
use_cases:
  - "add consented voice cloning to a product (branded voice, dubbing, voice restoration)"
  - "design consent, verification and abuse controls for a voice cloning feature"
  - "watermark generated speech so we can later prove it came from our system"
  - "detect deepfake or synthetic speech in calls before a voice-authenticated action"
  - "check the legal and licensing constraints on cloning a voice"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/SWivid/F5-TTS
  - https://elevenlabs.io/docs/cookbooks/voices/professional-voice-cloning
  - https://openai.com/index/navigating-the-challenges-and-opportunities-of-synthetic-voices/
  - https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/
  - https://github.com/facebookresearch/audioseal
  - https://arxiv.org/abs/2408.08739
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://cooley.com/news/insight/2024/2024-02-15-fcc-ai-generated-robocalls-illegal-under-the-tcpa
---

# Voice cloning and its ethics (zero-shot cloning, consent, watermarking, synthetic speech detection)

## Summary
Zero-shot voice cloning makes a text-to-speech model speak in a new voice from a few seconds of reference audio, with no per-speaker training. It enables dubbing, branded voices, accessibility (restoring the voice of someone losing speech) and game/media production, and equally enables fraud: impersonation calls, bypassing voice authentication, and political deepfakes. Responsible use rests on three controls: verified consent from the voice owner, marking generated audio (watermarks and disclosure), and detection of synthetic speech where voice is trusted. Several vendors have held back their strongest cloning models over misuse risk.

## Key concepts
- **Zero-shot vs fine-tuned cloning.** Zero-shot: the model conditions on a reference clip (often plus its transcript) at inference time. Fine-tuned ("professional") cloning trains on longer recordings of one speaker for higher fidelity and stability.
- **Model families.** Neural codec language models (VALL-E style: predict audio-codec tokens conditioned on a prompt), flow-matching/diffusion TTS (F5-TTS: reference audio + reference text + target text), and hosted APIs (ElevenLabs instant and professional cloning).
- **Release restraint.** OpenAI described Voice Engine (one 15-second sample) in March 2024 as a small-scale preview not broadly released because of misuse risk; Microsoft's VALL-E 2 page states it is research only with no plan to release it as a product or to the public.
- **Consent and verification.** Record who consented, to what uses, for how long, and verify the speaker is the person consenting. ElevenLabs' professional voice cloning, for example, has the owner read a CAPTCHA text aloud before training; its instant cloning requires the uploader to confirm rights.
- **Watermarking.** Embed an imperceptible, keyed signal in generated audio and detect it later. AudioSeal (Meta, MIT licence including weights) does sample-level localised watermarking with a fast detector. Watermarks only mark audio from cooperating generators.
- **Synthetic speech detection (anti-spoofing).** Classifiers that score bona fide vs spoofed speech. ASVspoof 5 (2024) uses crowdsourced data, adversarial attacks and codec conditions; its standalone detection track uses minDCF as the primary metric, with EER and Cllr also reported. Its overview reports that detectors degrade under adversarial attacks and neural codecs.
- **Law (examples, not legal advice).** US: the FCC's February 2024 declaratory ruling treats AI-generated voices in robocalls as "artificial" under the TCPA; Tennessee's ELVIS Act (2024) adds voice to right-of-publicity protection. EU: AI Act Article 50 requires providers to mark synthetic audio in a machine-readable way and deployers to disclose deepfakes. Biometric voiceprints can be special-category data under GDPR.

## When to use / scenarios
- Media localisation and dubbing with the original actor's contracted consent; audiobook and course narration in the author's voice.
- Accessibility: voice banking for people with ALS or before laryngectomy, then TTS in their own voice.
- Brand and product voices built from a hired voice actor under a licence that names the uses.
- Defensive: contact centres and banks screening calls for synthetic speech, and moving away from voice as the sole authentication factor ([[security-defensive]]).
- Not for: cloning anyone without documented consent (public figures included), voices of the deceased without estate permission, or any political or financial impersonation. For a generic natural voice, use a stock TTS voice instead ([[text-to-speech]]).

## Setup & code
Open zero-shot cloning (F5-TTS; code MIT, pretrained weights CC-BY-NC because of the Emilia training data, so not for commercial use):
```bash
pip install f5-tts            # needs FFmpeg and a PyTorch build for your hardware
f5-tts_infer-cli --model F5TTS_v1_Base \
  --ref_audio consented_sample.wav --ref_text "Exact transcript of the sample." \
  --gen_text "Text to speak in the consented voice."
```
Watermark generated audio with AudioSeal before it leaves your system (16 kHz audio):
<!-- skip-check: needs torch and a model download -->
```python
import torch
from audioseal import AudioSeal

wav = torch.randn(1, 1, 16000)                      # (batch, channels, samples): your generated 16 kHz audio
generator = AudioSeal.load_generator("audioseal_wm_16bits")
watermarked = wav + generator.get_watermark(wav)
detector = AudioSeal.load_detector("audioseal_detector_16bits")
prob, message = detector.detect_watermark(watermarked)
print(prob, message)                               # probability watermarked, 16-bit payload
```

Self-contained NumPy demo of the two safeguards: a keyed spread-spectrum watermark (the principle behind classic audio watermarks), tested against noise and filtering with its false-positive rate on clean speech, and the EER metric used to report synthetic-speech detectors:
```python
import numpy as np

rng = np.random.default_rng(7)
SR = 16000


def fake_speech(seconds=2.0):
    """Speech-like test signal: harmonic voiced segments with a syllable-rate envelope plus breath noise."""
    t = np.arange(int(SR * seconds)) / SR
    f0 = rng.uniform(90, 220) + 20 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6)), 0, None) ** 2
    return 0.3 * env * voiced + 0.01 * rng.standard_normal(t.size)


def pn_sequence(key, n):
    return np.random.default_rng(key).choice([-1.0, 1.0], n)


def embed(x, key, strength=0.005):
    """Add a keyed pseudo-noise sequence, scaled to the signal's local loudness so it hides under speech."""
    frame = 400
    loud = np.sqrt(np.convolve(x ** 2, np.ones(frame) / frame, mode="same"))
    return x + strength * (loud / (loud.max() + 1e-9) + 0.05) * pn_sequence(key, x.size)


def detect(y, key):
    """Normalised correlation with the keyed sequence; about N(0,1) for unmarked audio."""
    p = pn_sequence(key, y.size)
    y = y - np.convolve(y, np.ones(9) / 9, mode="same")       # whiten: remove low-frequency speech energy
    return float(np.dot(y, p) / (np.linalg.norm(y) + 1e-12))


def lowpass(y, taps=5):
    return np.convolve(y, np.ones(taps) / taps, mode="same")


KEY, THRESH = 1234, 4.0                                   # z-score threshold; about 3e-5 false-positive rate if N(0,1)
x = fake_speech()
wm = embed(x, KEY)
snr = 10 * np.log10((x ** 2).sum() / ((wm - x) ** 2).sum())
print(f"watermark SNR {snr:.1f} dB (higher = quieter watermark)")
attacks = {"none": lambda y: y,
           "noise 30 dB": lambda y: y + rng.normal(0, np.sqrt((y ** 2).mean() / 1e3), y.size),
           "lowpass": lowpass,
           "wrong key check": None}
for name, f in attacks.items():
    z = detect(wm, 999) if f is None else detect(f(wm), KEY)
    print(f"{name:16s} z = {z:6.1f}  -> {'WATERMARKED' if z > THRESH else 'not detected'}")
clean_z = np.array([detect(fake_speech(), KEY) for _ in range(200)])
print(f"false positives on 200 clean clips: {(clean_z > THRESH).sum()}  (mean z {clean_z.mean():+.2f}, sd {clean_z.std():.2f})")


def eer(bona_scores, spoof_scores):
    """Equal error rate: threshold where false acceptance of spoofs equals false rejection of genuine speech."""
    th = np.sort(np.concatenate([bona_scores, spoof_scores]))
    far = np.array([(spoof_scores >= t).mean() for t in th])
    frr = np.array([(bona_scores < t).mean() for t in th])
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2


# simulated detector scores (higher = more likely genuine): good on seen attacks, worse after a codec it never saw
bona = rng.normal(2.0, 1.0, 2000)
seen = rng.normal(-2.0, 1.0, 2000)
codec = rng.normal(0.5, 1.2, 2000)
print(f"EER seen attack {eer(bona, seen):.1%}   EER after unseen codec {eer(bona, codec):.1%}")
```
Reading the output: the watermark survives added noise, a wrong key finds nothing, and clean clips score about N(0, 1), so the threshold controls false positives. A simple low-pass filter, though, erases this naive watermark, because the detector discards low frequencies (to suppress speech) and so relies on the high-frequency part of the mark that the filter removes; robust schemes shape the mark into the bands that survive codecs and filtering. The EER part shows the pattern ASVspoof reports: a detector that looks strong on known attacks can be much weaker on new generators or codecs, so test on conditions the detector has not seen. This toy watermark is not robust to filtering or deliberate removal; use a maintained scheme such as AudioSeal plus provenance metadata.

## Choosing / trade-offs
- **Hosted API vs open model.** Hosted services bundle consent flows, verification and abuse monitoring but send voice data to a vendor; open models run privately but you must build those controls yourself. Check weight licences: many open cloning models are non-commercial (F5-TTS weights are CC-BY-NC).
- **Zero-shot vs professional cloning.** Zero-shot is instant and lower fidelity; fine-tuned voices sound closer and more stable but need more consented audio and stronger identity verification.
- **Watermark vs detector.** Watermarks are reliable for audio your own system made and useless for audio from others; passive detectors cover unknown generators but have error rates that grow with new models and codecs. Use both, plus provenance metadata (C2PA-style) where the format allows.
- **Voice authentication.** Treat voice biometrics as one weak factor; add liveness checks, out-of-band confirmation and transaction limits rather than relying on a deepfake detector alone.

## Gotchas
- Consent must cover the specific use, duration and revocation; keep the signed record and delete the voice model when consent is withdrawn.
- "Public" audio (podcasts, videos) is not consent to clone.
- Reference clips leak identity; store them as biometric data with access control and retention limits ([[ai-security-privacy-compliance]]).
- Watermarks are weakened by re-recording over the air, heavy compression, pitch shifting or regeneration; detection results are evidence, not proof.
- Detector scores do not transfer across languages, channels (telephone vs studio) and codecs; calibrate on your traffic ([[probability-calibration]]).
- Block prompts that request a specific real person's voice and rate-limit cloning endpoints; log who cloned what ([[guardrails-and-safety]]).
- Disclose synthetic voices to listeners where law or platform policy requires it, including in voice agents ([[voice-agents]]).

## Related
- [[text-to-speech]] - the synthesis models that cloning extends.
- [[speaker-diarization]] - speaker embeddings, the same representation used for similarity and verification.
- [[voice-agents]] - disclosure and consent in live calls.
- [[speech-to-speech-realtime-models]] - native audio models that can also imitate voices.
- [[guardrails-and-safety]] - abuse prevention and policy enforcement.
- [[ai-security-privacy-compliance]] - biometric data handling and vendor contracts.
- [[security-defensive]] - fraud and deepfake defence scenario.
- [[audio-classification-and-sound-events]] - audio classifiers, the base of spoof detectors.

## References
- F5-TTS: https://github.com/SWivid/F5-TTS
- ElevenLabs professional voice cloning (verification): https://elevenlabs.io/docs/cookbooks/voices/professional-voice-cloning
- OpenAI, "Navigating the Challenges and Opportunities of Synthetic Voices" (2024): https://openai.com/index/navigating-the-challenges-and-opportunities-of-synthetic-voices/
- Microsoft VALL-E 2 project page: https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/
- AudioSeal: https://github.com/facebookresearch/audioseal
- Wang et al., "ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale" (2024): https://arxiv.org/abs/2408.08739
- Regulation (EU) 2024/1689 (AI Act), Article 50: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- FCC declaratory ruling on AI voices in robocalls (FCC 24-17), law-firm summary: https://cooley.com/news/insight/2024/2024-02-15-fcc-ai-generated-robocalls-illegal-under-the-tcpa
