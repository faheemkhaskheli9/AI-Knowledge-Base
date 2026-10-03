---
title: Text-to-speech (TTS)
category: speech
tags: [tts, speech-synthesis, kokoro, voice-cloning, streaming, elevenlabs, piper]
use_cases:
  - "read articles or documents aloud in a natural voice"
  - "give a chatbot or voice agent a low-latency spoken voice"
  - "generate narration or voice-over for videos and courses"
  - "run offline text-to-speech on a laptop or edge device"
  - "clone a voice with consent for a branded assistant"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/hexgrad/kokoro
  - https://github.com/rhasspy/piper
  - https://platform.openai.com/docs/guides/text-to-speech
  - https://elevenlabs.io/docs
---

# Text-to-speech (TTS)

## Summary
TTS turns text into spoken audio. Hosted APIs (OpenAI, ElevenLabs, Cartesia, Google, Azure, Amazon Polly and others) give the most natural voices, streaming and voice cloning; open models (Kokoro, Piper, XTTS-style and other community models) run locally and cheaply. For conversational agents the key metric is time-to-first-audio, not just quality.

## Key concepts
- Streaming synthesis: audio chunks start before the full text is rendered; essential for voice agents ([[voice-agents]]).
- Voice selection, speaking rate, style/emotion controls; SSML supported by some cloud services for pauses and pronunciation.
- Voice cloning: reproduces a voice from a short sample; high misuse risk, needs documented consent.
- Phonemisation and text normalisation: numbers, dates, currencies and abbreviations must be expanded correctly ("$3.50", "Dr.").
- Audio output formats and sample rates (24 kHz common, PCM/MP3/Opus); telephony needs 8 kHz mu-law.
- Evaluation: human MOS-style listening tests, intelligibility (re-transcribe with [[speech-to-text]]), latency to first byte.

## When to use / scenarios
- Accessibility and content: audio versions of articles, e-learning narration, audiobooks (check licence of the voice).
- Contact centre and IVR: spoken answers from an LLM; low latency and telephony formats matter.
- Assistants/devices: offline TTS for privacy or no connectivity.
- Media/localisation: dubbing pipelines (STT -> translate -> TTS), with voice-actor and consent considerations.
- Not for: music generation; real-person impersonation.

## Setup & code
Open model, local (Kokoro, 82M parameters, Apache-2.0 per its repo):
```bash
pip install kokoro soundfile      # also needs espeak-ng installed on the system; see the repo README
```
```python
import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")                     # "a" = American English
for i, (graphemes, phonemes, audio) in enumerate(pipeline("Hello from the knowledge base.", voice="af_heart")):
    sf.write(f"out_{i}.wav", audio, 24000)
```
Hosted: use the provider SDK (for example the OpenAI speech endpoint) and request streamed output; see each provider's docs for current model and voice names.

## Choosing / trade-offs
- Quality and expressiveness: commercial APIs lead; open models are close for plain narration in major languages.
- Latency: Cartesia/ElevenLabs/OpenAI-style streaming offer low time-to-first-audio; local small models (Kokoro, Piper) are fast on a GPU or even CPU.
- Cost: per-character API pricing vs fixed local compute; high volume favours local.
- Languages: coverage and quality for Urdu, Arabic, Hindi and other languages vary; audition voices on your text.
- Licensing: check model and voice licences for commercial use; some open voices restrict it.
- Privacy: text sent to a cloud TTS is disclosed to the vendor.

## Gotchas
- Mispronounced names, acronyms and numbers; use lexicons/SSML or text normalisation before synthesis.
- Long inputs: chunk by sentence for latency and to avoid drift; stitch with short pauses.
- Voice cloning without explicit, recorded consent is illegal or against terms in many jurisdictions; disclose synthetic voice where required and consider watermarking.
- Telephony pipelines silently degrade quality when resampled; generate at the target rate.
- Barge-in: if the user interrupts, you must stop playback and flush queued audio.
- Rate limits and concurrency caps on APIs bite during spikes; pre-generate static prompts and cache.
- Model/voice names change between releases; confirm in the docs.

## Related
- [[voice-agents]] - TTS in a real-time loop.
- [[speech-to-text]] - verifying intelligibility, dubbing pipelines.
- [[edge-on-device]] - local TTS deployment.
- [[guardrails-and-safety]] - misuse prevention for voice cloning.
- [[education]] - narration and accessibility scenarios.

## References
- https://github.com/hexgrad/kokoro
- https://github.com/rhasspy/piper
- https://platform.openai.com/docs/guides/text-to-speech
- https://elevenlabs.io/docs
