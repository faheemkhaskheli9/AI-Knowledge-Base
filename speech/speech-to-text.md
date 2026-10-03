---
title: Speech-to-text (ASR)
category: speech
tags: [asr, stt, whisper, faster-whisper, transcription, streaming, vad, subtitles]
use_cases:
  - "transcribe meetings, interviews or calls into text with timestamps"
  - "generate subtitles for videos in many languages"
  - "run speech recognition locally for privacy-sensitive audio"
  - "add real-time streaming transcription to an app"
  - "transcribe medical or call-centre audio with domain vocabulary"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/SYSTRAN/faster-whisper
  - https://github.com/openai/whisper
  - https://platform.openai.com/docs/guides/speech-to-text
---

# Speech-to-text (ASR)

## Summary
Automatic speech recognition converts audio to text, optionally with word timestamps, language detection and translation. Whisper-family open models are the default for self-hosted batch transcription (faster-whisper is a CTranslate2 reimplementation reported as up to 4x faster); hosted APIs (OpenAI, Deepgram, AssemblyAI, Google, Azure, ElevenLabs and others) add streaming, diarization and managed scale.

## Key concepts
- Batch (file) vs streaming (partial results with low latency); Whisper is natively a 30-second-window batch model, so true streaming needs chunking or a streaming-first engine.
- Model size trades speed and VRAM for accuracy: tiny to large; `distil-large-v3` and `turbo` variants are faster (faster-whisper lists both).
- VAD (voice activity detection) trims silence and reduces hallucinated text; faster-whisper supports a VAD filter.
- Timestamps at segment or word level; needed for subtitles and alignment (see [[speaker-diarization]]).
- Language ID and translate-to-English tasks are built into Whisper; accuracy varies strongly by language and accent.
- Domain biasing: initial prompt / hotwords, or fine-tuning ([[fine-tuning-and-peft]]) for jargon.
- Metric: word error rate (WER), measured on your own audio.

## When to use / scenarios
- Meetings and interviews: transcript plus summary by an LLM; add diarization for who-said-what.
- Media: subtitles and captions (SRT/VTT) in multiple languages.
- Call centres: transcribe calls for QA and analytics; compliance review for recording consent.
- Healthcare dictation and legal: use local or BAA/DPA-covered services; review accuracy of drug and legal terms.
- Voice interfaces: feed STT into an LLM ([[voice-agents]]).
- Not for: music transcription or sound-event detection (different models); text in images ([[ocr]]).

## Setup & code
```bash
pip install faster-whisper     # GPU needs NVIDIA cuBLAS and cuDNN 9 for CUDA 12 per the project README
```
```python
from faster_whisper import WhisperModel

# CPU fallback: device="cpu", compute_type="int8"
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe("meeting.mp3", beam_size=5, vad_filter=True)
print(info.language, info.language_probability)
for s in segments:                       # generator: transcription runs while iterating
    print(f"[{s.start:.2f} -> {s.end:.2f}] {s.text}")
```
Hosted alternative: OpenAI audio transcription endpoint (see its speech-to-text guide for current model names and file size limits).

## Choosing / trade-offs
- Local Whisper (faster-whisper): private, fixed cost, strong multilingual; needs a GPU for large models (CPU with int8 works for small ones), no built-in diarization.
- Hosted API: simplest, streaming and diarization options, per-minute cost, audio leaves your network.
- Streaming latency: vendors with native streaming (Deepgram, AssemblyAI, etc.) beat chunked Whisper for live captions.
- Accuracy vs speed: large models for batch quality; small/turbo for real time.
- Low-resource languages and dialects (e.g. Urdu, Arabic dialects): benchmark several engines; consider fine-tuning.
- Cost estimate: self-host when audio hours are high and steady; API when spiky.

## Gotchas
- Whisper can hallucinate text on silence/noise ("thanks for watching"); use VAD and drop low-confidence segments.
- Long files: let the library chunk, and process the generator once (it is lazy).
- Sample rate and channels: models expect 16 kHz mono; resample stereo call audio and consider channel-per-speaker recordings.
- Numbers, names and acronyms mis-transcribed; post-correct with a glossary or LLM, but do not let an LLM invent content.
- Code-switching (mixed languages) is weak; set the language explicitly if known.
- Recording consent and retention laws apply to call audio; voice is personal data.
- CUDA library mismatches are the usual install failure ([[gpu-cuda-setup]]).

## Related
- [[speaker-diarization]] - who spoke when.
- [[voice-agents]] - real-time pipelines.
- [[text-to-speech]] - the reverse direction.
- [[huggingface-transformers]] - Whisper via transformers.
- [[customer-support]] - call analytics scenario.

## References
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/openai/whisper
- https://platform.openai.com/docs/guides/speech-to-text
