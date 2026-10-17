---
title: Speech translation
category: speech
tags: [speech-translation, s2tt, s2st, ast, cascade, end-to-end, whisper, seamlessm4t, canary, simultaneous-translation, dubbing, multilingual]
use_cases:
  - "translate spoken audio in one language into subtitles in another"
  - "build live interpretation for meetings or events"
  - "dub a video into other languages with translated speech"
  - "let a support agent understand callers who speak other languages"
  - "choose between a cascade (ASR + MT + TTS) and a direct speech translation model"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/openai/whisper
  - https://github.com/SYSTRAN/faster-whisper
  - https://github.com/facebookresearch/seamless_communication
  - https://huggingface.co/nvidia/canary-1b-v2
  - https://arxiv.org/abs/2509.14128
  - https://arxiv.org/abs/2312.05187
  - https://arxiv.org/abs/1810.08398
---

# Speech translation

## Summary
Speech translation turns speech in one language into text (S2TT, also called AST) or speech (S2ST) in another. There are two ways to build it. A **cascade** chains speech-to-text, then text machine translation (MT) or an LLM, then optionally text-to-speech. A **direct** (end-to-end) model maps source speech straight to target text or speech. Open direct options include Whisper's translate task (into English only), NVIDIA Canary-1B-v2 (25 European languages, to and from English) and Meta's SeamlessM4T v2 (around 100 languages, speech and text in and out). Cascades remain the most flexible and debuggable default.

## Key concepts
- **Task names**: ASR (speech to same-language text), S2TT/AST (speech to text in another language), S2ST (speech to speech), T2TT (text MT).
- **Cascade**: ASR, then MT or an LLM, then TTS. Each stage is swappable and inspectable, and the transcript is a free by-product. Errors compound, though: an ASR mistake becomes a confident mistranslation. Prosody and speaker voice are lost unless the TTS clones them.
- **Direct / end-to-end**: one model and lower latency. It avoids compounding errors and can carry paralinguistic cues. The cost is fewer language pairs, harder debugging and no editable intermediate transcript (unless the model also emits one).
- **Language direction coverage**: Whisper translates any supported language into English only. Canary-1B-v2 covers English to 24 languages and 24 languages to English, all European. SeamlessM4T v2 covers around 100 languages, with fewer for speech output than for speech input.
- **Simultaneous (streaming) translation**: output starts before the speaker finishes. A **read/write policy** decides when to emit. *Wait-k* reads k source units first, then alternates one write per read. SeamlessStreaming uses a learned monotonic-attention policy.
- **Latency metric**: Average Lagging (AL, from the STACL paper) measures how far behind an ideal real-time interpreter the system runs, in source units or seconds.
- **Quality metrics**: BLEU or chrF on text output ([[bleu-and-rouge-from-scratch]]), learned metrics such as COMET, and ASR-BLEU for speech output (transcribe the output speech, then score it). Human review is still the reference for anything customer-facing.
- **Segmentation**: long audio must be cut into sentence-like segments, for example with VAD ([[voice-activity-detection-from-scratch]]). Bad cuts split sentences and hurt translation more than they hurt ASR.

## When to use / scenarios
- Media and e-learning: translated subtitles (S2TT) for video libraries, then optional dubbing (S2ST or cascade with TTS).
- Customer support and contact centres: live agent assist showing translated caller text. A cascade fits better here because you also need the transcript for QA ([[customer-support]]).
- Meetings and events: live captions in several target languages. Pick a simultaneous model or chunked cascade, and tune the latency/quality trade-off.
- Healthcare and legal interpreting: AI can draft, but certified human interpreters remain required in many settings. Use it as an aid with human review ([[healthcare]], [[legal]]).
- Field research and journalism: translate interviews in low-resource languages, then verify with native speakers.
- Not for: translating text you already have (use MT or an LLM directly), or same-language transcription ([[speech-to-text]]).

## Setup & code
The core decision in simultaneous translation is latency. This self-contained block implements a wait-k read/write schedule and computes Average Lagging (Ma et al., 2019): `AL = (1/tau) * sum_{i=1..tau} [ g(i) - (i-1)/r ]`. Here `g(i)` is the number of source words read before writing target word i, `r = |target|/|source|`, and `tau` is the first target index written after the whole source has been read.
```python
def wait_k_schedule(n_src, n_tgt, k):
    """g(i) for i = 1..n_tgt: source units read before emitting target unit i."""
    return [min(k + i - 1, n_src) for i in range(1, n_tgt + 1)]

def average_lagging(g, n_src, n_tgt):
    r = n_tgt / n_src
    tau = next((i for i, gi in enumerate(g, start=1) if gi >= n_src), len(g))
    return sum(g[i - 1] - (i - 1) / r for i in range(1, tau + 1)) / tau

n_src, n_tgt = 20, 22                       # e.g. a 20-word sentence -> 22-word translation
offline = [n_src] * n_tgt                   # cascade on a full sentence: wait for the end
print(f"offline (full sentence): AL = {average_lagging(offline, n_src, n_tgt):.1f} source words")
for k in (1, 3, 5, 9):
    g = wait_k_schedule(n_src, n_tgt, k)
    print(f"wait-{k}: AL = {average_lagging(g, n_src, n_tgt):.2f} source words")

# With equal lengths, wait-k lags by exactly k words, and offline lags by the full source length
assert abs(average_lagging(wait_k_schedule(10, 10, 3), 10, 10) - 3) < 1e-9
assert average_lagging([10] * 10, 10, 10) == 10
```
Lower k means lower lag. It also means the model must commit to target words before seeing the verb, which hurts quality for language pairs with different word order (for example SOV to SVO).

Direct S2TT into English with Whisper (faster-whisper, local GPU or CPU):
<!-- skip-check: downloads a Whisper model and needs an audio file -->
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")   # CPU: device="cpu", compute_type="int8"
segments, info = model.transcribe("interview_ur.mp3", task="translate", vad_filter=True)
print("source language:", info.language)
for s in segments:                                   # English text, with timestamps for subtitles
    print(f"[{s.start:.1f}-{s.end:.1f}] {s.text}")
```
SeamlessM4T v2 (any of its supported directions, text or speech output) via Hugging Face Transformers:
<!-- skip-check: needs torch, a multi-GB model download and an audio file -->
```python
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2Model

proc = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
wav, sr = torchaudio.load("speech_en.wav")
wav = torchaudio.functional.resample(wav, sr, 16_000)
inputs = proc(audio=wav, sampling_rate=16_000, return_tensors="pt")   # transformers 5.x keyword
tokens = model.generate(**inputs, tgt_lang="fra", generate_speech=False)   # S2TT to French
print(proc.decode(tokens[0].tolist()[0], skip_special_tokens=True))
audio = model.generate(**inputs, tgt_lang="spa")[0].cpu().numpy().squeeze()  # S2ST to Spanish (16 kHz)
```
For a cascade, run [[speech-to-text]], then translate each segment with an LLM or MT model, giving it the previous segments as context and a glossary. Optionally finish with [[text-to-speech]].

## Choosing / trade-offs
| Option | Directions | Strengths | Limits |
|---|---|---|---|
| Cascade: ASR + LLM/MT (+ TTS) | Any pair your ASR and MT cover | Editable transcript, glossary control, best-in-class per stage, easy to debug | Higher latency, compounding errors, voice and prosody lost |
| Whisper `task="translate"` | Any Whisper language into English only | One local model, subtitles with timestamps | No other targets; translation quality below its transcription quality |
| NVIDIA Canary-1B-v2 | English to and from 24 European languages | CC-BY-4.0, ASR + AST in one model, NeMo toolkit | European languages only |
| SeamlessM4T v2 / SeamlessStreaming | Around 100 languages; speech output for a subset | S2ST, streaming variant, one model for many pairs | CC-BY-NC 4.0 (non-commercial); large |
| Hosted APIs (cloud speech translation, realtime voice models) | Vendor-specific | Managed streaming, scale | Per-minute cost, audio leaves your network; check each vendor's current language list |

- Licence decides many projects: SeamlessM4T and SeamlessStreaming weights are CC-BY-NC 4.0, so commercial products need a cascade, Canary, Whisper (MIT) or a commercial API.
- Latency budget: live interpretation needs a streaming policy (wait-k or learned). Offline subtitles can wait for full sentences and use the strongest models.
- Low-resource languages and dialects (Urdu, Punjabi, Sindhi, Arabic dialects): test both cascade and direct options on your own audio. Coverage claims at the language level hide large quality differences.
- Need a transcript anyway (compliance, search, QA)? Then a cascade costs little extra.

## Gotchas
- Whisper's translate task only outputs English. Asking it to translate into another language does not work reliably.
- Named entities, numbers and domain terms get mistranslated. Pass a glossary to the MT/LLM stage, or post-edit.
- LLM translators may "fix", summarise or drop disfluent speech. Instruct them to translate faithfully, and spot-check against the source.
- Sentence segmentation errors (cutting mid-clause) hurt badly. Segment on VAD pauses with a maximum length, and give the translator overlapping context.
- Simultaneous systems may need to revise earlier output. Either allow re-translation (flicker) or accept fixed output (lower quality), and make the choice explicit in the UI.
- Code-switching (for example Urdu-English) confuses language ID. Set the source language when you know it.
- Voice-preserving S2ST is voice cloning: get consent, and see the misuse controls in [[text-to-speech]].
- Evaluate with a metric suited to the language. BLEU penalises valid paraphrases and is unreliable for morphologically rich languages, so add chrF/COMET and human review.

## Related
- [[speech-to-text]] - the ASR stage of a cascade and Whisper basics.
- [[text-to-speech]] - the final stage for dubbing and S2ST cascades.
- [[voice-agents]] - real-time pipelines and latency budgets.
- [[sequence-to-sequence-and-ctc]] - the encoder-decoder models behind direct translation.
- [[bleu-and-rouge-from-scratch]] - text translation metrics.
- [[voice-activity-detection-from-scratch]] - segmenting audio before translation.
- [[customer-support]] - multilingual call centre scenario.
- [[music-information-retrieval]], [[wake-word-detection]] - other speech-category tasks in this batch.

## References
- Whisper (translate task into English): https://github.com/openai/whisper
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Seamless communication (SeamlessM4T v2, SeamlessStreaming, licences): https://github.com/facebookresearch/seamless_communication; paper https://arxiv.org/abs/2312.05187
- NVIDIA Canary-1B-v2 model card: https://huggingface.co/nvidia/canary-1b-v2; report https://arxiv.org/abs/2509.14128
- Ma et al. (2019), STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency (wait-k, Average Lagging): https://arxiv.org/abs/1810.08398
