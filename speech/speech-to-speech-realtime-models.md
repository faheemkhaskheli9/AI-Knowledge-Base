---
title: Speech-to-speech realtime models (native audio, full-duplex vs STT-LLM-TTS cascades)
category: speech
tags: [speech-to-speech, realtime, full-duplex, native-audio, moshi, mimi, openai-realtime, gemini-live, cascade, latency, turn-taking, barge-in]
use_cases:
  - "decide between a native speech-to-speech model and an STT-LLM-TTS cascade for a voice product"
  - "estimate and reduce time-to-first-audio for a voice assistant"
  - "run an open full-duplex speech model locally for research or a private assistant"
  - "build a voice agent that handles interruptions and backchannels naturally"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2410.00037
  - https://github.com/kyutai-labs/moshi
  - https://developers.openai.com/api/docs/guides/realtime
  - https://developers.openai.com/api/docs/guides/realtime-conversations
  - https://ai.google.dev/gemini-api/docs/live
---

# Speech-to-speech realtime models (native audio, full-duplex vs STT-LLM-TTS cascades)

## Summary
A speech-to-speech (S2S) model takes audio in and produces audio out with one network, instead of chaining speech-to-text, an LLM and text-to-speech. Native audio models keep prosody, emotion and non-speech cues, and avoid the waits between stages; full-duplex models go further and listen while speaking, so they can be interrupted, backchannel ("mm-hm") and overlap like people do. Hosted options include OpenAI's Realtime API and Google's Gemini Live API; Kyutai's Moshi is an open full-duplex model. Cascades remain the default when you need per-stage control, text-based guardrails, RAG or a specific voice and language mix. This file covers the model side; [[voice-agents]] covers the application around it.

## Key concepts
- **Cascade.** Streaming STT -> LLM -> streaming TTS. Latency is the sum of endpointing (deciding the user stopped), final transcript, LLM time to first token and TTS time to first audio, plus network hops. Tone of voice is lost at the STT step.
- **Native audio (half-duplex).** One model consumes audio tokens and emits audio tokens, but still takes turns: a voice activity detector (VAD) or turn detector decides when to respond.
- **Full-duplex.** The model processes the user's and its own audio streams in parallel at every step, so there are no explicit turns. Moshi models two audio streams plus an "inner monologue" of time-aligned text tokens that improves the content of what it says.
- **Neural audio codecs.** Audio is turned into discrete tokens a language model can predict. Moshi's Mimi codec compresses 24 kHz audio to a 12.5 Hz token rate at about 1.1 kbps with 80 ms frames, per its repo. Lower token rates mean fewer steps per second of speech.
- **Latency numbers.** The Moshi paper reports a theoretical latency of 160 ms and about 200 ms in practice, versus several seconds for the cascaded systems it compares against; treat vendor figures as best cases and measure your own end-to-end p95.
- **Turn detection.** Hosted APIs offer server-side VAD; OpenAI's Realtime API also has a `semantic_vad` mode that uses the content of speech to judge whether the user has finished. Manual turn control is available for push-to-talk.
- **Interruptions (barge-in).** When the user talks over the model, playback must stop and the conversation state must be truncated to what the user actually heard; per OpenAI's docs this is automatic over WebRTC/SIP and the client's job over WebSocket.
- **Transports.** WebRTC for browsers and mobile (handles jitter and echo cancellation), WebSocket for server-to-server, SIP for phone lines. Browsers connect with short-lived credentials minted by your server, never the API key.

## When to use / scenarios
- Consumer voice assistants and companions where natural turn-taking, emotion and interruptions matter more than exact control.
- Language tutoring and pronunciation practice, where the model needs to hear how something was said, not only the words.
- Live interpretation and conversational demos with tight latency targets.
- Research on dialogue, backchannels and overlap (open models such as Moshi).
- Not for: workflows that need verbatim transcripts, text guardrails or audit before anything is spoken (regulated advice, payments); per-language best-of-breed STT/TTS; specific cloned or branded voices; or long batch audio. Use a cascade ([[voice-agents]]) or batch [[speech-to-text]] instead.

## Setup & code
Open full-duplex model locally (Moshi; Python code MIT, Rust backend Apache, weights CC-BY 4.0 per the repo; the PyTorch version needs a GPU with about 24 GB because it has no quantisation, while the MLX version runs quantised on Apple silicon):
```bash
pip install -U moshi                 # Python 3.10+, 3.12 recommended
python -m moshi.server               # then open http://localhost:8998 and talk
# macOS: pip install -U moshi_mlx    (4-bit and 8-bit weights)
```
Hosted S2S: open a WebRTC, WebSocket or SIP session with the provider SDK, send a session configuration (instructions, voice, turn detection, tools), stream microphone audio in and play audio deltas out. Event names and model IDs change; copy them from the provider docs and [[openai-gpt]] / [[google-gemini]] at build time.

Self-contained latency model: why streaming at every stage matters more than any single fast component, and how p95 differs from the median. Stage timings are illustrative assumptions you replace with your own measurements, not vendor benchmarks:
```python
import numpy as np

rng = np.random.default_rng(3)
N = 20000


def lognormal_ms(median, spread):
    """Right-skewed stage latency: most calls near the median, a long tail."""
    return median * np.exp(spread * rng.standard_normal(N))


# ASSUMED stage latencies in ms (replace with traces from your own stack)
endpoint = lognormal_ms(300, 0.3)        # silence the VAD waits for before deciding the user finished
stt_final = lognormal_ms(150, 0.4)       # final transcript after end of speech
llm_first_tok = lognormal_ms(350, 0.5)   # time to first token
llm_tok_per_s = 60.0
tts_first_audio = lognormal_ms(150, 0.4)
net_hop = lognormal_ms(30, 0.5)          # one network hop between services
first_sentence_tokens = 18
full_reply_tokens = 120

# cascade, worst practice: wait for the full LLM reply, then synthesise
cascade_batch = (endpoint + stt_final + llm_first_tok + full_reply_tokens / llm_tok_per_s * 1000
                 + tts_first_audio + 3 * net_hop)
# cascade, good practice: stream tokens and send the first sentence to TTS as soon as it is complete
cascade_stream = (endpoint + stt_final + llm_first_tok + first_sentence_tokens / llm_tok_per_s * 1000
                  + tts_first_audio + 3 * net_hop)
# native S2S model: same endpointing, one model emits audio frames directly, one hop
s2s_first_audio = lognormal_ms(300, 0.5)
native = endpoint + s2s_first_audio + net_hop
# full-duplex: no endpointing wait; the model decides frame by frame (here: a fixed 200 ms reaction)
duplex = lognormal_ms(200, 0.3) + net_hop

for name, lat in [("cascade, no streaming", cascade_batch), ("cascade, sentence streaming", cascade_stream),
                  ("native S2S (VAD turns)", native), ("full-duplex", duplex)]:
    print(f"{name:28s} p50 {np.median(lat):6.0f} ms   p95 {np.percentile(lat, 95):6.0f} ms")

# which stage to attack first: share of the streaming-cascade p95 tail
tail = cascade_stream > np.percentile(cascade_stream, 95)
parts = {"endpointing": endpoint, "stt": stt_final, "llm first token": llm_first_tok, "tts": tts_first_audio}
print("mean stage time in the slowest 5% of turns:", {k: int(v[tail].mean()) for k, v in parts.items()})
```
Reading the output: streaming the first sentence into TTS removes seconds from a cascade; the endpointing wait is a fixed tax that only full-duplex models avoid; and the p95 is what users remember, so optimise the stage that dominates the tail, not the median.

## Choosing / trade-offs
- **Latency and naturalness** favour native and full-duplex models: fewer hops, prosody preserved, interruptions handled inside the model.
- **Control** favours cascades: choose STT, LLM and TTS per language and cost, inspect and filter text before it is spoken, run RAG and tools with any LLM, and get exact transcripts for logs and evaluation.
- **Intelligence per turn.** Native audio models may trail the strongest text LLMs on reasoning and tool use; a hybrid routes hard requests to a text LLM (the model says a filler while it waits).
- **Vendor lock-in and voices.** Hosted S2S ties you to one provider's voices, languages and pricing; open models (Moshi) give privacy and control but need a large GPU and fewer languages.
- **Cost.** S2S is usually priced per audio token or minute in and out; cascades sum three services. Compare on your call length and talk ratio with current pricing ([[cost-and-latency]]).
- **Session limits.** Hosted realtime sessions have maximum durations and context limits; long calls need session resumption or summarised context. Check the provider docs.

## Gotchas
- Benchmarked "first audio" times exclude endpointing and your network; measure from end of user speech to first audible sample on a real device.
- Over-eager VAD cuts users off mid-thought; lazy VAD feels slow. Tune eagerness or use semantic turn detection, and test with hesitant speakers.
- Echo: the model hears itself through the speaker and interrupts itself. Use WebRTC echo cancellation or headsets in testing.
- After barge-in, the model's context must contain only what was played, or it will refer to words the user never heard.
- Guardrails on audio are harder: there is no text to filter before speech starts. Use the provider's transcript events, constrain tools, and moderate asynchronously ([[guardrails-and-safety]]).
- Instructions and tool results can carry prompt injection just as in text agents ([[prompt-injection]]).
- Native models can imitate voices or styles they hear; restrict this and disclose that the user is talking to AI ([[voice-cloning-and-ethics]]).

## Related
- [[voice-agents]] - the application layer: transports, frameworks, telephony, tools.
- [[speech-to-text]] - the listening stage of a cascade.
- [[text-to-speech]] - the speaking stage of a cascade.
- [[voice-cloning-and-ethics]] - voice imitation risks and disclosure.
- [[multimodal-models]] - audio as one modality among others.
- [[cost-and-latency]] - latency budgets and pricing comparisons.
- [[tool-calling]] - actions during a live conversation.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/realtime-websockets-sse.md - WebSocket transport and connection handling.

## References
- Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue" (2024): https://arxiv.org/abs/2410.00037
- Moshi and Mimi code and weights: https://github.com/kyutai-labs/moshi
- OpenAI Realtime API guide: https://developers.openai.com/api/docs/guides/realtime
- OpenAI realtime conversations (turn detection, interruptions): https://developers.openai.com/api/docs/guides/realtime-conversations
- Gemini Live API: https://ai.google.dev/gemini-api/docs/live
