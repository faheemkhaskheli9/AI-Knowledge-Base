---
title: Voice agents (real-time STT-LLM-TTS and speech-to-speech)
category: speech
tags: [voice-agent, realtime, pipecat, livekit, speech-to-speech, vad, barge-in, telephony, webrtc]
use_cases:
  - "build a phone-based customer support agent that talks to callers"
  - "add a real-time voice assistant to a web or mobile app"
  - "choose between a cascaded STT-LLM-TTS pipeline and a speech-to-speech API"
  - "handle interruptions (barge-in) and turn-taking in a voice bot"
  - "build an appointment-booking or outbound-calling voice agent with tool use"
status: draft
last_verified: 2026-10-03
sources:
  - https://developers.openai.com/api/docs/guides/realtime
  - https://github.com/pipecat-ai/pipecat
  - https://docs.livekit.io/agents/
---

# Voice agents (real-time STT-LLM-TTS and speech-to-speech)

## Summary
A voice agent converses by speech with sub-second turn latency, calls tools and handles interruptions. Two architectures: a cascaded pipeline (streaming STT -> LLM -> streaming TTS) with full control over each part, or a speech-to-speech API where one model consumes and produces audio directly (for example OpenAI's Realtime API, which the docs describe as working directly with audio, maintaining conversation state and calling tools). Frameworks such as Pipecat and LiveKit Agents orchestrate transport, VAD, turn-taking and services.

## Key concepts
- Latency budget: time from user stop-speaking to first agent audio; conversational feel is roughly under about a second, and every stage (endpointing, STT final, LLM first token, TTS first byte, network) adds to it.
- Transport: WebRTC (browser/mobile, low latency), WebSocket (server-side), SIP/telephony (phone numbers via providers such as Twilio or Telnyx; 8 kHz audio).
- VAD and turn detection: decide when the user finished; semantic/model-based turn detection reduces cutting people off.
- Barge-in: when the user speaks over the agent, cancel TTS playback and generation, and truncate the context to what was actually heard.
- Streaming everywhere: partial STT, token streaming to TTS sentence by sentence.
- Tool calling mid-conversation (booking, lookups) with filler phrases to mask latency ([[tool-calling]]).
- Frameworks: Pipecat (open-source Python, many STT/LLM/TTS integrations, WebRTC/WebSocket transports), LiveKit Agents, vendor SDKs.

## When to use / scenarios
- Contact centres: Tier-1 support, order status, FAQs with handoff to a human ([[customer-support]]).
- Healthcare/clinics, salons, restaurants: scheduling and reminders; regulated data needs compliant vendors.
- In-app copilots, language tutors, kiosks, in-car or hands-free field tools.
- Outbound calls: follow laws on automated calling, disclosure and consent (TCPA/PECR-style rules vary).
- Not for: asynchronous voice notes (batch [[speech-to-text]] is cheaper); high-stakes decisions without human fallback.

## Setup & code
Pipecat quick start (from its README; requires `uv`):
```bash
uv tool install "pipecat-ai[cli]"
pipecat init                 # scaffolds a project and picks STT/LLM/TTS/transport services
```
Manual install: `uv add pipecat-ai` plus service extras (`uv add "pipecat-ai[...]"`; extras names are per provider, see the services docs). A cascaded pipeline is composed in code as ordered processors: transport input -> STT -> context aggregator -> LLM -> TTS -> transport output. Speech-to-speech: use the provider's session SDK or WebRTC/WebSocket connection and generate ephemeral credentials server-side so API keys never reach the browser.

## Choosing / trade-offs
- Speech-to-speech API: lowest latency, natural prosody and interruption handling, simplest build; less control over the exact STT/LLM/TTS, voice choices limited, harder to log/evaluate text intermediates, locked to one vendor.
- Cascaded pipeline: pick the best STT, LLM and TTS per language and cost; easy transcripts, guardrails and RAG; more latency tuning and moving parts.
- Self-hosted open models (Whisper-class STT, small LLM, Kokoro/Piper TTS) cut vendor dependence but need GPUs and engineering.
- Telephony vs web: phone adds codec and carrier latency and narrower audio.
- Cost: per-minute audio pricing for speech-to-speech vs summed per-stage costs; model and pricing change often, verify in provider docs.

## Gotchas
- Measure p95 latency, not average; network distance between STT, LLM and TTS regions adds up. Co-locate services.
- Echo and noise: speaker output retriggers VAD; use echo cancellation (WebRTC) or headset testing.
- Agent talking over the user or interrupted mid-sentence while the context says it finished; truncate history on barge-in.
- Numbers, spellings of names and emails over voice are error-prone; read back and confirm.
- Hallucinated promises (refunds, prices): ground with RAG, constrain tools, add guardrails ([[guardrails-and-safety]]).
- Always disclose the caller is talking to an AI where required, obtain recording consent, and provide a human escalation path.
- Prompt injection can come through speech or tool results; treat both as untrusted ([[prompt-injection]]).
- Evaluate with scripted test calls and transcripts, not only demos ([[llm-evaluation]]).

## Related
- [[speech-to-text]] - the listening stage.
- [[text-to-speech]] - the speaking stage.
- [[tool-calling]] - actions during a call.
- [[agents]] - agent logic behind the voice.
- [[customer-support]] - flagship scenario.
- [[llm-observability]] - tracing latency and transcripts.

## References
- https://developers.openai.com/api/docs/guides/realtime
- https://github.com/pipecat-ai/pipecat
- https://docs.livekit.io/agents/
