---
title: LLM API SDK setup (Anthropic, OpenAI, Google)
category: setup
tags: [anthropic, openai, google-genai, sdk, api-key, retries, timeouts, env-vars]
use_cases:
  - "make a first call to Claude, GPT or Gemini from Python"
  - "store API keys safely and load them in dev and production"
  - "add retries, timeouts and error handling around LLM calls"
  - "point the OpenAI SDK at a local or alternative OpenAI-compatible server"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/cli-sdks-libraries/overview
  - https://github.com/openai/openai-python
  - https://ai.google.dev/gemini-api/docs/quickstart
---

# LLM API SDK setup (Anthropic, OpenAI, Google)

## Summary
Each major provider ships an official SDK that reads an API key from an environment variable, handles auth, streaming, typed responses and automatic retries. Setup is the same everywhere: install the package, export the key, create a client, send a message. Model names change often, so keep them in config, not code.

## Key concepts
- Keys come from environment variables; the SDK clients read them by default.
- Retries: SDKs retry transient errors (connection failures, 408/409/429/5xx) with backoff; the OpenAI Python SDK defaults to 2 retries and a 10-minute timeout. Anthropic's SDK also retries by default (check its docs for current numbers).
- Always set an explicit timeout and a total-budget guard in your own code.
- Errors to handle: auth (401/403), rate limit (429), overloaded/5xx, invalid request (400), context too long.
- Many providers and local servers expose an OpenAI-compatible endpoint; the OpenAI SDK works with `base_url`.

## When to use / scenarios
- Any app calling hosted LLMs: chatbots, extraction, classification, agents.
- Prototype with one provider, keep the call behind a thin function so you can swap providers ([[model-selection]]).
- Local models: use the OpenAI SDK with `base_url` ([[ollama]], [[inference-servers-vllm]]).
- Use the raw HTTP API only when no SDK exists for your language.

## Setup & code
Install (use [[python-env-uv]]):
```bash
uv add anthropic openai google-genai     # or: pip install anthropic openai google-genai
```
Keys (Linux/macOS, then Windows PowerShell). Keep them out of git; use a `.env` file ignored by git or a secrets manager:
```bash
export ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...   GEMINI_API_KEY=...
```
```powershell
$env:ANTHROPIC_API_KEY = "..."   # session only; use setx or a secrets store for persistence
```
Anthropic:
```python
import os, anthropic
client = anthropic.Anthropic(max_retries=3, timeout=60.0)   # reads ANTHROPIC_API_KEY
msg = client.messages.create(
    model=os.environ["CLAUDE_MODEL"],     # model id from the models docs
    max_tokens=300,
    messages=[{"role": "user", "content": "Say hi in five words."}],
)
print(msg.content[0].text)
```
OpenAI:
```python
import os
from openai import OpenAI
client = OpenAI(max_retries=3, timeout=60.0)                # reads OPENAI_API_KEY
r = client.responses.create(model=os.environ["OPENAI_MODEL"], input="Say hi in five words.")
print(r.output_text)
```
Google Gemini (per the current quickstart):
```python
import os
from google import genai
client = genai.Client()                                     # reads GEMINI_API_KEY
r = client.interactions.create(model=os.environ["GEMINI_MODEL"], input="Say hi in five words.")
print(r.output_text)
```
Per-call override of retries/timeouts in OpenAI: `client.with_options(max_retries=5, timeout=5.0).responses.create(...)`.

Own retry wrapper for rate limits beyond SDK defaults: catch the SDK's rate-limit exception, sleep with exponential backoff plus jitter, cap attempts.

## Choosing / trade-offs
- Official SDK vs OpenAI-compat layer on other vendors: compat is quick for portability but may lack provider-specific features (caching controls, extended thinking, etc.).
- SDK retries vs your own: SDK retries are fine for transient faults; add your own for queueing, global rate limiting and fallbacks to another model/provider.
- Sync vs async clients: use async (`AsyncAnthropic`, `AsyncOpenAI`) in servers ([[serving-with-fastapi]]).

## Gotchas
- Never hard-code keys or print request headers; rotate any key that reaches git history.
- Model ids and "current model" names change; take them from config and check the vendor's models page.
- Streaming and long outputs need larger timeouts; a 60 s default may cut long generations.
- Retried non-idempotent calls (tools with side effects) can run twice; design for idempotency.
- Count output limits explicitly (`max_tokens` is required by Anthropic's Messages API).
- Corporate proxies: set `HTTPS_PROXY` or pass a custom `httpx` client.

## Related
- [[anthropic-claude]], [[openai-gpt]], [[google-gemini]] - model specifics.
- [[structured-output]] and [[tool-calling]] - next steps after the first call.
- [[cost-and-latency]] - controlling spend.
- [[ai-security-privacy-compliance]] - secrets and data handling.
- [[llm-observability]] - logging calls.

## References
- Anthropic SDKs: https://platform.claude.com/docs/en/cli-sdks-libraries/overview
- OpenAI Python: https://github.com/openai/openai-python
- Gemini quickstart: https://ai.google.dev/gemini-api/docs/quickstart
