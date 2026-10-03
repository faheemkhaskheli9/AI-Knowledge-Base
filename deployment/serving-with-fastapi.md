---
title: Serving LLM apps with FastAPI
category: deployment
tags: [fastapi, streaming, sse, async, rate-limiting, uvicorn, backpressure]
use_cases:
  - "wrap an LLM or RAG pipeline in an HTTP API with token streaming"
  - "stop one user from exhausting my LLM budget with rate limits and concurrency caps"
  - "build a chat backend for a web or mobile app that streams answers"
  - "put auth and validation in front of a self-hosted model server"
status: draft
last_verified: 2026-10-03
sources:
  - https://fastapi.tiangolo.com/
  - https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
  - https://github.com/openai/openai-python
---

# Serving LLM apps with FastAPI

## Summary
FastAPI is the common Python layer between clients and LLM providers or model servers: it validates input, authenticates, calls the model asynchronously, streams tokens back and enforces limits. LLM calls are slow I/O, so async handlers and streaming matter more than raw framework speed.

## Key concepts
- `async def` endpoints with async SDK clients (`AsyncOpenAI`, `AsyncAnthropic`) keep the event loop free while waiting on the model.
- Streaming: `StreamingResponse` with `text/event-stream` (Server-Sent Events) sends tokens as generated; the browser reads via `EventSource` or `fetch` streams.
- Concurrency cap: an `asyncio.Semaphore` bounds in-flight model calls (backpressure).
- Rate limiting: per-user/key token bucket (in-memory for one process, Redis for many).
- Timeouts and cancellation: stop the upstream call when the client disconnects.
- Run with `uvicorn`; scale with multiple worker processes behind a load balancer.

## When to use / scenarios
- Chatbot backend, document-Q&A API, internal tool exposing an LLM to other services.
- Gateway adding keys, quotas, logging and prompt-injection filters in front of a vLLM server ([[inference-servers-vllm]]).
- Not needed if the client can call the provider directly with a scoped token (rare; usually keys must stay server-side).

## Setup & code
```bash
uv add fastapi uvicorn openai
uv run uvicorn app:app --port 8000
```
`app.py`:
```python
import asyncio, os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI(timeout=60.0)               # OPENAI_API_KEY from env
MODEL = os.environ["OPENAI_MODEL"]
gate = asyncio.Semaphore(8)                      # max concurrent upstream calls
API_KEYS = set(os.environ.get("APP_API_KEYS", "").split(","))

class Chat(BaseModel):
    message: str = Field(max_length=4000)

@app.post("/chat")
async def chat(body: Chat, x_api_key: str = Header(default="")):
    if x_api_key not in API_KEYS:
        raise HTTPException(401, "bad key")

    async def events():
        async with gate:
            stream = await client.chat.completions.create(
                model=MODEL, stream=True,
                messages=[{"role": "user", "content": body.message}])
            async for ch in stream:
                tok = ch.choices[0].delta.content if ch.choices else None
                if tok:
                    yield f"data: {tok}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
```
Test: `curl -N -H "x-api-key: $KEY" -H "content-type: application/json" -d '{"message":"hi"}' localhost:8000/chat`

Simple per-key limiter (single process only):
```python
import time
from collections import defaultdict
hits = defaultdict(list)
def allow(key, limit=20, window=60):
    now = time.time(); hits[key] = [t for t in hits[key] if now - t < window]
    if len(hits[key]) >= limit: return False
    hits[key].append(now); return True
```
For several workers/hosts use Redis-backed limiting or the gateway/load balancer.

## Choosing / trade-offs
- SSE (simple, one-way, proxy friendly) vs WebSockets (bidirectional, voice/realtime agents).
- In-process limiter (simple) vs Redis/gateway (correct across replicas).
- Streaming gives faster perceived latency but complicates output moderation; validate after or per chunk ([[guardrails-and-safety]]).
- Background queue (Celery/RQ) for long jobs vs holding HTTP connections open.

## Gotchas
- Blocking calls (sync SDK, heavy CPU) inside `async def` stall every request; use the async client or `run_in_threadpool`.
- Reverse proxies buffer SSE by default; disable buffering for the route (e.g. `X-Accel-Buffering: no` on nginx) and mind idle timeouts.
- SSE data lines cannot contain raw newlines; JSON-encode tokens if they might.
- Client disconnects do not always cancel the upstream stream; handle `asyncio.CancelledError` / check `request.is_disconnected()`.
- Per-process semaphores multiply by worker count.
- Log token counts, not full prompts, unless you have a retention policy ([[ai-security-privacy-compliance]]).

## Related
- [[api-sdk-setup]] - clients, retries, timeouts.
- [[inference-servers-vllm]] - backend to proxy to.
- [[llm-observability]] - tracing the endpoint.
- [[cost-and-latency]] - time-to-first-token and budgets.
- [[structured-output]] - validated JSON responses.

## References
- FastAPI: https://fastapi.tiangolo.com/
- StreamingResponse: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
