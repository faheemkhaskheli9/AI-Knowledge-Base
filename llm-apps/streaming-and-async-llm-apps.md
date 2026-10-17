---
title: Streaming and async LLM apps
category: llm-apps
tags: [streaming, sse, websocket, asyncio, async, cancellation, timeouts, partial-json, fan-out]
use_cases:
  - "stream chatbot answers token by token to a web UI over SSE"
  - "call an LLM on 500 documents concurrently without hitting rate limits"
  - "show a form filling in live while the model streams JSON"
  - "stop paying for tokens when the user closes the tab or presses stop"
  - "add per-call timeouts and a total deadline to an LLM pipeline"
status: draft
last_verified: 2026-10-15
sources:
  - https://platform.claude.com/docs/en/build-with-claude/streaming
  - https://html.spec.whatwg.org/multipage/server-sent-events.html
  - https://docs.python.org/3/library/asyncio-task.html
  - https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
  - https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
---

# Streaming and async LLM apps

## Summary
LLM calls are slow (seconds) and I/O-bound, so two techniques matter in every
production app: **streaming** (send tokens to the user as they are generated,
cutting perceived latency to time-to-first-token) and **async concurrency**
(run many calls at once on one thread with `asyncio`). Both need explicit
**timeouts** and **cancellation**, or abandoned requests keep burning tokens.

## Key concepts
- **Time to first token (TTFT)** vs total latency: streaming does not make the
  answer faster, it makes the wait visible and lets the user stop early.
- **Provider streaming** is server-sent events (SSE). The Claude API sends
  `message_start`, then per content block `content_block_start`,
  `content_block_delta`*, `content_block_stop`, then `message_delta` and
  `message_stop`; `ping` and `error` events can appear anywhere (streaming docs).
  Tool-call arguments arrive as partial JSON deltas, not as finished objects.
- **SSE vs WebSocket** to your own browser client:
  - SSE: one-way server->client over plain HTTP, `text/event-stream`, frames are
    `data: ...\n\n`, browser `EventSource` reconnects automatically. Default for
    chat output.
  - WebSocket: bidirectional, good when the client also streams (voice audio,
    "stop"/interrupt messages, collaborative editing). More infra work (sticky
    connections, proxies, auth on upgrade).
- **Async fan-out**: `asyncio.TaskGroup` (Python 3.11+) or `gather` runs N
  calls concurrently; a `Semaphore` caps in-flight requests below the provider
  rate limit. Use the SDK's async client (`AsyncAnthropic`, `AsyncOpenAI`).
- **Timeouts**: per call (`asyncio.timeout` or SDK `timeout=`) plus an overall
  deadline for the whole request; a stalled stream needs an *idle* timeout
  between chunks, not just a total one.
- **Cancellation**: when the client disconnects or presses stop, cancel the
  task; closing the upstream stream stops generation and further billing of
  output tokens. Always clean up in `finally`.
- **Partial structured output**: streamed JSON is invalid until the end. Either
  repair it on the fly (close open strings/brackets) to render progressively,
  or buffer and validate at the end ([[structured-output]]).

## When to use / scenarios
- Chat UIs, copilots, writing assistants: stream text to the screen.
- Voice agents: stream LLM text straight into TTS ([[voice-agents]], [[text-to-speech]]).
- Batch enrichment (classify 10k tickets, summarize a document set): async
  fan-out with a semaphore; for non-urgent jobs a provider Batch API is cheaper
  ([[prompt-caching-and-cost]]).
- Agents: stream intermediate steps ("searching...") so long runs feel alive
  ([[agents]], [[long-running-agents]]).
- Not needed: offline jobs nobody watches, or very short outputs (a label);
  streaming adds parsing complexity for no UX gain.

## Setup & code
Self-contained demo (stdlib only): a fake streaming model, SSE framing, fan-out
with a concurrency cap and timeouts, cancellation on "client disconnect", and
progressive parsing of partial JSON.
```python
import asyncio
import json
import random

async def fake_llm_stream(prompt: str, delay: float = 0.01):
    """Stands in for client.messages.stream(...): yields text chunks."""
    text = f'{{"topic": "{prompt}", "summary": "streaming keeps users engaged", "score": 7}}'
    for i in range(0, len(text), 5):
        await asyncio.sleep(delay)
        yield text[i:i + 5]

def sse(data: str, event: str | None = None) -> str:
    """Format one server-sent event frame (multi-line data -> several data: lines)."""
    head = f"event: {event}\n" if event else ""
    return head + "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"

def parse_partial_json(buf: str):
    """Best-effort: close an unfinished JSON object so a UI can render it early."""
    stack, in_str, esc = [], False, False
    for ch in buf:
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        elif ch == '"': in_str = True
        elif ch in "{[": stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack: stack.pop()
    candidate = buf + ('"' if in_str else "")
    for tail in ("", "null"):                      # handle a dangling `"key":`
        try:
            return json.loads(candidate.rstrip().rstrip(",") + tail + "".join(reversed(stack)))
        except json.JSONDecodeError:
            continue
    return None

async def stream_to_client(prompt: str, idle_timeout: float = 1.0) -> list[str]:
    frames, buf = [], ""
    stream = fake_llm_stream(prompt)
    try:
        while True:
            async with asyncio.timeout(idle_timeout):      # max gap between chunks
                try:
                    chunk = await anext(stream)
                except StopAsyncIteration:
                    break
            buf += chunk
            frames.append(sse(json.dumps({"delta": chunk})))
            partial = parse_partial_json(buf)
            if partial:
                frames.append(sse(json.dumps(partial), event="partial"))
        frames.append(sse("[DONE]", event="done"))
        return frames
    finally:
        await stream.aclose()                              # stop upstream generation

async def fan_out(prompts, max_in_flight=3, per_call_timeout=2.0):
    sem = asyncio.Semaphore(max_in_flight)
    async def one(p):
        async with sem:
            async with asyncio.timeout(per_call_timeout):
                out = ""
                async for c in fake_llm_stream(p, delay=random.uniform(0.001, 0.005)):
                    out += c
                return json.loads(out)
    async with asyncio.TaskGroup() as tg:                  # one failure cancels the rest
        tasks = [tg.create_task(one(p)) for p in prompts]
    return [t.result() for t in tasks]

async def cancelled_by_user():
    task = asyncio.create_task(stream_to_client("long answer"))
    await asyncio.sleep(0.03)                              # user presses stop
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return "cancelled cleanly"

async def main():
    frames = await stream_to_client("sse")
    print(frames[0].strip(), "...", len(frames), "frames")
    last_partial = [f for f in frames if f.startswith("event: partial")][-1]
    print(last_partial.strip())
    results = await fan_out([f"doc-{i}" for i in range(8)])
    assert [r["topic"] for r in results] == [f"doc-{i}" for i in range(8)]
    print("fan-out ok:", len(results))
    print(await cancelled_by_user())
    assert parse_partial_json('{"a": "hel') == {"a": "hel"}
    assert parse_partial_json('{"a": [1, 2') == {"a": [1, 2]}

asyncio.run(main())
```

Real SDK equivalent (Anthropic async client; model ID from [[anthropic-claude]]):
<!-- skip-check: needs ANTHROPIC_API_KEY and network -->
```python
import asyncio
import anthropic

client = anthropic.AsyncAnthropic(timeout=60.0, max_retries=2)

async def answer(question: str):
    async with client.messages.stream(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        async for text in stream.text_stream:     # leaving the block closes the HTTP stream
            print(text, end="", flush=True)
        final = await stream.get_final_message()  # usage, stop_reason
    print("\n", final.stop_reason, final.usage.output_tokens)

asyncio.run(answer("Explain SSE in two sentences."))
```

Serving it to a browser with FastAPI (see [[serving-with-fastapi]]):
<!-- skip-check: needs a running server -->
```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/chat")
async def chat(q: str, request: Request):
    async def events():
        async for chunk in fake_llm_stream(q):          # swap for the SDK stream
            if await request.is_disconnected():          # stop paying for tokens
                break
            yield sse(chunk)
        yield sse("[DONE]", event="done")
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

## Choosing / trade-offs
| Choice | Pick A when | Pick B when |
|---|---|---|
| SSE vs WebSocket | output-only text streams, simple infra, HTTP auth/caching | client sends data mid-stream (audio, interrupts), multiplayer |
| Stream vs buffer | a human watches the output | output feeds code that needs complete, validated JSON |
| `asyncio` vs threads | many concurrent network calls, async SDK available | sync-only libraries; keep threads few |
| Semaphore cap vs Batch API | results needed in seconds/minutes | results can wait hours; batch is cheaper |
| Progressive partial JSON vs end validation | live form/preview UX | correctness only; validate once at the end |

## Gotchas
- **Proxy buffering**: nginx and some CDNs buffer responses, so the "stream"
  arrives all at once. Disable buffering (`X-Accel-Buffering: no`), avoid gzip
  on `text/event-stream`, and keep the connection alive with periodic comments
  or pings.
- **Total timeout only**: a stream that stalls mid-answer hangs until the total
  timeout; add an idle timeout between chunks.
- **Not cancelling upstream**: breaking out of your loop without closing the
  provider stream (context manager exit or `aclose()`) can leave generation
  running.
- **Errors after 200 OK**: once streaming starts the HTTP status is already
  200; errors arrive as `error` events in the stream (e.g. overload) and the UI
  must handle them.
- **Unbounded `gather`**: 1,000 concurrent calls trip rate limits; cap with a
  semaphore and retry 429/529 with backoff ([[api-sdk-setup]]).
- **Blocking calls inside async code** (sync SDK, CPU-heavy parsing) freeze the
  event loop; use the async client or `asyncio.to_thread`.
- **Rendering partial output as final**: moderation, citations and JSON
  validation only hold on the complete message; mark partial UI as provisional
  ([[guardrails-and-safety]]).
- **`TaskGroup` cancels siblings on first failure**; use `gather(...,
  return_exceptions=True)` when one bad document should not abort the batch.

## Related
- [[structured-output]] - validated JSON once the stream completes.
- [[serving-with-fastapi]] - the HTTP layer that forwards the stream.
- [[cost-and-latency]] - TTFT, throughput and budgets.
- [[api-sdk-setup]] - clients, retries and timeouts.
- [[voice-agents]] - streaming text into speech.
- [[long-running-agents]] - when a task outlives one request.
- [[tool-calling]] - tool arguments also stream as partial JSON.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/realtime-websockets-sse.md - SSE/WebSocket infrastructure, scaling and reconnects.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/resilience-patterns.md - timeouts, retries and backoff.

## References
- Claude streaming messages: https://platform.claude.com/docs/en/build-with-claude/streaming
- WHATWG HTML, server-sent events: https://html.spec.whatwg.org/multipage/server-sent-events.html
- MDN, Using server-sent events: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- Python asyncio tasks, TaskGroup and timeouts: https://docs.python.org/3/library/asyncio-task.html
- FastAPI StreamingResponse: https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse
