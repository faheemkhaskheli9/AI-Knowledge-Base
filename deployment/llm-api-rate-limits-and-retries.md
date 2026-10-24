---
title: LLM API rate limits, retries and backoff
category: deployment
tags: [rate-limits, retries, backoff, jitter, 429, 529, retry-after, token-bucket, throttling, anthropic, openai, reliability]
use_cases:
  - "stop our batch job from failing with 429 rate limit errors from the LLM API"
  - "decide which LLM API errors to retry and which to fail fast on"
  - "throttle many workers so together they stay under our tokens-per-minute limit"
  - "handle 529 overloaded and 5xx errors from a model provider without a retry storm"
status: draft
last_verified: 2026-10-24
sources:
  - https://platform.claude.com/docs/en/api/rate-limits
  - https://platform.claude.com/docs/en/api/errors
  - https://platform.claude.com/docs/en/api/sdks/python
  - https://platform.openai.com/docs/guides/rate-limits
  - https://github.com/openai/openai-python
  - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
  - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
  - https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after
---

# LLM API rate limits, retries and backoff

## Summary
Hosted LLM APIs cap how much each organization can send: requests per minute plus tokens per minute (Anthropic splits these into input and output tokens per minute), and sometimes daily or spend caps. Going over returns HTTP 429; a provider-wide overload returns 5xx (Anthropic uses 529 `overloaded_error`). A production client needs three things: retry only the errors that can succeed later, wait between attempts with exponential backoff plus jitter (and honour `retry-after`), and throttle itself before the provider has to, so many workers share one token budget instead of fighting over it. The official SDKs already retry a couple of times; this topic covers what to add on top for batch jobs, agents and multi-worker services.

## Key concepts
- **Limit dimensions**: Anthropic measures requests per minute (RPM), input tokens per minute (ITPM) and output tokens per minute (OTPM) per model class; exceeding any of them returns a 429 that says which one. OpenAI uses requests and tokens per minute (and per-day limits on some tiers). The numbers depend on your usage tier and change; read them from the console or the response headers, not from memory.
- **Token bucket**: Anthropic states its limits use the token bucket algorithm: capacity refills continuously up to the maximum rather than resetting each minute, so a short burst can trip a per-minute limit even when the minute's total is under it (a 60 RPM limit may be enforced as 1 request per second).
- **Cache-aware input limits**: for most Anthropic models, `cache_read_input_tokens` do not count toward ITPM, so prompt caching raises effective throughput as well as cutting cost ([[prompt-caching-and-cost]]).
- **Rate-limit headers**: Anthropic returns `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` and `retry-after`; OpenAI returns `x-ratelimit-limit-*`, `x-ratelimit-remaining-*` and `x-ratelimit-reset-*` for requests and tokens. Log them; they tell you which limit you are near.
- **Retryable vs not**: retry connection errors, 408, 409, 429 and 5xx (including 529). Do not retry 400 (bad request, context too long), 401/403 (auth, permission), 404 or 413: they fail the same way every time. Anthropic's spend-limit response is a 429 `rate_limit_error` with no `retry-after` and `error.details.error_code` set to `enforced_spend_limit_reached`; retrying it fails until access resumes.
- **SDK defaults**: the Anthropic and OpenAI Python SDKs retry connection errors, 408, 409, 429 and >=500 twice by default with short exponential backoff, configurable with `max_retries` per client or per request (`client.with_options(max_retries=5)`), and time out after 10 minutes by default. The OpenAI SDK retries a request only when its body can be safely resent.
- **Failed requests still count**: OpenAI's rate-limit cookbook notes that unsuccessful requests contribute to the per-minute limit, so resending in a tight loop keeps you limited; back off, and for bulk work add a steady delay of about 1/limit between requests.
- **`Retry-After` format**: RFC 9110 defines the value as either an HTTP-date or `delay-seconds` (a non-negative integer number of seconds), e.g. `Retry-After: 120`; handle both (the code below honours the seconds form and falls back to backoff on a date). RFC 9110 itself describes it for 503 and 3xx responses; LLM APIs also send it on 429.
- **Backoff with jitter**: wait `base * 2^attempt`, capped, and randomize it. "Full jitter" (uniform between 0 and the capped exponential) keeps many clients that failed together from retrying together.
- **Acceleration limits**: a sharp increase in an organization's usage can itself trigger 429s on the Anthropic API; ramp traffic up gradually.

## When to use / scenarios
- Batch enrichment or backfill (classify a million tickets, embed a corpus) that runs many workers in parallel and keeps hitting 429: throttle client-side to the tier's limit and run at a steady rate. If results can wait, a batch API removes the per-minute pressure altogether ([[batch-inference-and-batch-apis]]).
- An interactive app where a retry storm during a provider incident would multiply load: cap total retries per request and fall back to another model or provider ([[llm-gateways-and-routing]]).
- Agents and long-running jobs that make hundreds of calls per task: one 429 should not kill a multi-hour run ([[long-running-agents]]).
- Self-hosted models behind vLLM or a gateway that enforces its own quotas: the same client-side rules apply ([[inference-servers-vllm]]).
- Not a substitute for capacity: if you are always at the limit, request a higher tier, cache, shorten prompts, or batch ([[cost-and-latency]]).

## Setup & code
SDK-level settings (the SDK's own retries run first; add your own loop only for longer waits or cross-worker throttling):
<!-- skip-check: needs an API key and network -->
```python
import anthropic

MODEL_ID = "<model-id>"                                        # copy a current ID from models/anthropic-claude.md
client = anthropic.Anthropic(max_retries=4, timeout=120.0)     # defaults: 2 retries, 10-minute timeout
try:
    msg = client.with_options(max_retries=6).messages.create(  # per-request override
        model=MODEL_ID, max_tokens=256,
        messages=[{"role": "user", "content": "Classify: 'refund not received'"}],
    )
    print(msg.content[0].text)
except anthropic.RateLimitError as e:                          # still 429 after all retries
    print("429:", e.response.headers.get("retry-after"),
          e.response.headers.get("anthropic-ratelimit-output-tokens-remaining"))
except anthropic.APIStatusError as e:
    print("non-retryable or exhausted:", e.status_code)
```
Runnable, stdlib only: (1) a retry classifier that honours `retry-after` and treats the spend-cap 429 as permanent; (2) capped exponential backoff with full jitter; (3) a client-side token bucket shared by workers that blocks before sending instead of after a 429; (4) a simulation of a provider outage showing why jitter matters.
```python
import random
import threading
import time

# 1) Decide whether and when to retry.
RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}

def retry_decision(status, headers, body, attempt, base=1.0, cap=60.0, max_attempts=6):
    """Return seconds to wait before the next attempt, or None to give up."""
    if attempt + 1 >= max_attempts or status not in RETRYABLE:
        return None
    error = (body or {}).get("error", {})
    if error.get("details", {}).get("error_code") == "enforced_spend_limit_reached":
        return None                               # spend cap: retrying cannot succeed until access resumes
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), cap)   # seconds form; the HTTP-date form falls through to backoff
        except ValueError:
            pass
    return random.uniform(0, min(cap, base * 2 ** attempt))   # full jitter

random.seed(0)
print(retry_decision(429, {"retry-after": "7"}, None, attempt=0))            # 7.0, server told us
print(retry_decision(529, {}, None, attempt=3))                               # jittered, up to 8 s
print(retry_decision(400, {}, None, attempt=0))                               # None: bad request
print(retry_decision(429, {}, {"error": {"type": "rate_limit_error",
      "details": {"error_code": "enforced_spend_limit_reached"}}}, attempt=0))  # None: spend cap

# 2) A retry loop around any call that raises an exception carrying status and headers.
class APIError(Exception):
    def __init__(self, status, headers=None, body=None):
        super().__init__(status); self.status, self.headers, self.body = status, headers or {}, body

def call_with_retries(fn, sleep=time.sleep, **policy):
    attempt = 0
    while True:
        try:
            return fn()
        except APIError as e:
            wait = retry_decision(e.status, e.headers, e.body, attempt, **policy)
            if wait is None:
                raise
            sleep(wait); attempt += 1

outcomes = iter([APIError(529), APIError(429, {"retry-after": "2"}), "ok"])
def flaky():
    r = next(outcomes)
    if isinstance(r, Exception):
        raise r
    return r
waits = []
print(call_with_retries(flaky, sleep=waits.append), "after waits", [round(w, 2) for w in waits])

# 3) Client-side token bucket: reserve estimated tokens before sending, shared across threads.
class TokenBucket:
    def __init__(self, per_minute, burst=None):
        self.rate = per_minute / 60.0                   # refill per second
        self.capacity = burst or per_minute
        self.tokens, self.t = float(self.capacity), time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, n):
        n = min(n, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                wait = (n - self.tokens) / self.rate
            time.sleep(wait)

bucket = TokenBucket(per_minute=60_000, burst=2_000)    # e.g. an input-tokens-per-minute budget
sent = []
def worker(i):
    for _ in range(3):
        bucket.acquire(500)                             # estimated input tokens of the next request
        sent.append(time.monotonic())
start = time.monotonic()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in threads]; [t.join() for t in threads]
print(f"12 requests x 500 tokens through a 1000 tokens/s bucket: {time.monotonic() - start:.1f} s "
      f"(burst 2000 lets the first 4 go at once)")

# 4) Why jitter: 200 clients fail at t=0; the provider can serve 20 retries per second.
def simulate(jitter, clients=200, capacity_per_s=20, base=0.5, cap=30.0, seed=1):
    rnd = random.Random(seed)
    pending = [(0.0, 0) for _ in range(clients)]          # (next attempt time, attempt number)
    done, attempts = 0, 0
    while pending:
        pending.sort()
        second = int(pending[0][0])
        batch = [p for p in pending if int(p[0]) == second]
        pending = [p for p in pending if int(p[0]) != second]
        attempts += len(batch)
        rnd.shuffle(batch)
        for i, (t, a) in enumerate(batch):
            if i < capacity_per_s:
                done += 1
            else:
                delay = min(cap, base * 2 ** a)
                pending.append((t + (rnd.uniform(0, delay) if jitter else delay), a + 1))
    return attempts

print("total attempts without jitter:", simulate(jitter=False), "| with full jitter:", simulate(jitter=True))
```

## Choosing / trade-offs
| Approach | Pick when | Watch out for |
|---|---|---|
| SDK default retries only | low-volume interactive calls | two quick retries do not ride out a minute-long limit window |
| Own backoff loop on top of the SDK | batch jobs and agents that must finish | retries multiply: SDK retries x your retries x upstream retries; set the SDK to 0 or count both |
| Client-side token bucket / concurrency cap | many workers sharing one org limit | token counts are estimates before the call; reconcile with `usage` after it, keep headroom |
| Central gateway with quotas | many services or teams on one key | one more hop; it becomes the place limits and fallbacks live ([[llm-gateways-and-routing]]) |
| Batch API | results can wait hours | different limits and pricing; not for interactive paths |
| Fallback model or provider | availability matters more than identical output | the fallback's quality, prompt format and data terms must be tested too |

- **Concurrency vs rate**: a semaphore on in-flight requests is simple and adapts to latency; a token bucket matches the provider's own accounting. Large-output calls (long generations) are limited by OTPM, so budget output tokens with `max_tokens`, not just input.
- **Deadline over attempt count**: interactive requests should stop retrying when the user-facing deadline is near, whatever the attempt number.

## Gotchas
- A 429 with no `retry-after` on Anthropic may be the spend cap, not a rate limit; check `error.details.error_code` before retrying.
- Retrying a streamed response after tokens were already sent to the user duplicates text; resume or restart the UI message explicitly ([[streaming-and-async-llm-apps]]).
- Non-idempotent tool calls inside an agent loop (send email, create ticket) must not re-run when you retry the LLM call that requested them; key side effects by an idempotency ID ([[tool-calling]]).
- Fixed sleeps or backoff without jitter synchronize workers after an outage and cause a second spike (simulation above).
- Token estimates from character counts are rough, especially for code and non-English text; use the provider's token counting or the returned `usage` to calibrate ([[tokenization]]).
- Per-minute limits are per organization (and per model class), not per process: ten pods each throttling to the full limit exceed it tenfold. Share the budget through a gateway or a central store.
- Sudden jumps in traffic (a new batch job at full speed) can trigger acceleration limits; ramp up.
- Log the provider request ID (`request-id` header, `_request_id` on SDK objects) with each failure so support can trace it ([[llm-observability]]).

## Related
- [[api-sdk-setup]] - SDK installation, keys and default retry behaviour.
- [[llm-gateways-and-routing]] - central quotas, retries and provider fallback.
- [[batch-inference-and-batch-apis]] - moving bulk work off per-minute limits.
- [[prompt-caching-and-cost]] - cached input tokens and higher effective ITPM.
- [[streaming-and-async-llm-apps]] - async clients, concurrency and streaming retries.
- [[cost-and-latency]] - when limits are really a capacity or budget problem.
- [[llm-observability]] - logging status codes, rate-limit headers and request IDs.
- [[llm-load-testing-and-benchmarking]] - finding a self-hosted server's real limits.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/resilience-patterns.md - retries, backoff, circuit breakers and retry budgets in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/rate-limiting.md - token bucket and other rate-limiting algorithms on the server side.

## References
- Anthropic rate limits (RPM/ITPM/OTPM, token bucket, cache-aware ITPM, headers, spend limits, acceleration limits): https://platform.claude.com/docs/en/api/rate-limits
- Anthropic errors (429, 500, 504, 529 `overloaded_error`): https://platform.claude.com/docs/en/api/errors
- Anthropic Python SDK (retries, `max_retries`, timeouts, error classes, request IDs): https://platform.claude.com/docs/en/api/sdks/python
- OpenAI rate limits guide (limits, `x-ratelimit-*` headers, backoff advice): https://platform.openai.com/docs/guides/rate-limits
- OpenAI Python SDK (retries, safe-resend rule, 10-minute default timeout): https://github.com/openai/openai-python
- OpenAI Cookbook, How to handle rate limits (random exponential backoff, failed requests count toward limits, proactive delay, RPM/RPD vs TPM): https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
- Brooker, Exponential Backoff and Jitter (AWS Architecture Blog): https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
- RFC 9110, HTTP Semantics, section 10.2.3 `Retry-After` (HTTP-date or delay-seconds): https://www.rfc-editor.org/rfc/rfc9110.html#name-retry-after
