---
title: LLM load testing and serving benchmarks (TTFT, TPOT, goodput)
category: deployment
tags: [load-testing, benchmarking, ttft, tpot, itl, goodput, vllm-bench, guidellm, locust, k6, capacity-planning]
use_cases:
  - "find how many concurrent users one GPU running vLLM can serve within our latency SLO"
  - "compare two inference servers, quantizations or GPU types on our own traffic shape"
  - "load test an OpenAI-compatible endpoint before launch and size the autoscaling target"
  - "check whether a provider or gateway meets a time-to-first-token target at peak load"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/cli/bench/serve.html
  - https://docs.vllm.ai/en/latest/contributing/benchmarks.html
  - https://github.com/vllm-project/guidellm
  - https://docs.locust.io/en/stable/
  - https://grafana.com/docs/k6/latest/
  - https://arxiv.org/abs/2401.09670
---

# LLM load testing and serving benchmarks (TTFT, TPOT, goodput)

## Summary
Load testing an LLM endpoint means sending a realistic stream of streaming requests and measuring per-request latency split into time to first token (TTFT) and time per output token (TPOT), plus system throughput, as load rises. The useful answer is not "max tokens per second" but goodput: the highest request rate at which a chosen share of requests still meets both latency targets. Use it to size GPUs and autoscaling, compare servers, quantizations and hardware, and catch regressions before launch.

## Key concepts
- **Per-request latency metrics** (all need a streaming request so the first token is observable):
  - **TTFT**: send to first content token; dominated by queueing plus prefill, grows with prompt length.
  - **ITL (inter-token latency)**: gap between consecutive streamed tokens; spikes show prefills of other requests interrupting decode.
  - **TPOT**: `(e2e - ttft) / (output_tokens - 1)`, the average decode speed one user sees.
  - **E2E latency**: total request time, roughly `ttft + tpot * (output_tokens - 1)`.
- **System metrics**: requests per second, output tokens per second (the number vendors quote), and **goodput**: requests per second that meet the SLO (for example TTFT under 1 s and TPOT under 50 ms). The goodput framing comes from the DistServe paper.
- **Load models**:
  - **Closed loop** (fixed concurrency N, each virtual user sends the next request when the last ends): easy, but the load drops when the server slows, hiding overload.
  - **Open loop** (requests arrive at a target rate, typically Poisson): matches real traffic and exposes queue build-up. `vllm bench serve` uses `--request-rate` (Poisson by default) and can cap in-flight requests with `--max-concurrency`.
  - **Sweep**: step the rate from low to past saturation and plot p50/p99 TTFT and TPOT against rate; the knee is your capacity.
- **Workload shape matters more than anything else**: the input/output token-length distribution, the share of shared prefixes (prefix caching makes repeated prompts look far faster), and whether tool definitions or system prompts are long. Replay sampled production prompts, or fix lengths that match production percentiles.
- **Percentiles, not means**: report p50, p90, p99. Tail latency is what users and SLOs see ([[cost-and-latency]]).
- **Coordinated omission**: a closed-loop generator that waits for slow responses sends fewer requests during the slow period and under-reports tail latency. Open-loop arrival schedules avoid it.

## When to use / scenarios
- Before launching a self-hosted model: find the max request rate per replica that meets the SLO, then set the autoscaling target below it ([[kubernetes-for-llm-serving]], [[inference-servers-vllm]]).
- Choosing hardware or engine: same workload against vLLM vs SGLang vs TensorRT-LLM, FP16 vs FP8/INT4, or two GPU types ([[sglang-and-tensorrt-llm]], [[quantization]], [[gpu-cloud-options]]).
- Tuning server flags (max batched tokens, max sequences, chunked prefill, speculative decoding) with a before/after run.
- Hosted APIs and gateways: measure TTFT and rate-limit behaviour at expected peak, including retries and fallbacks ([[llm-gateways-and-routing]]).
- CI performance regression: a short fixed-rate run on every engine or model upgrade, failing if p99 TTFT degrades beyond a threshold.
- Not for model quality: a faster config can be a worse model. Run your eval set too ([[llm-evaluation]]).
- Not a substitute for production monitoring; keep tracing real traffic ([[llm-observability]]).

## Setup & code
Self-hosted vLLM: the bundled benchmark client (check `vllm bench serve --help` for the flags your version has).
```bash
pip install vllm
vllm serve <hf-org>/<model> --port 8000 &
vllm bench serve --backend vllm --model <hf-org>/<model> --base-url http://localhost:8000 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --num-prompts 500 --request-rate 4 --max-concurrency 64 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
```
GuideLLM (from the vLLM project) sweeps rates against any OpenAI-compatible server; see its README for the current flags:
```bash
pip install guidellm
guidellm benchmark --target http://localhost:8000 --rate-type sweep --max-seconds 30 \
  --data "prompt_tokens=1024,output_tokens=256"
```
Measuring TTFT/TPOT yourself against an OpenAI-compatible streaming endpoint (one request; wrap it in an arrival schedule like the simulation below):
<!-- skip-check: needs a running OpenAI-compatible server and the openai package -->
```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

def timed_request(prompt, model="<model-id>", max_tokens=256):
    t0 = time.perf_counter(); first = None; stamps = []
    stream = client.chat.completions.create(model=model, max_tokens=max_tokens, stream=True,
                                            messages=[{"role": "user", "content": prompt}])
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            first = first or now
            stamps.append(now)
    e2e = time.perf_counter() - t0
    n = len(stamps)     # chunks, not tokens: count tokens with the model's tokenizer for exact TPOT
    return {"ttft": first - t0, "e2e": e2e, "tpot": (e2e - (first - t0)) / max(n - 1, 1)}

print(timed_request("Explain goodput in two sentences."))
```
Runnable, stdlib only: an open-loop Poisson load generator against a simulated server whose decode slows as more requests share the GPU and whose prefill waits for a free slot. It sweeps the arrival rate and prints percentiles and goodput, the same table a real sweep produces. The simulated timings are placeholders, not measurements of any hardware.
```python
import asyncio, random, statistics

random.seed(0)
SLOTS = 8                                 # max concurrent sequences in the fake server
SLO_TTFT, SLO_TPOT = 0.10, 0.008          # seconds

class FakeServer:
    def __init__(self):
        self.sem = asyncio.Semaphore(SLOTS); self.active = 0
    async def generate(self, in_tok, out_tok):
        t0 = loop_time(); stamps = []
        async with self.sem:                                  # queueing when all slots are busy
            self.active += 1
            await asyncio.sleep(in_tok * 5e-6)                # prefill: proportional to prompt
            for _ in range(out_tok):                          # decode slows with batch size
                await asyncio.sleep(0.002 + 0.0005 * self.active)
                stamps.append(loop_time())
            self.active -= 1
        return t0, stamps

def loop_time():
    return asyncio.get_running_loop().time()

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

async def run(rate, n=80):
    server, tasks = FakeServer(), []
    for _ in range(n):                                        # open loop: arrivals do not wait
        in_tok = random.randint(200, 2000); out_tok = random.randint(20, 60)
        tasks.append(asyncio.create_task(server.generate(in_tok, out_tok)))
        await asyncio.sleep(random.expovariate(rate))        # Poisson inter-arrival gaps
    results = await asyncio.gather(*tasks)
    ttft, tpot, ok = [], [], 0
    for t0, s in results:
        a = s[0] - t0; b = (s[-1] - s[0]) / (len(s) - 1)
        ttft.append(a); tpot.append(b); ok += (a <= SLO_TTFT and b <= SLO_TPOT)
    span = max(s[-1] for _, s in results) - min(t0 for t0, _ in results)
    return {"rate": rate, "ttft_p50": statistics.median(ttft), "ttft_p99": pct(ttft, 99),
            "tpot_p99": pct(tpot, 99), "slo_ok": ok / n, "goodput_rps": ok / span}

async def main():
    print("rate  ttft_p50  ttft_p99  tpot_p99  slo_ok  goodput")
    for rate in (10, 20, 40, 80, 160):
        r = await run(rate)
        print(f"{r['rate']:>4}  {r['ttft_p50']:8.3f}  {r['ttft_p99']:8.3f}  {r['tpot_p99']:8.4f}"
              f"  {r['slo_ok']:6.0%}  {r['goodput_rps']:6.2f}")

asyncio.run(main())
```
Generic HTTP load tools also work: Locust (Python user classes) and k6 (JavaScript scripts) can drive streaming requests, but you must parse the SSE stream yourself to get TTFT; the LLM-specific tools above do that for you.

## Choosing / trade-offs
| Tool | Pick when | Watch out for |
|---|---|---|
| `vllm bench serve` | benchmarking vLLM (also other OpenAI-compatible backends), comparing flags and versions | flags change between releases; random datasets ignore your real prompt mix |
| GuideLLM | rate sweeps and SLO-style reports against any OpenAI-compatible server | check the README for the current CLI; synthetic data by default |
| Locust / k6 | you already run them, need mixed traffic (auth, RAG calls, non-LLM endpoints) or distributed load | no built-in TTFT/TPOT; write the SSE parsing and token counting |
| Your own asyncio script | replaying production traces with exact timing | easy to get coordinated omission or client-side bottlenecks wrong |

- **Throughput vs latency**: larger batches raise tokens per second and cost efficiency but raise TPOT and TTFT. Decide the SLO first, then maximize goodput under it, not raw throughput.
- **Synthetic vs replayed prompts**: synthetic fixed lengths are reproducible and comparable across runs; replayed traces predict production. Use both: fixed for regression CI, replay for capacity planning.
- **Single replica vs fleet**: benchmark one replica to get per-replica capacity, then test the fleet with the load balancer and autoscaler, which adds cold starts and uneven routing.

## Gotchas
- Benchmarking the client: a single Python process with a synchronous client saturates before the server does. Check client CPU, use async or several load-generator machines.
- Prefix caching inflates results when every request shares the same prompt; randomize or replay real prompts, or disable it for an apples-to-apples engine comparison.
- Counting chunks as tokens: a streamed chunk can carry several tokens. Use the response `usage` (if the server returns it for streams) or the model's tokenizer.
- Forgetting warm-up: the first requests pay CUDA graph capture, compilation and cold caches. Discard a warm-up phase.
- `max_tokens` set high with `ignore_eos` off gives variable output lengths; for comparable runs fix output length (vLLM's random dataset does) and state it in the report.
- Hosted APIs: you are measuring the provider plus your network and rate limits; 429s are part of the result, not noise. Do not load test someone else's API beyond your contracted limits.
- Report the full setup with every number: model, quantization, GPU, engine version and flags, input/output lengths, rate or concurrency, and duration. A tokens-per-second figure without these is not comparable.

## Related
- [[disaggregated-prefill-and-kv-cache-offloading]] - a fix for ITL spikes from long prefills; measure TTFT and ITL per pool.
- [[speculative-decoding-in-serving]] - a decode-latency lever to benchmark with real prompts and temperature.
- [[inference-servers-vllm]] - the server most often benchmarked, and the flags you tune.
- [[sglang-and-tensorrt-llm]] - alternative engines to compare on the same workload.
- [[cost-and-latency]] - what TTFT and throughput mean for user experience and cost.
- [[llm-cost-calculator]] - turn measured tokens per GPU-hour into cost per request.
- [[kubernetes-for-llm-serving]] - set autoscaling targets from the measured knee.
- [[llm-observability]] - the same metrics on production traffic.
- [[speculative-decoding-from-scratch-numpy]] - a decode-speed lever to verify with a benchmark.
- [[inference-server-and-gpu-metrics]] - server-side queue, KV-cache and GPU metrics to read during a load test.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/performance-and-load-testing.md - load testing practice in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - turning per-replica capacity into fleet size.

## References
- vLLM `bench serve` CLI: https://docs.vllm.ai/en/latest/cli/bench/serve.html
- vLLM benchmark suites: https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- GuideLLM: https://github.com/vllm-project/guidellm
- Locust documentation: https://docs.locust.io/en/stable/
- Grafana k6 documentation: https://grafana.com/docs/k6/latest/
- Zhong et al., DistServe (goodput under TTFT/TPOT SLOs): https://arxiv.org/abs/2401.09670
