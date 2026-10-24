---
title: Monitoring LLM inference servers and GPUs (vLLM and SGLang Prometheus metrics, DCGM exporter, alerts)
category: deployment
tags: [monitoring, prometheus, grafana, metrics, vllm, sglang, llama-cpp, dcgm, dcgm-exporter, gpu-utilization, kv-cache, ttft, alerting, slo]
use_cases:
  - "build a Grafana dashboard and alerts for a self-hosted vLLM or SGLang deployment"
  - "tell whether slow responses come from queueing, KV-cache pressure, long prompts or a sick GPU"
  - "export NVIDIA GPU memory, utilization, temperature and XID errors to Prometheus on Kubernetes"
  - "pick the server metric to autoscale on and the latency SLO to alert on"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/stable/usage/metrics/
  - https://docs.vllm.ai/en/latest/design/metrics/
  - https://docs.sglang.io/references/production_metrics.html
  - https://github.com/NVIDIA/dcgm-exporter
  - https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html
  - https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
  - https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile
---

# Monitoring LLM inference servers and GPUs (vLLM and SGLang Prometheus metrics, DCGM exporter, alerts)

## Summary
A self-hosted LLM endpoint needs two metric layers that request traces do not give you: the inference server's own scheduler metrics (queue depth, running requests, KV-cache usage, preemptions, time to first token, inter-token latency, token throughput) and the GPU's hardware metrics (memory, engine activity, temperature, power, XID errors). vLLM and SGLang expose the first as Prometheus metrics on `/metrics`; NVIDIA's DCGM exporter exposes the second on port 9400. Together they answer the operational questions: are we saturated, why is latency up, which pod to scale, and is a GPU failing. Per-request tracing of prompts and tool calls is a separate layer ([[llm-observability]]).

## Key concepts
- **Endpoints**: vLLM serves Prometheus text on `/metrics` of its OpenAI-compatible server (port 8000 by default). SGLang serves `/metrics` when started with `--enable-metrics`. llama.cpp's `llama-server` exposes a Prometheus endpoint with `--metrics`. DCGM exporter serves `/metrics` on port 9400, with the metric set read from a counters CSV (`/etc/dcgm-exporter/default-counters.csv` in the container).
- **Saturation signals (gauges)**: vLLM `vllm:num_requests_running`, `vllm:num_requests_waiting` and `vllm:kv_cache_usage_perc` (0-1); SGLang `sglang:num_running_reqs`, `sglang:num_queue_reqs` and `sglang:token_usage` (fraction of KV capacity). A non-zero waiting queue means the engine has no room for more sequences; it is the usual autoscaling signal ([[kubernetes-for-llm-serving]]).
- **Latency histograms**: `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds` (older releases: `vllm:time_per_output_token_seconds`), `vllm:e2e_request_latency_seconds`, `vllm:request_queue_time_seconds` and `vllm:request_prefill_time_seconds`; SGLang `sglang:time_to_first_token_seconds`. Prometheus exposes each as `_bucket`, `_sum` and `_count` series, and percentiles come from `histogram_quantile` over rates.
- **Throughput and cache counters**: `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:num_preemptions_total` (sequences evicted for lack of KV blocks and recomputed later), and `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` (counters, so `_total` in the scrape) for the prefix-cache hit rate. Use `rate()` on counters, never their raw values.
- **Names drift between versions**: vLLM renamed several metrics during the V1 engine migration (for example the KV-cache gauge and the prefix-cache counters, which had `gpu_` prefixes), and deprecated ones are hidden after a release or two. Read `/metrics` from your running version before writing dashboards, and pin dashboards to a server version.
- **GPU utilization is not GPU efficiency**: `DCGM_FI_DEV_GPU_UTIL` (the same signal `nvidia-smi` shows) is the fraction of time any kernel was running, so a GPU doing little work per kernel still reads near 100%. Profiling fields such as `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_SM_ACTIVE`, `DCGM_FI_PROF_DRAM_ACTIVE` and `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` show how busy the compute units, memory interface and tensor cores are; enable them in the counters CSV if your image does not.
- **Memory and health fields**: `DCGM_FI_DEV_FB_USED` / `DCGM_FI_DEV_FB_FREE` (framebuffer MB; vLLM preallocates most memory for the KV cache, so FB used is near full by design), `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_POWER_USAGE`, and `DCGM_FI_DEV_XID_ERRORS`, whose value is the code of the last XID error, not a count. XIDs are NVIDIA driver error events; some indicate a GPU or NVLink fault that needs the node drained.
- **Kubernetes attribution**: on Kubernetes (usually deployed by the NVIDIA GPU Operator or the dcgm-exporter Helm chart) the exporter maps GPUs to pods through the kubelet pod-resources API and adds `namespace`, `pod` and `container` labels, so GPU metrics join server metrics by pod.
- **Tensor-parallel caveat (SGLang)**: by default only rank 0 records request metrics, so sums across ranks count each request once; `--enable-metrics-for-all-schedulers` changes that.

## When to use / scenarios
- Any self-hosted vLLM, SGLang or llama.cpp endpoint serving real traffic: dashboard and alerts before launch ([[inference-servers-vllm]], [[sglang-and-tensorrt-llm]]).
- Capacity planning and load tests: read queue depth, KV usage and preemptions while the load generator ramps ([[llm-load-testing-and-benchmarking]]).
- Autoscaling GPU pods on queue depth instead of CPU ([[kubernetes-for-llm-serving]]).
- Diagnosing a latency regression after a model, quantization or config change ([[quantizing-llms-for-serving]], [[speculative-decoding-in-serving]]).
- Not for hosted APIs: there you see only latency, status codes and rate-limit headers from the client side ([[llm-api-rate-limits-and-retries]]).
- Not a replacement for traces and evals: these metrics say the server is fast, not that answers are right ([[llm-observability]], [[llm-evaluation]]).

## Setup & code
Run the DCGM exporter next to a vLLM server and scrape both. Image tags change with each DCGM release; take the current one from the dcgm-exporter README. `SYS_ADMIN` is needed for the profiling (`DCGM_FI_PROF_*`) fields.
```bash
docker run -d --gpus all --cap-add SYS_ADMIN -p 9400:9400 nvcr.io/nvidia/k8s/dcgm-exporter:<tag>
curl -s localhost:9400/metrics | grep -E 'DCGM_FI_DEV_(GPU_UTIL|FB_USED|XID_ERRORS)'

vllm serve <model> --port 8000
curl -s localhost:8000/metrics | grep -E '^vllm:(num_requests|kv_cache|time_to_first)'
```
```yaml
# prometheus.yml (outside Kubernetes; on Kubernetes use ServiceMonitor / PodMonitor objects)
scrape_configs:
  - job_name: vllm
    scrape_interval: 15s
    static_configs: [{targets: ["vllm-host:8000"]}]
  - job_name: dcgm
    scrape_interval: 15s
    static_configs: [{targets: ["vllm-host:9400"]}]
```
Queries and alert rules. Metric names follow the current vLLM docs; adjust to what your `/metrics` shows. Thresholds are placeholders to set from your SLO and load tests.
```yaml
groups:
  - name: llm-serving
    rules:
      - record: vllm:ttft_seconds:p95_5m
        expr: histogram_quantile(0.95, sum by (le, model_name) (rate(vllm:time_to_first_token_seconds_bucket[5m])))
      - record: vllm:generation_tokens:rate1m
        expr: sum by (model_name) (rate(vllm:generation_tokens_total[1m]))
      - record: vllm:prefix_cache_hit_ratio:5m
        expr: sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))
      - alert: LLMQueueBacklog
        expr: sum by (model_name) (vllm:num_requests_waiting) > 0
        for: 5m
        annotations: {summary: "Requests queued for 5m: scale out or cut max tokens"}
      - alert: LLMTTFTSLOBreach
        expr: vllm:ttft_seconds:p95_5m > 2      # placeholder SLO
        for: 10m
      - alert: LLMPreempting
        expr: sum by (pod) (rate(vllm:num_preemptions_total[5m])) > 0
        for: 10m
        annotations: {summary: "KV cache too small for the load: lower max_num_seqs, raise memory or add replicas"}
      - alert: GPUXidError
        expr: DCGM_FI_DEV_XID_ERRORS > 0
        annotations: {summary: "XID {{ $value }} on {{ $labels.Hostname }} GPU {{ $labels.gpu }}: check the XID catalog, drain if fatal"}
      - alert: LLMTargetDown
        expr: up{job=~"vllm|dcgm"} == 0
        for: 2m
```
Runnable, stdlib only: parse a Prometheus text scrape, compute `histogram_quantile` the way Prometheus does (linear interpolation inside the bucket), turn two scrapes of counters into rates, and classify the bottleneck from the combined server and GPU signals.
```python
import re

def parse(text):
    """Prometheus text format -> {(name, frozenset(labels)): value}. Ignores comments and timestamps."""
    out = {}
    for line in text.strip().splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r'([^{\s]+)(?:\{(.*)\})?\s+(\S+)', line)
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        lab = frozenset(re.findall(r'(\w+)="([^"]*)"', labels))
        out[(name, lab)] = value
    return out

def series(m, name, **match):
    return {k[1]: v for k, v in m.items() if k[0] == name and set(match.items()) <= set(k[1])}

def histogram_quantile(q, buckets):
    """buckets: {le: cumulative count}; Prometheus semantics incl. linear interpolation and +Inf handling."""
    bs = sorted(((float("inf") if le == "+Inf" else float(le)), c) for le, c in buckets.items())
    total = bs[-1][1]
    if total == 0:
        return float("nan")
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in bs:
        if c >= rank:
            if le == float("inf"):
                return prev_le                       # quantile in the +Inf bucket: highest finite bound
            return prev_le + (le - prev_le) * (rank - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c

def rate(m0, m1, name, seconds, **match):
    return (sum(series(m1, name, **match).values()) - sum(series(m0, name, **match).values())) / seconds

def ttft_p(m0, m1, q):
    def b(m):
        return {dict(l)["le"]: v for l, v in series(m, "vllm:time_to_first_token_seconds_bucket").items()}
    b0, b1 = b(m0), b(m1)
    return histogram_quantile(q, {le: b1[le] - b0.get(le, 0) for le in b1})   # increase over the window

def scrape(waiting, kv, preempt, ttft_counts, gen_tokens, sm_active, xid=0):
    les = ["0.1", "0.25", "0.5", "1.0", "2.5", "5.0", "+Inf"]
    lines = [f'vllm:num_requests_waiting{{model_name="m"}} {waiting}',
             f'vllm:kv_cache_usage_perc{{model_name="m"}} {kv}',
             f'vllm:num_preemptions_total{{model_name="m"}} {preempt}',
             f'vllm:generation_tokens_total{{model_name="m"}} {gen_tokens}',
             f'DCGM_FI_PROF_SM_ACTIVE{{gpu="0"}} {sm_active}',
             f'DCGM_FI_DEV_XID_ERRORS{{gpu="0"}} {xid}']
    lines += [f'vllm:time_to_first_token_seconds_bucket{{model_name="m",le="{le}"}} {c}' for le, c in zip(les, ttft_counts)]
    return parse("\n".join(lines))

def diagnose(m0, m1, window):
    waiting = sum(series(m1, "vllm:num_requests_waiting").values())
    kv = max(series(m1, "vllm:kv_cache_usage_perc").values())
    preempt = rate(m0, m1, "vllm:num_preemptions_total", window)
    sm = max(series(m1, "DCGM_FI_PROF_SM_ACTIVE").values())
    xid = max(series(m1, "DCGM_FI_DEV_XID_ERRORS").values())
    p95 = ttft_p(m0, m1, 0.95)
    tps = rate(m0, m1, "vllm:generation_tokens_total", window)
    if xid:
        verdict = f"hardware: XID {int(xid)} reported, check the XID catalog and drain the node if fatal"
    elif preempt > 0 or kv > 0.95:
        verdict = "KV-cache bound: preemptions/near-full cache; fewer concurrent seqs, shorter max tokens, more memory or replicas"
    elif waiting > 0 and sm > 0.8:
        verdict = "compute saturated: queue building with busy SMs; add replicas or a smaller/quantized model"
    elif waiting > 0:
        verdict = "queueing with idle compute: check max_num_seqs / batch limits and long-prompt prefill"
    else:
        verdict = "healthy: no queue"
    return f"p95 TTFT {p95:.2f}s, {tps:.0f} gen tok/s, waiting {waiting:.0f}, KV {kv:.0%}, SM {sm:.0%} -> {verdict}"

W = 60.0
base = scrape(0, 0.40, 10, [0, 0, 0, 0, 0, 0, 0], 0, 0.5)
cases = {
    "light load": scrape(0, 0.45, 10, [300, 380, 400, 400, 400, 400, 400], 30_000, 0.55),
    "kv pressure": scrape(6, 0.98, 70, [20, 60, 150, 260, 380, 400, 400], 26_000, 0.70),
    "compute bound": scrape(12, 0.70, 10, [0, 10, 40, 120, 300, 390, 400], 48_000, 0.93),
    "bad gpu": scrape(0, 0.30, 10, [300, 380, 400, 400, 400, 400, 400], 5_000, 0.10, xid=79),
}
for name, m in cases.items():
    print(f"{name:14s} {diagnose(base, m, W)}")
assert abs(histogram_quantile(0.5, {"1": 50, "2": 100, "+Inf": 100}) - 1.0) < 1e-9
assert histogram_quantile(0.99, {"1": 10, "+Inf": 100}) == 1.0
```
The rules in `diagnose` are a starting triage for a runbook, not thresholds from any vendor; calibrate them from your own load tests.

## Choosing / trade-offs
| Question | Primary signal | Secondary |
|---|---|---|
| Should we scale out? | Waiting requests (`num_requests_waiting`, `num_queue_reqs`) | p95 queue time, KV usage |
| Are users waiting too long to see text? | p95/p99 TTFT histogram | Queue time vs prefill time split |
| Is streaming smooth? | Inter-token latency histogram | Running requests (batch size) |
| Is the KV cache too small? | Preemption rate, KV usage near 1 | TTFT spikes under load |
| Is prompt caching working? | Prefix-cache hits / queries | Prompt tokens rate |
| Is the GPU healthy? | XID errors, `up`, temperature, power | Sudden throughput drop on one pod |
| Is the GPU well used? | `DCGM_FI_PROF_SM_ACTIVE`, tensor pipe active | Generation tokens/s per GPU |

- **Scale on queue vs latency**: queue depth reacts first and is cheap to compute; latency percentiles match the user experience but lag and are noisy at low traffic. Scale on queue, alert on latency SLOs ([[cost-and-latency]]).
- **Scrape interval**: 15 s is common; shorter intervals catch bursts but multiply series; histograms with many buckets and per-model labels are the main cardinality cost.
- **Profiling fields cost**: DCGM profiling metrics need extra privileges and add overhead on some GPUs; enable the few you chart.
- **Server metrics vs gateway metrics**: a gateway or client-side measure includes network and retries; server metrics isolate the engine. Keep both ([[llm-gateways-and-routing]]).

## Gotchas
- **Dashboards copied from an older version show nothing**: renamed metrics (`gpu_cache_usage_perc` to `kv_cache_usage_perc`, `gpu_prefix_cache_*` to `prefix_cache_*`, `time_per_output_token` to `inter_token_latency`) return empty series, and an empty series in an autoscaler query reads as zero ([[kubernetes-for-llm-serving]]).
- **Averaging percentiles**: averaging p95 across pods is wrong; sum the bucket rates by `le` first, then call `histogram_quantile`.
- **`rate()` on gauges, raw counters on dashboards**: counters reset on restart and only make sense through `rate`/`increase`; gauges must not be rated.
- **FB used near 100% is normal for vLLM**: it preallocates memory for the KV cache (`--gpu-memory-utilization`); alert on KV usage and preemptions instead of framebuffer.
- **GPU_UTIL 100% looks healthy**: it only says a kernel was running; a decode-heavy, small-batch server shows high utilization with low SM activity and low throughput.
- **XID value is a code, not a count**: `> 0` means an error was reported; the number identifies the error type in NVIDIA's XID catalog.
- **Metrics without the model**: label every series with model and revision (vLLM adds `model_name`); after a canary rollout you must compare old and new side by side ([[ab-testing-and-canary-for-models]]).
- **Prompts in labels**: never put user IDs, prompts or request IDs into metric labels; they explode cardinality and leak data. Keep those in traces.

## Related
- [[llm-observability]] - request-level traces, the layer above these metrics.
- [[opentelemetry-genai-tracing]] - OpenTelemetry for spans; Prometheus here for engine and GPU metrics.
- [[inference-servers-vllm]] - the server whose metrics most dashboards start from.
- [[sglang-and-tensorrt-llm]] - SGLang's own metrics and flags.
- [[kubernetes-for-llm-serving]] - KEDA autoscaling on the queue-depth metric.
- [[llm-load-testing-and-benchmarking]] - load tests that set the thresholds for these alerts.
- [[gpu-cuda-setup]] - drivers and `nvidia-smi`, the local view of the same GPU fields.
- [[disaggregated-prefill-and-kv-cache-offloading]] - when KV pressure and prefill time dominate.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/observability-logs-metrics-traces.md - metrics, logs and traces in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/slis-slos-error-budgets.md - turning TTFT and error rate into SLOs.

## References
- vLLM, Production Metrics (metric names, types, deprecations): https://docs.vllm.ai/en/stable/usage/metrics/
- vLLM, Metrics design (V1 metrics, prefix-cache and prefill-time metrics): https://docs.vllm.ai/en/latest/design/metrics/
- SGLang, Production Metrics (`--enable-metrics`, gauges and histograms, multi-rank behaviour): https://docs.sglang.io/references/production_metrics.html
- NVIDIA DCGM exporter repository (container, port 9400, counters CSV, Kubernetes): https://github.com/NVIDIA/dcgm-exporter
- NVIDIA, DCGM Exporter Metrics: https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html
- NVIDIA, DCGM Field Identifiers (`DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_XID_ERRORS`, profiling fields): https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
- Prometheus, `histogram_quantile` (interpolation, `+Inf` bucket behaviour): https://prometheus.io/docs/prometheus/latest/querying/functions/#histogram_quantile
