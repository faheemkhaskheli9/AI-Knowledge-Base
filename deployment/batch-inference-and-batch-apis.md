---
title: Batch inference and provider batch APIs
category: deployment
tags: [batch-api, message-batches, openai-batch, gemini-batch, vllm-run-batch, offline-inference, cost, throughput, jsonl]
use_cases:
  - "classify or summarize a million documents overnight at the lowest cost"
  - "run a large LLM evaluation or synthetic-data job without hitting rate limits"
  - "decide between a provider batch API and running vLLM offline on our own GPUs"
  - "backfill embeddings or extracted fields for an entire database table"
status: draft
last_verified: 2026-10-15
sources:
  - https://platform.claude.com/docs/en/build-with-claude/batch-processing
  - https://developers.openai.com/api/docs/guides/batch
  - https://help.openai.com/en/articles/9197833-batch-api-faq
  - https://ai.google.dev/gemini-api/docs/batch-api
  - https://docs.vllm.ai/en/latest/cli/run-batch/
  - https://docs.vllm.ai/en/latest/serving/offline_inference.html
---

# Batch inference and provider batch APIs

## Summary
Batch inference runs many LLM (or ML) requests asynchronously when nobody is waiting on each answer: nightly classification, document extraction, evals, embedding backfills, synthetic data. Hosted providers offer batch APIs at about half the normal per-token price in exchange for turnaround of up to 24 hours. Self-hosted, an offline engine (vLLM `LLM.generate` or `vllm run-batch`) keeps GPUs fully busy with no HTTP serving layer. Pick by volume, deadline, data-residency needs and whether you already own GPUs.

## Key concepts
- **Provider batch APIs** (as of last_verified; check each page for current terms):

  | Provider | Discount | Turnaround | Limits / notes |
  |---|---|---|---|
  | Anthropic Message Batches | 50% vs standard | most finish in under 1 h; expire if not done in 24 h | up to 100,000 requests or 256 MB per batch; results downloadable for 29 days |
  | OpenAI Batch | 50% vs synchronous | `completion_window: "24h"` (only option) | JSONL file uploaded with `purpose="batch"`; not every model supports it |
  | Gemini Batch Mode | 50% vs standard | target 24 h, often faster | inline requests under 20 MB, larger jobs via a JSONL file |

- **Request format**: each request carries a `custom_id` you choose. Results come back **in arbitrary order**, so join on `custom_id`, never on position. OpenAI and `vllm run-batch` use JSONL lines `{"custom_id", "method": "POST", "url": "/v1/chat/completions", "body": {...}}`. Anthropic takes `requests=[{custom_id, params}]` where `params` is a normal Messages request.
- **Per-request outcomes**: a batch can "end" while some requests failed. Anthropic reports `succeeded`, `errored`, `canceled`, `expired` per request (expired ones are not billed). OpenAI returns an output file plus an error file.
- **Stacking discounts**: batch pricing combines with prompt caching on providers that support both, so put the shared instructions in a cacheable prefix ([[prompt-caching-and-cost]]).
- **Offline self-hosted inference**: vLLM's `LLM.generate(list_of_prompts)` schedules everything with continuous batching. `vllm run-batch -i in.jsonl -o out.jsonl --model <m>` accepts the OpenAI batch file format, and input/output can be local paths or HTTP(S) URLs. SGLang and TensorRT-LLM have equivalent offline APIs ([[sglang-and-tensorrt-llm]]).
- **Throughput vs latency**: offline jobs care about total tokens per GPU-hour, so use large batches, long max-num-seqs and no streaming; latency per item is irrelevant.

## When to use / scenarios
- Insurance or legal: extract fields from 500k scanned contracts after OCR, results needed by morning ([[document-parsing]]).
- E-commerce: rewrite or tag the whole product catalog weekly ([[ecommerce-retail]]).
- Support analytics: classify yesterday's tickets for dashboards ([[customer-support]]).
- ML teams: run eval suites and LLM-as-judge scoring on every model candidate ([[llm-evaluation]]); generate synthetic training data ([[data-labeling-and-synthetic-data]]).
- RAG: re-embed a corpus after changing the embedding model ([[embedding-models]]).
- Not for anything a user waits on (chat, autocomplete, agents mid-task). Use normal APIs with streaming ([[cost-and-latency]]).
- Not for tiny jobs: under a few hundred requests the sync API with concurrency finishes sooner and the saving is negligible.

## Setup & code
Anthropic Message Batches (model ID from [[anthropic-claude]]):
<!-- skip-check: needs ANTHROPIC_API_KEY and network -->
```python
import time, anthropic
client = anthropic.Anthropic()
docs = {"t-001": "Refund not received after 10 days.", "t-002": "How do I export my data?"}
batch = client.messages.batches.create(requests=[
    {"custom_id": cid, "params": {
        "model": "claude-haiku-4-5", "max_tokens": 20,
        "system": "Classify the ticket as billing, account, bug or other. Reply with one word.",
        "messages": [{"role": "user", "content": text}]}}
    for cid, text in docs.items()])
while client.messages.batches.retrieve(batch.id).processing_status != "ended":
    time.sleep(60)
for r in client.messages.batches.results(batch.id):        # arbitrary order: key on custom_id
    if r.result.type == "succeeded":
        print(r.custom_id, r.result.message.content[0].text)
    else:
        print(r.custom_id, "->", r.result.type)                # errored / canceled / expired: retry
```
OpenAI Batch (model ID from [[openai-gpt]]):
<!-- skip-check: needs OPENAI_API_KEY and network -->
```python
from openai import OpenAI
client = OpenAI()
f = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
job = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
job = client.batches.retrieve(job.id)          # poll until status == "completed"
if job.status == "completed":
    print(client.files.content(job.output_file_id).text[:500])
```
Self-hosted with vLLM, same JSONL file:
```bash
vllm run-batch -i requests.jsonl -o results.jsonl --model <hf-org>/<model>
```
Provider-neutral job plumbing (stdlib, runs offline): build the JSONL, simulate out-of-order results with some failures, join on `custom_id`, and work out which requests to resubmit. The cost line uses the documented 50% discount on whatever per-token prices you plug in.
```python
import json, random

random.seed(1)
tickets = {f"t-{i:05d}": f"ticket text {i}" for i in range(1000)}

def to_line(cid, text, model="<model-id>"):
    return json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                       "body": {"model": model, "max_tokens": 5,
                                "messages": [{"role": "system", "content": "Label: billing/account/bug/other"},
                                             {"role": "user", "content": text}]}})

jsonl = "\n".join(to_line(c, t) for c, t in tickets.items())
assert len({json.loads(l)["custom_id"] for l in jsonl.splitlines()}) == len(tickets)  # ids must be unique

# Simulated results file: shuffled order, 2% failures
results = []
for cid in random.sample(list(tickets), k=len(tickets)):
    ok = random.random() > 0.02
    results.append({"custom_id": cid, "ok": ok, "label": random.choice(["billing", "bug"]) if ok else None})

labels = {r["custom_id"]: r["label"] for r in results if r["ok"]}
retry = sorted(set(tickets) - set(labels))
print(f"{len(labels)} labelled, {len(retry)} to resubmit, e.g. {retry[:3]}")

def job_cost(n, in_tok, out_tok, price_in, price_out, batch=True):   # prices per million tokens
    cost = n * (in_tok * price_in + out_tok * price_out) / 1e6
    return cost * (0.5 if batch else 1.0)

p_in, p_out = 1.0, 5.0     # placeholders: copy your model's prices from its pricing page
sync, bat = job_cost(1_000_000, 400, 5, p_in, p_out, False), job_cost(1_000_000, 400, 5, p_in, p_out)
print(f"1M tickets: sync {sync:,.0f} vs batch {bat:,.0f} (same currency units as the prices)")
```

## Choosing / trade-offs
| Option | Pick when | Watch out for |
|---|---|---|
| Provider batch API | no GPUs, frontier-model quality needed, deadline of hours is fine | 24 h worst case, per-batch size limits, data leaves your network |
| Sync API + client concurrency | small jobs, need results in minutes | full price, rate limits, you write retries |
| vLLM/SGLang offline on rented GPUs | very large or recurring volume, open model good enough, data must stay private | GPU ops, model quality gap, idle cost if jobs are rare ([[gpu-cloud-options]]) |
| Managed cloud batch (Bedrock, Vertex AI, Azure) | already on that cloud, governance needs | different limits and formats; check that cloud's docs ([[managed-cloud-ai-platforms]]) |

- **Break-even**: self-hosting wins once GPUs stay busy. Estimate tokens per GPU-hour on your own prompts with a short trial run, then compare with batch-API per-token price x total tokens. Do not use blog throughput numbers.
- **Turnaround SLO**: if a downstream job must start at 06:00, submit early and handle `expired` requests with a sync-API fallback.
- **Model choice**: batch makes larger models cheaper, but a smaller model at sync price can still be cheaper than a large one in batch ([[model-selection]]).

## Gotchas
- One malformed JSONL line can reject the whole OpenAI file; validate every line (JSON parses, unique `custom_id`, model name) before upload.
- Results are unordered; joining by index silently mislabels data.
- A batch that "ended" is not a batch that succeeded; count per-request outcomes and resubmit failures from an idempotent pipeline that skips already-stored `custom_id`s.
- Provider-side retention: results expire (29 days for Anthropic); download and store them promptly. Check zero-data-retention terms if data is sensitive ([[ai-security-privacy-compliance]]).
- Batches can slightly exceed workspace spend limits because of concurrent processing (Anthropic docs); set budgets with margin.
- Pin `temperature` and the model ID so reruns are comparable; log the batch ID with your dataset version ([[experiment-tracking]]).
- For self-hosted jobs, checkpoint progress (write results as you go) so a preempted spot GPU does not restart from zero.

## Related
- [[prompt-caching-and-cost]] - caching that stacks with batch discounts.
- [[cost-and-latency]] - when async processing is the right lever.
- [[inference-servers-vllm]] - the offline `LLM` engine and `run-batch`.
- [[sglang-and-tensorrt-llm]] - alternative engines for offline jobs.
- [[kubernetes-for-llm-serving]] - running offline jobs as Kubernetes Jobs on GPU nodes.
- [[onnx-and-triton-inference-server]] - batch scoring for non-LLM models.
- [[llm-evaluation]] - evals are a classic batch workload.
- [[anthropic-claude]], [[openai-gpt]], [[google-gemini]] - model IDs and provider specifics.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/batch-and-stream-processing.md - batch pipeline design in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/message-queues-and-streams.md - queuing work for async workers.

## References
- Anthropic Message Batches (discount, limits, result types, retention): https://platform.claude.com/docs/en/build-with-claude/batch-processing
- OpenAI Batch guide: https://developers.openai.com/api/docs/guides/batch
- OpenAI Batch API FAQ (50% discount, 24 h window): https://help.openai.com/en/articles/9197833-batch-api-faq
- Gemini Batch Mode: https://ai.google.dev/gemini-api/docs/batch-api
- vLLM `run-batch` CLI: https://docs.vllm.ai/en/latest/cli/run-batch/
- vLLM offline inference: https://docs.vllm.ai/en/latest/serving/offline_inference.html
