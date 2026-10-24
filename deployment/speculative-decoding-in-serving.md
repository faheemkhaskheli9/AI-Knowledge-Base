---
title: Speculative decoding in production serving (vLLM draft models, EAGLE, MTP, n-gram)
category: deployment
tags: [speculative-decoding, vllm, eagle, eagle3, mtp, draft-model, ngram, prompt-lookup, suffix-decoding, latency, inter-token-latency, acceptance-rate]
use_cases:
  - "cut inter-token latency of a self-hosted LLM without changing its outputs"
  - "choose between EAGLE, MTP, a draft model and n-gram speculation in vLLM"
  - "tune num_speculative_tokens and check the acceptance rate on our traffic"
  - "speed up a code-editing or document-rewriting endpoint whose output copies much of its input"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/features/speculative_decoding/
  - https://arxiv.org/abs/2211.17192
  - https://arxiv.org/abs/2302.01318
  - https://arxiv.org/abs/2401.15077
  - https://arxiv.org/abs/2503.01840
  - https://arxiv.org/abs/2401.10774
---

# Speculative decoding in production serving (vLLM draft models, EAGLE, MTP, n-gram)

## Summary
Speculative decoding lets a cheap proposer guess the next few tokens and the served (target) model check all of them in one forward pass, keeping the longest correct prefix. With the standard rejection sampler the output distribution is the target model's own, so quality does not change; what changes is how many tokens each expensive pass yields. In vLLM it is one `--speculative-config` JSON object choosing the proposer: an EAGLE head, the model's own multi-token-prediction (MTP) layers, a separate small draft model, or draft-free n-gram or suffix matching against the prompt and earlier outputs. The vLLM docs frame it as a way to reduce inter-token latency for medium-to-low QPS, memory-bound workloads; at high load the gain shrinks, so measure it on your traffic before keeping it on. The mechanism is built from scratch in [[speculative-decoding-from-scratch-numpy]]; this topic is about running it.

## Key concepts
- **Proposer types in vLLM** (from its method table): EAGLE / EAGLE-3 heads and native MTP give high gains at low QPS and medium-to-high at high QPS; a separate draft model gives high gains at low QPS, medium at high QPS; n-gram and suffix decoding give low-to-medium gains but add no model and little load. Exact numbers depend on model, traffic, hardware and sampling settings.
- **EAGLE**: a small head trained to predict the target's next features from its hidden states; EAGLE-3 drafts from several layers' features. Needs a head trained for your exact target checkpoint (pretrained heads exist on the Hugging Face Hub; vLLM points to its `speculators` project for training your own).
- **MTP**: some model families ship extra prediction layers trained with the model ([[multi-token-prediction]]); use `"method": "mtp"`, no separate draft. The vLLM docs suggest starting with `num_speculative_tokens: 1`.
- **Draft model**: a smaller model of the same family and, by default, the same tokenizer/vocabulary (`use_heterogeneous_vocab` relaxes this for `draft_model`). It costs GPU memory and its own KV cache.
- **N-gram (prompt lookup) and suffix decoding**: propose the tokens that followed the latest matching n-gram in the prompt (n-gram) or in a cache of earlier outputs (suffix). Strong when outputs copy their inputs: code edits, rewriting, extraction, RAG answers quoting the context, repeated agent outputs.
- **Acceptance**: `mean_acceptance_length = 1 + accepted_draft_tokens / verification_steps` (vLLM per-request metrics, including the bonus token); with per-token acceptance rate `a` and `k` drafts the expected tokens per step are `(1 - a^(k+1)) / (1 - a)` (Leviathan et al.).
- **Speed-up**: with draft cost per token `c` relative to one target pass, the expected wall-time gain is `(1 - a^(k+1)) / ((1 - a)(k*c + 1))` under the paper's assumptions (verification costs about one target pass). Larger `k` helps only while `a` is high; past that, wasted drafts cost more than they save.
- **Why load matters**: at small batch, a decode pass is memory-bandwidth bound, so verifying k+1 tokens costs about as much as one. At large batch the GPU is closer to compute-bound and the extra verification work is no longer free.
- **Losslessness**: vLLM states its standard rejection sampler is lossless up to floating-point and batching numerics, and tests that greedy output matches non-speculative greedy output. Its optional `fly` verifier is deliberately lossy.

## When to use / scenarios
- Interactive chat, coding assistants and agents on self-hosted models where per-user tokens per second (TPOT) is the SLO and the GPU is not saturated ([[llm-load-testing-and-benchmarking]]).
- Code-edit, "rewrite this document", JSON reformatting and extraction endpoints: n-gram or suffix speculation often accepts long runs because the output repeats the input.
- Models with native MTP layers: turn on `mtp` first; it needs no extra artefact.
- Not as a throughput fix for a saturated batch-serving cluster; quantization, more replicas or batch APIs come first ([[quantizing-llms-for-serving]], [[batch-inference-and-batch-apis]]).
- Not available on hosted APIs as a knob; providers decide that themselves ([[cost-and-latency]]).

## Setup & code
vLLM, online server (keys from the vLLM speculative decoding docs; `<draft-model>` must match the target's tokenizer unless you enable heterogeneous vocabularies):
```bash
# Separate draft model
vllm serve <target-model> --speculative-config '{"method": "draft_model", "model": "<draft-model>", "num_speculative_tokens": 5}'
# EAGLE-3 head trained for this exact target
vllm serve <target-model> --speculative-config '{"method": "eagle3", "model": "<eagle3-head>", "num_speculative_tokens": 2}'
# Native multi-token prediction (models that ship MTP layers)
vllm serve <target-model> --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
# Draft-free n-gram prompt lookup, plus per-request acceptance metrics in responses (experimental)
vllm serve <target-model> --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 5}' \
    --per-request-spec-decode-metrics summary
```
The same dictionary goes to the offline engine:
<!-- skip-check: needs vLLM, a GPU and model downloads -->
```python
from vllm import LLM, SamplingParams

llm = LLM(model="<target-model>",
          speculative_config={"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4})
out = llm.generate(["Rewrite in formal English: hey, the meeting moved to 3pm, pls confirm"],
                   SamplingParams(temperature=0.0, max_tokens=64))
print(out[0].outputs[0].text)
```
Runnable, stdlib only: (1) expected tokens per step and speed-up over `k` from the Leviathan et al. formula, (2) reading vLLM's per-request acceptance metrics, (3) a prompt-lookup (n-gram) proposer with greedy verification, showing why it pays off on copy-heavy outputs and not on free text.
```python
# 1) How many drafts? Expected tokens per target pass and wall-time gain.
#    The acceptance rates a and cost ratios c below are illustrative assumptions, not measurements.
#    The formula assumes a constant a per position; real acceptance usually falls with position,
#    so the best k in practice is smaller than this suggests. Read a from your server's metrics.
def expected_tokens(a, k):
    return (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1

def speedup(a, k, c):
    """Leviathan et al.: gain = E[tokens] / (k*c + 1); c = draft cost per token / target pass cost."""
    return expected_tokens(a, k) / (k * c + 1)

for a, c, label in [(0.8, 0.05, "EAGLE-like head, good match"),
                    (0.6, 0.15, "small draft model, weaker match"),
                    (0.8, 0.0, "n-gram on copy-heavy output (free proposer)")]:
    best = max(range(1, 9), key=lambda k: speedup(a, k, c))
    row = "  ".join(f"k={k}:{speedup(a, k, c):.2f}x" for k in (1, 2, 4, 8))
    print(f"{label:44s} {row}  best k={best}")

# 2) vLLM per-request metrics (response field metrics.speculative_decoding, experimental shape).
m = {"mean_acceptance_length": 1.2325581395348837, "draft_acceptance_rate": 0.07751937984496124,
     "acceptance_histogram": [39, 1, 0, 3], "num_spec_steps": 43,
     "num_accepted_draft_tokens": 10, "num_draft_tokens": 129, "num_spec_tokens": 3}
assert abs(m["mean_acceptance_length"] - (1 + m["num_accepted_draft_tokens"] / m["num_spec_steps"])) < 1e-9
steps = m["num_spec_steps"]
reach = [sum(m["acceptance_histogram"][j:]) / steps for j in range(1, m["num_spec_tokens"] + 1)]
print("P(at least j drafts accepted):", [round(r, 2) for r in reach],
      "-> mostly rejected; lower num_spec_tokens or switch method for this traffic")

# 3) Prompt-lookup proposer + greedy verification against a known target output.
def propose(context, k, n_max=3, n_min=1):
    for n in range(n_max, n_min - 1, -1):
        if len(context) < n:
            continue
        tail = context[-n:]
        for i in range(len(context) - n - 1, -1, -1):       # latest earlier occurrence of the tail
            if context[i:i + n] == tail:
                return context[i + n:i + n + k]
    return []

def run(prompt, target_output, k=4):
    """Greedy speculative decoding where the target's next token is known (the target_output)."""
    ctx, out, passes = list(prompt), [], 0
    while len(out) < len(target_output):
        draft = propose(ctx, k)
        passes += 1
        accepted = 0
        for tok in draft:                                  # target verifies all drafts in one pass
            if len(out) + accepted < len(target_output) and tok == target_output[len(out) + accepted]:
                accepted += 1
            else:
                break
        new = target_output[len(out):len(out) + accepted + 1]   # accepted drafts + one token from the target
        out += new; ctx += new
    return len(target_output) / passes

src = ("def total(items):\n    s = 0\n    for x in items:\n        s += x.price\n    return s\n").split(" ")
edit = ("def total(items):\n    s = 0\n    for x in items:\n        s += x.price * x.qty\n    return s\n").split(" ")
prompt = ["Add", "quantity:"] + src + ["Answer:"]
print(f"code edit (output copies input): {run(prompt, edit):.2f} tokens per target pass")
free = "Quarterly revenue grew because the new pricing tier converted more trial users than expected".split()
print(f"free-form answer (little to copy): {run(['Explain', 'the', 'revenue', 'change:'], free):.2f} tokens per target pass")
```

## Choosing / trade-offs
| Method | Needs | Best for | Cost |
|---|---|---|---|
| MTP | a model that ships MTP layers, supported by vLLM | those families, first thing to try | little extra memory |
| EAGLE / EAGLE-3 | a head trained for the exact target checkpoint | general chat and code at low-to-medium load | small head; retrain or re-download when the target changes |
| Draft model | a small same-tokenizer model | families with a good small sibling | draft weights + KV cache, its own latency |
| N-gram / suffix | nothing | copy-heavy outputs, agents repeating themselves | near zero; low gain on free text |

- **`num_speculative_tokens`**: start small (1-2 for MTP and EAGLE as in the vLLM examples; a few more for free proposers), then raise it while mean acceptance length keeps growing; watch the per-position acceptance in metrics.
- **Latency vs throughput**: at peak load speculation can lower total throughput; vLLM's dynamic speculative decoding adjusts to fluctuating load, or route latency-sensitive traffic to a speculative pool and batch traffic to a plain one.
- **Sampling temperature**: acceptance falls as temperature rises, because the draft and target disagree more on sampled tokens; benchmark with production sampling settings, not greedy.
- **Combining with quantization**: a quantized target changes acceptance of a head trained on the unquantized one; re-measure after quantizing ([[quantizing-llms-for-serving]]).

## Gotchas
- Benchmarks with greedy decoding or one repeated prompt overstate gains (and n-gram looks magical on templated prompts); replay real prompts at production temperature ([[llm-load-testing-and-benchmarking]]).
- An EAGLE head or draft model trained for a different checkpoint of the target (another fine-tune, another revision) still runs, but acceptance collapses; pin both together.
- `tensor_parallel_size` is not a key of `speculative_config`; use `draft_tensor_parallel_size`. Sampling parameters such as `temperature` do not go in it either.
- A draft model with a shorter `max_model_len` can lower the engine's maximum context length (vLLM docs).
- The vLLM docs list known incompatibilities by version (for example pipeline parallelism up to `vllm<=0.15.0`); check the list for your version.
- Outputs can still differ run to run with speculation on, through batch-dependent numerics, as they can without it; do not use exact string equality across configurations as a quality test.
- Per-request acceptance metrics are experimental and only reported for `n == 1`; in streaming they arrive with the final usage chunk only when usage reporting is on.

## Related
- [[speculative-decoding-from-scratch-numpy]] - the rejection-sampling rule and acceptance math built from scratch.
- [[multi-token-prediction]] - the training objective behind MTP drafting.
- [[inference-servers-vllm]] - the server these flags belong to.
- [[sglang-and-tensorrt-llm]] - alternative servers that also ship speculative decoding.
- [[llm-load-testing-and-benchmarking]] - measuring TPOT and goodput with and without it.
- [[quantizing-llms-for-serving]] - the other main lever for self-hosted speed and memory.
- [[decoding-and-sampling]] - temperature and sampling settings that drive acceptance.
- [[kv-cache-from-scratch-numpy]] - why a decode step is memory-bound at small batch.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/fundamentals/memory-and-performance.md - measuring before and after an optimization.

## References
- vLLM speculative decoding (methods table, `--speculative-config` keys, EAGLE, MTP, n-gram, suffix, per-request acceptance metrics, lossless guarantees, known incompatibilities): https://docs.vllm.ai/en/latest/features/speculative_decoding/
- Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative Decoding (expected tokens and wall-time improvement formulas): https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling: https://arxiv.org/abs/2302.01318
- Li et al., EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty: https://arxiv.org/abs/2401.15077
- Li et al., EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test: https://arxiv.org/abs/2503.01840
- Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads: https://arxiv.org/abs/2401.10774
