---
title: Code models (Qwen3-Coder, Codestral, Devstral) and FIM completion
category: models
tags: [code-models, fim, fill-in-the-middle, autocomplete, qwen3-coder, codestral, devstral, swe-bench, pass-at-k]
use_cases:
  - "self-host a code autocomplete model for an IDE inside our network"
  - "pick an open-weight model for an internal coding agent"
  - "add fill-in-the-middle completion to an editor or notebook product"
  - "decide which coding benchmarks to trust when comparing models"
  - "evaluate code models on our own repository before rolling one out"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/QwenLM/Qwen3-Coder
  - https://github.com/mistralai/client-python
  - https://mistral.ai/news/devstral-2-vibe-cli/
  - https://github.com/SWE-bench/SWE-bench
  - https://arxiv.org/abs/2207.14255
  - https://arxiv.org/abs/2107.03374
---

# Code models (Qwen3-Coder, Codestral, Devstral) and FIM completion

## Summary
Code-specialised models are LLMs trained heavily on source code. They serve
two different jobs: low-latency **completion** in the editor, usually with
fill-in-the-middle (FIM) prompts, and **agentic coding** (multi-file edits,
running tests, using tools). Small open code models are the usual choice for
self-hosted autocomplete; for agents, general frontier models and large
open-weight MoE models now compete directly with code-only models. Judge them
on contamination-resistant benchmarks and, above all, on your own repository.

## Key concepts
- **FIM (fill-in-the-middle)**: the model sees code before and after the cursor
  and generates the middle. Trained by moving a random span to the end with
  sentinel tokens (Bavarian et al. 2022). Common PSM order:
  `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>` (the format the
  Qwen3-Coder README documents; other families use different token names).
- **Completion vs chat**: FIM completion goes to a raw completion/FIM endpoint
  with the model's own sentinel tokens; chat-tuned agents use the chat API and
  tool calls ([[coding-agents]]).
- **Repo context**: completions improve a lot when you add retrieved snippets
  from other files (imports, called functions) ahead of the prefix.
- **Latency budget**: autocomplete needs first tokens in a few hundred ms, so
  small dense or low-active-parameter MoE models win; agents trade latency for
  quality.
- **pass@k**: probability that at least one of k samples passes the tests,
  estimated without bias from n >= k samples with c correct (Chen et al. 2021):
  1 - C(n-c, k) / C(n, k).

Models as of 2026-10 (from each source):

| Model | Access | Notes |
|---|---|---|
| Qwen3-Coder-480B-A35B-Instruct | open weights | 256K native context, YaRN to 1M; FIM in every Qwen3-Coder version |
| Qwen3-Coder-30B-A3B-Instruct | open weights | 3B active: fast enough for local completion |
| Qwen3-Coder-Next (+ Base) | open weights | built on Qwen3-Next-80B-A3B-Base; agentic training; non-thinking only; new tokenizer and tool parser in vLLM/SGLang |
| Codestral | Mistral API ("Premier", see [[mistral]]) | dedicated FIM endpoint in the SDK (`client.fim.complete`) |
| Devstral 2 / Devstral Small 2 | open weights | Mistral's announcement: Devstral 2 modified MIT, Small 2 Apache 2.0; a third-party review reports both as deprecated in the API docs (not verified) |
| GLM-5.x, Kimi K3, MiniMax M3, DeepSeek | open weights + APIs | general MoE models marketed for agentic coding; see [[chinese-open-weight-models]], [[deepseek]] |

Licence for the Qwen3-Coder weights is not stated in the GitHub README; read
each Hugging Face card ([[model-licenses]]).

## When to use / scenarios
- **Self-hosted autocomplete** for regulated code (banks, defence, healthcare)
  where source cannot leave the network: a small FIM-capable open model behind
  [[inference-servers-vllm]] or [[ollama]].
- **Internal coding agent** on open weights: a large agentic MoE (Qwen3-Coder,
  GLM, Kimi) behind an OpenAI-compatible server.
- **Code review, test generation, migrations**: chat-tuned models; see
  [[software-engineering]].
- Not for: the hardest agentic tasks when an API is allowed; compare with
  [[anthropic-claude]] and [[openai-gpt]] on your own tasks first.

## Setup & code
Runnable core ideas: build a FIM prompt, score samples with tests, and compute
unbiased pass@k.
```python
import numpy as np
from math import comb

def fim_prompt(prefix: str, suffix: str) -> str:
    """PSM-order FIM prompt using the Qwen3-Coder sentinel tokens."""
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

prefix = "def area(w, h):\n    "
suffix = "\n\nprint(area(3, 4))\n"
print(fim_prompt(prefix, suffix))

# Pretend the model returned these 5 middles; run hidden tests on each.
samples = ["return w * h", "return w + h", "return w*h", "return h * w", "return w ** h"]

def passes(middle: str) -> bool:
    ns: dict = {}
    try:
        exec(prefix + middle, ns)          # sandbox this in real use
        return ns["area"](3, 4) == 12 and ns["area"](2, 5) == 10
    except Exception:
        return False

results = [passes(m) for m in samples]

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. 2021."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)

n, c = len(results), int(np.sum(results))
print("correct:", c, "of", n)
for k in (1, 3):
    print(f"pass@{k} = {pass_at_k(n, c, k):.3f}")
assert abs(pass_at_k(5, 3, 1) - 0.6) < 1e-9
```

FIM with Qwen3-Coder locally (adapted from the Qwen3-Coder README; needs a
large GPU for this checkpoint):
<!-- skip-check: downloads a large model from Hugging Face; needs torch and a GPU -->
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-Next", device_map="auto").eval()
fim = "<|fim_prefix|>def area(w, h):\n    <|fim_suffix|>\n\nprint(area(3, 4))<|fim_middle|>"
messages = [{"role": "system", "content": "You are a code completion assistant."},
            {"role": "user", "content": fim}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok([text], return_tensors="pt").to(model.device)
out = model.generate(inputs.input_ids, max_new_tokens=64, do_sample=False)[0]
print(tok.decode(out[len(inputs.input_ids[0]):], skip_special_tokens=True))
```
The README also lists explicit `eos_token_id` values for FIM; copy them from
there. Hosted FIM: the Mistral Python SDK exposes `client.fim.complete(model=...,
prompt=prefix, suffix=suffix)`; take the Codestral model ID from Mistral's
models page.

## Choosing / trade-offs
- **Autocomplete**: smallest model that passes your acceptance tests; a 3B-active
  MoE or small dense model, quantized ([[quantization]]), on a local GPU.
- **Agents**: quality dominates; large MoE open weights need multi-GPU nodes
  ([[sglang-and-tensorrt-llm]]), APIs avoid the ops.
- **Benchmarks to trust**, roughly in order:
  1. Your own eval: real tickets and tests from your repos ([[llm-evaluation]]).
  2. SWE-bench Verified (500 issues confirmed solvable by engineers) and newer
     variants (Multilingual, Multimodal v2) for agentic repo work; compare only
     runs with the same scaffold/agent.
  3. Time-split sets such as LiveCodeBench for contamination resistance.
  4. Terminal-style agent benchmarks (Terminal-Bench) for shell-heavy agents.
  Treat HumanEval/MBPP-style function benchmarks as saturated and contamination
  prone ([[reading-ai-benchmarks]]).
- **Licence**: check each checkpoint; "modified MIT" licences can add revenue or
  attribution conditions.

## Gotchas
- Wrong FIM tokens or chat template: the model writes prose or repeats the
  suffix. Use the exact tokens from the model card and stop at the documented
  EOS tokens.
- Qwen3-Coder-Next changed special tokens and token IDs; use the new tokenizer
  (README warning).
- Trailing whitespace and the cursor position inside a line change outputs a
  lot; normalise the prefix/suffix split.
- Vendor scores often use custom agent scaffolds; the same model can score very
  differently in your agent.
- Executing generated code: always sandbox ([[sandboxed-code-execution]]).
- Generated code can include licensed snippets or insecure patterns; run SAST
  and licence scanners in CI.
- Autocomplete can leak secrets present in the prefix to logs; scrub before
  logging ([[llm-observability]]).

## Related
- [[coding-agents]] - agent loops built on these models.
- [[software-engineering]] - scenario view for developer tooling.
- [[reading-ai-benchmarks]] - how to read SWE-bench and contamination claims.
- [[qwen]], [[mistral]], [[deepseek]], [[chinese-open-weight-models]] - vendor files.
- [[open-weight-model-landscape]] - dated table of current open releases.
- [[inference-servers-vllm]] - serving open code models.
- [[sandboxed-code-execution]] - running generated code safely.

## References
- Qwen3-Coder README (models, FIM format): https://github.com/QwenLM/Qwen3-Coder
- Mistral Python SDK (FIM examples): https://github.com/mistralai/client-python
- Devstral 2 announcement: https://mistral.ai/news/devstral-2-vibe-cli/
- SWE-bench: https://github.com/SWE-bench/SWE-bench
- Bavarian et al., "Efficient Training of Language Models to Fill in the Middle": https://arxiv.org/abs/2207.14255
- Chen et al., "Evaluating Large Language Models Trained on Code" (pass@k): https://arxiv.org/abs/2107.03374
