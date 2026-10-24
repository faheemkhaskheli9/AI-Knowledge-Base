---
title: Benchmarking open models locally (lm-evaluation-harness, lighteval)
category: llm-apps
tags: [lm-evaluation-harness, lm-eval, lighteval, benchmarks, mmlu, gsm8k, hellaswag, loglikelihood, acc-norm, chat-template, vllm, regression-testing, fine-tuning]
use_cases:
  - "check that our fine-tuned or quantized model did not lose general ability before we ship it"
  - "run MMLU, GSM8K or HellaSwag on an open-weight model on our own GPUs with vLLM"
  - "benchmark a model already served behind an OpenAI-compatible endpoint (vLLM, llama-server)"
  - "turn our own multiple-choice or QA dataset into a reproducible benchmark task"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/EleutherAI/lm-evaluation-harness
  - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
  - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
  - https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
  - https://github.com/huggingface/lighteval
---

# Benchmarking open models locally (lm-evaluation-harness, lighteval)

## Summary
EleutherAI's **lm-evaluation-harness** (`lm_eval`) and Hugging Face's **lighteval** run standard academic benchmarks (MMLU, GSM8K, HellaSwag, ARC, IFEval and hundreds more) against a model you control: a Hugging Face checkpoint, a vLLM or SGLang engine, a GGUF file, or any OpenAI-compatible server. Use them to answer "did fine-tuning, quantization, merging or a new serving stack change the model's general ability?" with numbers comparable to published ones. They measure benchmark ability, not your product: pair them with an eval set built from your own task ([[llm-evaluation]]).

## Key concepts
- **Request types**: a task asks the model for `loglikelihood` (score fixed continuations, used by multiple-choice tasks), `loglikelihood_rolling` (perplexity) or `generate_until` (free generation, then answer extraction). Chat-completion APIs return no prompt logprobs, so they can only run generative tasks; local models and completions endpoints with logprobs can run all of them.
- **Multiple-choice scoring**: each choice is scored by its log-likelihood after the question. `acc` picks the highest log-likelihood; `acc_norm` divides by the choice's character length first, so long answers are not penalised for having more tokens. Report which one you compare.
- **Generative scoring**: the harness generates, applies filters (for example a regex that pulls the final number out of a GSM8K answer) and computes `exact_match` or similar. Metric keys in the results JSON are `"<metric>,<filter>"`, such as `"acc,none"` or `"exact_match,strict-match"`, with a matching `"<metric>_stderr,<filter>"` from bootstrap resampling.
- **Backends (`--model`)**: `hf` (transformers, with `peft=`, GGUF and `parallelize=True` for multi-GPU sharding), `vllm` and `sglang` (fast, tensor and data parallel), `local-completions` / `local-chat-completions` (any OpenAI-compatible server), `gguf` (a running `llama-server`), plus OpenVINO, ONNX Runtime, NeMo, Megatron-LM and hosted APIs. Since the December 2025 release notes, the base package ships without `transformers`/`torch`; install the extra for the backend you use (`lm_eval[hf]`, `lm_eval[vllm]`, `lm_eval[api]`).
- **Few-shot and chat templates**: `--num_fewshot` sets in-context examples; `--apply_chat_template` wraps prompts in the model's chat template, and `--fewshot_as_multiturn` (on by default with a chat template) turns few-shot examples into turns. Base and instruct models score differently with and without the template; never compare across settings.
- **Reasoning models**: `enable_thinking=True,think_end_token=</think>` in `--model_args` strips the reasoning trace before scoring (generative tasks only).
- **Custom tasks**: a YAML file names a Hugging Face or local dataset (`dataset_path: json` with `data_files`), how to render a prompt (`doc_to_text`), the target (`doc_to_target`, `doc_to_choice`), the output type and metrics; load it with `--include_path`.
- **lighteval**: Hugging Face's toolkit with its own task registry (1000+ tasks per its README), entry points per backend (`lighteval eval` on inspect-ai, `accelerate`, `vllm`, `sglang`, `endpoint`, `custom`) and sample-level result logs; its README says it is untested on Windows.

## When to use / scenarios
- Regression gate after fine-tuning, DPO, merging or abliteration: run the same task list on base and tuned checkpoints and compare with standard errors ([[fine-tuning-and-peft]], [[model-merging]], [[refusal-direction-and-abliteration]]).
- Before swapping a served model for an FP8, AWQ or GGUF quantized version ([[quantizing-llms-for-serving]], [[quantization]], [[llama-cpp-gguf]]).
- Checking that a new serving stack (vLLM version, chat template, tokenizer fix) reproduces the scores of the reference transformers implementation ([[inference-servers-vllm]]).
- Shortlisting open-weight models on public benchmarks in your language or domain before building a task-specific eval ([[model-selection]], [[open-weight-model-landscape]]).
- Not as the main evidence for product quality: benchmark scores say little about your prompts, tools or RAG pipeline; use [[llm-evaluation]], [[rag-evaluation]] or [[agent-evaluation]].
- Not to rank closed API models on multiple-choice tasks: chat APIs give no logprobs, so only generative variants run, and those scores depend on answer extraction ([[reading-ai-benchmarks]]).

## Setup & code
Install the harness with the backend you need, then list and run tasks (commands from the project README and CLI reference):
```bash
pip install "lm_eval[hf,vllm]"           # or lm_eval[api] for OpenAI-compatible servers
lm-eval ls tasks | head

# Quick smoke test: 20 examples per task (limit is for testing only, never report it)
lm_eval --model hf --model_args pretrained=./my-finetuned-model,dtype=bfloat16 \
    --tasks hellaswag,arc_easy --device cuda:0 --batch_size auto --limit 20

# Full run on vLLM with the chat template, saved per-sample outputs and fixed seeds
lm_eval --model vllm \
    --model_args pretrained=./my-finetuned-model,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096 \
    --tasks mmlu,gsm8k,ifeval --apply_chat_template --batch_size auto \
    --output_path results/tuned --log_samples --seed 0

# A LoRA adapter on its base model (hf backend)
lm_eval --model hf --model_args pretrained=<base-model>,peft=./lora-adapter --tasks arc_challenge

# Anything already served behind an OpenAI-compatible completions endpoint (vLLM, SGLang, TGI)
lm_eval --model local-completions \
    --model_args model=<served-name>,base_url=http://localhost:8000/v1/completions,num_concurrent=4,tokenized_requests=False \
    --tasks gsm8k

# A GGUF model served by llama-server
lm_eval --model gguf --model_args base_url=http://127.0.0.1:8080 --tasks hellaswag

# lighteval: separate install; entry points per backend (eval, accelerate, vllm, sglang, endpoint, custom)
pip install lighteval
lighteval --help
```
Python API, for a notebook or a CI job:
<!-- skip-check: needs lm_eval[hf], torch and a model download -->
```python
import json
import lm_eval

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=./my-finetuned-model,dtype=bfloat16",
    tasks=["hellaswag", "arc_easy"],
    num_fewshot=0,
    batch_size="auto",
    log_samples=False,
)
print(json.dumps(results["results"], indent=2))   # {"hellaswag": {"acc,none": ..., "acc_norm,none": ..., ...}}
```
A custom multiple-choice task over a local JSONL file (fields: `question`, `choices`, `label`), saved as `tasks/support_mc.yaml` and run with `--include_path tasks --tasks support_mc`:
```yaml
task: support_mc
dataset_path: json
dataset_kwargs:
  data_files: data/support_mc.jsonl
test_split: train
output_type: multiple_choice
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_choice: "{{choices}}"
doc_to_target: label
metric_list:
  - metric: acc
  - metric: acc_norm
```
Runnable, stdlib only: how `acc` and `acc_norm` pick different answers from the same log-likelihoods, and a comparison of two results files that flags only differences larger than the combined standard error (the files below mimic the harness's `"metric,filter"` keys; values are made up for the demo).
```python
import json
import math

def mc_predictions(choices, logliks):
    """acc: argmax log-likelihood. acc_norm: argmax log-likelihood / character length of the choice."""
    acc = max(range(len(choices)), key=lambda i: logliks[i])
    acc_norm = max(range(len(choices)), key=lambda i: logliks[i] / len(choices[i]))
    return acc, acc_norm

choices = [" Yes", " No, it must be returned within 30 days with a receipt"]
logliks = [-2.1, -9.0]                       # longer answers collect more negative log-probability
acc, acc_norm = mc_predictions(choices, logliks)
print("acc picks", repr(choices[acc]), "| acc_norm picks", repr(choices[acc_norm]))

base = json.loads("""{"results": {
  "arc_easy":  {"acc,none": 0.781, "acc_stderr,none": 0.0085, "acc_norm,none": 0.752, "acc_norm_stderr,none": 0.0089},
  "gsm8k":     {"exact_match,strict-match": 0.612, "exact_match_stderr,strict-match": 0.0134},
  "hellaswag": {"acc_norm,none": 0.801, "acc_norm_stderr,none": 0.0040}}}""")
tuned = json.loads("""{"results": {
  "arc_easy":  {"acc,none": 0.776, "acc_stderr,none": 0.0086, "acc_norm,none": 0.748, "acc_norm_stderr,none": 0.0089},
  "gsm8k":     {"exact_match,strict-match": 0.548, "exact_match_stderr,strict-match": 0.0137},
  "hellaswag": {"acc_norm,none": 0.799, "acc_norm_stderr,none": 0.0040}}}""")

def compare(base, tuned, z=1.96):
    rows = []
    for task, metrics in base["results"].items():
        for key, b in metrics.items():
            metric, filt = key.split(",", 1)
            if metric.endswith("_stderr") or key not in tuned["results"].get(task, {}):
                continue
            t = tuned["results"][task][key]
            se = math.hypot(metrics[f"{metric}_stderr,{filt}"], tuned["results"][task][f"{metric}_stderr,{filt}"])
            delta = t - b
            verdict = "REGRESSION" if delta < -z * se else "improved" if delta > z * se else "within noise"
            rows.append((task, key, b, t, delta, verdict))
    return rows

for task, key, b, t, delta, verdict in compare(base, tuned):
    print(f"{task:10s} {key:26s} {b:.3f} -> {t:.3f}  {delta:+.3f}  {verdict}")
assert any(v == "REGRESSION" and task == "gsm8k" for task, _, _, _, _, v in compare(base, tuned))
```
The standard-error test treats the two runs as independent; both models answer the same questions, so a paired test on `--log_samples` outputs is tighter when the difference is borderline ([[statistics-and-ab-testing]]).

## Choosing / trade-offs
| Need | lm-evaluation-harness | lighteval |
|---|---|---|
| Widest set of classic benchmarks, scores comparable to many papers | Yes; it is the backend of the Open LLM Leaderboard per its README | Large registry too, different task implementations |
| Backends | hf, vLLM, SGLang, GGUF/llama-server, OpenAI-compatible, ONNX Runtime, OpenVINO, NeMo, Megatron, hosted APIs | accelerate, vLLM, SGLang, nanotron, TGI, Inference Endpoints, LiteLLM, inspect-ai |
| Custom task format | YAML + optional `utils.py` | Python task definitions |
| Windows | Works with the hf backend | README: untested on Windows |

- **hf vs vllm backend**: hf is the reference implementation the harness compares against; vLLM is much faster for generative tasks and large models. The README notes outputs can differ slightly and ships a comparator script.
- **Server endpoint vs in-process**: `local-completions` measures exactly what production serves (template, quantization, sampling defaults) at the cost of HTTP overhead; in-process backends are faster and give full logprobs.
- **Which tasks**: pick a small fixed suite that covers what the change could break (knowledge: MMLU; reasoning: GSM8K; instruction following: IFEval; language: a benchmark in your users' language), and keep it fixed across runs.

## Gotchas
- **`--limit` scores are not results**: the CLI reference marks `--limit` as for testing only; small subsets have large standard errors and are not comparable to published numbers.
- **Settings must match**: number of shots, chat template on or off, system prompt, `acc` vs `acc_norm` and the task version all move scores. Compare runs only when the saved config in the results JSON matches.
- **Tokenizer and BOS issues**: GGUF loaded through the hf backend without a separate `tokenizer=` can take hours rebuilding the tokenizer (README warning); some quantized models are sensitive to a missing BOS token ([[quantizing-llms-for-serving]]).
- **Chat endpoints cannot run loglikelihood tasks**: `local-chat-completions` supports `generate_until` only; use `local-completions` against the same server for multiple-choice tasks.
- **Answer extraction fails silently**: generative tasks depend on a regex filter; a model that answers in a different format scores zero. Inspect `--log_samples` output on a few examples before trusting a drop.
- **Contamination**: public benchmarks may be in the training data of the model or your fine-tuning set; a jump after fine-tuning on web data can be leakage ([[reading-ai-benchmarks]], [[post-training-datasets-and-chat-templates]]).
- **Code-execution tasks run model output**: tasks that execute generated code need `--confirm_run_unsafe_code`; run them in a sandbox ([[sandboxed-code-execution]]).
- **`--trust_remote_code`** runs code from the model repo; only use it for repositories you trust.

## Related
- [[llm-evaluation]] - task-specific eval sets, the evidence that matters for a product.
- [[reading-ai-benchmarks]] - what benchmark numbers mean and how they mislead.
- [[quantizing-llms-for-serving]] - checking a quantized checkpoint against the original.
- [[fine-tuning-and-peft]] - the training whose side effects these benchmarks catch.
- [[unsloth-finetuning]] - a common fine-tuning setup to gate with a benchmark run.
- [[inference-servers-vllm]] - the backend most runs use for speed.
- [[llama-cpp-gguf]] - GGUF models scored through llama-server.
- [[llm-load-testing-and-benchmarking]] - serving speed benchmarks, the other meaning of "benchmark".
- [[model-registry-and-versioning]] - storing eval results next to each model version.

## References
- lm-evaluation-harness README (install extras, backends, API model table, caching, GGUF, vLLM notes): https://github.com/EleutherAI/lm-evaluation-harness
- CLI reference (`--limit`, `--apply_chat_template`, `--fewshot_as_multiturn`, `--seed`, thinking models): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md
- Python API (`simple_evaluate`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
- New task guide (YAML tasks, local datasets, `doc_to_text`/`doc_to_choice`): https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md
- `acc` / `acc_norm` computation: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py
- lighteval README (entry points, tasks, Python API): https://github.com/huggingface/lighteval
