---
title: Tool calling with self-hosted open models (vLLM tool parsers, Ollama, llama.cpp)
category: deployment
tags: [tool-calling, function-calling, vllm, tool-call-parser, chat-template, ollama, llama-cpp, jinja, open-weight, openai-compatible, strict-mode]
use_cases:
  - "make a self-hosted open-weight model return OpenAI-style tool_calls from vLLM"
  - "fix tool calls that come back as raw text in the content field instead of tool_calls"
  - "run an agent against Ollama or llama-server instead of a hosted API"
  - "test whether an open model picks the right tool with valid arguments before switching to it"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/features/tool_calling/
  - https://docs.ollama.com/capabilities/tool-calling
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
  - https://huggingface.co/docs/transformers/en/chat_extras
---

# Tool calling with self-hosted open models (vLLM tool parsers, Ollama, llama.cpp)

## Summary
Hosted APIs return tool calls as structured `tool_calls`; an open-weight model only emits text in whatever tool-call format it was trained on (JSON, XML-like tags, Python-style calls). A self-hosted server makes that look like the OpenAI API in two steps: a **chat template** renders your `tools` and earlier tool results into the prompt in the model's trained format, and a **tool parser** turns the generated text back into `tool_calls`. vLLM needs both chosen per model family (`--enable-auto-tool-choice --tool-call-parser <name>`, sometimes `--chat-template`); llama-server needs `--jinja` and a tool-aware template; Ollama handles it per model in its registry. Most "the model does not call tools" bugs are a missing flag, the wrong parser or a template without tool support, not the model.

## Key concepts
- **Template in, parser out**: the chat template (Jinja, shipped in the checkpoint's `tokenizer_config.json` or as a file) decides how tools and `tool`-role messages appear in the prompt; the parser must match the format the model writes. A mismatch shows up as tool-call markup leaking into `content` and an empty `tool_calls`.
- **vLLM flags**: `--enable-auto-tool-choice` (required for `tool_choice="auto"`), `--tool-call-parser` (per family, e.g. `hermes` for Hermes-style formats that the vLLM docs also recommend for Qwen2.5, `llama3_json`, `llama4_pythonic`, `mistral`, `pythonic`, `qwen3_xml` and many more), optional `--chat-template` (vLLM ships tool templates under `examples/`) and `--tool-parser-plugin` for your own parser.
- **`tool_choice` in vLLM**: named function and `"required"` use structured outputs, so the call is guaranteed to parse (not to be a good call; the docs note exceptions for recent Mistral-format models). `"auto"` extracts calls from free text unless structural-tag constraints apply: a tool with `strict: true`, or the server option `--tool-strict-level function|parameter`. Without them arguments can be malformed or violate the schema. `"none"` still puts tools in the prompt unless `--exclude-tools-when-tool-choice-none` is set.
- **Strict-schema style**: for strict enforcement, vLLM recommends `additionalProperties: false` on every object, every property listed in `required`, and optional fields expressed as nullable types.
- **llama.cpp**: `llama-server --jinja` enables OpenAI-style tools; recognised templates use a native handler, others fall back to a "Generic" format (more tokens, less reliable); parallel calls are off unless the request sets `"parallel_tool_calls": true`; `--chat-template-file` overrides a broken or tool-less template.
- **Ollama**: tools go in the `tools` field of `/api/chat`; tool support depends on the model. In the native API, `arguments` is a JSON object and results go back as `{"role": "tool", "tool_name": ..., "content": ...}`; the OpenAI-compatible endpoint uses the OpenAI shapes (arguments as a JSON string, `tool_call_id`).
- **Transformers**: `apply_chat_template(messages, tools=...)` accepts JSON schemas or Python functions with Google-style docstrings; use it to see exactly what prompt the model receives ([[huggingface-transformers]]).

## When to use / scenarios
- Data must stay on-prem (healthcare, government, finance) but the app is an agent that calls internal APIs ([[tool-calling]], [[agents]]).
- Cutting cost of a high-volume tool-routing step by moving it to a self-hosted model behind the same OpenAI-compatible client ([[inference-servers-vllm]], [[llm-gateways-and-routing]]).
- Local development of an agent or MCP client against Ollama or llama-server before pointing it at production ([[ollama]], [[llama-cpp-gguf]], [[model-context-protocol]]).
- Not when the open models you can host fail your tool-use eval: measure first with the harness below ([[agent-evaluation]]).
- If you only need one JSON object back, not a tool loop, structured output is simpler ([[structured-output]]).

## Setup & code
Serve with tool calling enabled (flags from the vLLM docs; pick the parser listed for your model family there):
```bash
vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
# Llama 3.1/3.2 JSON tool calls need vLLM's template as well:
vllm serve <model> --enable-auto-tool-choice --tool-call-parser llama3_json \
    --chat-template examples/tool_chat_template_llama3.1_json.jinja
# Constrain the tool-call envelope for clients that never set strict:
vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes --tool-strict-level function
# llama.cpp: --jinja is what turns on tool calling
llama-server --jinja -m model.gguf --port 8080
```
Call it with the OpenAI SDK exactly as a hosted API:
<!-- skip-check: needs a running vLLM, llama-server or Ollama server -->
```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")   # Ollama: :11434/v1, llama-server: :8080/v1
tools = [{"type": "function", "function": {
    "name": "get_order", "description": "Look up an order by ID", "strict": True,
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]
resp = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "Where is order A-1001?"}],
    tools=tools, tool_choice="auto")
msg = resp.choices[0].message
for call in msg.tool_calls or []:
    print(call.function.name, json.loads(call.function.arguments))
print("content:", msg.content)            # tool markup here means the parser or template is wrong
```
Runnable, stdlib only: normalize tool calls from OpenAI-compatible and Ollama-native responses, detect markup leaked into `content`, validate arguments against the tool schema, build the error message to send back to the model, and score a candidate model on a small tool-use test set.
```python
import json
import re

TOOLS = {
    "get_order": {"type": "object", "additionalProperties": False,
                  "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    "refund": {"type": "object", "additionalProperties": False,
               "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"},
                              "reason": {"type": "string", "enum": ["damaged", "late", "other"]}},
               "required": ["order_id", "amount", "reason"]},
}
TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "object": dict, "array": list}
LEAK = re.compile(r"<tool_call>|<function=|\[TOOL_CALLS\]|<\|python_tag\|>|\"name\"\s*:\s*\"\w+\"\s*,\s*\"(arguments|parameters)\"")

def normalize(message):
    """Return [(name, args_dict_or_None, raw)] from an OpenAI-style or Ollama-native assistant message."""
    out = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        raw = fn.get("arguments")
        if isinstance(raw, dict):                     # Ollama native API: arguments is already an object
            out.append((fn.get("name"), raw, raw))
            continue
        try:                                           # OpenAI-compatible: arguments is a JSON string
            out.append((fn.get("name"), json.loads(raw or "{}"), raw))
        except json.JSONDecodeError:
            out.append((fn.get("name"), None, raw))
    return out

def leaked(message):
    return bool(LEAK.search(message.get("content") or ""))

def validate(name, args, tools=TOOLS):
    """Small JSON-schema subset: known tool, object, required, types, enum, no extra keys."""
    if name not in tools:
        return [f"unknown tool {name!r}; choose one of {sorted(tools)}"]
    if args is None:
        return ["arguments are not valid JSON"]
    schema, errors = tools[name], []
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"missing required argument {key!r}")
    for key, value in args.items():
        spec = schema["properties"].get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected argument {key!r}")
            continue
        expected = TYPES[spec["type"]]
        if not isinstance(value, expected) or (spec["type"] in ("number", "integer") and isinstance(value, bool)):
            errors.append(f"{key!r} must be {spec['type']}, got {type(value).__name__}")
        elif "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key!r} must be one of {spec['enum']}")
    return errors

def tool_error_message(call_id, errors):
    """Message to append so the model can retry the call (OpenAI-compatible shape)."""
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"error": "invalid arguments", "details": errors})}

# Responses as different servers return them.
vllm_ok = {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
    "name": "refund", "arguments": '{"order_id": "A-1001", "amount": 25.0, "reason": "damaged"}'}}]}
ollama_ok = {"content": "", "tool_calls": [{"function": {"name": "get_order", "arguments": {"order_id": "A-7"}}}]}
wrong_parser = {"content": '<tool_call>{"name": "get_order", "arguments": {"order_id": "A-7"}}</tool_call>',
                "tool_calls": []}
bad_args = {"content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {
    "name": "refund", "arguments": '{"order_id": "A-1001", "amount": "25", "reason": "broken", "note": "x"}'}}]}

for label, m in [("vllm", vllm_ok), ("ollama", ollama_ok), ("wrong parser", wrong_parser), ("bad args", bad_args)]:
    calls = normalize(m)
    problems = [e for name, args, _ in calls for e in validate(name, args)]
    print(f"{label:12s} calls={[c[0] for c in calls]} leaked={leaked(m)} errors={problems}")
print(tool_error_message("c2", validate("refund", normalize(bad_args)[0][1]))["content"])

# Tool-use eval: expected tool and key arguments per prompt; replay a candidate model's responses.
cases = [
    ("Where is order A-1?", "get_order", {"order_id": "A-1"}),
    ("Refund 10 on A-2, it arrived broken", "refund", {"order_id": "A-2", "reason": "damaged"}),
    ("Hi, what are your opening hours?", None, {}),                # should answer, not call a tool
    ("Order A-3 came late, refund 5", "refund", {"order_id": "A-3", "reason": "late"}),
]
candidate = [   # recorded responses from the model under test, one per case
    {"tool_calls": [{"function": {"name": "get_order", "arguments": '{"order_id": "A-1"}'}}]},
    {"tool_calls": [{"function": {"name": "refund", "arguments": '{"order_id": "A-2", "amount": 10, "reason": "damaged"}'}}]},
    {"content": "We are open 9-17.", "tool_calls": []},
    {"tool_calls": [{"function": {"name": "refund", "arguments": '{"order_id": "A-3", "amount": 5, "reason": "other"}'}}]},
]

def score(cases, responses):
    right_tool = valid = right_args = 0
    for (_, want_tool, want_args), resp in zip(cases, responses):
        calls = normalize(resp)
        got_tool = calls[0][0] if calls else None
        right_tool += got_tool == want_tool
        if got_tool is None:
            valid += want_tool is None; right_args += want_tool is None
            continue
        args = calls[0][1]
        ok = not validate(got_tool, args)
        valid += ok
        right_args += ok and got_tool == want_tool and all(args.get(k) == v for k, v in want_args.items())
    n = len(cases)
    return {"tool_selection": right_tool / n, "schema_valid": valid / n, "args_correct": right_args / n}

print(score(cases, candidate))
```

## Choosing / trade-offs
| Server | Pick when | Watch out for |
|---|---|---|
| vLLM | GPU serving for many users, OpenAI-compatible tools, constrained calls | one parser per family; flags and parser names change between releases, re-check the docs on upgrade |
| SGLang | same role as vLLM with heavy prefix reuse ([[sglang-and-tensorrt-llm]]) | its own parser names; check its docs |
| llama-server (llama.cpp) | CPU or small GPU, GGUF quantized models | `--jinja` is required; Generic fallback is less reliable; aggressive KV-cache quantization can hurt tool calling (llama.cpp docs) |
| Ollama | local development, desktop apps | native vs OpenAI-compatible message shapes differ; tool support is per model |

- **`auto` vs `required` vs named**: `required` and named calls use constrained decoding in vLLM, so they parse every time; use them for steps that must call a tool (routing, extraction). Keep `auto` for chat turns where the model may answer directly, and add `strict: true` or `--tool-strict-level` to reduce malformed calls.
- **Native parser vs constrained JSON**: a model's native format with its own parser keeps the behaviour it was trained on; forcing JSON through a grammar guarantees syntax but can lower call quality if the model was trained on another format. Measure both on your eval set.
- **Model size**: small models often pick the right tool but get arguments wrong; score tool selection and argument correctness separately, as above.

## Gotchas
- Tool markup in `content` with empty `tool_calls`: wrong or missing `--tool-call-parser`, missing `--enable-auto-tool-choice`, or a chat template without tool support.
- The model calls a tool once, then ignores the result: the template does not render `tool`-role messages, or you dropped the assistant message with `tool_calls` from the history before appending the result.
- Llama 3.x JSON tool calls in vLLM do not support parallel calls, and the docs warn the model can serialize arrays as strings; validate argument types.
- Mistral's own chat template in Transformers format expects 9-digit tool-call IDs; vLLM ships adjusted templates for this.
- Named or `required` tool choice in vLLM compiles a grammar on first use, so the first request is slow; warm it up before load tests ([[llm-load-testing-and-benchmarking]]).
- Ollama native `arguments` is an object; code written for OpenAI that calls `json.loads` on it fails. Normalize once at the edge.
- Thinking or reasoning models may emit reasoning before the call; configure the matching reasoning parser in vLLM, or the reasoning ends up in `content`.
- Validated arguments are not authorized actions: check permissions and require approval for side effects as with any agent ([[guardrails-and-safety]], [[prompt-injection]]).

## Related
- [[tool-calling]] - the tool loop, schemas and safety, independent of the server.
- [[inference-servers-vllm]] - running vLLM itself.
- [[vllm-local-setup]] - installing vLLM locally and common errors.
- [[ollama]] - local runner with per-model tool support.
- [[llama-cpp-gguf]] - llama-server and GGUF models.
- [[structured-output]] - when one JSON object is enough.
- [[grammar-constrained-decoding-from-scratch]] - how constrained tool calls are enforced.
- [[agent-evaluation]] - scoring tool use end to end.
- [[model-context-protocol]] - exposing tools to any client, including self-hosted ones.
- [[open-weight-model-landscape]] - which open models to try for tool use.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/apis/rest-api-design.md - designing the internal APIs your tools wrap.

## References
- vLLM tool calling (flags, parsers per model family, `tool_choice` modes, strict mode, `--tool-strict-level`, known issues): https://docs.vllm.ai/en/latest/features/tool_calling/
- Ollama tool calling (single, parallel, multi-turn and streaming tool calls, Python SDK functions as tools): https://docs.ollama.com/capabilities/tool-calling
- llama.cpp function calling (`--jinja`, native and Generic handlers, `parallel_tool_calls`, template overrides): https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- Hugging Face Transformers, tool use with chat templates: https://huggingface.co/docs/transformers/en/chat_extras
