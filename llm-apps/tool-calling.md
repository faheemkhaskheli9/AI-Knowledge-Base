---
title: Tool calling (function calling)
category: llm-apps
tags: [tool-use, function-calling, tools, anthropic, openai, agentic]
use_cases:
  - "let a support bot look up order status and issue refunds through internal APIs"
  - "give an LLM access to a calculator, database or search engine"
  - "build a finance assistant that queries account balances safely"
  - "run several independent API lookups in parallel from one user question"
  - "force the model to always call a specific function for classification"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
  - https://platform.openai.com/docs/guides/function-calling
---

# Tool calling (function calling)

## Summary
Tool calling lets a model request that your code run a function: you describe tools with a JSON Schema, the model replies with a structured call, you execute it and return the result, and the model continues. It is the mechanism behind every agent ([[agents]]) and connects LLMs to live data and actions.

## Key concepts
- **Client tools** run in your app (your functions; Anthropic-schema tools like `bash`, `text_editor`). **Server tools** (web search, web fetch, code execution) run on the provider's infrastructure and return results inline.
- **Round trip (Anthropic)**: send `tools=[{name, description, input_schema}]` -> response with `stop_reason == "tool_use"` and `tool_use` blocks (`id`, `name`, `input`) -> append the assistant content, then a user message with `tool_result` blocks (`tool_use_id`, `content`, optional `is_error`) -> call again until the model stops with `end_turn`.
- **OpenAI**: `tools=[{"type":"function","function":{name, description, parameters}}]`; response `message.tool_calls`; reply with `role:"tool"` messages carrying `tool_call_id`.
- **tool_choice**: `auto` (model decides), `any`/`required` (must call some tool), a specific tool (forced), `none`. `disable_parallel_tool_use` limits to one call per turn.
- **Parallel calls**: one response may hold several tool calls; return all results in one user turn.
- **Strict mode**: `strict: true` guarantees arguments match the schema ([[structured-output]]).
- **Tool Runner** (SDK helper) executes the loop for you; MCP lets tools be shared across apps ([[model-context-protocol]]).
- **Tool descriptions are prompts**: name, purpose, when to use/not use, parameter meaning, units, examples.

## When to use / scenarios
- Data lookups (order status, patient schedule, stock price), actions (create ticket, send email), computation (calculator, code execution).
- Customer support: `get_order`, `issue_refund(order_id, amount)` with limits enforced server-side.
- Analytics assistants: `run_sql` read-only ([[text-to-sql]]).
- Not needed when the model only transforms provided text (use plain/structured output) or when a fixed pipeline always fetches the same data first (just do RAG, [[rag-basics]]).

## Setup & code
```python
# pip install anthropic
import json, anthropic
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

tools = [{
    "name": "get_order",
    "description": "Look up an order by id. Returns status and ETA. Use before answering any order question.",
    "input_schema": {"type": "object",
                     "properties": {"order_id": {"type": "string"}},
                     "required": ["order_id"]},
}]
def get_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "eta": "2026-10-07"}
REGISTRY = {"get_order": get_order}

messages = [{"role": "user", "content": "Where is order A123?"}]
for _ in range(8):                                   # hard iteration cap
    r = client.messages.create(model=MODEL, max_tokens=1024, tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": r.content})
    if r.stop_reason != "tool_use":
        break
    results = []
    for b in r.content:
        if b.type == "tool_use":
            try:
                out, err = json.dumps(REGISTRY[b.name](**b.input)), False
            except Exception as e:                   # report errors to the model
                out, err = f"error: {e}", True
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
    messages.append({"role": "user", "content": results})
print(next(b.text for b in r.content if b.type == "text"))
```

## Choosing / trade-offs
- **Few well-named tools vs many**: accuracy drops as tool count rises; group or use tool search / dynamic loading for dozens+.
- **Coarse vs fine tools**: one `search_orders(filters)` beats five near-duplicates, but too generic a tool pushes logic into prompts.
- **Auto vs forced**: force a tool for classification-style extraction; keep auto for agents.
- **Own loop vs SDK runner vs framework**: see [[agent-frameworks]].
- **Read vs write tools**: separate them; require confirmation or approval for irreversible writes.

## Gotchas
- Always return a `tool_result` for every `tool_use` id, in the next user message, or the API errors.
- Model-supplied arguments are untrusted input: validate, authorize against the end user (not the agent), parameterize SQL, enforce spend/limits in code.
- Tool results are an injection channel ([[prompt-injection]]); do not let fetched content trigger writes unchecked.
- Large tool outputs bloat context; truncate/summarize and paginate.
- Missing required info: some models guess a value instead of asking; add "ask the user if a required parameter is missing".
- Non-idempotent tools + retries cause duplicate actions; use idempotency keys.
- Always cap iterations and log every call ([[guardrails-and-safety]]).

## Related
- [[agents]] - the loop built on tool calling.
- [[model-context-protocol]] - standard way to expose tools.
- [[structured-output]] - strict schemas.
- [[prompt-injection]] - tool outputs as attack surface.
- [[text-to-sql]] - a common tool.

## References
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- https://platform.openai.com/docs/guides/function-calling
