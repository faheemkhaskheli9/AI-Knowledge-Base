---
title: LLM agents (agent loop, planning, stopping)
category: llm-apps
tags: [agents, agent-loop, planning, react, tool-use, termination, budgets]
use_cases:
  - "build a research assistant that searches, reads and synthesizes sources"
  - "automate multi-step IT or finance back-office tasks with tool access"
  - "debug why an agent loops forever or burns tokens"
  - "decide whether a workflow should be a fixed pipeline or an agent"
  - "build a customer-support agent that can look up and change orders"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.anthropic.com/engineering/building-effective-agents
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
  - https://arxiv.org/abs/2210.03629
---

# LLM agents (agent loop, planning, stopping)

## Summary
An agent is an LLM running in a loop: it decides the next step, calls tools, observes results, and repeats until the task is done. It suits open-ended tasks where the number of steps is unknown. Most production failures come from missing stop conditions, bloated context, and over-trusting tool output, not from the model.

## Key concepts
- **Workflow vs agent**: workflows follow code-defined paths (prompt chaining, routing, parallelization, evaluator-optimizer); agents choose their own path. Start with the simplest thing; add autonomy only when it measurably helps.
- **Loop (ReAct style)**: think -> act (tool call) -> observe -> repeat. Built on [[tool-calling]].
- **Planning**: let the model write a short plan first (or use reasoning/extended thinking), keep a visible to-do list, re-plan when a step fails. Plan-and-execute splits a planner from cheaper executors.
- **Stopping**: combine (1) the model ending its turn (`end_turn`/no tool call), (2) a hard max-iteration cap sized to the task (about 3 simple, 10-15 research), (3) a token/time/cost budget, (4) explicit failure paths. Never rely on self-reported "done" alone.
- **Handle every stop reason**: tool request, end turn, `max_tokens`, refusal, timeout, error - each needs a distinct branch.
- **Context management**: prune/summarize old tool results, offload to files/memory ([[agent-memory]]), keep the system prompt stable ([[prompt-caching-and-cost]]).
- **Human-in-the-loop**: approval gates before irreversible actions.
- **Observability**: log every step, tool args/results, tokens, latency ([[llm-observability]]).

## When to use / scenarios
- Coding assistants, deep research, support agents with several backend systems, data analysis, browser/computer tasks.
- Signals: steps unknown up front, branching on intermediate results, needs to recover from errors.
- Not agents: fixed extract->classify->write flows (use a chain), single lookup (RAG, [[rag-basics]]), anything needing deterministic compliance steps (code the workflow, call the LLM inside steps).

## Setup & code
Plain loop with caps and a budget (Anthropic SDK):
```python
# pip install anthropic
import anthropic
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

def run_agent(task: str, tools: list, registry: dict, max_steps=12, max_tokens_total=60_000):
    messages = [{"role": "user", "content": task}]
    spent = 0
    for step in range(max_steps):
        r = client.messages.create(model=MODEL, max_tokens=2048, tools=tools, messages=messages,
            system="Plan briefly, use tools, stop when the task is complete. If blocked, say what is missing.")
        spent += r.usage.input_tokens + r.usage.output_tokens
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason == "end_turn":
            return next(b.text for b in r.content if b.type == "text")
        if r.stop_reason != "tool_use":                 # max_tokens, refusal, ...
            raise RuntimeError(f"stopped: {r.stop_reason}")
        if spent > max_tokens_total:
            raise RuntimeError("token budget exceeded")
        results = []
        for b in r.content:
            if b.type == "tool_use":
                try: out, err = str(registry[b.name](**b.input))[:8000], False   # cap tool output
                except Exception as e: out, err = f"error: {e}", True
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("max steps reached")
```
Note: `spent` sums per-call input tokens, which re-counts the growing history each turn; it is a deliberate over-estimate guard against runaway cost.

## Choosing / trade-offs
- **Autonomy vs control**: more freedom = more capability and more variance. Constrain tools, scope, and spend.
- **Single agent vs multi-agent**: single agent with good tools first ([[multi-agent-systems]]).
- **Big model vs small**: use a strong model for planning and a cheaper one for routine sub-steps ([[model-selection]]).
- **Own loop vs framework**: [[agent-frameworks]].
- **Latency**: each step is a full LLM call; parallel tool calls and streaming help.

## Gotchas
- Infinite or oscillating loops (same failing call repeated): detect repeated identical calls and break out.
- Context explosion from large tool outputs; truncate and paginate.
- Compounding errors: 95% per step over 10 steps is about 60% overall; verify intermediate results and add checkpoints.
- Irreversible tools without approval or idempotency keys.
- Tool output and web content can hijack the agent ([[prompt-injection]]); least-privilege tools ([[guardrails-and-safety]]).
- Evaluate trajectories, not just final answers ([[llm-evaluation]]).
- Reasoning-model "thinking" blocks must be passed back unmodified when continuing tool turns on some APIs; use the SDK's content objects as in the sample.

## Related
- [[tool-calling]] - the primitive under the loop.
- [[multi-agent-systems]] - when one agent is not enough.
- [[agent-frameworks]] - libraries that implement the loop.
- [[agent-memory]] - state across turns/sessions.
- [[guardrails-and-safety]] - limits and approvals.
- [[reasoning-models]] - built-in planning.
- [[computer-use-agents]] - agents that drive a GUI from screenshots.

## References
- Anthropic, Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- Yao et al., ReAct: https://arxiv.org/abs/2210.03629
- Tool use overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
