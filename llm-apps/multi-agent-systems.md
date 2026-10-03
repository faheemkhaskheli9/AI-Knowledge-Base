---
title: Multi-agent systems
category: llm-apps
tags: [multi-agent, orchestrator, subagents, handoffs, supervisor, parallelization]
use_cases:
  - "build a deep-research system where parallel sub-agents search different sources"
  - "route support tickets to specialist agents (billing, technical, legal)"
  - "split a coding task among planner, implementer and reviewer agents"
  - "run independent analyses in parallel and merge them in a finance report"
  - "keep a long task within context limits by delegating to fresh-context workers"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.anthropic.com/engineering/multi-agent-research-system
  - https://www.anthropic.com/engineering/building-effective-agents
  - https://openai.github.io/openai-agents-python/handoffs/
---

# Multi-agent systems

## Summary
A multi-agent system splits work across several LLM agents, each with its own prompt, tools and context window. It helps when a task is parallelizable, needs different expertise or tool sets, or overflows one context. It also multiplies token cost and failure modes, so use it only where a single agent measurably falls short.

## Key concepts
- **Orchestrator-worker (supervisor)**: a lead agent plans, spawns sub-agents with focused briefs, and synthesizes their results. Anthropic's research system uses this and reports big gains on breadth-first research at roughly an order of magnitude more tokens than chat.
- **Handoffs / routing**: a triage agent transfers the conversation to a specialist who then owns it (OpenAI Agents SDK `handoffs`).
- **Agents as tools**: the parent calls a sub-agent like a function and gets back a result; the parent stays in control.
- **Parallelization**: independent sub-tasks run concurrently (sectioning) or the same task runs several times and votes.
- **Evaluator-optimizer**: generator plus critic loop until criteria are met.
- **Context isolation**: each sub-agent starts with a clean window and returns a compact summary, not its whole transcript - the main reason to split.
- **Shared state**: pass artifacts via files/DB or a shared store, not by chatting everything ([[agent-memory]]).
- **Briefs matter**: sub-agents know nothing; give objective, output format, tools, boundaries, and effort budget.

## When to use / scenarios
- Deep research: lead + N searchers + citation checker.
- Customer service: triage -> billing/tech/returns specialists with different tools and policies.
- Software tasks: explorer (read-only), implementer, tester, reviewer ([[coding-agents]]).
- Enterprise analytics: parallel per-region/per-document analyses merged by a synthesizer.
- Avoid for tightly coupled tasks sharing lots of state (e.g. most single-repo coding edits), or when a prompt chain works.

## Setup & code
Orchestrator with parallel sub-agents as plain async calls (Anthropic SDK):
```python
# pip install anthropic
import asyncio, anthropic
client = anthropic.AsyncAnthropic()
MODEL = "claude-sonnet-5-5"

async def worker(topic: str) -> str:
    r = await client.messages.create(model=MODEL, max_tokens=800,
        system="You are a research worker. Return <=150 words of findings with source names.",
        messages=[{"role": "user", "content": f"Investigate: {topic}"}])
    return r.content[0].text

async def lead(question: str) -> str:
    plan = await client.messages.create(model=MODEL, max_tokens=300,
        system="Split the question into 3 independent research topics, one per line, no numbering.",
        messages=[{"role": "user", "content": question}])
    topics = [t for t in plan.content[0].text.splitlines() if t.strip()][:3]   # cap fan-out
    findings = await asyncio.gather(*(worker(t) for t in topics))
    final = await client.messages.create(model=MODEL, max_tokens=1000,
        system="Synthesize the findings into one answer. Note disagreements.",
        messages=[{"role": "user", "content": question + "\n\n" + "\n\n".join(findings)}])
    return final.content[0].text

print(asyncio.run(lead("Compare pgvector and Qdrant for a 5M-chunk RAG app")))
```
OpenAI Agents SDK handoff (`pip install openai-agents`):
```python
from agents import Agent, Runner
billing = Agent(name="Billing", instructions="Handle billing questions.")
tech = Agent(name="Tech", instructions="Handle technical questions.")
triage = Agent(name="Triage", instructions="Route to the right specialist.", handoffs=[billing, tech])
print(Runner.run_sync(triage, "I was double charged").final_output)
```
The Claude Agent SDK supports subagents and Claude Code supports subagent definitions ([[agent-frameworks]], [[coding-agents]]).

## Choosing / trade-offs
- **Cost vs quality**: multi-agent runs use several times the tokens; justify with high-value tasks.
- **Orchestrator vs peer-to-peer/swarm**: orchestration is easier to debug and bound; free-form agent chat drifts and loops.
- **Parallel vs sequential**: parallel cuts latency but needs independent subtasks and a merge step.
- **Model mix**: strong model for lead, smaller for workers ([[model-selection]]).
- **Framework vs hand-rolled**: hand-rolled asyncio is often enough.

## Gotchas
- Vague delegation ("research X") causes duplicated work or gaps; specify scope and output schema.
- Agents agreeing with each other is not verification; add a grounded check (sources, tests).
- Unbounded spawning: cap fan-out, depth, steps and total budget.
- Lost information in summaries: ask workers for structured findings with evidence.
- Permission creep: each sub-agent gets only the tools it needs ([[guardrails-and-safety]]).
- Debugging needs per-agent traces ([[llm-observability]]).
- Prompt injection can propagate between agents ([[prompt-injection]]).

## Related
- [[agents]] - single-agent loop first.
- [[agent-frameworks]] - supervisors and handoffs in libraries.
- [[agent-memory]] - shared state.
- [[coding-agents]] - subagents in practice.
- [[llm-evaluation]] - evaluate end-to-end outcomes.

## References
- Anthropic, How we built our multi-agent research system: https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic, Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK handoffs: https://openai.github.io/openai-agents-python/handoffs/
