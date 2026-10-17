---
title: Workflows vs agents (fixed chains and routers vs autonomous loops)
category: llm-apps
tags: [workflows, agents, prompt-chaining, routing, orchestrator-workers, evaluator-optimizer, reliability, cost, debuggability, architecture]
use_cases:
  - "decide whether a support or back-office feature should be a fixed pipeline or an agent"
  - "replace a slow, expensive agent with a router plus fixed chains for the common cases"
  - "estimate how per-step reliability and loop length affect end-to-end success and cost"
  - "design a hybrid: deterministic workflow for known intents, agent for the long tail"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.anthropic.com/engineering/building-effective-agents
  - https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
  - https://arxiv.org/abs/2210.03629
---

# Workflows vs agents (fixed chains and routers vs autonomous loops)

## Summary
A workflow runs LLM calls and tools along paths your code defines (chain,
router, parallel fan-out, generate-then-check); an agent lets the model choose
its next step in a loop until it decides it is done. Workflows are cheaper,
faster, easier to test and to debug, because every request takes a known path;
agents handle open-ended tasks whose steps cannot be known in advance, at the
cost of more tokens, a long latency tail and errors that compound across
steps. Start with a single well-prompted call, move to a workflow, and use an
agent only where measured results show the autonomy pays for itself, often
just for the long tail behind a router.

## Key concepts
- **Definitions** (Anthropic, Building effective agents): workflows orchestrate
  LLMs and tools through predefined code paths; agents dynamically direct their
  own process and tool use.
- **Workflow patterns**:
  - *Prompt chaining*: fixed sequence of calls, each consuming the previous
    output, with programmatic checks (gates) between steps.
  - *Routing*: classify the input, send it to a specialised prompt, tool or
    model (cheap model for easy cases, strong model for hard ones).
  - *Parallelisation*: split into independent sections, or run the same task
    several times and vote ([[self-consistency-voting-from-scratch]]).
  - *Orchestrator-workers*: an LLM splits the task at runtime and hands
    pieces to workers; the pieces are dynamic but the shape is fixed.
  - *Evaluator-optimizer*: one call generates, another critiques against
    criteria, loop until it passes or a cap is hit.
- **Agent**: model-chosen tool calls in a loop with stop conditions and
  budgets ([[agents]]). Autonomy spectrum: single call -> chain -> router ->
  orchestrator-workers -> agent with tools -> multi-agent
  ([[multi-agent-systems]]).
- **Errors compound**: if each step succeeds with probability p, n dependent
  steps succeed with roughly p^n. Workflows keep n small and put a validator
  after each step; agents rely on the model noticing and recovering.
- **Cost and latency shape**: a workflow's call count is fixed per path, so
  cost and p95 latency are predictable. An agent's call count varies per
  request, and each step re-sends a growing context, so tokens grow faster
  than linearly in steps ([[prompt-caching-and-cost]]).
- **Debuggability**: a workflow failure points to a step with typed inputs
  and outputs; an agent failure needs a full trajectory trace to replay
  ([[llm-observability]]).

## When to use / scenarios
Choose a **workflow** when:
- The steps are known: extract -> classify -> draft -> check (invoice
  processing, claim intake, document summarisation, ticket triage).
- Compliance requires the same steps every time, with audit logs (finance,
  healthcare, legal).
- Latency or cost budgets are tight (voice agents, high-volume back office).
- Most traffic falls into a few intents (support FAQs, order status).

Choose an **agent** when:
- The number and order of steps depends on what is found along the way:
  research, debugging, data exploration, multi-system troubleshooting
  ([[coding-agents]], [[web-search-and-grounding]]).
- Success can be checked by the environment (tests pass, query returns rows)
  so the loop has a reliable signal ([[agent-evaluation]]).
- The task value justifies variable cost and minutes of latency.

Use a **hybrid** (router + workflows + agent fallback) when a few intents
cover most traffic and a long tail is open-ended: route known intents to
fixed chains, send the rest to an agent or a human.

Not either: if one call with good context and examples meets the quality bar,
stop there ([[prompt-engineering]], [[rag-basics]]).

## Setup & code
A NumPy simulation of the trade-off. Part 1 shows how per-step reliability
compounds. Part 2 compares a router + fixed chain (escalates unknown requests
to a human), an agent for everything, and a router with an agent fallback,
on simulated traffic. All rates, step counts and token sizes are assumptions
for illustration; plug in numbers from your own traces and evals.

```python
import numpy as np

rng = np.random.default_rng(7)

# 1) Errors compound: end-to-end success of n steps that each succeed with probability p.
for p in (0.99, 0.95, 0.90):
    print(f"p={p}: " + "  ".join(f"{n} steps {p ** n:.2f}" for n in (3, 10, 20)))

# 2) Simulated traffic: 85% of requests are known intents, 15% are open-ended.
# All numbers below are illustrative assumptions; replace them with your own measurements.
N = 20_000
known = rng.random(N) < 0.85

def workflow(known):
    """Router (1 call) + fixed 3-call chain for known intents; open-ended requests go to a human."""
    calls = np.where(known, 4, 1)
    ok = np.where(known, rng.random(N) < 0.99 ** 4, False)       # 4 narrow, validated steps
    escalated = ~known
    tokens = calls * 1_500                                       # short, focused prompts
    return ok, escalated, calls, tokens

def agent(known):
    """Tool loop: steps vary per request; context grows each step; capped at 15 steps."""
    steps = np.minimum(np.where(known, rng.geometric(1 / 4, N), rng.geometric(1 / 8, N)), 15)
    capped = steps == 15
    ok = (rng.random(N) < 0.97 ** steps) & ~capped
    ok &= np.where(known, True, rng.random(N) < 0.75)             # open-ended tasks are harder
    tokens = np.array([sum(3_000 + 1_200 * i for i in range(s)) for s in steps])  # re-sent context
    return ok, np.zeros(N, bool), steps, tokens

def router_plus_agent(known):
    """Workflow for known intents, agent only for the open-ended tail."""
    w_ok, _, w_calls, w_tok = workflow(known)
    a_ok, _, a_calls, a_tok = agent(known)
    return (np.where(known, w_ok, a_ok), np.zeros(N, bool),
            np.where(known, w_calls, 1 + a_calls), np.where(known, w_tok, 1_500 + a_tok))

print(f"\n{'design':18s} {'solved':>7s} {'to human':>9s} {'calls p50/p95':>14s} {'tokens/request':>15s}")
for name, f in [("workflow", workflow), ("agent", agent), ("router + agent", router_plus_agent)]:
    ok, esc, calls, tok = f(known)
    print(f"{name:18s} {ok.mean():7.2f} {esc.mean():9.2f} "
          f"{int(np.median(calls)):>6d}/{int(np.percentile(calls, 95)):<7d} {tok.mean():15,.0f}")
```
Under these assumptions the workflow is the cheapest and has a flat latency
tail but hands every unknown request to a human; the agent solves the most
open-ended requests but costs several times more per request and has a long
p95 tail; the router-plus-agent hybrid gets the best solve rate at about
half the agent's token cost. Change the assumptions and the ranking can flip,
which is why the decision should come from your own evals and traces.

A router + fixed chain with the Anthropic SDK (model IDs from
[[anthropic-claude]]; a cheap model routes, a stronger one drafts):
<!-- skip-check: needs an API key -->
```python
# pip install anthropic pydantic
from typing import Literal
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()
ROUTER, WRITER = "claude-haiku-4-5-20251001", "claude-sonnet-5-5"

class Route(BaseModel):
    intent: Literal["order_status", "refund", "other"]

def route(msg: str) -> str:
    r = client.messages.parse(model=ROUTER, max_tokens=50, output_format=Route,
        messages=[{"role": "user", "content": f"Classify this customer message:\n{msg}"}])
    return r.parsed_output.intent

def handle(msg: str, lookup_order, refund_policy: str, run_agent) -> str:
    intent = route(msg)
    if intent == "order_status":                       # fixed chain: tool in code, LLM only writes
        order = lookup_order(msg)
        prompt = f"Order data: {order}\nWrite a short reply to: {msg}"
    elif intent == "refund":
        prompt = f"Policy: {refund_policy}\nApply the policy to: {msg}. Do not promise anything outside it."
    else:
        return run_agent(msg)                          # long tail: agent loop with caps (see agents topic)
    r = client.messages.create(model=WRITER, max_tokens=400,
                               messages=[{"role": "user", "content": prompt}])
    draft = r.content[0].text
    if intent == "refund" and "guarantee" in draft.lower():   # programmatic gate between steps
        raise ValueError("draft makes a promise outside policy; send to a human")
    return draft
```

## Choosing / trade-offs
| Dimension | Workflow | Agent |
|---|---|---|
| Cost per request | fixed, low | variable, grows with steps and context |
| Latency | predictable, p95 close to p50 | long tail; caps needed |
| Reliability | per-step validation, small n | compounding errors, self-recovery |
| Flexibility | only the paths you coded | handles novel requests |
| Testing | unit-test each step, eval each path | trajectory evals, repeated trials, sandboxes |
| Debugging | failing step is obvious | replay full traces |
| Security | tools called by code with fixed arguments | model chooses tools and arguments; injection risk grows ([[prompt-injection]]) |
| Change management | code change per new path | prompt/tool change, less code |

- **Frameworks**: graph-based frameworks express both workflows and agents;
  Anthropic's guidance is to start with direct API calls and understand what
  any framework does underneath ([[agent-frameworks]]).
- **Model strength vs structure**: stronger models make agents more reliable
  per step, but a workflow with a smaller model is often cheaper at the same
  quality for well-defined tasks ([[model-selection]]).
- **Reasoning models** can fold some planning into one call, shrinking the
  need for either a chain or a loop ([[reasoning-models]]).

## Gotchas
- Calling a fixed chain an "agent" (or the reverse) muddles requirements;
  name the pattern so budgets and tests match.
- Agents without hard step, token and time caps run away on edge cases.
- A router's misclassifications become silent failures; log route decisions
  and include routing cases in the eval set ([[llm-evaluation]]).
- Workflows grow into tangles of special cases; when the number of paths
  explodes, that is the signal to try an agent for that slice.
- Comparing designs on average cost hides the p95; report the distribution.
- Agents given write tools on untrusted input need approval gates; workflows
  can call the same tools with code-checked arguments instead
  ([[guardrails-and-safety]]).

## Related
- [[agents]] - the agent loop, stop conditions and budgets.
- [[multi-agent-systems]] - when one agent is split into several.
- [[agent-frameworks]] - libraries that implement graphs, chains and loops.
- [[tool-calling]] - the primitive both designs use.
- [[agent-evaluation]] - measuring agent success, trajectories and budgets.
- [[llm-as-judge-and-eval-driven-development]] - per-step and end-to-end eval gates.
- [[llm-gateways-and-routing]] - model routing infrastructure.
- [[cost-and-latency]] - budgeting tokens and latency per design.
- [[self-consistency-voting-from-scratch]] - the voting form of parallelisation.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/resilience-patterns.md - timeouts, retries and fallbacks around each step.

## References
- Anthropic, Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- Anthropic tool use overview: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models: https://arxiv.org/abs/2210.03629
