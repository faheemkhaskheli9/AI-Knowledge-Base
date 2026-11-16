---
title: Task decomposition, prompt chains and agent handoffs
category: llm-apps
tags: [task-decomposition, prompt-chaining, handoffs, planner, plan-and-execute, replanning, granularity, multi-agent]
use_cases:
  - "decide how many agents a workflow needs, or whether to split it at all"
  - "have a planner agent turn a variable request into a dependency-ordered plan of subtasks"
  - "chain extract -> analyse -> write steps with validation between them"
  - "let one specialist agent hand the rest of a task to another without losing context"
  - "stop handoff loops and agents bouncing a task they cannot do"
status: draft
last_verified: 2026-11-07
sources:
  - https://learn.microsoft.com/en-us/training/modules/aaai-apply-task-decomposition-multi-agent-azure/
  - https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/handoff
  - https://www.anthropic.com/engineering/building-effective-agents
---

# Task decomposition, prompt chains and agent handoffs

## Summary
How to split one complex request into steps or agents, and how to pass work between them without losing information. It covers prompt chains (fixed steps inside one agent), planner-driven decomposition (an LLM writes the plan at runtime), and handoffs (one specialist passes the rest of the task to another). The core rule: split only at clean interfaces and only when measurements show the split pays for its overhead. [[multi-agent-systems]] covers the topologies; this file covers where to cut and how to hand over.

## Key concepts
- **Prompt chain**: step N's structured output is step N+1's input, inside one agent. *Linear* chains (A -> B -> C) suit natural dependencies. *Branching* chains classify first and then route to a specialised step (for example a valuation method chosen per company profile).
- **Step contract**: each step declares the input schema it accepts and the output schema it produces, and validates its input before working ([[structured-output]]).
- **Static vs dynamic decomposition**: static plans are fixed at design time, which is predictable and easy to debug. A dynamic *planner* (meta-agent) reads the request and an agent registry, then emits a plan. That adapts to varied requests but adds a planning call and makes runs less predictable.
- **Plan-and-execute**: planning and execution are separate phases. The plan (task id, description, assigned agent, `depends_on`, expected output) is a reviewable artefact and an audit trail, so a human or policy check can reject it before anything runs.
- **Reflection / replanning**: after each task (or parallel batch) the planner decides *continue*, *modify remaining tasks* or *replan from current state*. Do the deep review only on low-confidence or unexpected results; otherwise do a cheap confirm.
- **Handoff**: a peer-to-peer transfer of control, as opposed to returning results to a hub. The handoff payload needs: the task still to do, the completed work (kept separate), source agent, reason, expected output format, and a deadline.
- **Context collapse**: each summary-of-a-summary loses detail. Rule: raw data is summarised at most once. Pass full structured outputs or shared history, not re-compressed notes.

## When to use / scenarios
- Requests that vary widely in complexity (a one-line lookup vs a multi-source investment memo): use a planner whose plan depth follows its complexity assessment.
- A pipeline where each stage produces a complete artefact (extracted metrics JSON -> peer benchmark -> written commentary): use a chain or a split into agents.
- Regulated domains (finance, healthcare): use plan-and-execute so the plan can be inspected before it touches restricted data.
- A specialist finds mid-task that part of the work is outside its domain (a credit analyst meets convertible securities): use a partial handoff.
- **Do not split** when:
  - subtasks keep re-reading the same data;
  - each step depends on all the earlier ones (no parallelism);
  - the answer needs everything considered together (integrative reasoning such as macro factors that interact);
  - the coordination costs more than the task;
  - one agent already does the job well.

  In the course's macroeconomics example, merging four agents into one gave higher quality and lower latency.
- Nano-tasks (fetch a price, format JSON) are tool calls or plain functions, not agents.

## Setup & code
A plan executor, stdlib only. It validates a planner's JSON plan (unknown agent, missing or cyclic dependencies), runs tasks in dependency order, passes each task only its dependencies' outputs, and caps retries per task:
```python
import json

REGISTRY = {"retriever": "fetch market data", "analyst": "analyse metrics", "writer": "write the thesis"}

def validate(plan):
    ids = {t["task_id"] for t in plan["subtasks"]}
    for t in plan["subtasks"]:
        if t["assigned_agent"] not in REGISTRY:
            raise ValueError(f"{t['task_id']}: unknown agent {t['assigned_agent']}")
        if missing := set(t["depends_on"]) - ids:
            raise ValueError(f"{t['task_id']}: unknown deps {missing}")

def execute(plan, run_agent, max_attempts=2):
    validate(plan)
    done, pending = {}, list(plan["subtasks"])
    while pending:
        ready = [t for t in pending if all(d in done for d in t["depends_on"])]
        if not ready:  # nothing runnable but work remains -> cycle
            raise ValueError(f"dependency cycle among {[t['task_id'] for t in pending]}")
        for t in ready:  # tasks in one `ready` batch are independent: run them in parallel in production
            context = {d: done[d] for d in t["depends_on"]}  # only what this step needs
            for attempt in range(1, max_attempts + 1):
                try:
                    done[t["task_id"]] = run_agent(t["assigned_agent"], t["description"], context)
                    break
                except RuntimeError:
                    if attempt == max_attempts:
                        raise  # escalate to the hub / a human, do not keep rerouting
            pending.remove(t)
    return done[plan["subtasks"][-1]["task_id"]]

plan = json.loads("""{"subtasks": [
  {"task_id": "T1", "assigned_agent": "retriever", "description": "prices", "depends_on": []},
  {"task_id": "T2", "assigned_agent": "analyst", "description": "ratios", "depends_on": ["T1"]},
  {"task_id": "T3", "assigned_agent": "writer", "description": "thesis", "depends_on": ["T1", "T2"]}]}""")

def fake_agent(agent, task, context):  # stand-in for an LLM call
    return {"agent": agent, "task": task, "used": sorted(context)}

result = execute(plan, fake_agent)
assert result["used"] == ["T1", "T2"], result
plan["subtasks"][0]["depends_on"] = ["T3"]
try:
    execute(plan, fake_agent)
    raise AssertionError("cycle not caught")
except ValueError as e:
    print("rejected:", e)
print(result)
```
Framework route: Microsoft Agent Framework's `HandoffBuilder` injects handoff tools into each agent and shares the full conversation history with every participant. It lets you restrict who may hand off to whom, and it detects circular routing. In the OpenAI Agents SDK the same idea is `handoffs=[...]` ([[multi-agent-systems]], [[agent-frameworks]]).

## Choosing / trade-offs
- **Granularity**: coarse (1-2 agents) suits simple or latency-bound tasks. Medium (3-5) suits most analytical workflows. Fine (6+) only pays when the components are truly distinct, and it fails when context is shared throughout. Expect diminishing returns: in the course's illustrative numbers, going from 3 to 6 agents added a few quality points for much more cost.
- **Decide with data**: run the same queries single-agent vs decomposed. Compare latency, calls, tokens and an [[llm-evaluation]] quality score, then compare quality gain per % cost increase. The best granularity often differs by query class (public vs private company vs macro vs quick lookup), so route by class rather than using one architecture for everything.
- **Hub vs handoff**: a hub re-plans after every step, which is easier to bound and debug but costs a round trip and loses specialist judgement. Handoffs are faster and use the specialist's judgement, but they need a protocol and loop limits.
- **Shared history vs explicit payload**: broadcasting full history avoids collapse, but tokens grow with every hop. An explicit structured payload is cheaper but can drop fields, so make the required fields mandatory.

## Gotchas
- **Cut at capability interfaces.** Test: can the next specialist do its step using only this step's output, without knowing how it was produced? Splitting a document "pages 1-50 / 51-end" fails this test.
- **Splitting shared context.** Sending assets, liabilities and equity to three agents breaks debt-to-equity reasoning. One agent should own context that belongs together.
- **Decomposition for show.** Don't add agents to a task that rarely fails with one agent.
- **Validate between chain steps**, both schema and meaning (for example a margin of 5000% is impossible), and halt on failure rather than passing corrupted data downstream.
- **Checkpoint expensive steps** (a 45 s filing parse) so a retry of step 3 reuses step 2's cached output. Back off exponentially on transient errors and fail fast on logic errors.
- **Instructions that say "summarise before handing off"** bring context collapse back. Ask for full structured findings instead.
- **Cap handoff depth** at about 3-4 hops. Beyond that, accumulated context hurts quality and cost, so return to the hub for a fresh plan.
- **Cap failed attempts**: after about two failed handoffs to different agents, escalate to the hub or a human. Otherwise agents bounce a task none of them can do.
- **Give each specialist a domain and an out-of-scope list in its instructions**, so it declines and routes back instead of attempting work outside its domain.
- **Log every handoff**: source, target, reason, accepted or rejected, result. A high rejection rate usually means the payload lacks a field. In the course, 40% of currency-risk handoffs were rejected until the payload required currency details.
- **Planner output is untrusted JSON.** Validate agent names and dependencies before executing (the code above), and cap plan length.

## Related
- [[multi-agent-systems]] - topologies (orchestrator-worker, handoffs, agents-as-tools); this file covers where to split and the handoff protocol.
- [[agents]] - the single-agent loop to try first.
- [[structured-output]] - schemas for step contracts and planner plans.
- [[llm-evaluation]] - quality scores for comparing decomposition strategies.
- [[agent-frameworks]] - built-in handoff and workflow orchestration.

## References
- Microsoft Learn, Apply task decomposition and agent collaboration strategies in Microsoft Foundry: https://learn.microsoft.com/en-us/training/modules/aaai-apply-task-decomposition-multi-agent-azure/
- Microsoft Agent Framework, Handoff orchestration: https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/handoff
- Anthropic, Building effective agents (prompt chaining, routing, orchestrator-workers): https://www.anthropic.com/engineering/building-effective-agents
