---
title: Long-running agents (checkpointing, durable execution, approvals, budgets)
category: llm-apps
tags: [agents, long-running, checkpointing, durable-execution, human-in-the-loop, approvals, budgets, temporal, langgraph]
use_cases:
  - "make an agent survive a crash or deploy and resume where it stopped"
  - "pause an agent for a manager's approval before it sends money or emails"
  - "run a multi-hour research or coding agent with a hard cost cap"
  - "run an agent over many sessions and context windows on one project"
  - "choose between Temporal, LangGraph checkpoints and a hand-rolled job table"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
  - https://docs.langchain.com/oss/python/langgraph/persistence
  - https://docs.langchain.com/oss/python/langgraph/human-in-the-loop
  - https://docs.temporal.io/workflows
  - https://www.anthropic.com/engineering/building-effective-agents
---

# Long-running agents (checkpointing, durable execution, approvals, budgets)

## Summary
An agent that runs for minutes to days cannot live inside one HTTP request or
one process: servers restart, APIs time out, humans take hours to approve, and
the context window fills up. Long-running agents persist their state after each
step (checkpointing), resume from the last checkpoint instead of starting over
(durable execution), pause for human approval without holding a process open,
and stop when a token, money or time budget runs out.

## Key concepts
- **Checkpoint**: after every step, save everything needed to continue:
  messages (or a summary), plan/to-do list, tool results, step counter, budget
  spent. Key it by a run/thread ID. LangGraph does this per graph step with a
  checkpointer and a `thread_id` (persistence docs).
- **Durable execution**: an engine (Temporal, LangGraph with a durable
  checkpointer, or a DB job table you write) records each completed step, so a
  restarted worker replays or skips finished steps and continues. In Temporal,
  workflow code must be deterministic; LLM and tool calls go in *activities*,
  whose results are recorded and retried (Temporal docs).
- **Idempotent side effects**: a crash between "tool ran" and "checkpoint
  saved" means the step runs again on resume. Give every side-effecting call an
  idempotency key (run ID + step number) so a retry does not double-charge,
  double-email or double-merge.
- **Human-in-the-loop (HITL) approval**: before an irreversible action the agent
  writes a pending-approval record, checkpoints and *exits*. An approval (UI,
  Slack, email link) resumes the run with the decision. LangGraph's
  `interrupt()` + `Command(resume=...)` is this pattern built in.
- **Budgets**: hard caps on steps, tokens, dollars and wall-clock time, checked
  before each model call and stored in the checkpoint so a resume cannot reset
  them. Exceeding a cap ends the run in a clear `budget_exceeded` state, not an
  exception lost in logs.
- **Many context windows**: long work spans several sessions. Anthropic's
  harness write-up has an initializer session set up a feature list, a progress
  file and an init script, then each later session reads the progress notes and
  git history, works on one item, tests it, commits and updates the notes. The
  notes on disk, not the chat history, carry the state ([[agent-memory]]).
- **Run states**: `running -> waiting_approval -> running -> done | failed |
  budget_exceeded | cancelled`. Make them explicit and queryable.

## When to use / scenarios
- Coding agents working through a backlog over hours ([[coding-agents]]).
- Deep research reports that browse for tens of minutes ([[web-search-and-grounding]]).
- Back-office automations with sign-off: refunds, vendor payments, contract
  changes (finance, legal, procurement).
- Data migrations or bulk edits run by an agent in chunks.
- Not needed: chat turns that finish in seconds - a plain loop with caps
  ([[agents]]) and streaming ([[streaming-and-async-llm-apps]]) is enough.
  Fixed pipelines with no model decisions belong in an ordinary workflow/queue
  system with LLM calls inside steps.

## Setup & code
Stdlib demo: an agent loop that checkpoints to SQLite after every step, pauses
for approval before a risky tool, survives a simulated crash, enforces a budget
and uses idempotency keys so a replayed step is not executed twice. The model
is a scripted stub so the block runs offline.
```python
import json
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE runs (id TEXT PRIMARY KEY, state TEXT)")
db.execute("CREATE TABLE effects (key TEXT PRIMARY KEY, result TEXT)")  # idempotency ledger

SCRIPT = [  # what a model would decide at each step
    {"tool": "lookup_invoice", "args": {"id": "INV-7"}},
    {"tool": "issue_refund", "args": {"invoice": "INV-7", "amount": 120}},
    {"final": "Refunded 120 for INV-7."},
]
RISKY = {"issue_refund"}
CALLS = {"refunds": 0}

def fake_model(state):
    return SCRIPT[state["step"]], 400          # (decision, tokens used)

def run_tool(name, args, key):
    row = db.execute("SELECT result FROM effects WHERE key=?", (key,)).fetchone()
    if row:                                     # already done before a crash: do not repeat
        return json.loads(row[0])
    if name == "issue_refund":
        CALLS["refunds"] += 1
        result = {"ok": True, "refund_id": "R-1"}
    else:
        result = {"amount_due": 120}
    db.execute("INSERT INTO effects VALUES (?, ?)", (key, json.dumps(result)))
    return result

def save(run_id, state):
    db.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (run_id, json.dumps(state)))
    db.commit()

def load(run_id):
    return json.loads(db.execute("SELECT state FROM runs WHERE id=?", (run_id,)).fetchone()[0])

def start(run_id, task, max_tokens=5_000, max_steps=10):
    save(run_id, {"task": task, "step": 0, "tokens": 0, "status": "running", "log": [],
                  "budget": {"tokens": max_tokens, "steps": max_steps}, "approval": None})

def advance(run_id, crash_after_tool=False):
    """Run until done, paused for approval, out of budget, or crashed."""
    s = load(run_id)
    while s["status"] == "running":
        if s["step"] >= s["budget"]["steps"] or s["tokens"] >= s["budget"]["tokens"]:
            s["status"] = "budget_exceeded"; break
        decision, used = fake_model(s)
        s["tokens"] += used
        if "final" in decision:
            s["status"], s["answer"] = "done", decision["final"]; break
        if decision["tool"] in RISKY and s["approval"] != s["step"]:
            s["status"], s["pending"] = "waiting_approval", decision
            break                                # checkpoint and exit; no process waits
        key = f"{run_id}:{s['step']}"
        result = run_tool(decision["tool"], decision["args"], key)
        if crash_after_tool:
            raise RuntimeError("worker died before checkpoint")
        s["log"].append({"step": s["step"], "tool": decision["tool"], "result": result})
        s["step"] += 1
        save(run_id, s)
    save(run_id, s)
    return s["status"]

def approve(run_id, ok: bool):
    s = load(run_id)
    if ok:
        s["status"], s["approval"] = "running", s["step"]
    else:
        s["status"] = "cancelled"
    s.pop("pending", None)
    save(run_id, s)

start("run-1", "Refund invoice INV-7 if it is a duplicate charge")
print(advance("run-1"))                          # waiting_approval (after the lookup)
approve("run-1", ok=True)
try:
    advance("run-1", crash_after_tool=True)      # refund executes, then the worker dies
except RuntimeError as e:
    print("crash:", e)
print(advance("run-1"))                          # resume from checkpoint -> done
s = load("run-1")
assert s["status"] == "done" and CALLS["refunds"] == 1, (s, CALLS)
print(s["answer"], "| refunds executed:", CALLS["refunds"], "| tokens:", s["tokens"])

start("run-2", "same task, tiny budget", max_tokens=500)
print("run-2:", advance("run-2"))                # budget_exceeded after one model call
```

LangGraph's built-in version (checkpointer + `interrupt`); swap `InMemorySaver`
for a Postgres/SQLite saver in production:
<!-- skip-check: needs langgraph installed; illustrative fragment -->
```python
from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class State(TypedDict):
    amount: int
    status: str

def refund(state: State):
    decision = interrupt({"question": f"Approve refund of {state['amount']}?"})
    return {"status": "refunded" if decision == "yes" else "rejected"}

g = StateGraph(State)
g.add_node("refund", refund)
g.add_edge(START, "refund")
g.add_edge("refund", END)
graph = g.compile(checkpointer=InMemorySaver())

cfg = {"configurable": {"thread_id": "run-42"}}
graph.invoke({"amount": 120, "status": "new"}, cfg)        # pauses at interrupt
print(graph.invoke(Command(resume="yes"), cfg))           # resumes the same thread
```

## Choosing / trade-offs
| Option | Good for | Cost |
|---|---|---|
| DB job table + your loop (like the demo) | one service, simple steps, full control | you build retries, timers, visibility |
| LangGraph + durable checkpointer | graph-shaped agents, built-in `interrupt`, time-travel debugging | tied to the framework's state model |
| Temporal (or similar durable workflow engine) | many workers, long timers, strong retry semantics, cross-service sagas | extra cluster/service, determinism rules for workflow code |
| Managed agent platforms / agent SDK sessions | fastest start | less control over storage, retention and region |

- Checkpoint full message history (exact resume, large storage, may contain
  PII) vs a compact summary plus progress notes (small, loses detail; better
  across context windows).
- Approval granularity: every tool call (safe, slow, approval fatigue) vs only
  irreversible or high-value actions with thresholds (amount, recipient
  outside the org, production systems).

## Gotchas
- **Resetting budgets on resume**: keep spent tokens/dollars in the checkpoint,
  not in process memory.
- **Non-idempotent tools**: retries after a crash repeat side effects unless
  each call carries a stable idempotency key the downstream system honours.
- **Non-determinism in replayed workflow code** (Temporal): time, random and
  LLM calls in workflow code break replay; put them in activities.
- **Stale approvals**: an approval granted hours later may refer to a changed
  world. Re-check preconditions (balance, status) after resume and expire
  pending approvals.
- **Approval prompts that hide details**: show the exact action and arguments
  the agent will execute, not the agent's own summary of them
  ([[prompt-injection]] can make that summary lie).
- **Context bloat over long runs**: summarize or offload tool results; long
  runs also benefit from a stable cached prefix ([[prompt-caching-and-cost]]).
- **Premature "done"**: agents declare victory early; require a verification
  step (tests pass, checklist complete) before the final state
  ([[agent-evaluation]]).
- **Model ID retirement mid-run**: a run lasting weeks can outlive a model ID;
  store the model ID in the checkpoint and handle a migration explicitly.
- **Checkpoints are sensitive data**: they hold prompts, tool outputs and
  secrets passed to tools; encrypt, set retention, restrict access
  ([[ai-security-privacy-compliance]]).

## Related
- [[agents]] - the basic loop, stop conditions and caps.
- [[agent-memory]] - what to keep across sessions and how.
- [[agent-frameworks]] - frameworks that ship checkpointers and interrupts.
- [[guardrails-and-safety]] - which actions need approval.
- [[coding-agents]] - the most common long-running agent today.
- [[streaming-and-async-llm-apps]] - progress updates while a run works.
- [[llm-observability]] - tracing long runs step by step.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/distributed-systems/idempotency-and-exactly-once.md - idempotency keys for replayed steps.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/distributed-systems/distributed-transactions-and-sagas.md - compensating actions when a multi-step run fails.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/message-queues-and-streams.md - queues and workers behind background runs.

## References
- Anthropic, Effective harnesses for long-running agents: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- Anthropic, Building effective agents: https://www.anthropic.com/engineering/building-effective-agents
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph human-in-the-loop: https://docs.langchain.com/oss/python/langgraph/human-in-the-loop
- Temporal workflows: https://docs.temporal.io/workflows
