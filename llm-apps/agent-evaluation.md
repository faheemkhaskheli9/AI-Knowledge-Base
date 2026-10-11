---
title: Agent evaluation (task success, trajectories, SWE-bench, tau-bench)
category: llm-apps
tags: [evaluation, agent-eval, trajectory, tool-call-accuracy, swe-bench, tau-bench, gaia, webarena, osworld, terminal-bench, sandbox, pass-k]
use_cases:
  - "measure whether our coding agent actually fixes issues, not just sounds right"
  - "score an agent's recorded tool-call trajectory against an expected call list"
  - "set turn and cost budgets and flag runs that exceed them in CI"
  - "choose a public benchmark (SWE-bench, tau-bench, GAIA, WebArena, OSWorld, Terminal-Bench) for my agent type"
  - "make agent evals reproducible with a replayable sandbox and state-based grading"
status: draft
last_verified: 2026-10-11
sources:
  - https://arxiv.org/abs/2310.06770
  - https://www.swebench.com/verified.html
  - https://github.com/SWE-bench/SWE-bench
  - https://arxiv.org/abs/2406.12045
  - https://arxiv.org/abs/2311.12983
  - https://arxiv.org/abs/2307.13854
  - https://arxiv.org/abs/2404.07972
  - https://github.com/laude-institute/terminal-bench
---

# Agent evaluation

## Summary
Evaluating an agent means scoring a multi-step run, not one output. Grade the outcome (did the task succeed, checked against environment state or tests), optionally grade the trajectory (right tools, right arguments, no wasted steps), and enforce turn and cost budgets. Run it in a sandbox you can reset and replay, repeat trials because agents are non-deterministic, and use public benchmarks to compare models, not to replace evals on your own tasks. Single-output LLM-as-judge is covered in [[llm-evaluation]].

## Key concepts
- **Task success (outcome) scoring**: a deterministic check of the end state. SWE-bench runs tests on the patched repo; tau-bench compares the final database state with an annotated goal state; Terminal-Bench ships a test script per task; OSWorld uses a custom execution-based evaluation script per task. State checks beat judging the agent's final message, because an agent can claim success it did not achieve.
- **Trajectory scoring**: judge the path (tool choice, arguments, order, redundant calls, policy violations). Useful for debugging and for catching right-answer-wrong-way runs, but over-strict path matching penalises valid alternative routes. Prefer outcome as the primary metric and trajectory metrics as diagnostics.
- **Tool-call accuracy**: compare the recorded calls with an expected set: right tool name, right (or sufficiently matching) arguments, no forbidden or extra calls. Report per-call precision and recall, not one pass/fail.
- **Replayable sandbox**: each trial starts from an identical, resettable environment (container, database snapshot, simulated user) and records the full trajectory so a run can be re-graded or inspected later. SWE-bench uses Docker for reproducible evaluations; WebArena provides self-hosted websites; OSWorld a real computer environment.
- **Budgets**: cap turns, tokens or dollars and time per task. Count a budget breach as a failure (or report it separately) so a model that succeeds by burning 10x the steps does not look equal to an efficient one. Report cost per success next to success rate.
- **Repeated trials and reliability**: tau-bench proposes `pass^k`, a metric for the reliability of agent behaviour over multiple trials, since solving a task once is weaker than solving it every time. Report variance across trials, not one run.
- **Simulated users**: for conversational agents, an LLM plays the user (tau-bench); the simulator is itself a source of noise to pin and version.

## When to use / scenarios
- Coding agent: SWE-bench style (issue + repo -> patch, graded by tests) on your own repos; see [[coding-agents]].
- Customer-support or back-office agent with tools and policy rules: tau-bench style (final DB state vs goal state, policy adherence).
- Browser, desktop or terminal agent: WebArena, OSWorld, Terminal-Bench style; see [[computer-use-agents]].
- General assistant needing browsing and tool use: GAIA style (question with a short checkable answer).
- Regression gate in CI: replay a fixed task set on every prompt, tool or model change.
- Not for: judging single-turn answers (use [[llm-evaluation]]); ranking many models by pairwise human votes (see [[elo-and-bradley-terry-leaderboard-from-scratch]]).

### Main public benchmarks (what each measures)
| Benchmark | Measures | Facts from the official source |
|---|---|---|
| SWE-bench | Resolving real GitHub issues in Python repos by editing the codebase | 2,294 task instances from 12 popular Python repositories (paper); Docker-based evaluation (repo) |
| SWE-bench Verified | Human-validated subset of SWE-bench | 500 instances; human annotators reviewed each for sound problem descriptions, tests and solvability; built with OpenAI (swebench.com) |
| tau-bench | Tool-using agent talking to a simulated user under domain policy rules | Success = end-of-conversation database state matches the annotated goal state; introduces `pass^k` |
| GAIA | General assistant questions: reasoning, multimodality, web browsing, tool use | 466 questions, answers to 300 withheld for a leaderboard; simple for humans, hard for AI |
| WebArena | Long-horizon tasks on self-hosted functional websites | Four domains: e-commerce, social forum, collaborative software development, content management; judged on functional correctness of completion |
| OSWorld | Open-ended tasks in a real computer environment | 369 tasks across real web and desktop apps, OS file operations and multi-app workflows; per-task execution-based evaluation script |
| Terminal-Bench | End-to-end tasks in a sandboxed terminal | Each task: English instruction, test script, reference (oracle) solution; a dataset plus an execution harness |

Scores move constantly and benchmark versions change; look up current numbers on the official leaderboard rather than in this file.

## Setup & code
Stdlib-only scorer for one recorded trajectory: outcome check, tool-call accuracy against an expected list, and budget violation.
```python
from collections import Counter

def call_key(c):
    return (c["tool"], tuple(sorted(c["args"].items())))

def score(traj, expected, final_state, goal_state, max_turns, max_cost):
    actual = Counter(call_key(c) for c in traj["calls"])
    want = Counter(call_key(c) for c in expected)
    hit = sum((actual & want).values())              # matched calls (multiset)
    return {
        "success": final_state == goal_state,         # outcome: compare state, not the agent's claim
        "call_precision": hit / max(sum(actual.values()), 1),
        "call_recall": hit / max(sum(want.values()), 1),
        "over_turns": len(traj["calls"]) > max_turns,
        "over_cost": traj["cost_usd"] > max_cost,
    }

traj = {"cost_usd": 0.12, "calls": [
    {"tool": "get_order", "args": {"id": "A1"}},
    {"tool": "get_order", "args": {"id": "A1"}},      # redundant repeat
    {"tool": "cancel_order", "args": {"id": "A1"}},
]}
expected = [{"tool": "get_order", "args": {"id": "A1"}},
            {"tool": "cancel_order", "args": {"id": "A1"}}]
r = score(traj, expected, {"A1": "cancelled"}, {"A1": "cancelled"}, max_turns=2, max_cost=0.50)
print(r)
# {'success': True, 'call_precision': 0.6666666666666666, 'call_recall': 1.0, 'over_turns': True, 'over_cost': False}
```
Reading the result: the task succeeded and every expected call was made (recall 1.0), but one redundant call lowered precision and pushed the run over its turn budget. In a real harness, record `traj` from the agent loop ([[agents]]), reset the sandbox before each trial, and run each task several times.

## Choosing / trade-offs
- **Outcome vs trajectory**: outcome is robust to alternative valid paths; trajectory catches policy violations and waste. Gate on outcome, alert on trajectory.
- **Exact vs partial argument matching**: exact is simple but brittle (free-text arguments); match key fields only, or judge free text with an LLM ([[llm-evaluation]]).
- **Public benchmark vs your own tasks**: public ones give comparability; your own tasks predict your product. Do both, and treat a benchmark as a filter for model choice ([[model-selection]]).
- **Budget as failure vs separate metric**: failing on breach is strict and simple; a separate cost-per-success column keeps more information.
- **Trials**: more trials give tighter estimates and expose reliability gaps (`pass^k`) at multiplied cost.
- **Real vs simulated environment**: real services are realistic but non-reproducible; mocks and snapshots are repeatable but can drift from production.

## Gotchas
- Grading on the agent's final message instead of environment state rewards confident false claims.
- Non-resettable environments: leftover state from trial N changes trial N+1. Reset every time.
- Tests that do not actually pin the behaviour let wrong patches pass; the SWE-bench Verified effort exists because annotators checked tests and problem statements for soundness.
- Benchmark contamination: public tasks and their solutions may be in training data; keep a private held-out set (GAIA withholds the answers to 300 of its questions and scores them only through its leaderboard).
- Scaffold differences (prompt, tools, retries, turn limits) change scores as much as the model; compare under the same harness and budgets.
- A single run is one sample from a distribution; report repeats and variance.
- LLM-simulated users add noise; fix their prompt and model version.
- Don't let the agent see grader files or expected outputs inside the sandbox.

## Related
- [[llm-evaluation]] - golden sets and LLM-as-judge for single outputs; this topic extends it to multi-step agents.
- [[agents]] - the loop, stop conditions and budgets being evaluated.
- [[tool-calling]] - what tool-call accuracy checks.
- [[coding-agents]] - SWE-bench style tasks.
- [[computer-use-agents]] - OSWorld and WebArena style tasks.
- [[elo-and-bradley-terry-leaderboard-from-scratch]] - ranking models from pairwise comparisons.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/testing-strategy-and-pyramid.md - where evals sit next to other tests.

## References
- Jimenez et al., SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-bench Verified: https://www.swebench.com/verified.html
- SWE-bench repository (Docker evaluation): https://github.com/SWE-bench/SWE-bench
- tau-bench paper: https://arxiv.org/abs/2406.12045
- GAIA paper: https://arxiv.org/abs/2311.12983
- WebArena paper: https://arxiv.org/abs/2307.13854
- OSWorld paper: https://arxiv.org/abs/2404.07972
- Terminal-Bench repository: https://github.com/laude-institute/terminal-bench
