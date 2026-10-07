---
title: Software engineering with AI
category: scenarios
tags: [coding, code-review, testing, devops, agents, documentation]
use_cases:
  - "use a coding agent to implement features and fix bugs"
  - "automate code review or PR summaries"
  - "generate tests and documentation"
  - "migrate or refactor a large codebase"
  - "build an internal developer assistant over our repos and docs"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.anthropic.com/en/docs/claude-code/overview
  - https://owasp.org/www-project-top-10-for-large-language-model-applications/
---

# Software engineering with AI

## Summary
Coding is where agents currently deliver the most verifiable value because
there is an automatic feedback loop: compilers, linters and tests. The key
is giving the agent context (conventions, architecture), a way to check its
work, and tight permissions, with a human reviewing what merges.

## Key concepts
- Verification beats trust: tests, type checks and CI are the agent's
  feedback signal and your safety net ([[coding-agents]]).
- Context is the lever: project instructions files, a file map and
  conventions outperform longer prompts ([[prompt-engineering]],
  [[long-context]]).
- Small, reviewable diffs; one task per branch; no unreviewed merges.
- Agents run commands: sandboxing, least privilege and secret hygiene matter
  ([[guardrails-and-safety]]).
- Licence and provenance: generated code may resemble licensed code; check
  your organisation's policy and tool terms ([[model-licenses]]).

## When to use / scenarios
1. **Feature work and bug fixing with a coding agent.** Problem: backlog.
   Approach: agent in the repo with test runner access, plan then implement,
   human review. Read: [[coding-agents]], [[agents]], [[tool-calling]].
2. **Code review assistant.** Approach: LLM reviews diffs for bugs and style
   against team rules; comments are advisory. Read: [[prompt-engineering]],
   [[llm-evaluation]].
3. **Test generation.** Approach: generate tests, run them, keep only those
   that fail on mutated code or cover new branches. Read: [[coding-agents]].
4. **Documentation and PR/changelog summaries.** Approach: diff + commit
   history -> summary. Read: [[structured-output]].
5. **Legacy migration (language/framework upgrades).** Approach: agent per
   module with tests as the oracle; batch across the repo. Read:
   [[coding-agents]], [[multi-agent-systems]], [[cost-and-latency]].
6. **Codebase Q&A / internal dev portal.** Approach: RAG over code, docs,
   tickets with symbol-aware chunking. Read: [[rag-basics]], [[advanced-rag]],
   [[vector-databases]], [[model-context-protocol]].
7. **Text-to-SQL and data tooling for engineers.** Read: [[text-to-sql]],
   [[data-analytics]].
8. **Incident response and log analysis.** Approach: summarise logs and
   propose hypotheses; humans act. Read: [[security-defensive]],
   [[llm-observability]].
9. **Local/private coding model.** Approach: open-weights coder model behind
   an OpenAI-compatible server. Read: [[qwen]], [[deepseek]], [[ollama]],
   [[inference-servers-vllm]].
10. **Building LLM features into your product.** Read: [[structured-output]],
    [[llm-evaluation]], [[serving-with-fastapi]], [[prompt-caching-and-cost]].
11. **Tool integrations for agents (issue trackers, CI).** Read:
    [[model-context-protocol]], [[agent-frameworks]].

## Setup & code
Reference architecture (flagship: agent-assisted change with CI gate):

```
ticket -> agent (sandboxed checkout, read repo + instructions file)
  -> plan -> edit -> run tests/lint/typecheck -> iterate (bounded steps)
  -> open PR with summary + test evidence
  -> CI + AI review comments -> human review -> merge
  controls: no prod credentials, allowlisted commands, step/cost budget
```

<!-- skip-check: illustrative pipeline fragment, needs your inputs -->
```python
# Minimal agent loop skeleton: tool results (test output) drive iteration
for step in range(MAX_STEPS):                     # bounded, never while True
    msg = llm(messages, tools=[read_file, edit_file, run_tests])
    if msg.stop_reason != "tool_use":
        break
    messages += run_tools(msg)                    # allowlisted tools only
assert tests_pass(), "do not open PR with failing tests"
```
See [[agent-frameworks]] for ready-made loops and [[api-sdk-setup]] for SDKs.

## Choosing / trade-offs
- Autocomplete vs agent: completion is low-risk and low-leverage; agents
  are higher leverage and need guardrails and review.
- Hosted frontier model vs local: quality vs code confidentiality
  ([[model-selection]], [[ollama]]).
- One agent vs several: start with one; split roles only when context
  overflows ([[multi-agent-systems]]).

## Gotchas
- Plausible but wrong code that passes weak tests; review test quality.
- Hallucinated packages ("slopsquatting"): verify dependencies exist and are
  legitimate before installing.
- Secrets in prompts, logs or committed files; use scoped tokens.
- Prompt injection through issues, READMEs or web pages the agent reads
  ([[prompt-injection]]); OWASP LLM Top 10 lists it first.
- Skill atrophy and review fatigue on large generated diffs; keep diffs small.
- Costs balloon on long agent loops; set budgets ([[cost-and-latency]]).
- Line endings, formatting and lockfile churn on cross-platform repos.

## Related
- [[hallucination-and-grounding]] - fabricated APIs and flags; check docs.
- [[mlops-lifecycle]] - CI/CD for ML code.
- [[python-env-uv]] - reproducible Python environments for agents and CI.
- [[windows-wsl-setup]] - Windows dev environment for local tooling.

## References
- Claude Code overview: https://docs.anthropic.com/en/docs/claude-code/overview
- OWASP Top 10 for LLM applications: https://owasp.org/www-project-top-10-for-large-language-model-applications/
