---
title: DSPy and automatic prompt optimization
category: llm-apps
tags: [dspy, prompt-optimization, mipro, gepa, few-shot, signatures, llm-pipelines]
use_cases:
  - "improve a multi-step LLM pipeline's accuracy without hand-tuning every prompt"
  - "port a prompt pipeline to a cheaper or newer model and re-tune it automatically"
  - "pick few-shot examples for a classifier prompt from a labelled set"
  - "decide whether DSPy is worth adopting over hand-written prompts"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/stanfordnlp/dspy
  - https://dspy.ai/learn/optimization/optimizers/
  - https://pypi.org/project/dspy/
  - https://arxiv.org/abs/2310.03714
  - https://arxiv.org/abs/2406.11695
  - https://arxiv.org/abs/2507.19457
---

# DSPy and automatic prompt optimization

## Summary
DSPy (Stanford, MIT licence) lets you write an LLM pipeline as Python modules with typed input/output **signatures** instead of prompt strings, then **compile** it: an optimizer searches instructions and few-shot demonstrations (or fine-tunes weights) to maximize a metric you define on a training set. It pays off when you have a measurable task, a few dozen to a few hundred examples, and a pipeline with several LLM steps or frequent model swaps. For a one-off prompt with no evaluation data, hand-writing ([[prompt-engineering]]) is faster.

## Key concepts
- **Signature**: declared inputs/outputs with types and a docstring, e.g. `ticket -> label: Literal[...]`. DSPy turns it into a prompt through an adapter and parses the output.
- **Module**: a strategy for calling the LM with a signature: `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct` (tool use). Modules compose into programs like ordinary Python.
- **Metric**: a function `(example, prediction, trace) -> float|bool`. Everything the optimizer does is driven by it; a bad metric produces a confidently bad program ([[llm-evaluation]]).
- **Optimizer (teleprompter)**: searches the program's "prompt parameters" (instructions, demos) or weights. From the DSPy docs:
  - `LabeledFewShot`, `BootstrapFewShot` (teacher generates demos, keeps ones that pass the metric), `BootstrapFewShotWithRandomSearch`, `KNNFewShot` (nearest examples per input).
  - `COPRO` (instruction refinement by coordinate ascent), `MIPROv2` (proposes instructions and demos, then searches combinations with Bayesian optimization), `SIMBA`, `GEPA` (LM reflects on execution traces and textual feedback to evolve prompts).
  - `BootstrapFinetune` (distil the program into fine-tuned weights), `BetterTogether`, `Ensemble`.
- **Docs' rule of thumb by data size**: ~10 examples -> `BootstrapFewShot`; 50+ -> `BootstrapFewShotWithRandomSearch`; ~200+ examples and longer runs -> `MIPROv2`; instruction-only (0-shot) -> `MIPROv2` configured for 0-shot; efficient small model after a good large-model program -> `BootstrapFinetune`.
- **Compiled artifact**: the optimized program is saved as JSON (instructions + demos) and loaded at inference; no optimizer runs in production.

## When to use / scenarios
- Multi-stage pipelines (retrieve -> rewrite -> answer -> verify) where tuning each prompt by hand interacts badly with the others ([[advanced-rag]]).
- Classification/extraction with a labelled set: support triage, claims coding, document field extraction.
- Model migration or cost reduction: re-compile the same program for a cheaper model and compare on the dev set ([[model-selection]]).
- Agents with measurable task success where reflective optimizers (GEPA) can use failure traces ([[agent-evaluation]]).
- Not worth it: no metric or labels, a single short prompt, or a task judged only by taste. Also skip it when strict prompt wording is required by compliance review, since the optimizer rewrites instructions.

## Setup & code
`pip install dspy` (3.4.1 on PyPI at verification; requires Python >= 3.10). The core idea, without any LLM, is search over prompt parameters scored by a metric on held-out data. This runnable toy uses a stub "LM" whose behaviour depends on which instruction and demos are in its prompt, and does what `BootstrapFewShotWithRandomSearch` + instruction proposal do: try candidates on a train split, keep the best by dev-set metric.
```python
import itertools, random
random.seed(0)

data = [("card declined at checkout", "billing"), ("refund not received", "billing"),
        ("charged twice this month", "billing"), ("invoice shows wrong VAT", "billing"),
        ("app crashes on login", "tech"), ("password reset email missing", "tech"),
        ("page loads forever", "tech"), ("error 500 when uploading", "tech"),
        ("how do I change my username", "other"), ("love the new design", "other"),
        ("where is your office", "other"), ("can I talk to sales", "other")]
random.shuffle(data)
train, dev = data[:6], data[6:]

def stub_lm(instruction, demos, ticket):
    """Pretend LM: uses words from demos as evidence; 'money' hint helps billing recall."""
    votes = {"billing": 0.0, "tech": 0.0, "other": 0.1}
    for text, label in demos:
        votes[label] += len(set(text.split()) & set(ticket.split()))
    if "money" in instruction and any(w in ticket for w in ("charge", "refund", "invoice", "card")):
        votes["billing"] += 2
    if "errors" in instruction and any(w in ticket for w in ("error", "crash", "loads", "reset")):
        votes["tech"] += 2
    return max(votes, key=votes.get)

def metric(gold, pred): return gold == pred
def score(instr, demos, split): return sum(metric(y, stub_lm(instr, demos, x)) for x, y in split) / len(split)

instructions = ["Classify the ticket.",                                     # baseline (hand-written)
                "Classify the ticket. Tickets about money are billing.",   # proposed candidates
                "Classify the ticket. Tickets about money are billing; errors are tech."]
demo_sets = [list(c) for c in itertools.combinations(train, 2)]             # bootstrapped demo candidates

baseline = score(instructions[0], [], dev)
best = max(((i, d) for i in instructions for d in demo_sets), key=lambda c: score(*c, train))
print(f"baseline dev acc={baseline:.2f}")
print(f"optimized dev acc={score(*best, dev):.2f} with instruction={best[0]!r} and {len(best[1])} demos")
```
The same with DSPy (model ID from [[anthropic-claude]]; DSPy routes `provider/model` strings through LiteLLM):
<!-- skip-check: needs an API key and the dspy package -->
```python
# pip install dspy
from typing import Literal
import dspy

dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-5-5"))

class Triage(dspy.Signature):
    """Route a customer support ticket."""
    ticket: str = dspy.InputField()
    label: Literal["billing", "tech", "other"] = dspy.OutputField()

program = dspy.ChainOfThought(Triage)
rows = [("card declined at checkout", "billing"), ("app crashes on login", "tech"), ("where is your office", "other")]
trainset = [dspy.Example(ticket=t, label=l).with_inputs("ticket") for t, l in rows]  # use 50-300 in practice

def metric(example, pred, trace=None):
    return example.label == pred.label

optimizer = dspy.MIPROv2(metric=metric, auto="light")       # "light" | "medium" | "heavy" search budgets
compiled = optimizer.compile(program, trainset=trainset)
compiled.save("triage.json")                                  # ship this; load with program.load("triage.json")
print(compiled(ticket="I was charged twice").label)
```

## Choosing / trade-offs
| Approach | Needs | Strength | Weakness |
|---|---|---|---|
| Hand-written prompt | A few test cases | Fast, readable, reviewable | Does not scale to many steps or model swaps |
| Few-shot bootstrapping | ~10-50 labelled examples | Cheap, often the biggest single gain | Demo tokens on every call |
| Instruction search (MIPROv2, COPRO) | 50-300+ examples, budget for many LM calls | Finds wording humans miss; can be 0-shot | Optimization run costs real money; can overfit a small dev set |
| Reflective evolution (GEPA) | Traces and ideally textual feedback | Uses *why* it failed, sample-efficient per the paper | Newer; needs a good feedback signal |
| Fine-tuning (BootstrapFinetune / [[fine-tuning-and-peft]]) | Hundreds+ examples, a tunable model | Lowest per-call cost and latency | Training infra; less flexible |

- **DSPy vs other optimizers**: TextGrad-style and provider "prompt improver" tools also exist; DSPy's advantage is optimizing whole multi-module programs against one metric.
- **Budget**: an optimization run makes many LM calls; use a cheaper task model with a stronger proposer/teacher model where the optimizer supports it, and cache.

## Gotchas
- Overfitting: tiny dev sets reward lucky prompts. Hold out a test set the optimizer never sees and report on that.
- The metric is the spec. Exact-match metrics on free text punish correct paraphrases; use an LLM judge or structured outputs ([[structured-output]]) and validate the judge.
- Compiled prompts are model-specific: re-compile when the model changes instead of reusing the JSON.
- Optimized instructions can be odd or contain leaked training examples; review them before production, especially with PII in the trainset.
- DSPy's API has changed across major versions (2.x -> 3.x); pin the version and follow current docs, not old tutorials.
- Optimization does not fix missing context or bad retrieval; check what the model sees first ([[context-engineering]]).

## Related
- [[prompt-engineering]] - the manual baseline DSPy automates.
- [[llm-evaluation]] - metrics, golden sets and LLM judges that drive the optimizer.
- [[structured-output]] - typed outputs make metrics reliable.
- [[context-engineering]] - what goes into the window besides the optimized instruction.
- [[fine-tuning-and-peft]] - the weight-update alternative when prompts plateau.
- [[advanced-rag]] - multi-stage retrieval pipelines that benefit from joint optimization.
- [[agent-evaluation]] - task-success metrics for optimizing agents.
- [[agent-skills]] - hand-authored procedures, the opposite end from learned prompts.
- [[graphrag-and-knowledge-graphs]] - extraction prompts that can be tuned with a labelled sample.

## References
- DSPy repository: https://github.com/stanfordnlp/dspy
- DSPy optimizers guide: https://dspy.ai/learn/optimization/optimizers/
- PyPI package: https://pypi.org/project/dspy/
- Khattab et al., DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines: https://arxiv.org/abs/2310.03714
- Opsahl-Ong et al., Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPRO): https://arxiv.org/abs/2406.11695
- Agrawal et al., GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning: https://arxiv.org/abs/2507.19457
