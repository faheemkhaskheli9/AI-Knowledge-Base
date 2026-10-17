---
title: LLM-as-judge and eval-driven development
category: llm-apps
tags: [llm-as-judge, evals, eval-driven-development, golden-dataset, pairwise, rubric, judge-bias, position-bias, ci, regression-gate]
use_cases:
  - "set up eval-driven development for a prompt or RAG change so regressions block the merge"
  - "write a judge prompt and check that it agrees with human graders"
  - "choose between pairwise comparison and rubric scoring to compare two model versions"
  - "remove position and verbosity bias from an LLM judge"
  - "run a golden dataset in CI and fail the build on a significant quality drop"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2306.05685
  - https://arxiv.org/abs/2305.17926
  - https://arxiv.org/abs/2303.16634
  - https://arxiv.org/abs/2404.12272
  - https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
  - https://platform.openai.com/docs/guides/evals
---

# LLM-as-judge and eval-driven development

## Summary
Eval-driven development treats the eval set like a test suite: write cases
for the behaviour you want before changing the prompt, model or retriever, run
them on every change, and block merges that make things significantly worse.
Where code cannot score an output, an LLM judge does, but a judge is itself a
model with biases (answer position, length, its own style), so it needs a
precise rubric, order swapping for pairwise tests, and a measured agreement
with human graders before its numbers gate anything. The basic harness and
scoring ladder are in [[llm-evaluation]]; this topic covers the development
loop, judge design and CI gating.

## Key concepts
- **The loop**: collect failures (production logs, user reports, red-team
  findings) -> turn each into a golden case -> change the system -> run the
  suite -> compare against the baseline run -> ship only if no significant
  regression. Every bug found in production becomes a case.
- **Golden dataset**: inputs plus per-case expectations (reference answer,
  must-include facts, must-not behaviours, or a rubric). Slice it by
  category (intent, language, difficulty, adversarial) so a regression in one
  slice is not averaged away. Keep a held-out slice you do not tune against.
  Version it next to the code.
- **Rubric (pointwise) scoring**: the judge scores one output against explicit
  criteria. Binary pass/fail per criterion is more stable than a 1-10 scale.
  G-Eval (Liu et al., 2023) has the judge write evaluation steps first, then
  fill a score form.
- **Pairwise scoring**: the judge picks the better of two outputs. Good for
  "is the new version better than the old" and for subjective qualities;
  produces win rates, which aggregate into Elo / Bradley-Terry rankings
  ([[elo-and-bradley-terry-leaderboard-from-scratch]]).
- **Reference-guided judging**: give the judge the reference answer;
  Zheng et al. (MT-Bench, 2023) report it helps on maths and reasoning items
  where judges otherwise accept wrong answers.
- **Judge biases**: position bias (Wang et al., 2023 showed reordering two
  answers can flip which model wins), verbosity bias, self-enhancement
  (preferring its own outputs), and limited ability to check maths and
  facts (Zheng et al., 2023).
- **Mitigations**: run pairwise in both orders and count a win only when both
  agree (else a tie); tell the judge length is not quality and cap answer
  length; use a judge from a different family than the generator; ask for
  reasoning before the verdict; give reference answers.
- **Judge calibration**: hand-label 50-200 outputs, compare with the judge
  using agreement and Cohen's kappa (agreement corrected for chance). Re-check
  after changing the judge model or prompt. Shankar et al. (2024) found
  graders' criteria shift as they read outputs ("criteria drift"), so expect
  to revise the rubric while labelling.
- **Regression gate**: compare per-case scores of the candidate with the
  baseline on the same cases (paired), and fail on a drop beyond noise, on any
  failing must-pass case, or on a slice falling below its threshold.

## When to use / scenarios
- Prompt and model changes in a support bot or RAG assistant: rubric judge
  for faithfulness and tone, deterministic checks for format and facts.
- Model migration or cost-down (switching to a smaller model): pairwise
  judge, old vs new, both orders, plus the rubric suite ([[model-selection]]).
- Summarisation, writing and other open-ended output where exact match is
  impossible.
- Safety regression after red-teaming: findings become must-pass cases
  ([[red-teaming-llm-apps]]).
- Do not use a judge where code can decide: JSON validity, SQL result
  equality, unit tests, exact labels ([[structured-output]], [[text-to-sql]]).
- Agents: judge the final outcome and check the trajectory with code
  ([[agent-evaluation]]).

## Setup & code
Three pieces in NumPy, on simulated data: (1) a judge with position bias and
the swap-and-aggregate fix, (2) judge-human agreement with Cohen's kappa,
(3) a CI gate using a paired bootstrap over per-case scores plus must-pass
cases.

```python
import numpy as np

rng = np.random.default_rng(1)

# 1) Position bias in pairwise judging, and the swap-and-aggregate fix.
n = 400
quality_a = rng.normal(0.0, 1.0, n)
quality_b = quality_a + rng.normal(0.0, 1.0, n)     # B is a different system; truth = higher quality wins
truth_b_wins = quality_b > quality_a

def biased_judge(first, second, position_bias=0.8):
    """Returns True if it prefers the FIRST answer shown. position_bias adds logit to slot 1."""
    p_first = 1 / (1 + np.exp(-(2.0 * (first - second) + position_bias)))
    return rng.random(first.shape) < p_first

b_first = biased_judge(quality_b, quality_a)           # B shown first
a_first = biased_judge(quality_a, quality_b)           # A shown first (swapped order)
b_wins_one_order = b_first
b_wins_swapped = ~a_first
consistent = b_wins_one_order == b_wins_swapped
print(f"true B win rate              {truth_b_wins.mean():.2f}")
print(f"judge, B always shown first  {b_wins_one_order.mean():.2f}")
print(f"judge, A always shown first  {b_wins_swapped.mean():.2f}")
# Count a win only when both orders agree; disagreement = tie (0.5).
score = np.where(consistent, b_wins_one_order.astype(float), 0.5)
print(f"swap-and-aggregate           {score.mean():.2f}  (ties: {1 - consistent.mean():.0%})")

# 2) Calibrate the judge against human labels before trusting it: Cohen's kappa.
def cohens_kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = (x == y).mean()
    pe = sum((x == c).mean() * (y == c).mean() for c in np.union1d(x, y))
    return (po - pe) / (1 - pe)

human = rng.random(120) < 0.7                          # human pass/fail on 120 sampled outputs
judge = np.where(rng.random(120) < 0.85, human, ~human) # judge agrees 85% of the time
always_pass = np.ones(120, dtype=bool)
print(f"judge vs human: agreement {np.mean(judge == human):.2f}, kappa {cohens_kappa(judge, human):.2f}")
print(f"'always pass' : agreement {np.mean(always_pass == human):.2f}, kappa {cohens_kappa(always_pass, human):.2f}")

# 3) CI regression gate: paired bootstrap on per-case scores, plus must-pass cases.
def regression_gate(base, cand, must_pass_ids, tol=0.02, n_boot=2000, seed=0):
    base, cand = np.asarray(base, float), np.asarray(cand, float)
    r = np.random.default_rng(seed)
    d = cand - base
    boots = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    broken = [i for i in must_pass_ids if cand[i] < 1]
    ok = not broken and d.mean() >= -tol and hi >= 0
    return ok, dict(delta=round(float(d.mean()), 3), ci=(round(float(lo), 3), round(float(hi), 3)), broken=broken)

base = (rng.random(200) < 0.80).astype(float)          # per-case pass/fail of the current prompt
same = base.copy()                                     # 6 cases flip each way: run-to-run noise
same[np.flatnonzero(base == 1)[-6:]] = 0
same[np.flatnonzero(base == 0)[:6]] = 1
worse = base.copy(); worse[np.flatnonzero(base)[:16]] = 0
for name, cand in [("noise-level change", same), ("real regression", worse)]:
    ok, info = regression_gate(base, cand, must_pass_ids=[int(np.flatnonzero(base)[0])])
    print(f"{name:20s} -> {'PASS' if ok else 'FAIL'} {info}")

assert regression_gate(base, same, [])[0] and not regression_gate(base, worse, [])[0]
```
What it shows: one-order pairwise results swing from 0.60 to 0.36 with the
same answers, while swap-and-aggregate lands near the true 0.50 (with many
ties, which is the honest outcome when quality is close). An "always pass"
judge reaches 0.76 raw agreement but kappa 0, which is why raw agreement
alone is not a calibration. The gate lets run-to-run noise through and blocks
a real drop and a broken must-pass case.

A pairwise judge call with both orders (model IDs from [[anthropic-claude]];
use a judge model different from the one that generated the answers):
<!-- skip-check: needs an API key -->
```python
# pip install anthropic pydantic
from typing import Literal
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()
JUDGE = "claude-opus-5-5"

class Pick(BaseModel):
    reasoning: str
    winner: Literal["1", "2", "tie"]

RUBRIC = ("Compare two answers to the user's question. Criteria: factually correct, answers the question "
          "asked, follows the policy. Length and style are not quality. Reason first, then pick.")

def pick(question: str, first: str, second: str) -> str:
    r = client.messages.parse(model=JUDGE, max_tokens=600, output_format=Pick, system=RUBRIC,
        messages=[{"role": "user", "content":
            f"<question>{question}</question>\n<answer_1>{first}</answer_1>\n<answer_2>{second}</answer_2>"}])
    return r.parsed_output.winner

def compare(question: str, a: str, b: str) -> str:
    x, y = pick(question, a, b), pick(question, b, a)      # both orders
    if x == "1" and y == "2":
        return "a"
    if x == "2" and y == "1":
        return "b"
    return "tie"                                           # inconsistent or tied
```

Running it in CI (GitHub Actions; the eval script exits non-zero when the
gate fails, and writes the per-case results as an artifact for review):
```yaml
# .github/workflows/evals.yml
on: pull_request
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: python evals/run.py --baseline evals/baseline.json --out results.json
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: eval-results, path: results.json}
```
Frameworks that implement the same loop: Promptfoo (YAML assertions,
`llm-rubric`, CI integration), Inspect, DeepEval, OpenAI Evals, and
LangSmith / Braintrust / Langfuse datasets with experiment comparison.

## Choosing / trade-offs
| Choice | Pick A when | Pick B when |
|---|---|---|
| Pairwise vs rubric | comparing two versions, subjective quality, small differences | absolute quality bar, tracking over time, many versions |
| Binary criteria vs scale | stable, easy to calibrate, clear gate | you need to rank outputs that all pass |
| Strong vs cheap judge | gate decisions, subtle criteria | high volume monitoring, after calibration shows a cheap judge agrees |
| Every PR vs nightly | small suite, fast, cheap judge | large suite, slow or costly judge; run a smoke subset per PR |
| Real vs synthetic cases | real logs exist (better coverage of what users do) | cold start, rare edge cases; review synthetic cases by hand |

- **Cost**: pairwise in both orders doubles judge calls; cache results keyed
  by (case, output hash, judge prompt, judge model) so unchanged outputs are
  not re-judged.
- **Noise vs sensitivity**: more cases and repeated samples narrow the
  confidence interval; a tiny suite can only detect large regressions.

## Gotchas
- A judge with high raw agreement can still be useless when one class
  dominates; report kappa or per-class agreement.
- Judging with the generator's own model inflates scores (self-preference).
- Changing the judge prompt or model changes every score; re-baseline and
  re-calibrate, do not compare across judge versions.
- Averages hide slice regressions; gate per slice and on must-pass cases.
- Temperature and sampling make both the system and the judge noisy; run
  multiple samples for borderline cases and compare distributions.
- Tuning the prompt until the suite passes overfits it; keep a held-out
  slice and keep adding fresh production failures.
- Judges asked for a score before reasoning anchor on a first impression; ask
  for reasoning (or criteria checks) first.
- Never let the judge see which system produced which answer.

## Related
- [[llm-evaluation]] - golden sets, the scoring ladder and a basic judge harness.
- [[agent-evaluation]] - outcome and trajectory scoring for agents.
- [[red-teaming-llm-apps]] - adversarial findings that become regression cases.
- [[elo-and-bradley-terry-leaderboard-from-scratch]] - turning pairwise wins into rankings.
- [[reading-ai-benchmarks]] - public benchmarks vs your own evals.
- [[prompt-engineering]] - what the loop iterates on.
- [[llm-observability]] - sampling production traffic into the golden set.
- [[workflows-vs-agents]] - per-step evals are easier for fixed workflows.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/testing/snapshot-and-golden-testing.md - golden-file testing, the deterministic cousin of golden eval sets.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/ci-cd-pipelines.md - where the eval gate sits in the pipeline.

## References
- Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena: https://arxiv.org/abs/2306.05685
- Wang et al., Large Language Models are not Fair Evaluators: https://arxiv.org/abs/2305.17926
- Liu et al., G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment: https://arxiv.org/abs/2303.16634
- Shankar et al., Who Validates the Validators?: https://arxiv.org/abs/2404.12272
- Anthropic, Define success and build evaluations: https://platform.claude.com/docs/en/test-and-evaluate/develop-tests
- OpenAI Evals guide: https://platform.openai.com/docs/guides/evals
