---
title: RL with verifiable rewards (RLVR) for reasoning models
category: concepts
tags: [rlvr, verifiable-rewards, reinforcement-learning, reasoning-models, grpo, reward-hacking, pass-at-k, trl, verifier, post-training]
use_cases:
  - "train a small open model to get better at math, code or SQL using automatic correctness checks instead of human labels"
  - "write reward functions (answer checkers, unit tests, format checks) for GRPO training in TRL"
  - "diagnose a reasoning RL run where reward goes up but real accuracy does not"
  - "decide whether RL fine-tuning is worth it versus SFT or best-of-n sampling"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2411.15124
  - https://arxiv.org/abs/2501.12948
  - https://arxiv.org/abs/2402.03300
  - https://arxiv.org/abs/2503.14476
  - https://arxiv.org/abs/2504.13837
  - https://arxiv.org/abs/2107.03374
  - https://huggingface.co/docs/trl/grpo_trainer
---

# RL with verifiable rewards (RLVR) for reasoning models

## Summary
Reinforcement learning with verifiable rewards (RLVR) fine-tunes a language model with RL where the reward comes from a *program* that checks the output (does the final answer match the reference, do the unit tests pass, does the SQL return the right rows) instead of a learned reward model or human preference. The term was introduced in the Tulu 3 report, and the recipe (sample several attempts per prompt, score each with a checker, push up the attempts that pass) is how DeepSeek-R1 and many open reasoning models learned long chains of thought. The hard parts are not the RL algorithm ([[grpo-from-scratch-numpy]]) but the *verifier* (it must not be gameable), the *prompt set* (problems the model sometimes, not always or never, solves) and honest evaluation.

## Key concepts
- **Verifier.** A deterministic function `reward(prompt, completion, reference) -> float`. Typical rewards: exact or symbolic math-answer match, unit tests in a sandbox, SQL result equality, JSON schema validity, instruction constraints ("exactly 3 bullet points"). Usually binary; partial credit invites hacking.
- **Outcome vs process reward.** RLVR normally scores only the final outcome; the reasoning in between is free. Process reward models score each step but need labels or a learned model.
- **Format reward.** A small extra reward for the expected structure (e.g. thinking tags, one `\boxed{}` answer) so the answer can be parsed. DeepSeek-R1 used rule-based accuracy and format rewards, and no neural reward model for reasoning tasks.
- **Algorithm.** Mostly GRPO-family methods (group-relative advantages, no critic) or PPO. Variants such as DAPO and Dr. GRPO change loss normalisation, clipping and sampling; see [[grpo-from-scratch-numpy]] and [[ppo-from-scratch]].
- **Learnable band.** A group where every sample passes or every sample fails gets zero advantage, so prompts that are too easy or too hard waste compute. Filter prompts to those with intermediate pass rates, or resample (DAPO's dynamic sampling).
- **Reward hacking.** The policy maximises the checker, not your intent: listing several answers, printing the expected output instead of computing it, special-casing tests, or exploiting parser bugs.
- **pass@k.** Probability that at least one of `k` samples is correct; the unbiased estimator from `n ≥ k` samples with `c` correct is `1 − C(n−c, k)/C(n, k)`. Yue et al. (2025) report that RLVR models beat their base model at small `k` while the base model catches up or overtakes at large `k`: RLVR mostly makes good reasoning paths more likely rather than creating new ones.
- **Cold start.** R1-Zero ran RL directly on a base model; the released R1 first did SFT on curated long chain-of-thought data, then RLVR, which gave more readable output.

## When to use / scenarios
- Math tutoring or quantitative QA where final answers are checkable ([[reasoning-models]]).
- Code generation with unit tests: internal codegen assistants, competitive-programming models ([[coding-agents]], [[sandboxed-code-execution]]).
- Text-to-SQL with execution match against a reference query ([[text-to-sql]]).
- Structured extraction with schema and field-level checks ([[structured-output]]).
- Tool-using agents where the end state is checkable (file contents, API result) ([[agents]]).
- NOT when quality is subjective (tone, helpfulness, creative writing): use preference data and DPO or a reward model ([[rlhf-and-preference-optimization]]).
- NOT as a first step: if the base model never solves the task, there is no signal; do SFT first ([[fine-tuning-and-peft]]). If you only need better answers at inference, try best-of-n with the same verifier, which needs no training ([[self-consistency-voting-from-scratch]]).
- NOT for closed API models unless the provider offers RL fine-tuning with custom graders ([[hosted-fine-tuning-apis]]).

## Setup & code
NumPy only. Two verifiers score completions to 200 addition problems: a lax one ("the right number appears somewhere") and a strict one (exactly one `\boxed{int}` that matches). A toy policy chooses among four behaviours and is trained with GRPO-style group-normalised advantages. Against the lax verifier it learns to hack (list several boxed answers, or skip the format); against the strict verifier it learns to actually solve. The block ends with the unbiased pass@k estimator.
```python
import re
from math import comb
import numpy as np

rng = np.random.default_rng(0)
problems = [(int(a), int(b)) for a, b in rng.integers(10, 99, (200, 2))]

# --- Verifiers: deterministic programs that score a completion against the reference. ---
BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")

def lax_verifier(completion, answer):            # "answer appears somewhere": hackable
    return float(str(answer) in completion)

def strict_verifier(completion, answer):         # exactly one \boxed{int}, and it must match
    found = BOXED.findall(completion)
    return float(len(found) == 1 and int(found[0]) == answer)

# --- A toy "policy": a softmax over four response strategies (stand-ins for behaviours). ---
STRATEGIES = ["solve", "guess", "hedge", "no_format"]

def respond(strategy, a, b):
    if strategy == "solve":                      # real skill: right 60% of the time
        ans = a + b if rng.random() < 0.6 else a + b + int(rng.integers(1, 10))
        return f"{a} + {b}: add tens then units. \\boxed{{{ans}}}"
    if strategy == "guess":
        return f"\\boxed{{{int(rng.integers(20, 200))}}}"
    if strategy == "hedge":                      # reward hack: list many candidates
        return " ".join(f"\\boxed{{{a + b + d}}}" for d in range(-3, 4))
    return f"The answer is {a + b}"              # right but unparseable by the strict checker

def train(verifier, steps=300, G=8, lr=0.5):
    logits = np.zeros(len(STRATEGIES))
    for _ in range(steps):
        a, b = problems[rng.integers(len(problems))]
        p = np.exp(logits - logits.max()); p /= p.sum()
        acts = rng.choice(len(STRATEGIES), size=G, p=p)
        r = np.array([verifier(respond(STRATEGIES[i], a, b), a + b) for i in acts])
        if r.std() == 0:                         # all-same group: no learning signal
            continue
        adv = (r - r.mean()) / r.std()           # GRPO group-relative advantage
        grad = sum(A * (np.eye(len(STRATEGIES))[i] - p) for i, A in zip(acts, adv)) / G
        logits += lr * grad
    p = np.exp(logits - logits.max()); return p / p.sum()

for name, v in [("lax", lax_verifier), ("strict", strict_verifier)]:
    p = train(v)
    shares = {k: round(float(v), 2) for k, v in zip(STRATEGIES, p)}
    true_acc = np.mean([strict_verifier(respond(STRATEGIES[rng.choice(4, p=p)], a, b), a + b)
                        for a, b in problems])
    print(f"{name:6s} verifier -> policy {shares}, "
          f"true accuracy {true_acc:.2f}")

# --- pass@k: unbiased estimator from n samples with c correct (Chen et al. 2021). ---
def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)

n, c = 16, 3
print("pass@1, pass@8 with 3/16 correct:", round(pass_at_k(n, c, 1), 3), round(pass_at_k(n, c, 8), 3))
p_lax, p_strict = train(lax_verifier), train(strict_verifier)
assert p_lax[0] < 0.05 and p_lax[2] + p_lax[3] > 0.9   # lax: reward hacks win, real solving dies
assert p_strict[0] > 0.8                               # strict: only real solving is rewarded
```
The lax run ends with zero true accuracy even though its training reward is near 1: training reward measures the verifier, not the task. Always evaluate with a separate, stricter check (and spot-read samples).

With TRL, a verifier is just a reward function; extra dataset columns arrive as keyword arguments, and it returns one float per completion. Model ID is the one TRL's own quick start uses:
<!-- skip-check: needs torch, trl and a model download; run on a GPU -->
```python
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

rows = [{"prompt": f"What is {a} + {b}? Put the final answer in \\boxed{{}}.", "answer": a + b}
        for a in range(10, 60) for b in range(10, 30)]
BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")

def correct(completions, answer, **kwargs):           # `answer` = dataset column
    out = []
    for text, ref in zip(completions, answer):
        found = BOXED.findall(text if isinstance(text, str) else text[0]["content"])
        out.append(1.0 if len(found) == 1 and int(found[0]) == ref else 0.0)
    return out

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[correct],
    args=GRPOConfig(output_dir="rlvr-addition", num_generations=8, max_completion_length=256),
    train_dataset=Dataset.from_list(rows),
)
trainer.train()
```
For real math answers use a symbolic checker (e.g. Hugging Face `math-verify`) instead of a regex; TRL ships ready-made reward functions in `trl.rewards` (e.g. `accuracy_reward`). Larger runs use frameworks such as verl or OpenRLHF that add vLLM rollouts and multi-node training.

## Choosing / trade-offs
- **RLVR vs SFT on correct solutions (rejection sampling / STaR).** SFT on filtered self-generated solutions is simpler and stable; RLVR also learns from failures through negative advantages and usually goes further on hard reasoning, at higher cost and instability.
- **RLVR vs best-of-n at inference.** Best-of-n with the verifier costs `n` times the inference per query but no training; RLVR moves that cost to training time. If the verifier is available at inference (tests can run), best-of-n is a strong baseline.
- **Binary vs partial rewards.** Binary is hard to hack but sparse; partial credit (fraction of tests passed) gives denser signal but rewards half-solutions and test special-casing.
- **Group size.** Larger groups find a passing sample on harder prompts and give lower-variance advantages, at proportionally more generation cost.
- **KL penalty.** A KL term to a reference model limits drift and protects other skills; several recent recipes (and TRL's default `beta=0.0`) drop it for reasoning RL. Keep it if you see general capabilities regress.
- **Length.** Reasoning RL tends to grow response length; cap `max_completion_length` and decide how to treat truncated responses (zero reward vs masked out), because it changes what is learned.

## Gotchas
- Training reward going up is not success. Track accuracy on a held-out set with a strict checker, and read samples for hacks.
- Verifier bugs become model behaviour: lenient regexes, string matching that accepts "12" inside "112", tests that check output format only, timeouts counted as passes.
- Run generated code only in a sandbox with time and memory limits; the policy will eventually write infinite loops or touch the filesystem ([[sandboxed-code-execution]]).
- Contamination: if eval problems or their answers are in the RL prompt set, scores are meaningless. Deduplicate by normalised text and near-duplicates.
- Prompts the model already always solves or never solves give zero gradient; measure per-prompt pass rate before training and filter.
- Chat template mismatches (missing generation prompt, wrong end-of-turn token) produce garbage completions that all score 0.
- Gains are often narrow: improved math can coincide with regressions elsewhere. Run a broad eval suite before and after ([[llm-evaluation]]).
- Compare at matched inference compute: an RLVR model producing 4x longer outputs should be compared to the base model with best-of-n or more thinking budget, and report pass@k at several `k`.

## Related
- [[grpo-from-scratch-numpy]] - the GRPO update used in most RLVR runs, built in NumPy.
- [[reasoning-models]] - the models RLVR produces and how to use them.
- [[rlhf-and-preference-optimization]] - RLHF, DPO and reward models, with TRL setup.
- [[ppo-from-scratch]] - the clipped objective and critic that GRPO simplifies.
- [[reinforce-policy-gradient-from-scratch]] - the policy-gradient estimator underneath.
- [[self-consistency-voting-from-scratch]] - the inference-time alternative: sample and vote.
- [[sandboxed-code-execution]] - running code-based verifiers safely.
- [[llm-evaluation]] - held-out evaluation that a reward curve cannot replace.

## References
- Lambert et al. (2024), Tulu 3: Pushing Frontiers in Open Language Model Post-Training (introduces RLVR): https://arxiv.org/abs/2411.15124
- DeepSeek-AI (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning: https://arxiv.org/abs/2501.12948
- Shao et al. (2024), DeepSeekMath (introduces GRPO): https://arxiv.org/abs/2402.03300
- Yu et al. (2025), DAPO: An Open-Source LLM Reinforcement Learning System at Scale: https://arxiv.org/abs/2503.14476
- Yue et al. (2025), Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?: https://arxiv.org/abs/2504.13837
- Chen et al. (2021), Evaluating Large Language Models Trained on Code (pass@k estimator): https://arxiv.org/abs/2107.03374
- Hugging Face TRL, GRPO Trainer: https://huggingface.co/docs/trl/grpo_trainer
