---
title: Synthetic data for LLM training
category: concepts
tags: [synthetic-data, self-instruct, distillation, evol-instruct, data-filtering, decontamination, model-collapse, licences, sft, preference-data]
use_cases:
  - "generate instruction-tuning data for a domain model when we have few real examples"
  - "distil a large API model's behaviour into a small open model we can self-host"
  - "filter and deduplicate LLM-generated training examples before fine-tuning"
  - "check whether a model's terms of use allow training on its outputs"
  - "make sure synthetic training data does not leak our evaluation set"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2212.10560
  - https://arxiv.org/abs/2304.12244
  - https://arxiv.org/abs/2306.02707
  - https://arxiv.org/abs/2306.11644
  - https://arxiv.org/abs/2212.08073
  - https://www.nature.com/articles/s41586-024-07566-y
  - https://openai.com/policies/terms-of-use/
  - https://www.anthropic.com/legal/commercial-terms
  - https://www.llama.com/llama3_1/license/
---

# Synthetic data for LLM training

## Summary
Synthetic data is training data written by a model instead of people: instructions and answers for supervised fine-tuning, preference pairs for DPO/RLHF, reasoning traces, or domain documents for continued pretraining. It is cheap and scalable and is how many small models are distilled from larger ones, but its value comes almost entirely from filtering (quality, diversity, deduplication, decontamination), and its use is constrained by the generating model's licence or terms of service.

## Key concepts
- **Self-Instruct** (Wang et al.): start from a small pool of human-written seed tasks (175 in the paper), prompt a model to write new instructions, inputs and outputs, filter out low-quality and near-duplicate ones (ROUGE-L similarity above 0.7 to any existing instruction is dropped), add survivors to the pool, repeat.
- **Evol-Instruct** (WizardLM): rewrite existing instructions to be harder (add constraints, deepen, concretise) or broader, to raise difficulty and diversity.
- **Distillation from a stronger model**: the teacher answers prompts, the student is fine-tuned on the answers; richer traces (step-by-step explanations, as in Orca) transfer more than bare answers ([[knowledge-distillation-and-compression]]).
- **Textbook-style data** (phi-1, "Textbooks Are All You Need"): small, carefully generated, high-educational-value corpora can beat much larger scraped corpora for a narrow skill.
- **AI feedback for preferences**: a model critiques or ranks responses against written principles (Constitutional AI / RLAIF) to build preference data ([[rlhf-and-preference-optimization]]).
- **Rejection sampling with a verifier**: generate many solutions, keep only those that pass tests or match a known answer; the basis of reasoning-trace datasets ([[test-time-compute-scaling]]).
- **Filtering pipeline**: format/length checks -> exact and near-duplicate removal -> quality scoring (rules, a judge model, or a verifier) -> diversity balancing across topics -> decontamination against eval sets.
- **Decontamination**: remove training examples that overlap eval items (e.g. by long n-gram overlap); otherwise you measure memorisation ([[llm-evaluation]]).
- **Model collapse**: models trained recursively on their own generated data lose the tails of the original distribution (Shumailov et al., Nature 2024). Keep real data in the mix and do not train generation after generation on synthetic output only.
- **Licences and terms**: OpenAI's terms prohibit using output "to develop models that compete with OpenAI"; Anthropic's commercial terms restrict using the services to build competing products or train competing AI models without approval; the Llama 3.1 Community License requires a model trained on Llama outputs and distributed to have "Llama" at the start of its name. Open-weight models under Apache-2.0 or MIT usually carry no such restriction ([[model-licenses]]).

## When to use / scenarios
- Domain assistant (insurance claims, internal IT support) with a few hundred real tickets: generate paraphrases and new cases seeded from real ones, then fine-tune a small model ([[fine-tuning-and-peft]], [[hosted-fine-tuning-apis]]).
- Distilling a classifier or extractor from a frontier model into a small self-hosted model to cut cost or keep data on-prem, where the provider's terms permit it ([[small-language-models]]).
- Structured-output training: generate many valid JSON examples against a schema and validate each programmatically ([[structured-output]]).
- Maths and code: rejection-sampled solutions checked by tests or exact answers.
- Privacy-sensitive data (health, finance): synthetic records instead of real ones, after checking they do not reproduce real individuals ([[data-labeling-and-synthetic-data]]).
- Not a replacement for an evaluation set: keep a real, human-checked test set. Not useful when the generator itself cannot do the task; you will distil its errors.

## Setup & code
A standard-library filtering stage for generated instruction data: normalisation, length and refusal filters, near-duplicate removal with a ROUGE-L style similarity (longest common subsequence), and n-gram decontamination against an eval set.

```python
import re

def tokens(s):
    return re.findall(r"\w+", s.lower())

def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]

def rouge_l_f(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    l = lcs_len(ta, tb)
    p, r = l / len(ta), l / len(tb)
    return 0.0 if l == 0 else 2 * p * r / (p + r)

def ngrams(s, n):
    t = tokens(s)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}

REFUSAL = re.compile(r"\b(as an ai|i cannot|i can't help|i'm unable to)\b", re.I)

def filter_examples(candidates, pool, eval_items, sim_max=0.7, ngram_n=8,
                    min_words=3, max_words=300):
    eval_grams = set().union(*(ngrams(e, ngram_n) for e in eval_items))
    kept, reasons = [], {}
    def drop(why):
        reasons[why] = reasons.get(why, 0) + 1
    for ex in candidates:
        instr, out = ex["instruction"].strip(), ex["output"].strip()
        n_words = len(tokens(instr + " " + out))
        if not instr or not out or n_words < min_words or n_words > max_words:
            drop("length"); continue
        if REFUSAL.search(out):
            drop("refusal"); continue
        if ngrams(instr + " " + out, ngram_n) & eval_grams:
            drop("eval-contamination"); continue
        if any(rouge_l_f(instr, p) > sim_max for p in pool):
            drop("near-duplicate"); continue
        pool.append(instr)
        kept.append(ex)
    return kept, reasons

seed_pool = ["Classify the sentiment of a customer review as positive, negative or neutral.",
             "Summarise an insurance claim note in two sentences."]
eval_set = ["The policyholder reports water damage in the kitchen after a pipe burst on Monday night"]
generated = [
    {"instruction": "Classify the sentiment of this customer review as positive, negative or neutral.",
     "output": "negative"},                                                     # near-duplicate of a seed
    {"instruction": "Extract the incident date from the claim note.",
     "output": "The policyholder reports water damage in the kitchen after a pipe burst on Monday night: Monday"},
    {"instruction": "Write a polite reply refusing a refund outside the 30-day window.",
     "output": "Thank you for contacting us. Our policy allows refunds within 30 days of purchase, "
               "so we cannot process this one, but we can offer store credit."},
    {"instruction": "Explain the claim.", "output": "As an AI, I cannot see the claim."},
    {"instruction": "List three questions an adjuster should ask about a stolen bicycle claim.",
     "output": "1. Was the bicycle locked? 2. Is there a police report number? 3. Do you have proof of purchase?"},
]
kept, reasons = filter_examples(generated, list(seed_pool), eval_set)
print(f"kept {len(kept)}/{len(generated)}; dropped: {reasons}")
for ex in kept:
    print("-", ex["instruction"])
assert reasons == {"near-duplicate": 1, "eval-contamination": 1, "refusal": 1}
```

The generation side, Self-Instruct style, with a teacher model from [[anthropic-claude]]. Check the provider's terms first: training a competing model on Claude or OpenAI outputs is restricted.

<!-- skip-check: needs ANTHROPIC_API_KEY and network -->
```python
import anthropic, json, random

client = anthropic.Anthropic()
pool = ["Summarise an insurance claim note in two sentences.",
        "Extract the policy number and incident date from a claim email as JSON."]

def generate_batch(pool, k=8):
    shots = "\n".join(f"- {t}" for t in random.sample(pool, min(4, len(pool))))
    r = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=4000,
        messages=[{"role": "user", "content":
            "Here are example tasks for an insurance-claims assistant:\n" + shots +
            f"\nWrite {k} new, different tasks with realistic inputs and ideal outputs. "
            'Return only a JSON list of {"instruction": ..., "input": ..., "output": ...}.'}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return json.loads(text[text.find("["): text.rfind("]") + 1])

candidates = generate_batch(pool)
# then: kept, reasons = filter_examples(candidates, pool, eval_set)  (block above), judge-score, repeat
```

## Choosing / trade-offs
- **Teacher choice**: a stronger teacher gives better data but higher cost and stricter terms; an open-weight teacher under a permissive licence avoids competitive-use clauses ([[model-licenses]], [[meta-llama]], [[qwen]], [[deepseek]]).
- **Quantity vs quality**: a small, filtered, diverse set usually beats a large raw one (phi-1, Orca results); spend generation budget on harder and more varied prompts, not more of the same.
- **Bare answers vs explanations**: explanation traces transfer reasoning but cost more tokens and can teach the student to be verbose.
- **Judge filtering**: an LLM judge scales quality checks but shares the generator's blind spots; calibrate it on a human-labelled sample ([[llm-evaluation]]).
- **Real/synthetic mix**: more synthetic data raises coverage but drifts from the real distribution; keep a real-data share and evaluate on real data only.

## Gotchas
- Synthetic data inherits the generator's errors, biases and style; the student cannot exceed a teacher it only imitates, except via verifier filtering.
- Diversity collapses fast: the same prompt template yields the same phrasing; vary seeds, personas, topics and difficulty explicitly and measure n-gram diversity.
- Benchmark contamination: the teacher may have memorised public benchmarks and reproduce them in "new" examples; decontaminate against every eval you report.
- Privacy: models can regurgitate training data, including personal data; scan synthetic records for real identifiers ([[ai-security-privacy-compliance]]).
- Terms change and apply per provider and per product tier; record which model, version and terms produced each dataset (lineage, [[model-registry-and-versioning]]).
- Licence clauses can propagate: a model trained on Llama outputs carries Llama naming obligations when distributed.
- Synthetic refusals ("As an AI, I cannot...") slip into datasets and teach over-refusal; filter them.

## Related
- [[data-labeling-and-synthetic-data]] - labelling workflows and synthetic data for classic ML and eval sets.
- [[knowledge-distillation-and-compression]] - distillation objectives and student models.
- [[fine-tuning-and-peft]] - training on the generated data.
- [[rlhf-and-preference-optimization]] - preference data and AI feedback.
- [[rl-with-verifiable-rewards]] - training on verifier-checked outputs instead of fixed synthetic targets.
- [[model-licenses]] - which model outputs you may train on.
- [[test-time-compute-scaling]] - rejection sampling and verifiers for higher-quality traces.
- [[label-noise-and-data-cleaning]] - finding bad labels after generation.
- [[pretraining-and-scaling-laws]] - data quantity vs quality at pretraining scale.

## References
- Wang et al. (2023), Self-Instruct: https://arxiv.org/abs/2212.10560
- Xu et al. (2023), WizardLM / Evol-Instruct: https://arxiv.org/abs/2304.12244
- Mukherjee et al. (2023), Orca: Progressive Learning from Complex Explanation Traces: https://arxiv.org/abs/2306.02707
- Gunasekar et al. (2023), Textbooks Are All You Need: https://arxiv.org/abs/2306.11644
- Bai et al. (2022), Constitutional AI: https://arxiv.org/abs/2212.08073
- Shumailov et al. (2024), AI models collapse when trained on recursively generated data, Nature: https://www.nature.com/articles/s41586-024-07566-y
- OpenAI terms of use: https://openai.com/policies/terms-of-use/
- Anthropic commercial terms: https://www.anthropic.com/legal/commercial-terms
- Llama 3.1 Community License: https://www.llama.com/llama3_1/license/
