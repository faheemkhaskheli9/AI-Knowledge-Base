---
title: Low-budget AI (cheapest viable stack)
category: scenarios
tags: [low-budget, free-tier, cost, local-models, small-models, ollama, cpu, students, nonprofits, bootstrapped]
use_cases:
  - "build an AI feature with no budget, on a laptop or free tiers"
  - "find the cheapest stack that still passes my quality bar"
  - "run a model locally on CPU or a small GPU instead of paying per token"
  - "cut an existing LLM bill without losing quality"
  - "prototype on free API tiers without leaking sensitive data"
status: draft
last_verified: 2026-10-15
sources:
  - https://ai.google.dev/gemini-api/docs/pricing
  - https://openrouter.ai/pricing
  - https://console.groq.com/docs/rate-limits
  - https://huggingface.co/docs/inference-providers/pricing
  - https://github.com/ollama/ollama
  - https://scikit-learn.org/stable/
---

# Low-budget AI (cheapest viable stack)

## Summary
For students, nonprofits, side projects and bootstrapped teams: how to get a
working AI feature for little or no money. The order of savings is: do not
use an LLM where classic ML or rules work; use the smallest model that
passes your eval; run it locally if you already own the hardware; use free
API tiers for prototyping only, with non-sensitive data; and cut tokens
(cache, batch, shorter prompts) before paying for a bigger plan. No prices
are quoted here because they change monthly; check each provider's page.

## Key concepts
- **Your eval set is the budget tool**: 50-200 labelled examples tell you
  which cheap option is good enough. Without it you overpay for safety or
  ship something broken ([[llm-evaluation]]).
- **Cost ladder (cheapest first)**: rules/regex -> classic ML on CPU
  ([[classic-ml-scikit-learn]], [[gradient-boosting-tabular]]) ->
  embeddings + nearest neighbour ([[embeddings]]) -> small local model
  ([[small-language-models]], [[ollama]], [[llama-cpp-gguf]]) -> cheap hosted
  tier -> frontier API for the hard tail only.
- **Free tiers trade money for data and limits**: free API tiers are rate
  limited and some use your prompts to improve products. Google's Gemini API
  pricing page, for example, lists free-tier content as used to improve
  products and paid-tier content as not. Read the terms before sending
  anything personal or confidential ([[ai-security-privacy-compliance]]).
- **Local is free per call, not free**: you pay in hardware you already own,
  electricity, setup time and lower quality. Quantised 1-8B models run on a
  laptop CPU or a consumer GPU ([[quantization]]).
- **Cascades**: a cheap model handles confident cases, an expensive one
  only the uncertain rest. This often cuts most of the spend while keeping
  quality ([[llm-gateways-and-routing]], [[conformal-prediction-and-uncertainty]]).
- **Token hygiene**: short system prompts, capped outputs, cached stable
  prefixes, batch APIs for non-urgent work ([[prompt-caching-and-cost]],
  [[cost-and-latency]]).

## When to use / scenarios
1. **Text classification or routing (tickets, emails, intents).** Cheapest:
   TF-IDF or embeddings + logistic regression on CPU; LLM only for the
   uncertain band. Read: [[nlp-classic-tasks]], [[embeddings]].
2. **Chat over your documents.** Cheapest: local embedding model + a
   file-based vector store + a small local or cheap hosted model; BM25 alone
   is a strong free baseline. Read: [[rag-basics]], [[embedding-models]],
   [[bm25-from-scratch]], [[vector-databases]].
3. **Summarisation and extraction at volume.** Cheapest: a batch API or a
   small model with schema-constrained output. Read: [[structured-output]],
   [[prompt-caching-and-cost]].
4. **Transcription.** Cheapest: open speech-to-text models locally. Read:
   [[speech-to-text]].
5. **Image tasks.** Cheapest: a pretrained classifier/detector fine-tuned on
   free notebook GPUs, run on CPU at inference. Read:
   [[image-classification]], [[object-detection]],
   [[transfer-learning-and-domain-adaptation]].
6. **Coding help on a budget.** Local coding models through Ollama or a
   cheap tier; keep the frontier model for hard tasks. Read:
   [[coding-agents]].
7. **Tabular prediction or forecasting.** Never pay for an LLM here. Read:
   [[gradient-boosting-tabular]], [[time-series-forecasting]].
8. **Fine-tuning on a budget.** LoRA/QLoRA on a single consumer or free
   notebook GPU; only after prompting fails. Read: [[fine-tuning-and-peft]].
- **When cheapest is wrong**: regulated or sensitive data on free tiers;
  safety-relevant outputs where a weaker model's error rate is unacceptable;
  when developer time spent self-hosting costs more than the API bill.

## Setup & code
Zero-cost stacks:

| Need | Free / near-free option | Read |
|---|---|---|
| Run an LLM | Ollama or llama.cpp with a quantised small model | [[ollama]], [[llama-cpp-gguf]] |
| Embeddings | Open embedding model via sentence-transformers or Ollama | [[embedding-models]] |
| Vector store | In-process store (FAISS, Chroma, SQLite-based) | [[vector-databases]] |
| Hosted LLM for prototyping | Free tiers (Gemini API, Groq, OpenRouter `:free` models, Hugging Face Inference Providers credits); all rate limited, check terms | [[model-selection]], [[llm-gateways-and-routing]] |
| Training GPU | Free notebook tiers, or rent by the hour | [[gpu-cloud-options]] |
| Serving | One small VM with FastAPI; scale later | [[serving-with-fastapi]] |

Runnable cascade: a free CPU classifier answers confident requests and only
the uncertain ones go to a paid LLM.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
intents = {
    "refund": ["refund my order", "money back please", "i want a refund", "return and refund"],
    "shipping": ["where is my parcel", "track my package", "delivery is late", "when will it ship"],
    "account": ["reset my password", "cannot log in", "change my email", "delete my account"],
}
fillers = ["", "hi,", "please", "asap", "thanks", "hello team"]
texts, labels = [], []
for label, phrases in intents.items():
    for _ in range(80):
        texts.append(f"{rng.choice(fillers)} {rng.choice(phrases)} {rng.choice(fillers)}")
        labels.append(label)
for i in range(30):                                       # ambiguous: humans label it either way
    texts.append("my parcel arrived broken what now")
    labels.append("refund" if i % 2 else "shipping")

Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.3, random_state=0, stratify=labels)
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(Xtr)
clf = LogisticRegression(max_iter=1000).fit(vec.transform(Xtr), ytr)

proba = clf.predict_proba(vec.transform(Xte))
confident = proba.max(axis=1) >= 0.6                     # tune on a validation set
pred = clf.classes_[proba.argmax(axis=1)]
acc_cheap = (pred[confident] == np.array(yte)[confident]).mean()
print(f"handled free on CPU: {confident.mean():.0%}  accuracy there: {acc_cheap:.0%}")
print(f"escalate to an LLM: {(~confident).sum()} of {len(yte)} requests")
assert 0.5 < confident.mean() < 1 and acc_cheap > 0.9
```

In production the escalated requests go to whichever hosted or local model
passed your eval; log both paths and re-check the threshold monthly.

## Choosing / trade-offs
- **Free tier vs paid tier**: free tiers are fine for prototypes and public
  data; paid tiers usually bring higher limits and stricter data handling.
- **Local vs cheapest hosted**: local wins when data must stay on your
  machine, you need offline use, or volume is steady and hardware is already
  owned; hosted wins on quality per unit of effort and for spiky low volume
  ([[cost-and-latency]]).
- **Small model + fine-tune vs big model + prompt**: fine-tuning a small
  model costs time and data up front and pays off at volume on a narrow task
  ([[knowledge-distillation-and-compression]], [[hosted-fine-tuning-apis]]).
- **Aggregators vs direct providers**: an aggregator (e.g. OpenRouter)
  makes switching to the cheapest model easy; going direct gives you the
  provider's own data terms and features ([[llm-gateways-and-routing]]).

## Gotchas
- Free tiers change or vanish; keep the provider behind a thin interface
  so swapping costs an afternoon, not a rewrite.
- Rate-limit errors (429) on free tiers look like outages; add retries with
  backoff and a local fallback.
- Some free tiers count failed requests against the daily cap.
- Quantising too far (very low bit widths) breaks quality on reasoning and
  multilingual tasks; check on your eval, not on vibes.
- Unbounded agent loops and long chat histories turn a cheap model into an
  expensive bill; cap iterations, tokens and spend per user.
- Model licences: "free to download" is not always free for commercial use
  ([[model-licenses]]).
- Never put API keys in client-side code or public repos; a leaked key on a
  paid account is the most expensive outcome of a "free" project.

## Related
- [[startup-mvp-with-ai]] - budget and stage decisions for a product team.
- [[small-language-models]] - which small models to run locally.
- [[cost-and-latency]] - the token and latency arithmetic.
- [[edge-on-device]] - running on phones and small boards.
- [[education]] - learners and teachers on tight budgets.
- [[llm-cost-calculator]] - estimate the bill before choosing a tier.
- [[batch-inference-and-batch-apis]] - discounted asynchronous processing for non-urgent work.
- [[semantic-caching]] - skip repeat calls for near-duplicate queries.
- See also (SE KB): <https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/caching-strategies.md> - caching repeated requests.

## References
- Gemini API pricing (free vs paid tier, data use): https://ai.google.dev/gemini-api/docs/pricing
- OpenRouter pricing and free-model limits: https://openrouter.ai/pricing
- Groq rate limits: https://console.groq.com/docs/rate-limits
- Hugging Face Inference Providers pricing and credits: https://huggingface.co/docs/inference-providers/pricing
- Ollama: https://github.com/ollama/ollama
- scikit-learn: https://scikit-learn.org/stable/
