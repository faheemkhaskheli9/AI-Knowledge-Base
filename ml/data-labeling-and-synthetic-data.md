---
title: Data labeling and synthetic data
category: ml
tags: [labeling, annotation, label-studio, synthetic-data, llm-labeling, active-learning, weak-supervision]
use_cases:
  - "build a labelled dataset for a custom classifier or detector"
  - "use an LLM to pre-label data and have humans verify it"
  - "generate synthetic training or test data when real data is scarce or private"
  - "measure annotation quality and inter-annotator agreement"
  - "create an evaluation set for an LLM application"
status: draft
last_verified: 2026-10-03
sources:
  - https://labelstud.io/guide/
  - https://cvat.ai
  - https://docs.cleanlab.ai/
  - https://sdv.dev/
---

# Data labeling and synthetic data

## Summary
Model quality is capped by label quality. Practical labelling now combines model pre-labelling (LLMs, foundation vision models), human review of the uncertain cases, and clear guidelines. Synthetic data (LLM-generated text, simulated or generated images, tabular generators) fills gaps and protects privacy, but must be validated against real data.

## Key concepts
- Annotation guidelines: written definitions with examples and edge cases; most label disagreement is guideline ambiguity.
- Inter-annotator agreement: Cohen's kappa / Krippendorff's alpha; low agreement means the task, not the annotators, is the problem.
- Gold set: a small, expert-labelled, held-out set used for evaluation and annotator QA; never train on it.
- Pre-labelling + review: a model proposes, humans correct; much faster than labelling from scratch (SAM-assisted masks, LLM-suggested classes).
- Active learning: label the examples the model is least sure about; weak supervision: combine noisy heuristics/labelling functions.
- Label-noise detection: find likely mislabelled items with confident-learning tools (cleanlab).
- Synthetic data: LLM-generated text and paraphrases, rendered/diffusion images, simulation, tabular synthesisers (SDV); rarely a full replacement for real data.

## When to use / scenarios
- Computer vision: bounding boxes, masks, keypoints in CVAT or Label Studio (see [[object-detection]], [[segmentation]]).
- NLP: intents, entities, sentiment ([[nlp-classic-tasks]]); LLM pre-labelling then human audit of a sample.
- LLM apps: build a golden question/answer set for [[llm-evaluation]]; synthetic questions generated from documents for RAG tests.
- Healthcare/finance: synthetic tabular data to share a dataset without exposing individuals (verify privacy, it is not automatically anonymous).
- Rare events (defects, fraud): synthesise or augment minority cases, but test on real ones.

## Setup & code
```bash
pip install label-studio     # then: label-studio start   (UI on port 8080 by default; check it is free)
```
LLM pre-labelling with a schema (sketch using the Anthropic SDK; see [[structured-output]]):
```python
import anthropic, json

client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
LABELS = ["billing", "bug", "feature_request", "other"]

def prelabel(text: str) -> str:
    msg = client.messages.create(
        model="MODEL_ID",               # replace with a current model id from the Anthropic docs
        max_tokens=50,
        messages=[{"role": "user",
                   "content": f"Classify into one of {LABELS}. Reply with the label only.\n\n{text}"}],
    )
    out = msg.content[0].text.strip()
    return out if out in LABELS else "other"

print(prelabel("I was charged twice this month"))
```
Treat output as a proposal: sample, review, measure agreement with humans, then train or evaluate.

## Choosing / trade-offs
- Human-only: highest quality for subtle tasks; slowest and costliest.
- LLM-only labels: cheap and consistent but encode the LLM's biases; validate on a human gold set before trusting.
- Hybrid (LLM proposes, humans resolve disagreements/low confidence): best cost/quality for most teams.
- Tools: Label Studio (general, open source), CVAT (images/video), Argilla (NLP/LLM feedback), Prodigy (paid, fast active learning); managed vendors for large crews.
- Synthetic vs real: synthetic for coverage, privacy and bootstrapping; real for final validation.

## Gotchas
- Training a model on LLM-generated data and evaluating with the same LLM inflates scores (circularity).
- Synthetic text is more uniform than real text; models trained on it can fail on messy inputs. Mix real data in.
- Model collapse risk if models are repeatedly trained on their own outputs.
- Test-set leakage: duplicates between train and test, or labellers seeing model predictions (anchoring bias).
- Annotator PII exposure: redact before sending data to crowd workers or third-party APIs.
- Tabular synthetic data can leak records (memorisation); run a privacy evaluation, do not assume anonymity.
- Class definitions drift mid-project; version the guidelines and re-label affected data.

## Related
- [[nlp-classic-tasks]] - the models you train on these labels.
- [[object-detection]] - box labelling and formats.
- [[segmentation]] - SAM-assisted mask labelling.
- [[llm-evaluation]] - golden sets for LLM apps.
- [[anomaly-detection]] - validation sets for rare events.

## References
- https://labelstud.io/guide/
- https://docs.cleanlab.ai/
- https://sdv.dev/
