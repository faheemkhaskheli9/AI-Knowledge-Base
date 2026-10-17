---
title: Insurance AI
category: scenarios
tags: [insurance, claims-triage, fraud, underwriting, explainability, pricing, naic, ai-act]
use_cases:
  - "triage incoming insurance claims and route them to the right handler"
  - "flag potentially fraudulent claims for investigation"
  - "extract data from underwriting submissions, medical reports or loss runs"
  - "build a pricing or risk model regulators and customers can understand"
  - "summarise a claim file for an adjuster"
status: draft
last_verified: 2026-10-15
sources:
  - https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://gdpr-info.eu/art-22-gdpr/
  - https://scikit-learn.org/stable/modules/ensemble.html#monotonic-constraints
---

# Insurance AI

## Summary
Insurance runs on two things AI can help with: risk prediction (pricing,
underwriting, fraud, reserving) where tabular ML dominates, and documents
(applications, medical reports, loss runs, claim files, adjuster notes) where
LLMs and OCR help. Decisions affect consumers' access to cover and payouts,
so regulators expect governance, fairness testing and explanations. The
safe pattern is AI that ranks and drafts, with people deciding denials.

## Key concepts
- Explainability is a requirement, not a nice-to-have: underwriting and
  claim decisions often need specific reasons. Prefer interpretable or
  constrained models (GLMs, GAMs, monotonic gradient boosting) and per-case
  reason codes ([[model-interpretability]], [[generalized-linear-models]]).
- US governance: the NAIC Model Bulletin on the Use of AI Systems by
  Insurers (adopted December 2023, since adopted by many states) expects a
  written AI systems program, governance and risk management, oversight of
  third-party models and data, and attention to unfair discrimination.
  Some states (e.g. Colorado) have stricter rules. Check your states.
- EU: the AI Act lists AI for risk assessment and pricing of natural
  persons in life and health insurance as high-risk (Annex III). Its
  application dates were amended in 2026; check the current consolidated
  text. GDPR Article 22 limits solely automated decisions with legal or
  similarly significant effects.
- Fairness and proxies: postcode, credit-based scores and external data
  can act as proxies for protected classes; test outcomes by group where
  law allows ([[ai-security-privacy-compliance]]).
- Fraud is rare and adversarial: precision at the top of the queue matters
  more than AUC; labels arrive late and only for investigated claims
  ([[imbalanced-data]]).
- Health and medical data in claims is special-category data; see
  [[healthcare]] for the privacy rules.

## When to use / scenarios
1. **Claims triage (first notice of loss).** Problem: simple claims wait
   behind complex ones. Approach: extract fields from FNOL text and photos,
   score complexity and severity, route to fast-track or specialist; no
   auto-denial. Read: [[structured-output]], [[gradient-boosting-tabular]],
   [[nlp-classic-tasks]].
2. **Fraud detection and SIU referral.** Approach: gradient boosting on
   claim features + rules + network features (shared phones, addresses,
   repair shops), ranked queue for investigators. Read:
   [[anomaly-detection]], [[graph-neural-networks]], [[imbalanced-data]],
   [[finance]] (similar transaction fraud patterns).
3. **Underwriting document intake.** Approach: parse submissions,
   schedules of values, loss runs and medical reports into a schema;
   validate totals; underwriter reviews. Read: [[document-processing]],
   [[document-parsing]], [[ocr]].
4. **Pricing and risk models.** Approach: GLM/GAM baseline (the actuarial
   standard), gradient boosting with monotonic constraints if it adds
   lift, calibration and reason codes. Read: [[generalized-linear-models]],
   [[generalized-additive-models-and-splines]], [[probability-calibration]].
5. **Claim file summarisation for adjusters.** Approach: long-context or
   RAG summary with citations to page and document. Read: [[long-context]],
   [[advanced-rag]], [[hallucination-and-grounding]].
6. **Damage assessment from photos (motor, property).** Approach: detection
   and segmentation of damage, estimate as a suggestion; adjuster confirms.
   Read: [[object-detection]], [[segmentation]],
   [[vision-language-models]].
7. **Reserving and claim duration.** Approach: time-to-close and severity
   models. Read: [[survival-analysis]], [[quantile-regression]].
8. **Policyholder service assistant.** Approach: RAG over policy wording
   with exact clause citations; coverage decisions handed to staff. Read:
   [[customer-support]], [[guardrails-and-safety]].
9. **Parametric and crop insurance.** Approach: trigger on an index
   (rainfall, satellite vegetation index) instead of loss adjustment. Read:
   [[agriculture]].
10. **Subrogation and recovery opportunity detection.** Read:
    [[nlp-classic-tasks]], [[gradient-boosting-tabular]].

## Setup & code
Reference architecture (flagship: claims triage with fraud screening):

```
FNOL (web/phone/email) + photos -> STT/OCR -> LLM extracts schema fields
  -> validation (policy active? dates in cover period?) via policy system
  -> severity/complexity model + fraud score (separate models)
  -> routing: fast-track | standard | specialist | SIU review (never auto-deny)
  -> reason codes + model version logged per decision
  -> monitoring: leakage, subgroup outcomes, override rates, drift
```

Runnable sketch: a transparent fraud-referral score with per-claim reason
codes (logistic regression contributions; synthetic data).

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(2)
names = ["days_since_policy_start", "claim_amount_k", "prior_claims",
         "late_report_days"]
n = 5000
X = np.column_stack([rng.exponential(300, n), rng.lognormal(1.5, 0.8, n),
                     rng.poisson(0.4, n), rng.exponential(5, n)])
logit = -4 - 0.004 * X[:, 0] + 0.15 * X[:, 1] + 0.6 * X[:, 2] + 0.08 * X[:, 3]
y = rng.random(n) < 1 / (1 + np.exp(-logit))

scaler = StandardScaler().fit(X)
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

def reason_codes(x, k=2):
    contrib = clf.coef_[0] * scaler.transform(x[None])[0]
    top = np.argsort(contrib)[::-1][:k]
    return [(names[i], round(float(contrib[i]), 2)) for i in top if contrib[i] > 0]

claim = np.array([12.0, 25.0, 2.0, 30.0])  # new policy, large, late report
score = clf.predict_proba(scaler.transform(claim[None]))[0, 1]
print(f"referral score {score:.2f}", reason_codes(claim))
```

## Choosing / trade-offs
- GLM/GAM vs gradient boosting: GLMs are filed, explained and reviewed
  easily; boosting can add lift but needs monotonic constraints, reason
  codes and more validation. Check whether the regulator will accept it.
- Fraud precision vs customer friction: every false referral delays a
  genuine claim; set thresholds from investigator capacity and the cost of
  delay, not from F1.
- Hosted vs in-house LLM for claim documents: claim files hold health and
  financial data; confirm data terms, retention and region, or run
  open-weights models in your own environment ([[inference-servers-vllm]]).
- External data: third-party scores and data can lift accuracy but bring
  proxy-discrimination and vendor-oversight duties.

## Gotchas
- Training fraud models on investigated claims only learns the old
  referral rules (selection bias); add random audits.
- An LLM summary that omits an exclusion or misstates a date can drive a
  wrong coverage decision; require citations and adjuster sign-off.
- Denials or price increases driven by a black-box score with no specific
  reason create regulatory and complaint risk.
- Photo damage models can be fooled by reused or edited images; check
  image metadata and near-duplicates.
- Drift after product, regulation or catastrophe events changes claim mix;
  monitor and retrain ([[mlops-lifecycle]]).
- Log model version, inputs and reasons per decision for audits and
  complaints ([[llm-observability]], [[experiment-tracking]]).

## Related
- [[finance]] - credit and transaction-fraud patterns that transfer.
- [[healthcare]] - medical data rules for health and life claims.
- [[legal]] - policy wording, disputes and litigation documents.
- [[agriculture]] - remote-sensing evidence for crop insurance.
- [[government-public-sector]] - public schemes and social insurance.
- [[graph-ml-for-fraud]] - fraud rings and shared-entity networks.
- [[ai-regulation-checklist]] - EU AI Act duties for high-risk pricing and underwriting.

## References
- NAIC Model Bulletin: Use of Artificial Intelligence Systems by Insurers (Dec 2023): https://content.naic.org/sites/default/files/inline-files/2023-12-4%20Model%20Bulletin_Adopted_0.pdf
- Regulation (EU) 2024/1689 (AI Act), Annex III point 5: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- GDPR Article 22 (automated individual decision-making): https://gdpr-info.eu/art-22-gdpr/
- scikit-learn monotonic constraints: https://scikit-learn.org/stable/modules/ensemble.html#monotonic-constraints
