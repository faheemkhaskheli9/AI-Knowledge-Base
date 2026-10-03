---
title: Finance and banking AI
category: scenarios
tags: [finance, fraud, credit-scoring, forecasting, compliance, kyc, personal-finance]
use_cases:
  - "detect fraudulent transactions"
  - "build a credit or risk model that regulators can accept"
  - "extract data from financial statements, invoices or bank SMS"
  - "categorise transactions and build a personal finance assistant"
  - "build an analyst copilot over filings and research"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
  - https://www.consumerfinance.gov/rules-policy/regulations/1002/
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://www.sec.gov/search-filings
---

# Finance and banking AI

## Summary
Finance mixes numeric prediction (fraud, credit, forecasting) where classic
ML wins, with document-heavy work (filings, statements, KYC) where LLMs help.
Accuracy needs are high, models must be explainable and auditable, and
regulators care about governance, bias and customer communication.

## Key concepts
- Most core problems are tabular: use [[gradient-boosting-tabular]] with
  proper time-based validation, not an LLM.
- Model risk management: in the US, supervisory guidance SR 11-7 expects
  validation, documentation and ongoing monitoring of models. Check the
  current guidance for your institution.
- Credit decisions: adverse-action notices need specific reasons (e.g. US
  ECOA / Regulation B); the EU AI Act lists creditworthiness assessment of
  natural persons as high-risk. Black-box models need explanation tooling
  (SHAP or inherently interpretable models). Check counsel.
- Fairness: proxy variables (postcode, device) can reproduce protected-class
  bias; test disparate impact.
- Never use an LLM to do arithmetic; have it call code or a database
  ([[tool-calling]], [[text-to-sql]]).
- Market-data leakage: random splits on time-ordered data inflate results.

## When to use / scenarios
1. **Card/transaction fraud detection.** Problem: rare, adversarial, delayed
   labels. Approach: gradient boosting + anomaly features + rules, graph
   features, human review queue. Read: [[anomaly-detection]],
   [[gradient-boosting-tabular]], [[mlops-lifecycle]].
2. **Credit scoring / underwriting.** Approach: interpretable or monotonic GBM,
   calibration, reason codes, fairness audit. Read:
   [[gradient-boosting-tabular]], [[classic-ml-scikit-learn]],
   [[experiment-tracking]].
3. **Transaction categorisation and merchant matching.** Approach: rules +
   embeddings + LLM fallback; multilingual merchant names. Read:
   [[embeddings]], [[nlp-classic-tasks]], [[structured-output]].
4. **Parsing bank SMS / statements / invoices.** Approach: regex first, LLM for
   the long tail, measured accuracy on labelled samples. Read:
   [[document-processing]], [[ocr]], [[llm-evaluation]].
5. **KYC / AML document and name screening.** Approach: OCR + extraction,
   fuzzy matching on watchlists, analyst review. Read: [[document-parsing]],
   [[embedding-models]].
6. **Demand/revenue/cash-flow forecasting.** Approach: time-series baselines
   first, GBM with lags, intervals. Read: [[time-series-forecasting]].
7. **Analyst copilot over filings and earnings calls.** Approach: RAG with
   citations to page/paragraph. Read: [[advanced-rag]], [[rag-basics]],
   [[long-context]].
8. **Personal finance / budgeting assistant.** Approach: local data, SQL
   tools for numbers, LLM for explanation. Read: [[personal-assistants]],
   [[text-to-sql]], [[agent-memory]].
9. **Customer support in banking.** Read: [[customer-support]],
   [[guardrails-and-safety]].
10. **Trading signals.** Be sceptical: low signal-to-noise, backtest overfit.
    Read: [[time-series-forecasting]]; treat claims as unproven until
    out-of-sample and cost-adjusted.
11. **Compliance monitoring (communications surveillance).** Approach:
    classifier + LLM triage, reviewer feedback. Read: [[nlp-classic-tasks]].

## Setup & code
Reference architecture (flagship: fraud scoring):

```
event stream -> feature store (velocity, device, merchant stats)
  -> GBM score + rules -> threshold bands: approve | step-up | review | decline
  -> analyst review -> labels back to training (watch label delay)
  -> monitoring: drift, precision@k, alert volume, subgroup rates
```

```python
import lightgbm as lgb
from sklearn.metrics import average_precision_score
# time-ordered split: train on past, validate on the following period
train, valid = df[df.ts < cut], df[df.ts >= cut]
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                       scale_pos_weight=50)  # tune for class imbalance
m.fit(train[feats], train.is_fraud)
print(average_precision_score(valid.is_fraud, m.predict_proba(valid[feats])[:, 1]))
```

## Choosing / trade-offs
- Explainability vs accuracy: a constrained GBM or scorecard is often
  accepted where a deep net is not.
- Precision vs recall: set by cost of false declines vs fraud loss; pick the
  threshold with business costs, not F1.
- Local vs hosted LLM: financial records are sensitive; check provider data
  terms ([[ai-security-privacy-compliance]]), consider [[ollama]].

## Gotchas
- Label delay and feedback loops: declined transactions never get labels.
- Concept drift as fraudsters adapt; monitor and retrain.
- LLM-made numeric errors in summaries; compute in code and cite the source.
- Forecasts presented as advice may fall under investment-advice regulation;
  check local rules before offering to retail users.
- Data protection: transaction data is personal data (GDPR where relevant);
  minimise and secure.
- Backtests that ignore fees, slippage and look-ahead look great and fail live.

## Related
- [[hallucination-and-grounding]] - numbers must be traceable.
- [[llm-observability]] - audit trail of model decisions.
- [[data-analytics]] - dashboards and ad hoc analysis.

## References
- Federal Reserve SR 11-7 model risk guidance: https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
- CFPB Regulation B (ECOA): https://www.consumerfinance.gov/rules-policy/regulations/1002/
- Regulation (EU) 2024/1689 (AI Act), Annex III: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
