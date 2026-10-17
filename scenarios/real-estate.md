---
title: Real estate and property AI
category: scenarios
tags: [real-estate, proptech, avm, property-valuation, listing-generation, lease-extraction, fair-housing, geospatial, mortgage]
use_cases:
  - "build an automated valuation model (AVM) for homes or rents"
  - "generate property listing descriptions from photos and attributes"
  - "extract terms from leases, appraisals and title documents"
  - "add property search and recommendations to a listings site"
  - "forecast rents, occupancy or maintenance needs for a portfolio"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.fhfa.gov/regulation/federal-register/final-rule/quality-control-standards-for-automated-valuation-models
  - https://www.justice.gov/crt/fair-housing-act-1
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
---

# Real estate and property AI

## Summary
Real estate combines a classic tabular-plus-location prediction problem
(valuation, rent, time-to-sell) with document-heavy work (leases, appraisals,
title, inspection reports) and content generation (listings, photos,
search). Valuation is gradient boosting on structure, location and market
features with spatial and time-aware validation; LLMs help with extraction,
listings and search. Fair housing and lending rules apply to valuations,
ads and tenant screening, so bias checks and wording guardrails are part of
the design.

## Key concepts
- **AVM (automated valuation model).** Predicts price or rent from property
  attributes (size, rooms, age, condition), location
  ([[geospatial-ml]]), market timing and comparable sales. Approaches:
  hedonic regression, GBMs, repeat-sales indices for market trends, and
  comparable-sales (k-nearest) models that appraisers can follow.
- **Accuracy metrics used in the industry:** median absolute percentage
  error and the share of predictions within ±5/10/20% of the sale price;
  report them by price band and region, not only overall.
- **Uncertainty matters.** Lenders and buyers need a range or confidence
  score, not one number: quantile models or conformal intervals
  ([[quantile-regression]], [[conformal-prediction-and-uncertainty]]).
- **US AVM quality-control rule.** A 2024 interagency final rule (OCC, Fed,
  FDIC, NCUA, CFPB, FHFA), effective 1 October 2025, requires mortgage
  originators and secondary-market issuers using AVMs in covered credit
  decisions to have quality controls for confidence in estimates, protection
  against data manipulation, conflicts of interest, random sample testing
  and reviews, and compliance with nondiscrimination laws.
- **Fair housing.** The US Fair Housing Act bars discrimination in sale,
  rental, financing and advertising on protected characteristics (race,
  colour, religion, sex, disability, familial status, national origin), and
  ad wording can itself violate it. Location features can act as proxies
  for protected classes. Other countries have equivalent laws.
- **EU AI Act:** creditworthiness assessment of natural persons is
  high-risk (Annex III), which covers mortgage scoring; valuation alone is
  not listed, but check how it feeds credit decisions.
- **Market regime change.** Prices move with rates and local cycles; models
  trained in a rising market overestimate in a falling one.

## When to use / scenarios
1. **Home value / rent AVM.** Approach: GBM on structure + spatial features
   + time, quantile outputs, spatial and time-based validation, monitoring
   against new sales. Read: [[gradient-boosting-tabular]],
   [[geospatial-ml]], [[quantile-regression]],
   [[data-leakage-and-validation-splits]].
2. **Comparable selection ("comps").** Approach: nearest neighbours in a
   learned or weighted feature space, filtered by distance and recency, shown
   with adjustments. Read: [[distance-metrics-and-similarity]],
   [[metric-learning-and-few-shot]].
3. **Listing description generation.** Approach: LLM writes from structured
   attributes and photo captions only; fair-housing wording filter; agent
   approves. Read: [[prompt-engineering]], [[structured-output]],
   [[guardrails-and-safety]], [[marketing-content]].
4. **Photo understanding.** Room-type classification, feature tagging
   (pool, renovated kitchen), quality/condition scores, floor-plan reading.
   Read: [[image-classification]], [[vision-language-models]].
5. **Virtual staging and image edits.** Generative fill of furniture;
   disclose edited images and never alter structural features. Read:
   [[image-generation-models]].
6. **Lease, appraisal, title and inspection extraction.** Approach: parser +
   LLM extraction to a schema (rent, term, break clauses, escalations,
   defects), page citations, human check for exceptions. Read:
   [[document-processing]], [[document-parsing]], [[ocr]], [[legal]].
7. **Property search and recommendations.** Natural-language search
   ("3-bed near a good school under X") mapped to filters plus semantic
   ranking; recommender on viewing behaviour. Read: [[text-to-sql]],
   [[recommender-systems]], [[learning-to-rank]].
8. **Lead scoring and agent copilots.** Approach: GBM on engagement for
   leads; RAG assistant over listings and policies. Read:
   [[rag-basics]], [[customer-support]].
9. **Portfolio forecasting and maintenance.** Rent and occupancy
   forecasts; predictive maintenance from work orders and sensors. Read:
   [[time-series-forecasting]], [[manufacturing-iot]].
10. **Tenant screening.** High legal risk (fair housing, credit reporting
    rules); prefer transparent criteria and human review. Read:
    [[finance]], [[model-interpretability]].

## Setup & code
Reference architecture (flagship: AVM):

```
sales + listings + property records + geo layers (POIs, schools, flood)
  -> entity resolution (address -> parcel) -> feature store
  -> GBM point + quantile models -> calibrate intervals on recent sales
  -> output: value, range, confidence, top comps, reason codes
  -> monitoring: error vs new sales by region/price band, drift, fairness by area
```

Quantile AVM with industry-style metrics on synthetic data (time split:
train on older sales, test on the latest):
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(0)
n = 4000
size = rng.uniform(40, 250, n); rooms = np.round(size / 30 + rng.normal(0, 0.7, n))
age = rng.uniform(0, 100, n); dist_km = rng.uniform(0, 25, n)
month = np.sort(rng.integers(0, 36, n))                      # sale month index
market = 1 + 0.004 * month                                    # rising market
price = (3000 * size * np.exp(-0.03 * dist_km) * (1 - 0.002 * age) * market
         * rng.lognormal(0, 0.12, n))
X = np.column_stack([size, rooms, age, dist_km, month])
tr, te = month < 30, month >= 30                              # time-based split
y = np.log(price)                                             # log price: errors become relative

models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                           random_state=0).fit(X[tr], y[tr])
          for q in (0.1, 0.5, 0.9)}
pred = {q: np.exp(m.predict(X[te])) for q, m in models.items()}
ape = np.abs(pred[0.5] - price[te]) / price[te]
cover = np.mean((price[te] >= pred[0.1]) & (price[te] <= pred[0.9]))
print(f"median APE {np.median(ape):.3f}, within 10%: {np.mean(ape <= 0.10):.2f}, "
      f"80% interval coverage: {cover:.2f}")
```
Coverage below 0.80 on the newest months is typical when the market moves
beyond the training range (trees cannot extrapolate the time trend); fix with
a market index feature, recalibration on recent sales, or
[[conformal-prediction-and-uncertainty]].

Listing generation with a wording guardrail (model ID from
[[anthropic-claude]]):
<!-- skip-check: needs an API key and network -->
```python
import anthropic, re

facts = {"type": "flat", "beds": 2, "baths": 1, "size_m2": 68, "features": ["balcony", "lift", "new kitchen"]}
client = anthropic.Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=400,
    system=("Write a property listing using ONLY the given facts. Describe the property, "
            "not the people who should live there. No claims about neighbours, schools' "
            "demographics, religion, nationality, family status, age or disability."),
    messages=[{"role": "user", "content": str(facts)}],
)
text = msg.content[0].text
risky = re.findall(r"\b(perfect for (couples|families|singles)|no kids|adults only|"
                   r"exclusive neighbou?rhood|walking distance to (church|mosque|synagogue))\b", text, re.I)
print(text if not risky else f"needs human review: {risky}")
```
A keyword list is a backstop, not compliance; have the phrase list reviewed
by counsel and keep an agent approval step.

## Choosing / trade-offs
- **GBM vs comps vs hedonic regression:** GBMs are usually most accurate;
  comps and linear hedonic models are easier to explain to appraisers,
  lenders and regulators. Many products show a GBM value with comps as
  evidence.
- **Coverage vs accuracy:** a model can refuse (low confidence) in sparse or
  unusual areas; publishing values everywhere spreads large errors.
- **Hosted vs local LLM:** leases and tenant data are confidential; check
  data terms ([[ai-security-privacy-compliance]]).
- **Generated vs edited images:** staging improves engagement but misleading
  images create legal and trust risk; label and keep originals.

## Gotchas
- Leakage: listing price, days on market or post-sale features in a price
  model; repeat sales of the same property across train and test; random
  splits instead of time and spatial splits.
- Using a valuation model to buy homes at scale is a different risk: errors
  are adversely selected (sellers accept when the model overpays) and
  market turns hit inventory; a widely reported iBuying programme was wound
  down in 2021 after forecasting problems.
- Data quality: wrong sizes, duplicate listings, non-arm's-length sales
  (family transfers, foreclosures) distort training; filter them.
- Feedback loops: published estimates influence asking prices and then
  training data.
- Location features (postcode, distance to amenities) can reproduce
  historical segregation in values; test error and bias by neighbourhood
  demographics.
- LLM listings invent features ("sea view", "recently renovated") not in
  the facts; constrain to inputs and diff against them.
- Extraction errors on rent escalations and dates are costly; require page
  citations and validate arithmetic in code ([[hallucination-and-grounding]]).

## Related
- [[geospatial-ml]] - location features and spatial validation for AVMs.
- [[finance]] - mortgage credit, fraud and model risk overlap.
- [[document-processing]] - lease and appraisal extraction pipelines.
- [[ecommerce-retail]] - listings search and recommendation patterns.
- [[legal]] - contract review overlap.

## References
- FHFA, Quality Control Standards for Automated Valuation Models (final rule, effective 2025-10-01): https://www.fhfa.gov/regulation/federal-register/final-rule/quality-control-standards-for-automated-valuation-models
- US Department of Justice, The Fair Housing Act: https://www.justice.gov/crt/fair-housing-act-1
- Regulation (EU) 2024/1689 (AI Act), Annex III: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- scikit-learn, prediction intervals for gradient boosting regression: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
