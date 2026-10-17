---
title: Logistics and supply chain AI
category: scenarios
tags: [logistics, supply-chain, demand-forecasting, vehicle-routing, eta, inventory, document-automation]
use_cases:
  - "forecast demand per SKU and location to set inventory levels"
  - "plan delivery routes for a fleet with time windows and capacities"
  - "extract data from bills of lading, invoices and customs documents"
  - "predict shipment ETAs and alert on delays or anomalies"
  - "build an assistant that answers where-is-my-shipment questions"
status: draft
last_verified: 2026-10-15
sources:
  - https://developers.google.com/optimization/routing
  - https://developers.google.com/optimization/routing/vrptw
  - https://otexts.com/fpp3/
  - https://www.statsmodels.org/stable/tsa.html
---

# Logistics and supply chain AI

## Summary
Supply chain work combines forecasting (how much, where, when), combinatorial
optimisation (routing, loading, scheduling) and a lot of paperwork (bills of
lading, invoices, customs declarations). Classic forecasting and operations
research solvers do the core decisions; ML improves inputs such as demand and
travel times; LLMs handle documents, exceptions and natural-language queries
over shipment data.

## Key concepts
- Predict, then optimise: a forecast or ETA is only an input. The decision
  (order quantity, route, slot) comes from a solver or policy that uses its
  uncertainty, e.g. a quantile forecast for safety stock.
- Forecast at the level the decision is made (SKU x location x week) and
  reconcile across levels (hierarchical forecasting). Intermittent demand
  (many zeros) needs special methods or aggregation.
- Baselines first: seasonal naive and exponential smoothing are strong and
  cheap; ML must beat them on time-based backtests
  ([[time-series-forecasting]], [[data-leakage-and-validation-splits]]).
- Vehicle routing (VRP) is NP-hard; use a solver such as Google OR-Tools
  (capacities, time windows, pickups and deliveries) instead of asking an
  LLM to plan routes.
- Censored demand: sales are not demand when shelves are empty; stock-outs
  bias forecasts low.
- Documents are semi-structured and multilingual; extraction needs
  validation against master data (ports, HS codes, consignees).

## When to use / scenarios
1. **Demand forecasting and inventory planning.** Problem: stock-outs and
   overstock. Approach: baselines + gradient boosting with lags, promotions
   and holidays, quantile outputs feeding reorder points. Read:
   [[time-series-forecasting]], [[gradient-boosting-tabular]],
   [[quantile-regression]], [[ecommerce-retail]].
2. **Route and fleet planning.** Approach: OR solver with time windows and
   capacities; ML-predicted travel and service times as inputs. Read:
   [[black-box-and-evolutionary-optimization]],
   [[reinforcement-learning]] (research for dynamic dispatch, rarely
   needed in production).
3. **Document automation (bills of lading, invoices, packing lists,
   customs forms).** Approach: OCR/parse -> schema extraction -> validate
   against master data -> human review of low-confidence fields. Read:
   [[document-processing]], [[document-parsing]], [[ocr]],
   [[structured-output]].
4. **ETA prediction and delay alerts.** Approach: regression on route,
   carrier, weather and history; alert when predicted lateness crosses a
   threshold; calibrated intervals. Read: [[gradient-boosting-tabular]],
   [[conformal-prediction-and-uncertainty]], [[survival-analysis]].
5. **Anomaly alerts (temperature excursions, unusual dwell, invoice
   mismatches).** Approach: rules plus anomaly detectors on telemetry and
   billing; rank alerts by cost. Read: [[anomaly-detection]],
   [[online-learning-and-concept-drift]].
6. **Customs classification assistance (HS codes).** Approach: retrieval
   over the tariff nomenclature and past rulings, LLM suggests code with
   cited heading, licensed broker confirms. Read: [[rag-basics]],
   [[embedding-models]], [[nlp-classic-tasks]].
7. **Where-is-my-shipment and ops copilot.** Approach: tool calls into the
   TMS/WMS for facts, LLM for wording; never let the model invent status.
   Read: [[tool-calling]], [[text-to-sql]], [[customer-support]].
8. **Warehouse vision (damage, pallet counting, label reading).** Read:
   [[object-detection]], [[ocr]], [[video-analytics]].
9. **Supplier risk monitoring.** Approach: news and filings watch,
   entity matching, summarised alerts with sources. Read:
   [[web-search-and-grounding]], [[nlp-classic-tasks]].
10. **Exception handling agents (rebooking, claims).** Approach: agent
    with narrow tools and approval steps for money-moving actions. Read:
    [[agents]], [[guardrails-and-safety]].

## Setup & code
Reference architecture (flagship: forecast-driven replenishment):

```
orders, stock, promotions, calendar -> feature table (SKU x location x week)
  -> baseline + ML forecast (median and upper quantile), backtested by time
  -> reorder policy: order-up-to = upper-quantile demand over lead time - stock
  -> planner review of large changes -> purchase orders
  -> monitoring: forecast error by segment, stock-outs, overstock days
```

Runnable sketch: seasonal-naive baseline vs a lag-feature model, then a
quantile-based reorder point (synthetic weekly demand).

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(1)
weeks = np.arange(156)
demand = 100 + 20 * np.sin(2 * np.pi * weeks / 52) + rng.poisson(10, weeks.size)

lags = [1, 2, 52]
X = np.column_stack([demand[52 - l:-l] for l in lags])
y = demand[52:]
cut = len(y) - 26  # last 26 weeks are the backtest period

naive = demand[52 - 52:-52][cut:]  # same week last year
mae_naive = np.mean(np.abs(y[cut:] - naive))
model = GradientBoostingRegressor(random_state=0).fit(X[:cut], y[:cut])
mae_ml = np.mean(np.abs(y[cut:] - model.predict(X[cut:])))
print(f"MAE seasonal-naive {mae_naive:.1f} vs ML {mae_ml:.1f}")

# 90th-percentile forecast for safety stock
q90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=0)
q90.fit(X[:cut], y[:cut])
lead_time_weeks, on_hand = 2, 150
reorder_point = lead_time_weeks * q90.predict(X[-1:])[0]
print("order now" if on_hand < reorder_point else "no order",
      round(reorder_point))
```

For routing, use the OR-Tools VRP guides (see References) rather than
hand-written heuristics once there are real constraints.

## Choosing / trade-offs
- Statistical vs ML vs foundation forecasting models: statistical methods
  are robust with short histories; gradient boosting wins with many related
  series and rich covariates; pretrained time-series models are worth a
  backtest when histories are short or new SKUs are frequent.
- Exact vs heuristic routing: exact methods only scale to small instances;
  metaheuristics in OR-Tools give good routes fast with tunable time limits.
- Template extraction vs LLM extraction: templates and regex are cheap and
  exact for a few high-volume layouts; LLM extraction covers the long tail
  but needs field-level accuracy measurement ([[llm-evaluation]]).
- Central cloud vs edge: warehouse vision and scanners often need local
  inference for latency ([[edge-on-device]]).

## Gotchas
- Backtesting with random splits or future covariates (actual promotions
  known only later) overstates accuracy.
- Optimising average error hides the SKUs that matter; weight error by
  value or service-level impact.
- A forecast that changes every run causes "nervous" plans; smooth or
  freeze near-term horizons.
- LLMs miscount and misread numbers in documents; validate totals, weights
  and quantities with code and cross-document checks.
- Driver telematics and location data are personal data in many
  jurisdictions; minimise and set retention ([[ai-security-privacy-compliance]]).
- Customs and trade-compliance decisions carry legal liability; keep a
  qualified human approver and an audit trail ([[llm-observability]]).

## Related
- [[ecommerce-retail]] - retail demand, pricing and recommendations.
- [[manufacturing-iot]] - production planning and sensor anomalies.
- [[agriculture]] - harvest and commodity forecasting.
- [[document-processing]] - document pipelines in depth.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/case-studies/design-ride-sharing.md

## References
- Google OR-Tools vehicle routing: https://developers.google.com/optimization/routing
- OR-Tools VRP with time windows: https://developers.google.com/optimization/routing/vrptw
- Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed.): https://otexts.com/fpp3/
- statsmodels time-series analysis: https://www.statsmodels.org/stable/tsa.html
