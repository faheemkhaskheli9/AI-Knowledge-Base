---
title: Telecom AI
category: scenarios
tags: [telecom, network-anomaly-detection, churn, kpi, nwdaf, o-ran, customer-support, fraud, capacity-planning]
use_cases:
  - "detect anomalies in cell or network KPIs without drowning the NOC in alerts"
  - "predict which subscribers will churn and target retention offers"
  - "automate first-line customer support and bill explanations for a mobile operator"
  - "forecast traffic per cell for capacity planning"
  - "build an assistant for NOC engineers over runbooks, tickets and alarms"
status: draft
last_verified: 2026-10-15
sources:
  - https://www.3gpp.org/dynareport/23288.htm
  - https://www.o-ran.org/specifications
  - https://eur-lex.europa.eu/eli/dir/2002/58/oj
  - https://www.fcc.gov/consumers/guides/protecting-your-privacy
  - https://scikit-learn.org/stable/modules/outlier_detection.html
---

# Telecom AI

## Summary
Telecom operators sit on huge volumes of time-series (network KPIs,
alarms), tabular customer data (usage, billing, plan) and support text.
The core wins are classic ML: anomaly detection and forecasting on network
KPIs, and churn and fraud models on subscriber data. LLMs help with
customer-care automation, bill and plan explanations, and NOC assistants
over runbooks and tickets. Standards bodies now define where analytics run
in the network (3GPP NWDAF in the 5G core, O-RAN RAN Intelligent
Controllers), so placement matters as much as the model.

## Key concepts
- **KPIs are seasonal and hierarchical**: traffic per cell follows
  daily and weekly cycles and rolls up cell -> site -> region. Compare each
  point with its own hour-of-week baseline, not a global mean
  ([[time-series-forecasting]], [[anomaly-detection]]).
- **Alarm storms**: one fibre cut raises thousands of alarms downstream.
  Correlate by topology and time before alerting; root-cause the parent,
  suppress the children.
- **Where analytics run**: 3GPP TS 23.288 defines the Network Data
  Analytics Function (NWDAF) that collects data and serves analytics to 5G
  core functions; O-RAN defines near-real-time and non-real-time RAN
  Intelligent Controllers that host xApps/rApps. Latency budget decides
  placement: per-second RAN control needs the edge, churn runs nightly in
  the data warehouse ([[edge-on-device]]).
- **Churn is a ranking and causal problem**: a churn score ranks who is at
  risk; whether an offer changes their decision is uplift
  ([[causal-inference-and-uplift]], [[survival-analysis]]).
- **Regulated data**: call records, location and traffic data are
  specially protected (EU ePrivacy Directive; US CPNI rules). Purpose
  limitation and consent constrain what can feed a model or an LLM prompt
  ([[ai-security-privacy-compliance]]).

## When to use / scenarios
1. **Network KPI anomaly detection (NOC).** Problem: thousands of cells,
   each with dozens of KPIs, static thresholds either miss or spam.
   Approach: per-cell seasonal baseline + robust z-score or forecast
   residuals, then group anomalies by topology; isolation forest on
   multivariate KPIs. Read: [[anomaly-detection]],
   [[time-series-forecasting]], [[cusum-change-point-detection-from-scratch]].
2. **Alarm correlation and root cause.** Approach: topology graph + time
   windows; graph models where you have labelled incidents. Read:
   [[graph-neural-networks]], [[clustering]].
3. **Churn prediction and retention.** Approach: gradient boosting on usage,
   billing, complaints and network-experience features with a time-based
   split; uplift model for who to target. Read:
   [[gradient-boosting-tabular]], [[causal-inference-and-uplift]],
   [[survival-analysis]], [[imbalanced-data]].
4. **Customer-care automation.** Approach: RAG over plans, tariffs and
   troubleshooting guides; tools for account lookup, bill breakdown and
   outage status; hand off to an agent. Read: [[customer-support]],
   [[rag-basics]], [[tool-calling]], [[voice-agents]].
5. **Bill shock and plan explanation.** Approach: compute charges in code
   from the billing system, let the LLM explain them. Read:
   [[text-to-sql]], [[hallucination-and-grounding]].
6. **Capacity planning.** Approach: per-cell traffic forecasts with
   intervals, aggregated to sites. Read: [[time-series-forecasting]],
   [[quantile-regression]].
7. **Fraud (SIM-swap, subscription, international revenue share).**
   Approach: rules + anomaly features + GBM, analyst queue. Read:
   [[anomaly-detection]], [[finance]].
8. **NOC engineer assistant.** Approach: RAG over runbooks, vendor manuals
   and past tickets; summarise an active incident from alarms. Read:
   [[advanced-rag]], [[long-context]], [[cybersecurity-soc]] (same triage
   pattern).
9. **Field technician support.** Approach: VLM over photos of equipment,
   offline-capable app. Read: [[vision-language-models]], [[edge-on-device]].
- **When NOT to use an LLM**: KPI anomaly detection, churn scoring,
  forecasting and RAN optimisation are numeric problems; an LLM adds cost
  and error. Use it to explain and route, not to compute.

## Setup & code
Reference architecture (flagship: KPI anomaly detection):

```
PM counters / KPIs per cell (15-min) -> per-cell hour-of-week baseline
  -> robust score (median / MAD) -> anomaly flags
  -> group by topology (site, transport link) + time -> one incident per parent
  -> NOC ticket with top KPIs and neighbours; LLM drafts the summary
  -> engineer feedback -> threshold tuning and labelled incidents
```

Runnable numpy sketch: a robust z-score against each hour's own baseline
(weekday and weekend kept apart) catches a busy-hour outage that a single
global threshold misses, while the global threshold fires on quiet nights.

```python
import numpy as np

rng = np.random.default_rng(0)
hours = np.arange(24 * 7 * 6)                               # six weeks, hourly
hod, weekend = hours % 24, (hours // 24) % 7 >= 5
bucket = hod + 24 * weekend                                 # 48 buckets: hour x weekday/weekend
level = 10 + 90 * np.sin(np.pi * (hod - 6) / 16).clip(0)    # quiet night, busy day
level = level * np.where(weekend, 0.7, 1.0)
traffic = level * (1 + rng.normal(0, 0.05, hours.size))    # 5% noise
outage = hours.size - 24 * 3 + 14                           # 14:00, three days before the end
traffic[outage] *= 0.45                                     # busy-hour drop, still above night levels

train = hours < hours.size - 24 * 7                         # baseline from the first five weeks
med = np.array([np.median(traffic[train & (bucket == b)]) for b in range(48)])
mad = np.array([np.median(np.abs(traffic[train & (bucket == b)] - med[b])) for b in range(48)])
z = (traffic - med[bucket]) / (1.4826 * mad[bucket] + 1e-6)

test = ~train
seasonal_flags = np.where(test & (np.abs(z) > 6))[0]
global_flags = np.where(test & (traffic < np.percentile(traffic[train], 1)))[0]
print("seasonal:", seasonal_flags, "global:", global_flags)  # global: misses the outage, flags quiet nights
assert list(seasonal_flags) == [outage] and outage not in global_flags
```

## Choosing / trade-offs
- **Statistical baseline vs deep model**: per-cell seasonal baselines are
  cheap, explainable and scale to millions of series; multivariate or deep
  models catch subtler degradations but need labelled incidents and more
  ops. Start with the baseline and measure ([[model-evaluation-and-metrics]]).
- **Central vs edge**: RAN optimisation loops need near-real-time
  inference at the edge (O-RAN near-RT RIC); churn, fraud and planning run
  centrally in batch.
- **Vendor analytics vs in-house**: network equipment vendors ship
  analytics tied to their gear; in-house models work across vendors but need
  normalised counters.
- **Hosted vs self-hosted LLM for care**: customer and traffic data
  residency rules often push operators to models in their own cloud tenancy
  ([[managed-cloud-ai-platforms]]) or self-hosted ([[inference-servers-vllm]]).

## Gotchas
- Counter definitions change with vendor software upgrades; a "drop" can be
  a firmware change. Track configuration changes alongside KPIs.
- Planned maintenance and holidays look like anomalies; feed the change
  calendar into suppression.
- Churn label leakage: features recorded after the customer gave notice
  (port-out request, retention call) inflate offline metrics
  ([[data-leakage-and-validation-splits]]).
- Retention offers to everyone with a high churn score waste budget on
  customers who would have stayed; measure uplift with a holdout.
- Care bots quoting wrong prices or plan terms create liability; ground
  every price in the billing system, not the model's memory.
- Location and call-detail data in prompts or training sets needs a lawful
  basis; anonymisation of location traces is hard.

## Related
- [[customer-support]] - care automation patterns in depth.
- [[manufacturing-iot]] - the same sensor-anomaly and edge patterns.
- [[online-learning-and-concept-drift]] - traffic patterns shift with new devices and plans.
- [[mlops-lifecycle]] - retraining and monitoring many per-cell models.
- [[cybersecurity-soc]] - alert triage pattern reused for NOC tickets.

## References
- 3GPP TS 23.288, architecture enhancements for 5G System to support network data analytics services (NWDAF): https://www.3gpp.org/dynareport/23288.htm
- O-RAN Alliance specifications (RAN Intelligent Controller): https://www.o-ran.org/specifications
- Directive 2002/58/EC (ePrivacy): https://eur-lex.europa.eu/eli/dir/2002/58/oj
- FCC, Protecting your privacy: phone and cable records (CPNI): https://www.fcc.gov/consumers/guides/protecting-your-privacy
- scikit-learn outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html
