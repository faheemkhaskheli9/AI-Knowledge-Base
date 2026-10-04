---
title: Kaplan-Meier survival curve and log-rank test from scratch (censoring, Greenwood CI)
category: ml
tags: [survival-analysis, kaplan-meier, censoring, log-rank-test, greenwood, time-to-event, churn, numpy, from-scratch, ml-basics]
use_cases:
  - "estimate a survival / retention curve from right-censored data in NumPy"
  - "see why dropping censored rows or treating them as events biases survival estimates"
  - "compare two groups' time-to-event curves with a log-rank test"
  - "explain censoring and the Kaplan-Meier estimator in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1080/01621459.1958.10501452
  - https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
  - https://scikit-survival.readthedocs.io/
---

# Kaplan-Meier survival curve and log-rank test from scratch (censoring, Greenwood CI)

## Summary
Time-to-event data (time to churn, machine failure, death, loan default) is usually right-censored. For some subjects you only know that the event had not happened yet when observation stopped. The Kaplan-Meier (KM) estimator handles this. At each event time it multiplies the running survival by the fraction of at-risk subjects who did not have the event: `S(t) = ∏_{t_i ≤ t} (1 − d_i / n_i)`. A censored subject counts as at risk until it leaves, and is never counted as an event. Below, 400 Weibull lifetimes with 39% random censoring are estimated three ways. KM tracks the true curve, and its 95% Greenwood intervals cover the truth 93-95% of the time over 500 repeats. Dropping censored rows or counting them as events both underestimate survival badly: 0.25 instead of 0.37 at t = 12, and a median of 7.9 instead of 9.4. A from-scratch log-rank test detects a 30% longer lifetime in a treatment arm (p = 0.0002) and finds nothing between two arms from the same distribution (p = 0.62).

## Key concepts
- **Censoring.** For a right-censored subject, all you know is `T > c`. Dropping such rows removes the long-lived subjects preferentially. Treating censoring as the event shortens lifetimes. Both bias the curve downward.
- **Risk set.** `n_i` counts subjects with observed time `≥ t_i`, so it includes those censored at or after `t_i`. Censored subjects contribute to the denominator until they leave.
- **Product-limit.** `S(t)` is a step function that drops only at event times. Censoring moves no probability itself but shrinks later risk sets. This assumes **non-informative censoring**: subjects who leave are no more or less likely to have the event than those who stay.
- **Greenwood variance.** `Var[S(t)] ≈ S(t)² Σ d_i / (n_i (n_i − d_i))`. A pointwise 95% CI is `S ± 1.96·SE`. The CI widens in the tail as risk sets shrink. A log-log transformed CI stays inside [0, 1] and behaves better there.
- **Log-rank test.** At each event time, compare the events observed in group 1 with those expected if both groups shared one hazard (`d · n1/n`). Sum `O − E` and its hypergeometric variance; `(O − E)² / V` is χ² with 1 degree of freedom under the null. It has the most power when hazards are proportional.

## When to use / scenarios
- Learning: the core non-parametric survival tool, and a clear example of how ignoring censoring biases an estimate.
- Interviews: "what is censoring", "why not just average the observed durations", "KM vs Cox", "how do you compare two retention curves".
- Practice: SaaS customer retention and churn cohorts, time to first purchase, equipment reliability and time between failures, clinical trial arms, time to loan default, time to hire. Plot KM curves per cohort or arm, then test them with log-rank ([[survival-analysis]]).
- Not for: adjusting for many covariates at once (use Cox proportional hazards or survival forests); individual risk prediction (use Cox, accelerated failure time models or gradient-boosted survival); or competing risks, such as churn vs downgrade (use cumulative incidence / Aalen-Johansen, because 1 − KM overstates incidence when there are competing events).

## Setup & code
NumPy plus `math` from the stdlib. Runs in about ten seconds, most of it the 500-repeat coverage check.

```python
import numpy as np
from math import erf, sqrt

rng = np.random.default_rng(0)


def simulate(n, scale):
    t_event = rng.weibull(1.5, n) * scale               # true event time
    t_censor = rng.uniform(0, 30, n)                    # dropout / end of study
    time = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(int)           # 1 = event seen, 0 = censored
    return time, event


def kaplan_meier(time, event):
    """Return distinct event times, S(t), and Greenwood standard errors."""
    times = np.unique(time[event == 1])
    s, var_sum, surv, se = 1.0, 0.0, [], []
    for t in times:
        n_at_risk = np.sum(time >= t)
        d = np.sum((time == t) & (event == 1))
        s *= 1 - d / n_at_risk
        var_sum += d / (n_at_risk * (n_at_risk - d)) if n_at_risk > d else 0.0
        surv.append(s)
        se.append(s * np.sqrt(var_sum))
    return times, np.array(surv), np.array(se)


def median_survival(times, surv):
    below = np.nonzero(surv <= 0.5)[0]
    return times[below[0]] if len(below) else np.inf


def logrank(time, event, group):
    """Two-group log-rank test: chi-square statistic and p-value (1 df)."""
    o_minus_e, var = 0.0, 0.0
    for t in np.unique(time[event == 1]):
        at_risk = time >= t
        n, n1 = at_risk.sum(), (at_risk & (group == 1)).sum()
        d = ((time == t) & (event == 1)).sum()
        d1 = ((time == t) & (event == 1) & (group == 1)).sum()
        o_minus_e += d1 - d * n1 / n
        var += d * (n1 / n) * (1 - n1 / n) * (n - d) / max(n - 1, 1)
    chi2 = o_minus_e ** 2 / var
    p = 1 - erf(sqrt(chi2 / 2))                          # chi2(1) survival = 2*(1-Phi(sqrt))
    return chi2, p


time, event = simulate(400, scale=12)
print(f"n=400, censored {1 - event.mean():.1%}")
times, surv, se = kaplan_meier(time, event)
true_s = lambda t: np.exp(-(t / 12) ** 1.5)
naive_t = np.sort(time[event == 1])                    # wrong: drop censored rows
naive_s = lambda t: np.mean(naive_t > t)
ignore_s = lambda t: np.mean(time > t)                 # wrong: treat censoring as event
print("\n   t   true S(t)   KM S(t)  95% CI           drop-censored  censor-as-event")
for t in [3, 6, 12, 18, 24]:
    i = np.searchsorted(times, t, side="right") - 1
    s, e = surv[i], se[i]
    print(f"{t:>4}   {true_s(t):>8.3f}  {s:>8.3f}  [{max(s - 1.96 * e, 0):.3f}, {min(s + 1.96 * e, 1):.3f}]"
          f"   {naive_s(t):>12.3f}  {ignore_s(t):>15.3f}")
hits = {t: 0 for t in [6, 12, 18]}
for _ in range(500):                                   # does the 95% Greenwood CI cover the truth?
    tt, ee = simulate(400, scale=12)
    ts, ss, es = kaplan_meier(tt, ee)
    for t in hits:
        i = np.searchsorted(ts, t, side="right") - 1
        hits[t] += abs(ss[i] - true_s(t)) <= 1.96 * es[i]
print("\nGreenwood 95% CI coverage over 500 samples: " + ", ".join(f"t={t}: {h / 500:.3f}" for t, h in hits.items()))
print(f"median survival: true {12 * np.log(2) ** (1 / 1.5):.2f}, KM {median_survival(times, surv):.2f}, "
      f"drop-censored {np.median(naive_t):.2f}")

# two arms: treatment lengthens time-to-event by 30%
ta, ea = simulate(200, scale=12)
tb, eb = simulate(200, scale=15.6)
chi2, p = logrank(np.r_[ta, tb], np.r_[ea, eb], np.r_[np.zeros(200), np.ones(200)])
ma = median_survival(*kaplan_meier(ta, ea)[:2])
mb = median_survival(*kaplan_meier(tb, eb)[:2])
print(f"\nlog-rank control vs treatment: medians {ma:.2f} vs {mb:.2f}, chi2={chi2:.2f}, p={p:.4f}")
tc, ec = simulate(200, scale=12)
chi2, p = logrank(np.r_[ta, tc], np.r_[ea, ec], np.r_[np.zeros(200), np.ones(200)])
print(f"log-rank control vs same-distribution arm: chi2={chi2:.2f}, p={p:.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
n=400, censored 39.2%

   t   true S(t)   KM S(t)  95% CI           drop-censored  censor-as-event
   3      0.882     0.890  [0.859, 0.921]          0.827            0.828
   6      0.702     0.751  [0.706, 0.795]          0.626            0.603
  12      0.368     0.428  [0.371, 0.484]          0.247            0.260
  18      0.159     0.225  [0.169, 0.281]          0.082            0.083
  24      0.059     0.060  [0.014, 0.106]          0.012            0.013

Greenwood 95% CI coverage over 500 samples: t=6: 0.932, t=12: 0.950, t=18: 0.940
median survival: true 9.40, KM 10.12, drop-censored 7.92

log-rank control vs treatment: medians 8.16 vs 11.00, chi2=13.91, p=0.0002
log-rank control vs same-distribution arm: chi2=0.24, p=0.622
```

The two naive estimators are not noisy, they are biased: at every `t` both sit well below the truth, and at t = 18 they report about half the true survival. This particular KM curve runs high (0.428 vs 0.368 at t = 12, and the truth at t = 18 is just below the CI). That is sampling noise, and the errors are correlated along the curve because each step multiplies the previous ones. The 500-repeat check shows the estimator and its Greenwood intervals are calibrated: 93-95% coverage against a 95% target. The log-rank test separates the arms clearly (χ² = 13.9) with 200 subjects each, while the null comparison gives χ² = 0.24, well under the 3.84 critical value.

## Choosing / trade-offs
- **KM vs parametric (Weibull, log-normal).** KM makes no shape assumption and is the default descriptive curve. A parametric fit extrapolates past the last observed time and is more efficient when the shape is right, but it can be badly wrong when it is not.
- **KM vs Cox.** KM handles one grouping variable at a time. Cox proportional hazards adjusts for many covariates and gives hazard ratios. Check its proportional-hazards assumption, for example with Schoenfeld residuals or crossing KM curves.
- **Log-rank vs Wilcoxon (Breslow) / Fleming-Harrington.** Log-rank weights all event times equally and is optimal under proportional hazards. Weighted tests emphasize early or late differences. When the curves cross, consider restricted mean survival time (RMST) instead.
- **Libraries.** `lifelines` (`KaplanMeierFitter`, `logrank_test`, `CoxPHFitter`) and `scikit-survival` are the Python standards. Use them for log-log CIs, left truncation, weights and plotting.

## Gotchas
- Never compute "average lifetime" from observed durations alone. With censoring, the mean may not even be estimable, because the KM curve need not reach 0. Report the median or RMST up to a fixed horizon.
- Informative censoring breaks KM. If customers who downgrade are also removed from the dataset, or sick patients drop out, the curve is biased. Check how censoring is generated.
- Ties: compute `d_i` as the number of events at exactly `t_i`, and count subjects censored at `t_i` as still at risk at `t_i`. That is the standard convention, and it is what `time >= t` does here.
- The tail is unreliable. When only a handful of subjects remain at risk, each event drops the curve sharply. Show the number at risk under the plot, and truncate where it gets small.
- Define time zero consistently (signup date, diagnosis, install). Mixing origins, or starting the clock after a qualifying event (immortal-time bias), invalidates comparisons.
- Pointwise CIs are not a simultaneous band. To claim "the whole curve lies in here", use Hall-Wellner or equal-precision bands.

## Related
- [[survival-analysis]] - Cox, AFT and survival ML with lifelines and scikit-survival.
- [[statistics-and-ab-testing]] - hypothesis testing basics behind the log-rank test.
- [[maximum-likelihood-and-map-estimation]] - KM as the non-parametric MLE of the survival function.
- [[causal-inference-and-uplift]] - when group differences are not randomized.

## References
- Kaplan and Meier (1958), "Nonparametric Estimation from Incomplete Observations", JASA: https://doi.org/10.1080/01621459.1958.10501452
- lifelines documentation, survival analysis introduction: https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html
- scikit-survival documentation: https://scikit-survival.readthedocs.io/
