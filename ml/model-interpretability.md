---
title: Model interpretability (feature importance, PDP, SHAP)
category: ml
tags: [interpretability, explainability, xai, shap, permutation-importance, partial-dependence, feature-importance, scikit-learn]
use_cases:
  - "explain to a loan applicant or regulator why the model declined them"
  - "find out which features a churn or fraud model actually relies on"
  - "catch data leakage by spotting a suspiciously important feature"
  - "show how predicted risk changes as one input (age, price, dosage) changes"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/inspection.html
  - https://scikit-learn.org/stable/modules/permutation_importance.html
  - https://scikit-learn.org/stable/modules/partial_dependence.html
  - https://shap.readthedocs.io/
  - https://christophm.github.io/interpretable-ml-book/
---

# Model interpretability (feature importance, PDP, SHAP)

## Summary
Interpretability methods explain what a trained model has learned: which features matter overall (global) and why it made one specific prediction (local). Simple models are interpretable by design (linear weights, a shallow tree); for forests, boosting and neural networks you use model-agnostic tools such as permutation importance, partial dependence plots and SHAP values. These explain the model, not the world: they show what the model uses, which is also how you catch leakage and bias before deployment.

## Key concepts
- **Global vs local.** Global: which features drive the model on average. Local: why this one customer got a score of 0.83.
- **Intrinsic vs post-hoc.** Intrinsic: linear/logistic weights, tree rules, GAMs (see [[linear-models]], [[decision-trees-and-random-forests]]). Post-hoc: methods applied to any fitted model.
- **Permutation importance.** Shuffle one feature on held-out data and measure the drop in score. Model-agnostic and tied to a real metric; `sklearn.inspection.permutation_importance`.
- **Impurity importance** (`feature_importances_` on trees): cheap but computed on training data and biased toward high-cardinality features.
- **Partial dependence (PDP).** Average prediction as one feature varies, others held at their observed values. **ICE** curves show the same per row, revealing heterogeneous effects.
- **SHAP values.** Split one prediction into additive contributions per feature (from Shapley values in game theory): base value + sum of contributions = prediction. Fast exact versions exist for tree models (`TreeExplainer`); summary plots aggregate them into a global view.
- **LIME.** Fit a simple local surrogate model around one prediction; easier to compute, less stable than SHAP.
- **Correlated features** share or split credit unpredictably in every method; explanations of one of a correlated pair are unreliable.
- **Counterfactuals.** "Income 5k higher would have approved the loan": the most actionable local explanation for end users.

## When to use / scenarios
- Credit, insurance, hiring and healthcare decisions where regulation or policy requires reasons for an individual decision (adverse-action reasons): SHAP or reason codes from a linear scorecard.
- Model debugging before launch: a top feature like `account_closed_date` in a churn model is leakage, see [[ml-fundamentals]].
- Fairness review: check whether protected attributes or their proxies (postcode) drive predictions.
- Stakeholder trust: a PDP of "predicted demand vs price" a planner can sanity-check.
- Not a substitute for causal analysis: importance does not mean changing the feature changes the outcome.

## Setup & code
```bash
pip install scikit-learn matplotlib
# optional, for SHAP:
pip install shap
```
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split

data = fetch_california_housing()          # downloads ~400 KB on first use
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, random_state=0)
names = data.feature_names
model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
print("test R2:", round(model.score(X_te, y_te), 3))

# Global: permutation importance on held-out data.
imp = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0)
for i in imp.importances_mean.argsort()[::-1][:4]:
    print(f"{names[i]:>10}: {imp.importances_mean[i]:.3f} +/- {imp.importances_std[i]:.3f}")

# Shape of one effect: partial dependence of price on median income.
mi = names.index("MedInc")
# method="brute" averages real predictions; the faster "recursion" method for
# boosting omits the baseline, so its values are offsets, not predictions.
pd_res = partial_dependence(model, X_te, features=[mi], grid_resolution=5,
                            method="brute")
for x, avg in zip(pd_res["grid_values"][0], pd_res["average"][0]):
    print(f"MedInc={x:.1f} -> predicted value {avg:.2f}")
# Plot: from sklearn.inspection import PartialDependenceDisplay
#       PartialDependenceDisplay.from_estimator(model, X_te, [mi], kind="both")
```
Local explanation of one prediction with SHAP (needs `shap`):
```python
import shap
explainer = shap.TreeExplainer(model)
sv = explainer(X_te[:200])
row = sv[0]
print("base", row.base_values, "+ contributions", row.values.sum(), "=", model.predict(X_te[:1])[0])
shap.plots.waterfall(row)                   # why this one prediction
shap.plots.beeswarm(sv)                     # global summary
```

## Choosing / trade-offs
- **Need reasons per decision, regulated setting:** prefer an intrinsically interpretable model (scorecard, monotonic-constrained boosting) if it is close in accuracy; otherwise SHAP with documented limits.
- **Which features matter, any model:** permutation importance on a validation set. Cheap, honest, metric-based.
- **How a feature matters:** PDP plus ICE; switch to ALE plots (third-party) when features are strongly correlated.
- **SHAP vs LIME:** SHAP for consistency and tree models (fast exact); LIME only when SHAP is too slow for the model type.
- **Neural networks on images/text:** gradient-based attributions (Integrated Gradients via Captum) or attention visualisation; these are noisier and harder to validate.

## Gotchas
- Importance on training data rewards memorised noise; compute on held-out data.
- Correlated features split importance: dropping either one barely changes the score, so both look unimportant while together they matter. Group them or drop one and re-check.
- PDPs average over unrealistic combinations when features are correlated (a 1-room house with 10 bedrooms) and can hide opposite effects that cancel; look at ICE curves.
- SHAP explains the model output in its own units (log-odds for many classifiers, not probability); say which when presenting.
- Explanations of a bad model are confident explanations of nonsense; check accuracy first.
- Explanation methods can be gamed and do not prove fairness; pair them with group metrics on outcomes.

## Related
- [[linear-models]] - interpretable by design through coefficients and odds ratios.
- [[decision-trees-and-random-forests]] - readable rules, and why impurity importance misleads.
- [[gradient-boosting-tabular]] - the models SHAP is most often used with.
- [[model-evaluation-and-metrics]] - measure performance before explaining it.
- [[ai-security-privacy-compliance]] - regulatory requirements that drive explanation needs.

## References
- scikit-learn inspection module: https://scikit-learn.org/stable/inspection.html
- Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
- Partial dependence and ICE: https://scikit-learn.org/stable/modules/partial_dependence.html
- SHAP documentation: https://shap.readthedocs.io/
- Molnar, *Interpretable Machine Learning* (free book): https://christophm.github.io/interpretable-ml-book/
