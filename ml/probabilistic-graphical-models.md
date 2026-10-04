---
title: Probabilistic graphical models (Bayesian networks, Markov random fields, CRFs)
category: ml
tags: [probabilistic-graphical-models, bayesian-network, markov-random-field, crf, inference, pgmpy, sklearn-crfsuite, causal, explainable]
use_cases:
  - "model how a handful of variables influence each other and query probabilities given evidence"
  - "build an explainable diagnostic or risk model from expert knowledge plus a little data"
  - "reason under uncertainty with missing inputs (some symptoms unknown)"
  - "label a sequence (named entities, part-of-speech) with dependencies between neighbouring labels"
status: draft
last_verified: 2026-10-04
sources:
  - https://pgmpy.org/
  - https://sklearn-crfsuite.readthedocs.io/
  - https://www.cs.ubc.ca/~murphyk/MLbook/
---

# Probabilistic graphical models (Bayesian networks, Markov random fields, CRFs)

## Summary
A probabilistic graphical model (PGM) represents a joint probability
distribution over many variables as a graph: nodes are variables, edges are
direct dependencies. That factorization makes the model small, explainable
and queryable: "what is P(disease | these symptoms, test unknown)?" works even
with missing evidence. Bayesian networks (directed), Markov random fields
(undirected) and conditional random fields (CRFs, for structured labels) are
the main families.

## Key concepts
- **Bayesian network (BN).** A directed acyclic graph; each node has a
  conditional probability table (CPT) `P(node | parents)`. The joint is the
  product of the CPTs. Edges often follow cause -> effect, which makes the
  model readable by domain experts.
- **Markov random field (MRF).** Undirected graph with potential functions
  over cliques; natural for symmetric relations such as neighbouring pixels.
- **Conditional random field (CRF).** An MRF over the *outputs* conditioned
  on the inputs; a linear-chain CRF labels sequences so neighbouring labels
  are consistent (I-PER cannot follow O directly in BIO tagging).
- **Inference.** Computing posteriors given evidence. Exact: variable
  elimination, belief propagation / junction tree (fine for small or
  tree-like graphs). Approximate: sampling (MCMC, likelihood weighting) or
  variational methods for large, loopy graphs.
- **Learning.** Parameters (CPTs) from counts with a Dirichlet/Laplace prior
  (MLE or Bayesian estimation); structure from data via score-based (hill
  climbing with BIC) or constraint-based (PC algorithm) search, or from experts.
- **HMMs and Kalman filters** are dynamic Bayesian networks (see
  [[hidden-markov-models-and-kalman-filters]]).

## When to use / scenarios
- Small domains (tens of variables) where explanations and "what-if" queries
  matter: medical or equipment fault diagnosis, risk assessment, root-cause
  analysis in IT/manufacturing (see [[manufacturing-iot]], [[healthcare]]).
- Expert knowledge is available but data is scarce: experts draw the graph,
  data fills or refines the CPTs.
- Inputs are often missing at prediction time; a BN marginalizes them out
  naturally.
- Sequence labelling with a small feature-engineered model (linear-chain CRF),
  or a CRF head on top of a neural encoder (see [[nlp-classic-tasks]]).
- NOT for: high-dimensional raw data (images, text) where deep networks win;
  pure prediction accuracy on tabular data (see [[gradient-boosting-tabular]]);
  causal effect estimation from observational data without a defended graph
  (see [[causal-inference-and-uplift]]).

## Setup & code
```bash
pip install pgmpy
```
```python
from pgmpy.models import DiscreteBayesianNetwork   # older pgmpy: BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Rain -> WetGrass <- Sprinkler
model = DiscreteBayesianNetwork([("Rain", "WetGrass"), ("Sprinkler", "WetGrass")])
model.add_cpds(
    TabularCPD("Rain", 2, [[0.8], [0.2]]),          # state 0 = no, 1 = yes
    TabularCPD("Sprinkler", 2, [[0.6], [0.4]]),
    TabularCPD("WetGrass", 2,
               # columns: (R=0,S=0) (R=0,S=1) (R=1,S=0) (R=1,S=1)
               [[0.99, 0.10, 0.20, 0.01],
                [0.01, 0.90, 0.80, 0.99]],
               evidence=["Rain", "Sprinkler"], evidence_card=[2, 2]),
)
assert model.check_model()

infer = VariableElimination(model)
print(infer.query(["Rain"], evidence={"WetGrass": 1}))
# Explaining away: also knowing the sprinkler was on lowers P(Rain).
print(infer.query(["Rain"], evidence={"WetGrass": 1, "Sprinkler": 1}))
```
Learn CPTs from a DataFrame with the same column names:
`model.fit(df, estimator=BayesianEstimator, prior_type="BDeu")`
(`from pgmpy.estimators import BayesianEstimator`). For a linear-chain CRF on
tokens, `sklearn-crfsuite` takes a list of per-token feature dicts per sentence.

## Choosing / trade-offs
- **BN vs a discriminative model.** BNs model the full joint (any query, any
  missing pattern, readable); a classifier only models `P(y | x)` but is
  usually more accurate with plenty of data.
- **Expert vs learned structure.** Expert graphs are interpretable and work
  with little data but encode bias; learned structures need much more data
  and are only identified up to equivalence classes (edge directions are
  often not determined by data alone).
- **Exact vs approximate inference.** Exact cost grows exponentially with the
  graph's treewidth; switch to sampling for large or densely connected graphs.
- **Discrete vs continuous.** Discretizing continuous variables is simple but
  loses information; Gaussian or hybrid BNs avoid that with less tooling.

## Gotchas
- CPT size grows exponentially with the number of parents; keep in-degree
  small or the tables cannot be estimated from data.
- Zero probabilities in a CPT make some evidence "impossible" and inference
  fail or return NaN; use a prior (BDeu/Laplace) when learning from counts.
- Edge direction in a learned BN is not causal proof; treat it as a hypothesis.
- CPT column order follows the `evidence` list order; a swapped order gives a
  silently wrong model. Check with a few known queries.
- pgmpy renames classes between versions (`BayesianModel` ->
  `BayesianNetwork` -> `DiscreteBayesianNetwork`); pin the version.

## Related
- [[hidden-markov-models-and-kalman-filters]] - dynamic Bayesian networks over time.
- [[bayesian-and-gaussian-processes]] - Bayesian inference for continuous models.
- [[causal-inference-and-uplift]] - when the graph is used to estimate effects.
- [[nlp-classic-tasks]] - CRFs for sequence labelling.
- [[svm-knn-naive-bayes]] - Naive Bayes is the simplest Bayesian network.
- [[model-interpretability]] - other routes to explainable models.

## References
- https://pgmpy.org/
- https://sklearn-crfsuite.readthedocs.io/
- https://www.cs.ubc.ca/~murphyk/MLbook/ (Murphy, Probabilistic Machine Learning)
