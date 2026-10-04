---
title: Black-box and evolutionary optimization (GA, ES, CMA-ES, differential evolution)
category: ml
tags: [optimization, black-box, derivative-free, genetic-algorithm, evolution-strategies, cma-es, differential-evolution, nelder-mead, simulated-annealing, scipy]
use_cases:
  - "optimize parameters of a simulation or process that has no gradient"
  - "tune a scheduling, routing or design problem with a custom cost function"
  - "search a bumpy objective with many local minima"
  - "calibrate a physical or financial model to match observed data"
  - "optimize a non-differentiable metric or a policy without backpropagation"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.scipy.org/doc/scipy/reference/optimize.html
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
  - https://github.com/CMA-ES/pycma
  - https://arxiv.org/abs/1604.00772
---

# Black-box and evolutionary optimization (GA, ES, CMA-ES, differential evolution)

## Summary
Black-box optimizers find good inputs to a function you can only evaluate,
with no gradient: a simulator, a lab process, a cost computed by business
rules, a non-differentiable metric. Local methods (Nelder-Mead, pattern search)
are cheap but stop at the nearest minimum; population-based methods (genetic
algorithms, evolution strategies, CMA-ES, differential evolution) explore
globally at the cost of many more evaluations. If evaluations are expensive
(minutes each), use Bayesian optimization instead.

## Key concepts
- **Black box:** you get `f(x)` and nothing else. Budget is counted in
  function evaluations, not iterations.
- **Local derivative-free:** Nelder-Mead (simplex), Powell, COBYLA
  (constraints). Fast on smooth, low-dimensional problems; trapped by local minima.
- **Population-based (evolutionary):** keep a set of candidates, generate new
  ones by mutation/recombination, keep the best. Selection pressure vs
  diversity is the core trade-off.
- **Genetic algorithm (GA):** encoded candidates, crossover + mutation; natural
  for discrete/combinatorial encodings (permutations, bitstrings).
- **Evolution strategies (ES):** real vectors plus Gaussian mutation with an
  adapted step size. **CMA-ES** also adapts the full covariance, learning the
  problem's scaling and correlations; the default for continuous problems
  up to a few hundred dimensions.
- **Differential evolution (DE):** mutate with scaled differences between
  population members; simple, robust, built into SciPy.
- **Simulated annealing:** single candidate, accepts worse moves with a
  probability that cools over time (`scipy.optimize.dual_annealing`).
- **Constraints:** penalty terms, bounds clipping, or repair operators that map
  invalid candidates back into the feasible set.

## When to use / scenarios
- Manufacturing/engineering: tune process setpoints or a design against a
  simulator ([[manufacturing-iot]]).
- Operations: shift scheduling, routing, layout with custom cost rules (GA,
  or a proper solver, see below).
- Calibration: fit parameters of an ODE, agent-based or pricing model to data
  ([[finance]]).
- RL: evolution strategies as a parallel, gradient-free alternative to policy
  gradients ([[reinforcement-learning]]).
- NOT when gradients exist (use gradient descent, see [[math-for-machine-learning]]),
  NOT when each evaluation is expensive (use Bayesian optimization, see
  [[bayesian-and-gaussian-processes]] and [[hyperparameter-tuning]]), and NOT for
  linear/integer programs with a known structure (use a MILP/CP solver such as
  OR-Tools or HiGHS: exact and far faster).

## Setup & code
```bash
pip install numpy scipy        # cma for CMA-ES, pymoo for multi-objective / GA
```
```python
import numpy as np
from scipy.optimize import differential_evolution, minimize

def rastrigin(x):                      # many local minima, global min 0 at x = 0
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

d, bounds = 5, [(-5.12, 5.12)] * 5
rng = np.random.default_rng(0)

# Local, derivative-free: Nelder-Mead from a random start gets stuck
x0 = rng.uniform(-5.12, 5.12, d)
nm = minimize(rastrigin, x0, method="Nelder-Mead", options={"maxfev": 5000})
print(f"nelder-mead  f={nm.fun:.3f} evals={nm.nfev}")

# Random search with the same budget
pts = rng.uniform(-5.12, 5.12, (5000, d))
print(f"random       f={min(rastrigin(p) for p in pts):.3f} evals=5000")

# Population-based global: differential evolution
de = differential_evolution(rastrigin, bounds, seed=0, tol=1e-8)
print(f"diff-evol    f={de.fun:.3f} evals={de.nfev}")

# Minimal (mu, lambda) evolution strategy with step-size decay
mu, lam, sigma = 10, 50, 1.0
parents = rng.uniform(-5.12, 5.12, (mu, d))
evals = 0
for gen in range(200):
    kids = parents[rng.integers(mu, size=lam)] + sigma * rng.normal(size=(lam, d))
    kids = np.clip(kids, -5.12, 5.12)
    f = np.array([rastrigin(k) for k in kids]); evals += lam
    parents = kids[np.argsort(f)[:mu]]
    sigma *= 0.98
print(f"simple ES    f={rastrigin(parents[0]):.3f} evals={evals}")
```
Output (numpy 2.5.1, scipy 1.18.0), 5-D Rastrigin, global minimum 0:

| Method | Best f | Evaluations |
|---|---|---|
| Nelder-Mead (one start) | 66.662 | 216 (converged to a local minimum) |
| Random search | 19.507 | 5,000 |
| Differential evolution | 0.000 | 14,406 |
| Simple ES (hand-rolled) | 3.027 | 10,000 |

For CMA-ES: `import cma; xbest, es = cma.fmin2(rastrigin, x0, 2.0)`.
For multi-objective problems (cost vs quality, a Pareto front), use `pymoo`
(NSGA-II/III).

## Choosing / trade-offs
- Smooth, few dimensions, decent start: Nelder-Mead or Powell, with several
  random restarts.
- Continuous, bumpy, up to a few hundred dims, cheap evaluations: CMA-ES
  first, differential evolution second.
- Discrete / permutation / mixed encodings: GA with problem-specific
  crossover and mutation, or simulated annealing; check whether a MILP/CP
  solver can model it first.
- Expensive evaluations (under ~a few hundred affordable): Bayesian
  optimization (Optuna TPE, GP-based) instead of any population method.
- Evaluations are independent within a generation: population methods
  parallelize trivially (`differential_evolution(..., workers=-1)`).

## Gotchas
- Evaluation budget, not wall-clock iterations, is the fair comparison;
  report both best value and evaluations used.
- Noisy objectives (simulations with random seeds): the "best" candidate is
  often a lucky draw. Re-evaluate the top candidates several times, or use
  averaging inside `f`.
- Scale the search space: parameters spanning very different ranges hurt
  everything except CMA-ES. Search in log space for rates and scales.
- A single run proves little; evolutionary results vary by seed. Run several
  seeds and report the spread.
- GA hyperparameters (population, mutation rate) matter as much as the
  problem; start from library defaults before hand-tuning.
- Penalty constraints with the wrong weight make the optimizer either ignore
  the constraint or never leave the feasible boundary.

## Related
- [[hyperparameter-tuning]] - Bayesian optimization when evaluations are expensive.
- [[bayesian-and-gaussian-processes]] - the surrogate models behind BO.
- [[math-for-machine-learning]] - gradient-based optimization when gradients exist.
- [[reinforcement-learning]] - evolution strategies as a policy-search method.
- [[automl]] - search over model configurations.
- [[manufacturing-iot]] - process and design optimization scenarios.

## References
- SciPy optimize reference: https://docs.scipy.org/doc/scipy/reference/optimize.html
- SciPy differential_evolution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
- pycma (CMA-ES): https://github.com/CMA-ES/pycma
- Hansen, The CMA Evolution Strategy: A Tutorial: https://arxiv.org/abs/1604.00772
