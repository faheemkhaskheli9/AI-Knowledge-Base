---
title: Metropolis-Hastings MCMC from scratch (random-walk proposals, burn-in, ESS, R-hat)
category: ml
tags: [mcmc, metropolis-hastings, markov-chain-monte-carlo, bayesian-inference, posterior-sampling, effective-sample-size, r-hat, burn-in, numpy, from-scratch, ml-basics]
use_cases:
  - "implement random-walk Metropolis in NumPy to sample a Bayesian posterior"
  - "tune the proposal step size and see how it changes acceptance rate and effective sample size"
  - "check MCMC convergence with multiple chains, burn-in and the Gelman-Rubin R-hat"
  - "explain the acceptance ratio, detailed balance and why MCMC samples are correlated in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1063/1.1699114
  - https://doi.org/10.1093/biomet/57.1.97
  - https://sites.stat.columbia.edu/gelman/book/
  - https://mc-stan.org/docs/reference-manual/analysis.html
---

# Metropolis-Hastings MCMC from scratch (random-walk proposals, burn-in, ESS, R-hat)

## Summary
Markov chain Monte Carlo draws samples from a distribution you can only evaluate up to a constant, which is exactly the situation with a Bayesian posterior `p(θ | y) ∝ p(y | θ) p(θ)`. Metropolis-Hastings builds a random walk whose long-run distribution is the target: propose a move, accept it with probability `min(1, p(proposal) / p(current))`, otherwise stay put. Below, random-walk Metropolis infers the mean and standard deviation of 50 Gaussian draws. Four chains agree (R-hat 1.002), and the posterior sd of the mean (0.265) matches the exact Student-t answer (0.266). The step size decides everything: 0.02 gives 91% acceptance but only 5 effective samples out of 4000, 0.3 gives 29% acceptance and 206, and 3.0 gives 1% acceptance and 24.

## Key concepts
- **Unnormalised target.** The acceptance ratio `p(x') / p(x)` cancels the normalising constant, so you only need `log p(y | θ) + log p(θ)`. That is why MCMC works where the evidence integral is intractable.
- **Acceptance rule.** For a proposal density `q`, accept with probability `min(1, p(x') q(x | x') / (p(x) q(x' | x)))`. A symmetric proposal such as `x + step · N(0, I)` makes the `q` terms cancel (plain Metropolis).
- **Detailed balance.** The rule makes probability flow from `x` to `x'` equal the flow back, so the target is the chain's stationary distribution. A rejected proposal still counts: the current state is recorded again.
- **Burn-in.** Early draws reflect the starting point, not the target. Discard them.
- **Autocorrelation and ESS.** Consecutive draws are correlated, so N draws carry less information than N independent ones. The effective sample size `N / (1 + 2 Σ ρ_k)` is the number that matters for Monte Carlo error.
- **R-hat.** Run several chains from dispersed starts and compare between-chain and within-chain variance. Values near 1 (below about 1.01) mean the chains agree. It cannot prove convergence, only detect its absence.

## When to use / scenarios
- Learning: the simplest general-purpose posterior sampler, and the foundation for Gibbs sampling, HMC and NUTS.
- Interviews: "derive the acceptance probability", "why is the normalising constant not needed", "how do you know the chain converged", "what does a 99% acceptance rate tell you".
- Practice: posteriors with a handful of parameters and a cheap likelihood, such as small hierarchical models in epidemiology, A/B test analysis with custom likelihoods, pharmacokinetic parameters, and calibrating simulators. Also as a fallback when gradients are unavailable.
- Not for: dozens or more continuous parameters (use HMC/NUTS through Stan, PyMC or NumPyro, which scale far better); large datasets where approximate answers are fine (use variational inference); or conjugate models (compute the posterior in closed form, see [[bayesian-linear-regression-from-scratch]]).

## Setup & code
NumPy only. Runs in a few seconds.

```python
import numpy as np

rng = np.random.default_rng(0)

# data: 50 draws from N(mu=3, sigma=2); infer (mu, log_sigma) with a flat prior on mu, log_sigma
y = rng.normal(3.0, 2.0, size=50)


def log_post(theta):
    mu, log_s = theta
    s = np.exp(log_s)
    return -len(y) * log_s - 0.5 * np.sum((y - mu) ** 2) / s**2   # log-likelihood, flat prior


def metropolis(log_p, x0, step, n, rng):
    """Random-walk Metropolis: propose x' = x + step * N(0, I), accept with prob min(1, p(x')/p(x))."""
    x, lp = np.array(x0, float), log_p(x0)
    out, accepted = np.empty((n, len(x))), 0
    for i in range(n):
        prop = x + step * rng.standard_normal(len(x))
        lp_prop = log_p(prop)
        if np.log(rng.random()) < lp_prop - lp:      # compare in log space, never exp the densities
            x, lp = prop, lp_prop
            accepted += 1
        out[i] = x
    return out, accepted / n


def ess(chain):
    """Effective sample size from the initial positive sequence of autocorrelations."""
    x = chain - chain.mean()
    n = len(x)
    acf = np.correlate(x, x, "full")[n - 1:] / (x.var() * n)
    s = 0.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        s += acf[k]
    return n / (1 + 2 * s)


def rhat(chains):
    """Gelman-Rubin potential scale reduction for one parameter, chains shape (m, n)."""
    m, n = chains.shape
    W = chains.var(axis=1, ddof=1).mean()
    B = n * chains.mean(axis=1).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + B / n) / W)


print(f"data: n={len(y)}, sample mean {y.mean():.3f}, sample sd {y.std(ddof=1):.3f}")
print("\nstep   accept   ESS(mu) per 5000 draws")
for step in [0.02, 0.3, 3.0]:
    ch, acc = metropolis(log_post, [0.0, 0.0], step, 5000, np.random.default_rng(1))
    print(f"{step:<5}  {acc:>6.2f}   {ess(ch[1000:, 0]):>7.0f}")

# 4 chains from dispersed starts, burn-in 1000, keep 4000 each
starts = [[-10, 2], [10, -1], [0, 3], [5, 0]]
chains = [metropolis(log_post, s, 0.3, 5000, np.random.default_rng(10 + i))[0][1000:] for i, s in enumerate(starts)]
chains = np.array(chains)                                   # (4, 4000, 2)
mu, sigma = chains[..., 0].ravel(), np.exp(chains[..., 1]).ravel()
print(f"\nR-hat mu {rhat(chains[..., 0]):.3f}, log_sigma {rhat(chains[..., 1]):.3f}")
print(f"posterior mu    mean {mu.mean():.3f}  95% CI [{np.quantile(mu, .025):.3f}, {np.quantile(mu, .975):.3f}]")
print(f"posterior sigma mean {sigma.mean():.3f}  95% CI [{np.quantile(sigma, .025):.3f}, {np.quantile(sigma, .975):.3f}]")
se = y.std(ddof=1) / np.sqrt(len(y))
print(f"exact check: mu | y ~ t(49) centred {y.mean():.3f}, sd {se * np.sqrt(49 / 47):.3f}; MCMC sd {mu.std():.3f}")

# a bad chain: started far away, no burn-in
bad, _ = metropolis(log_post, [-10, 2], 0.3, 300, np.random.default_rng(2))
print(f"\nfirst 300 draws from mu0=-10, no burn-in: mean mu {bad[:, 0].mean():.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
data: n=50, sample mean 3.258, sample sd 1.841

step   accept   ESS(mu) per 5000 draws
0.02     0.91         5
0.3      0.29       206
3.0      0.01        24

R-hat mu 1.002, log_sigma 1.000
posterior mu    mean 3.279  95% CI [2.769, 3.797]
posterior sigma mean 1.869  95% CI [1.536, 2.292]
exact check: mu | y ~ t(49) centred 3.258, sd 0.266; MCMC sd 0.265

first 300 draws from mu0=-10, no burn-in: mean mu -3.20
```

The step-size table is the main lesson. Tiny steps are almost always accepted but crawl, so 4000 kept draws hold about 5 independent ones. Huge steps land in low-density regions and are almost always rejected, so the chain sits still. The middle setting, with acceptance near the textbook 23% to 44% range, mixes about 40 times better than the tiny step. With four chains started far apart (including μ = −10 and μ = 10), R-hat is about 1, and the posterior for μ matches the analytical Student-t result to the third decimal. Sampling `log σ` instead of σ keeps the parameter unconstrained, so the random walk never proposes a negative standard deviation. Under this flat prior on `log σ`, the exact marginal for μ is a t distribution with n − 1 degrees of freedom. The last line shows why burn-in matters: a chain started at μ = −10 averages −3.2 over its first 300 draws, nowhere near the truth.

## Choosing / trade-offs
- **Step size.** Aim for roughly 0.23 acceptance in many dimensions and up to about 0.44 in one. Tune during burn-in only. Adapting the step forever breaks the stationary distribution.
- **Proposal shape.** An isotropic proposal struggles when parameters are correlated or on different scales. Reparameterise (standardise, log-transform positive parameters) or use the covariance of a pilot run as the proposal covariance.
- **Random-walk Metropolis vs Gibbs vs HMC/NUTS.** Random walks need roughly O(d) steps per independent sample and are fine for a few parameters. Gibbs sampling needs full conditionals, which conjugate models give you. HMC and NUTS use gradients to make long, accepted moves and are the default in Stan, PyMC and NumPyro.
- **MCMC vs variational inference.** MCMC is asymptotically exact but slow. VI optimises an approximate posterior, is fast, and usually underestimates uncertainty.
- **Thinning.** Keeping every k-th draw saves memory but never increases ESS per unit of compute. Report ESS instead of thinning.

## Gotchas
- Work in log space. Ratios of raw likelihoods underflow to 0/0 with even modest data.
- Record the current state again when a proposal is rejected. Recording only accepted moves biases the sample toward rarely visited regions.
- A proposal that is not symmetric (for example a log-normal step on a positive parameter) needs the Hastings correction `q(x | x') / q(x' | x)`. Leaving it out samples the wrong distribution with no visible error.
- Sampling a transformed parameter (here `log σ`) changes the prior. A flat prior on `log σ` is not a flat prior on σ, so add the log-Jacobian when you want a prior stated on the original scale.
- One chain that looks stable can still be stuck in one mode. Use several dispersed chains, R-hat and trace plots, and expect multimodal targets to need tempering or better samplers.
- Report ESS next to every posterior summary. 10,000 draws with an ESS of 20 is a 20-sample estimate.
- MCMC is serial within a chain. Parallelise across chains, not across steps.

## Related
- [[bayesian-and-gaussian-processes]] - Bayesian modelling with PyMC/Stan, where NUTS replaces hand-written samplers.
- [[bayesian-linear-regression-from-scratch]] - a conjugate posterior in closed form, no sampling needed.
- [[maximum-likelihood-and-map-estimation]] - point estimates of the same posterior.
- [[bayesian-deep-learning-and-uncertainty]] - approximate posteriors over neural network weights.
- [[probabilistic-graphical-models]] - models whose posteriors MCMC samples.
- [[particle-filter-from-scratch]] - sequential Monte Carlo, the time-series counterpart.

## References
- Metropolis, Rosenbluth, Rosenbluth, Teller and Teller (1953), "Equation of state calculations by fast computing machines": https://doi.org/10.1063/1.1699114
- Hastings (1970), "Monte Carlo sampling methods using Markov chains and their applications", Biometrika: https://doi.org/10.1093/biomet/57.1.97
- Gelman et al., Bayesian Data Analysis, 3rd ed. (chapters 11-12 on MCMC and R-hat): https://sites.stat.columbia.edu/gelman/book/
- Stan reference manual, posterior analysis (ESS, R-hat): https://mc-stan.org/docs/reference-manual/analysis.html
