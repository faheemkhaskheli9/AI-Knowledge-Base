---
title: Particle filter from scratch (bootstrap SIR filter, systematic resampling, ESS, vs EKF)
category: ml
tags: [particle-filter, sequential-monte-carlo, smc, bootstrap-filter, resampling, effective-sample-size, nonlinear-filtering, state-space-model, tracking, numpy, from-scratch, ml-basics]
use_cases:
  - "implement a bootstrap particle filter with systematic resampling in NumPy"
  - "track a hidden state through nonlinear dynamics and a nonlinear measurement where a Kalman filter fails"
  - "see weight degeneracy, effective sample size and why resampling is needed"
  - "explain particle filters vs EKF/UKF and sequential importance resampling in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1049/ip-f-2.1993.0015
  - https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
  - https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
  - https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html
---

# Particle filter from scratch (bootstrap SIR filter, systematic resampling, ESS, vs EKF)

## Summary
A particle filter tracks a hidden state through a state-space model by representing its distribution with a cloud of weighted samples ("particles") instead of a single Gaussian. Each step moves every particle through the dynamics, reweights it by how well it explains the new measurement, and resamples when the weights collapse onto a few particles. Because it never linearises anything, it handles nonlinear dynamics, non-Gaussian noise and multimodal posteriors where a Kalman filter breaks. On the classic nonlinear benchmark `y = x²/20 + noise`, the extended Kalman filter gets an RMSE of 20.95. A 500-particle filter gets 4.04, and the same filter without resampling degrades to 8.73.

## Key concepts
- **State-space model.** A hidden state follows `x_t ~ p(x_t | x_{t-1})`, and you observe `y_t ~ p(y_t | x_t)`. Filtering means computing `p(x_t | y_1..t)` online.
- **Sequential importance sampling.** Particles are samples from a proposal. Weights correct for the mismatch with the target. The bootstrap filter uses the dynamics as the proposal, so each weight update is just the measurement likelihood `w ∝ w · p(y_t | x_t)`.
- **Weight degeneracy.** After a few steps almost all the weight sits on one particle and the rest are wasted computation.
- **Effective sample size.** `ESS = 1 / Σ wᵢ²` with normalised weights. It is N for equal weights and 1 when one particle holds everything. Resample when it drops below a threshold such as N/2.
- **Resampling.** Draw N particles with replacement in proportion to their weights, then reset the weights to equal. Systematic resampling uses one random offset and N evenly spaced points, which adds less noise than multinomial draws.
- **Estimates.** Any posterior summary is a weighted average over particles: the mean `Σ wᵢ xᵢ`, quantiles, or the probability of a region.

## When to use / scenarios
- Learning: the cleanest introduction to sequential Monte Carlo and importance sampling, and a direct contrast with the [[kalman-filter-from-scratch]] predict/update loop.
- Interviews: "why does a Kalman filter fail here", "what is weight degeneracy", "how many particles do you need", "particle filter vs UKF".
- Practice: robot localisation against a map (Monte Carlo localisation), indoor positioning from Wi-Fi or beacon signal strength, tracking in clutter, battery state-of-charge estimation, epidemiological models fitted to case counts, and stochastic-volatility models in finance.
- Not for: linear-Gaussian models (the Kalman filter is exact and far cheaper); mildly nonlinear models with unimodal posteriors (try an EKF or UKF first); or high-dimensional states (dozens of dimensions or more), where the particle count needed grows exponentially. Use ensemble Kalman filters or Rao-Blackwellised filters there.

## Setup & code
NumPy only. Runs in about a second.

```python
import numpy as np

rng = np.random.default_rng(0)

# nonlinear benchmark (Gordon et al. 1993 / Kitagawa 1996):
#   x_t = x/2 + 25 x/(1+x^2) + 8 cos(1.2 t) + N(0, 10)
#   y_t = x_t^2 / 20 + N(0, 1)
T, Q, R = 100, 10.0, 1.0


def f(x, t):
    return 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * t)


x_true, ys, x = np.empty(T), np.empty(T), 0.1
for t in range(T):
    x = f(x, t) + rng.normal(0, np.sqrt(Q))
    x_true[t], ys[t] = x, x**2 / 20 + rng.normal(0, np.sqrt(R))


def systematic_resample(w, rng):
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)


def particle_filter(ys, n, rng, resample=True):
    """Bootstrap (SIR) filter: propagate through the dynamics, weight by likelihood, resample."""
    p = rng.normal(0.1, np.sqrt(2.0), n)
    logw = np.zeros(n)
    means, ess_hist = np.empty(len(ys)), np.empty(len(ys))
    for t, y in enumerate(ys):
        p = f(p, t) + rng.normal(0, np.sqrt(Q), n)                 # predict: sample the transition
        logw += -0.5 * (y - p**2 / 20) ** 2 / R                     # update: weight by p(y | x)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        means[t] = np.sum(w * p)
        ess_hist[t] = 1 / np.sum(w**2)
        if resample and ess_hist[t] < n / 2:                       # resample only when weights degenerate
            p = p[systematic_resample(w, rng)]
            logw = np.zeros(n)
        else:
            logw -= logw.max()                                      # keep weights, avoid underflow
    return means, ess_hist


def ekf(ys):
    """Extended Kalman filter: linearise f and h around the current estimate."""
    m, P, out = 0.1, 2.0, np.empty(len(ys))
    for t, y in enumerate(ys):
        F = 0.5 + 25 * (1 - m**2) / (1 + m**2) ** 2
        m, P = f(m, t), F * P * F + Q
        H = m / 10
        S = H * P * H + R
        K = P * H / S
        m, P = m + K * (y - m**2 / 20), (1 - K * H) * P
        out[t] = m
    return out


rmse = lambda est: np.sqrt(np.mean((est - x_true) ** 2))
print(f"T={T}; RMSE of the posterior-mean estimate of x_t")
print(f"EKF                          {rmse(ekf(ys)):6.2f}")
for n in [50, 500, 5000]:
    m, e = particle_filter(ys, n, np.random.default_rng(1))
    print(f"particle filter, N={n:<5}     {rmse(m):6.2f}   (min ESS {e.min():.0f})")
m, e = particle_filter(ys, 500, np.random.default_rng(1), resample=False)
print(f"N=500, never resample        {rmse(m):6.2f}   (ESS at end {e[-1]:.1f})")

# sign ambiguity: y depends on x^2, so the posterior is bimodal when x is far from 0
m, _ = particle_filter(ys, 5000, np.random.default_rng(1))
wrong_sign = np.mean(np.sign(m) != np.sign(x_true))
print(f"\nsteps where the posterior mean has the wrong sign (N=5000): {wrong_sign:.0%}")
```

Output (Python 3.14, NumPy 2.5):
```
T=100; RMSE of the posterior-mean estimate of x_t
EKF                           20.95
particle filter, N=50          5.83   (min ESS 1)
particle filter, N=500         4.04   (min ESS 1)
particle filter, N=5000        4.12   (min ESS 18)
N=500, never resample          8.73   (ESS at end 1.5)

steps where the posterior mean has the wrong sign (N=5000): 14%
```

The EKF fails because the measurement `x²/20` cannot tell `x` from `−x`, and its local slope `x/10` is zero at the origin. Linearising there throws away the measurement entirely, and a single Gaussian cannot represent "either +10 or −10". The particle filter keeps both modes and lands at an RMSE about 5 times lower. Going from 50 to 500 particles helps. Going to 5000 does not improve this run further: the remaining error comes from the bimodal posterior itself, since its mean sits between the two modes, and from Monte Carlo noise in a single run. `min ESS 1` shows that some measurements are so informative that one particle takes almost all the weight, which is exactly when resampling must happen. Without resampling the weights degenerate for good (ESS 1.5 at the end) and the error doubles. The last line is the honest limit: on 14% of steps even 5000 particles put the posterior mean on the wrong side of zero, because the data genuinely cannot resolve the sign.

## Choosing / trade-offs
- **Particle filter vs Kalman family.** The Kalman filter is exact for linear-Gaussian models. The EKF and UKF are cheap and good when the posterior stays roughly Gaussian and unimodal. Particle filters handle anything, at a cost of N model evaluations per step and Monte Carlo noise.
- **Number of particles.** Error falls roughly as `1/√N` once the filter is tracking. Increase N until estimates stop changing across seeds. Hundreds suffice in 1 to 3 dimensions, and the need grows fast with state dimension.
- **Proposal.** The bootstrap proposal ignores the current measurement, so a sharp likelihood leaves few particles in the right place. Auxiliary particle filters, optimal or locally linearised proposals, and a UKF-based proposal (unscented particle filter) put particles where the measurement says.
- **Resampling scheme.** Systematic and stratified resampling are low-variance and O(N), and are the usual defaults. Multinomial is simplest and noisiest. Resample adaptively on ESS rather than every step.
- **Point estimate.** The weighted mean is wrong for a bimodal posterior: it can sit between the modes in a region of low probability. Report the full particle distribution, or the mode, when the posterior can split.
- **Libraries.** `filterpy` provides resampling routines. `particles` (Chopin) implements SMC samplers, smoothing and parameter estimation. Robotics stacks ship Monte Carlo localisation (for example the ROS 2 AMCL package).

## Gotchas
- Work with log weights and subtract the maximum before exponentiating. Raw likelihood products underflow to zero within a few steps.
- Resampling without jitter collapses diversity: many copies of one particle. With little process noise, add small roughening noise or a regularisation (kernel) step after resampling.
- Static parameters inside the state never get new values after resampling and degenerate quickly. Estimate them with particle MCMC, SMC², or by giving them artificial dynamics.
- The process noise is a tuning knob. Too little and the cloud cannot follow the truth (sample impoverishment); too much and estimates get noisy.
- A single run is one Monte Carlo draw. Compare filters over several seeds before trusting a difference like 4.04 vs 4.12.
- Particle smoothing (estimating past states with future data) needs more than the filter output. The resampled ancestry paths degenerate, so use forward-filtering backward-smoothing.
- Vectorise across particles. A Python loop over particles is the usual reason particle filters are called slow.

## Related
- [[kalman-filter-from-scratch]] - the exact linear-Gaussian filter and its predict/update structure.
- [[hidden-markov-models-and-kalman-filters]] - library usage for HMMs and Kalman filters.
- [[hmm-from-scratch]] - exact filtering when the hidden state is discrete.
- [[metropolis-hastings-mcmc-from-scratch]] - Monte Carlo for a static posterior instead of a sequence.
- [[time-series-forecasting]] - state-space models for forecasting.
- [[probabilistic-graphical-models]] - the state-space model as a dynamic Bayesian network.

## References
- Gordon, Salmond and Smith (1993), "Novel approach to nonlinear/non-Gaussian Bayesian state estimation", IEE Proceedings F: https://doi.org/10.1049/ip-f-2.1993.0015
- Doucet and Johansen, "A tutorial on particle filtering and smoothing: fifteen years later": https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
- Särkkä, Bayesian Filtering and Smoothing (chapter 7, particle filtering): https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
- FilterPy resampling routines: https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html
