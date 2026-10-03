---
title: Hidden Markov models and Kalman filters
category: ml
tags: [hmm, hidden-markov-model, viterbi, forward-backward, kalman-filter, state-estimation, regime-switching, sensor-fusion, tracking, hmmlearn]
use_cases:
  - "detect hidden regimes (calm vs volatile market, machine idle vs running) from a noisy signal"
  - "smooth noisy sensor or GPS readings and estimate velocity that is never measured"
  - "segment a sequence into states that persist over time instead of classifying each point alone"
  - "fuse two sensors with different noise levels into one estimate"
  - "track objects between detector frames in video"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
  - https://hmmlearn.readthedocs.io/en/latest/
  - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
  - https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html
---

# Hidden Markov models and Kalman filters

## Summary
Both are state-space models: a hidden state evolves over time, and you see
only noisy measurements of it. A hidden Markov model (HMM) has a discrete
state (a regime, a phoneme, a machine mode); a Kalman filter has a continuous
state (position, velocity, temperature) with linear dynamics and Gaussian
noise. They use the fact that states persist, so they beat any method that
looks at one time step alone, and they are small, fast and interpretable.

## Key concepts
- Ingredients: initial state distribution, transition model (how the state
  moves), emission/measurement model (what you see given the state).
- HMM inference: forward-backward gives `p(state_t | all data)` (smoothing);
  Viterbi gives the single most likely state path. Baum-Welch (EM) learns the
  parameters from unlabeled sequences.
- Kalman filter: alternate predict (`x = F x`, `P = F P F' + Q`) and update
  (blend in the measurement with gain `K`). `Q` is process noise (how much the
  world can surprise you), `R` is measurement noise (how much the sensor lies).
  Their ratio is the main tuning knob.
- Filtering uses data up to now (online); smoothing (RTS smoother,
  forward-backward) uses the whole sequence (offline, more accurate).
- Nonlinear dynamics: extended Kalman filter (linearize), unscented Kalman
  filter (sigma points), particle filter (sampling, any distribution).
- Work in log space for HMMs; products of many probabilities underflow.

## When to use / scenarios
- Finance: market regime detection (bull/bear, low/high volatility) with
  Markov-switching models.
- Manufacturing/IoT: machine state from vibration or current; sensor smoothing
  and fusion ([[manufacturing-iot]], [[anomaly-detection]]).
- Robotics, drones, vehicles: position/velocity from GPS + IMU.
- Video: track boxes between detections (SORT/ByteTrack use a Kalman filter,
  see [[video-analytics]]).
- Bioinformatics and simple speech/gesture segmentation.
- NOT for: long-range, nonlinear patterns with lots of training data (use
  sequence models from [[cnn-and-rnn-architectures]], [[state-space-models]]
  or [[transformers-and-attention]]); plain forecasting of a target series
  (start with [[time-series-forecasting]]).

## Setup & code
```bash
pip install numpy   # hmmlearn, filterpy or statsmodels for packaged versions
```
```python
import numpy as np
rng = np.random.default_rng(0)

# --- HMM: 2 hidden regimes (calm, volatile), Gaussian emissions, Viterbi ---
A = np.array([[0.95, 0.05], [0.10, 0.90]])   # transition matrix
pi = np.array([0.5, 0.5])
mu, sd = np.array([0.0, 0.0]), np.array([0.5, 2.0])
T = 500
z = np.zeros(T, int)
for t in range(1, T):
    z[t] = rng.choice(2, p=A[z[t - 1]])
x = rng.normal(mu[z], sd[z])

def log_gauss(x, m, s):
    return -0.5 * np.log(2 * np.pi * s**2) - (x - m) ** 2 / (2 * s**2)

def viterbi(x, A, pi, mu, sd):
    logB, logA = log_gauss(x[:, None], mu, sd), np.log(A)
    d = np.log(pi) + logB[0]
    back = np.zeros((len(x), len(pi)), int)
    for t in range(1, len(x)):
        s = d[:, None] + logA                    # from i -> to j
        back[t] = s.argmax(0)
        d = s.max(0) + logB[t]
    path = [d.argmax()]
    for t in range(len(x) - 1, 0, -1):
        path.append(back[t, path[-1]])
    return np.array(path[::-1])

zhat = viterbi(x, A, pi, mu, sd)
naive = (np.abs(x) > 1.0).astype(int)           # per-point threshold, no memory
print(f"viterbi={np.mean(zhat == z):.3f} threshold={np.mean(naive == z):.3f}")

# --- Kalman filter: 1D position, constant-velocity model ---
dt, n = 1.0, 100
F = np.array([[1, dt], [0, 1]])                 # state (pos, vel)
H = np.array([[1.0, 0.0]])                      # observe position only
Q, R = 0.01 * np.eye(2), np.array([[4.0]])      # process / measurement noise
truth = np.zeros((n, 2)); truth[0] = [0, 1]
for t in range(1, n):
    truth[t] = F @ truth[t - 1] + rng.multivariate_normal([0, 0], Q)
meas = truth[:, 0] + rng.normal(0, 2.0, n)

xk, P, est = np.zeros(2), 10 * np.eye(2), []
for zk in meas:
    xk, P = F @ xk, F @ P @ F.T + Q                       # predict
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)          # gain
    xk = xk + (K @ (zk - H @ xk)).ravel()                 # update
    P = (np.eye(2) - K @ H) @ P
    est.append(xk[0])
rmse = lambda a: np.sqrt(np.mean((np.array(a)[10:] - truth[10:, 0]) ** 2))
print(f"kalman RMSE={rmse(est):.3f} raw RMSE={rmse(meas):.3f} "
      f"vel est={xk[1]:.3f} true={truth[-1, 1]:.3f}")
```
Output (numpy 2.5.1): Viterbi recovers the hidden regime 94.6% of the time
against 79.8% for a per-point threshold on `|x|`, because the HMM knows regimes
last. The Kalman filter cuts position RMSE from 2.036 (raw sensor) to 0.937,
and estimates velocity (0.566 vs true 0.762), which no sensor measured.

Here the parameters are known. On real data, fit them: `hmmlearn.hmm.GaussianHMM(n_components=2).fit(X)` (Baum-Welch), or
`statsmodels` `MarkovRegression(..., switching_variance=True)` for regime
switching; for Kalman, tune `Q`/`R` from data (or by maximum likelihood with
`statsmodels` `UnobservedComponents`).

## Choosing / trade-offs
- Discrete states: HMM. Continuous state, roughly linear: Kalman filter.
  Mildly nonlinear: EKF/UKF. Strongly nonlinear or multimodal: particle filter.
- Online (real time): filter. Offline analysis: smoother (better, uses future).
- Labeled sequences and features: a CRF or neural sequence model often beats a
  generative HMM; HMMs win with little or no labeled data.
- Number of HMM states is a modeling choice; compare with BIC and check the
  states mean something.

## Gotchas
- Baum-Welch finds a local optimum and the state labels are arbitrary (state 0
  in one fit can be state 1 in the next). Run several seeds; map states by
  their emission means.
- Too-high `R` (or too-low `Q`) makes the filter sluggish and lag behind real
  changes; the reverse just passes the noise through. Plot the innovations
  (`z - H x`); they should look like white noise.
- The filter's velocity estimate starts poor: discard the first steps (the
  example skips 10) or initialize `P` honestly large.
- Plain HMM durations are geometric; if regimes have a minimum length, use a
  semi-Markov HMM or add sticky self-transitions.
- Non-uniform time steps: rebuild `F` and `Q` with the actual `dt` each step.

## Related
- [[time-series-forecasting]] - forecasting, which often uses state-space forms.
- [[anomaly-detection]] - state changes and large innovations as anomalies.
- [[state-space-models]] - the deep-learning descendants (S4, Mamba).
- [[bayesian-and-gaussian-processes]] - the same Bayesian updating idea.
- [[video-analytics]] - Kalman-based multi-object tracking.
- [[manufacturing-iot]], [[finance]] - sensor and regime scenarios.

## References
- Rabiner, A Tutorial on Hidden Markov Models: https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
- hmmlearn documentation: https://hmmlearn.readthedocs.io/en/latest/
- Labbe, Kalman and Bayesian Filters in Python: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
- statsmodels MarkovRegression: https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html
