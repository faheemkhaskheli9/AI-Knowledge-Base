---
title: Kalman filter from scratch (predict/update, RTS smoother, noise tuning)
category: ml
tags: [kalman-filter, state-space-model, rts-smoother, bayesian-filtering, tracking, sensor-fusion, time-series, numpy, from-scratch, ml-basics]
use_cases:
  - "implement a linear Kalman filter and Rauch-Tung-Striebel smoother in NumPy"
  - "track position and velocity from noisy position-only measurements"
  - "check whether a filter's uncertainty is honest (consistency) and tune Q and R"
  - "explain the Kalman gain, predict/update steps and filter vs smoother in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
  - https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
  - https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
---

# Kalman filter from scratch (predict/update, RTS smoother, noise tuning)

## Summary
A Kalman filter estimates a hidden continuous state (position, velocity, a level and trend) from noisy measurements. It assumes linear dynamics and Gaussian noise. Each step predicts the state forward with the motion model, then corrects the prediction with the new measurement, weighting the two by their uncertainties. The NumPy version below tracks a 1-D object from position-only readings. It halves the measurement error, estimates velocity it never measures, and its own 2-sigma band covers 96.5% of the true errors. The offline RTS smoother halves the error again by also using future measurements. Mis-tuned process noise `Q` is the usual failure: too small and the filter ignores the data and drifts away.

## Key concepts
- **State-space model.** `x_t = F x_{t−1} + w`, `w ~ N(0, Q)` (dynamics); `z_t = H x_t + v`, `v ~ N(0, R)` (measurement). The filter keeps a Gaussian belief `N(x, P)` over the state.
- **Predict.** `x ← F x`, `P ← F P Fᵀ + Q`. Uncertainty grows by the process noise.
- **Update.** Innovation `y = z − H x`, its covariance `S = H P Hᵀ + R`, gain `K = P Hᵀ S⁻¹`, then `x ← x + K y`, `P ← (I − K H) P`.
- **Kalman gain.** How far to move toward the measurement. Large `P` or small `R` gives a gain near 1 (trust the sensor); small `P` or large `R` gives a gain near 0 (trust the model). For a fixed model it converges to a steady-state value.
- **Hidden components.** A constant-velocity model estimates velocity from position-only data because `P` couples the two.
- **RTS smoother.** A backward pass after filtering: `x_t^s = x_t + G (x_{t+1}^s − x̂_{t+1|t})`, `G = P_t Fᵀ P_{t+1|t}⁻¹`. It conditions on all measurements, so it only works offline.
- **Consistency.** If `Q` and `R` are right, errors fall inside `±2√P` about 95% of the time and the innovations are white. This is the main tuning diagnostic.

## When to use / scenarios
- Learning: Bayesian inference with closed-form Gaussian updates; the continuous-state counterpart of the HMM forward algorithm ([[hmm-from-scratch]]).
- Interviews: "derive the Kalman gain", "what do Q and R do", "filter vs smoother", "what if the system is non-linear".
- Tracking and navigation: object tracking (SORT-style multi-object trackers use a Kalman filter per box), GPS/IMU fusion, robot localisation.
- Time series: local level and trend models, smoothing noisy sensor streams, online estimation of a slowly drifting quantity, handling missing readings (skip the update step).
- Not for: strongly non-linear dynamics or multi-modal beliefs. Use an extended or unscented Kalman filter for mild non-linearity, a particle filter for multi-modal cases, and an HMM when the state is discrete.

## Setup & code
`pip install numpy`. Runs in under a second on CPU.

```python
import numpy as np


def kalman_filter(zs, F, H, Q, R, x0, P0):
    """Linear Kalman filter. Returns filtered means, covariances, and predicted (prior) ones for smoothing."""
    x, P = x0, P0
    xs, Ps, xps, Pps = [], [], [], []
    I = np.eye(len(x0))
    for z in zs:
        x, P = F @ x, F @ P @ F.T + Q                    # predict
        xps.append(x); Pps.append(P)
        S = H @ P @ H.T + R                               # innovation covariance
        K = P @ H.T @ np.linalg.inv(S)                    # Kalman gain
        x = x + K @ (z - H @ x)                           # update with the innovation
        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  # Joseph form: stays symmetric PSD
        xs.append(x); Ps.append(P)
    return np.array(xs), np.array(Ps), np.array(xps), np.array(Pps)


def rts_smoother(xs, Ps, xps, Pps, F):
    """Rauch-Tung-Striebel backward pass: uses future measurements too (offline only)."""
    xs_s, Ps_s = xs.copy(), Ps.copy()
    for t in range(len(xs) - 2, -1, -1):
        G = Ps[t] @ F.T @ np.linalg.inv(Pps[t + 1])
        xs_s[t] = xs[t] + G @ (xs_s[t + 1] - xps[t + 1])
        Ps_s[t] = Ps[t] + G @ (Ps_s[t + 1] - Pps[t + 1]) @ G.T
    return xs_s, Ps_s


# constant-velocity model: state [position, velocity], we only measure a noisy position
dt, T, q, r = 1.0, 200, 0.01, 4.0
F = np.array([[1, dt], [0, 1]])
H = np.array([[1.0, 0.0]])
Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])   # white-noise acceleration
R = np.array([[r]])
rng = np.random.default_rng(0)
x_true = np.zeros((T, 2)); x = np.array([0.0, 1.0])
for t in range(T):
    x = F @ x + rng.multivariate_normal([0, 0], Q); x_true[t] = x
zs = x_true[:, :1] + rng.normal(0, np.sqrt(r), (T, 1))

x0, P0 = np.array([0.0, 0.0]), np.diag([10.0, 10.0])
xs, Ps, xps, Pps = kalman_filter(zs, F, H, Q, R, x0, P0)
xs_s, _ = rts_smoother(xs, Ps, xps, Pps, F)
rmse = lambda a: np.sqrt(np.mean((a - x_true[:, 0]) ** 2))
print(f"position RMSE: raw measurements {rmse(zs[:, 0]):.3f} | filter {rmse(xs[:, 0]):.3f} | RTS smoother {rmse(xs_s[:, 0]):.3f}")
print(f"velocity RMSE: filter {np.sqrt(np.mean((xs[:, 1] - x_true[:, 1])**2)):.3f} (never measured directly)")

# consistency check: errors should fall inside the filter's own 2-sigma band ~95% of the time
err, sd = xs[:, 0] - x_true[:, 0], np.sqrt(Ps[:, 0, 0])
print(f"share of position errors within 2 sigma: {np.mean(np.abs(err) < 2 * sd):.3f} (expect ~0.95)")
print(f"steady-state gain K: {(Ps[-1] @ H.T / r).ravel().round(3)} | position sd {sd[-1]:.3f} vs measurement sd {np.sqrt(r):.3f}")

# mis-tuned noise: Q too small = sluggish over-confident filter, too large = noisy filter
for scale in [1e-4, 1, 1e3]:
    xs_m, Ps_m, _, _ = kalman_filter(zs, F, H, Q * scale, R, x0, P0)
    e = xs_m[:, 0] - x_true[:, 0]
    print(f"Q x {scale:g}: position RMSE {rmse(xs_m[:, 0]):.3f} | within 2 sigma {np.mean(np.abs(e) < 2 * np.sqrt(Ps_m[:, 0, 0])):.3f}")
```

Output (numpy 2.5):
```
position RMSE: raw measurements 1.999 | filter 1.089 | RTS smoother 0.575
velocity RMSE: filter 0.271 (never measured directly)
share of position errors within 2 sigma: 0.965 (expect ~0.95)
steady-state gain K: [0.271 0.043] | position sd 1.041 vs measurement sd 2.000
Q x 0.0001: position RMSE 17.997 | within 2 sigma 0.190
Q x 1: position RMSE 1.089 | within 2 sigma 0.965
Q x 1000: position RMSE 1.718 | within 2 sigma 0.975
```

The filter cuts position error from 2.0 (raw sensor) to 1.09 and recovers velocity to within 0.27 units/step without ever measuring it. Its reported uncertainty is honest: 96.5% of errors land inside its own 2-sigma band. The gain settles at 0.27, so each new reading moves the position estimate about a quarter of the way toward it. The RTS smoother, which also sees later readings, halves the error again to 0.58. With `Q` 10,000× too small the filter believes the object moves at exactly constant velocity, stops listening to the sensor, and drifts to an RMSE of 18. Only 19% of its errors fall inside its (far too narrow) band, which is how you spot over-confidence. With `Q` 1,000× too large it trusts every noisy reading and its error rises to 1.72. Its band is wide enough to stay "consistent", so the 2-sigma check alone does not catch this case; compare RMSE or innovation whiteness too.

## Choosing / trade-offs
- **Filter vs smoother.** Use the filter online (each estimate uses past data only). Use RTS for offline analysis, backfilling, or training labels; it is always at least as accurate.
- **Tuning Q and R.** Set `R` from the sensor spec or the variance of readings of a stationary target. Tune `Q` with the consistency check, innovation whiteness, or by maximising the innovation log-likelihood (`statsmodels` and `pykalman` can fit them by EM or MLE).
- **Model choice.** Constant position (random walk), constant velocity, or constant acceleration. A richer model tracks manoeuvres better but needs more data to pin down.
- **Non-linear systems.** EKF linearises with Jacobians and is cheap but can diverge. UKF propagates sigma points, needs no Jacobians and is usually more robust. A particle filter handles multi-modal beliefs at much higher cost.
- **Library.** `filterpy` for teaching-style filters (KF, EKF, UKF), `pykalman` for EM fitting and smoothing, `statsmodels` `UnobservedComponents` for structural time-series models.

## Gotchas
- Use the Joseph form (or symmetrise `P`) for the covariance update. The short form `(I − K H) P` loses symmetry and positive-definiteness through rounding over long runs.
- Use `np.linalg.solve` or a Cholesky factor in place of `inv(S)` when the measurement dimension is large.
- Under-estimated `Q` is the most common bug: the filter becomes over-confident, the gain goes to zero, and it ignores the data (RMSE 18 above).
- Measurement units and time step must match `F`, `Q` and `R`. With irregular sampling, rebuild `F` and `Q` from each step's `dt`.
- Missing measurement: run predict only and skip the update. Do not feed a zero.
- An outlier pulls the estimate hard because Gaussian noise has thin tails. Gate measurements by the Mahalanobis distance `yᵀ S⁻¹ y` (chi-squared test) before updating.
- A vague initial `P0` is fine; a confident wrong `x0` with a small `P0` takes many steps to recover.

## Related
- [[hidden-markov-models-and-kalman-filters]] - library usage and the discrete-state counterpart.
- [[hmm-from-scratch]] - same predict/correct recursion with a discrete hidden state.
- [[time-series-forecasting]] - state-space models for level, trend and seasonality.
- [[bayesian-and-gaussian-processes]] - Gaussian conditioning, the maths behind the update step.
- [[probabilistic-graphical-models]] - the Kalman filter as a linear-Gaussian dynamic Bayesian network.
- [[state-space-models]] - neural sequence models (S4, Mamba) built on the same linear recurrence.

## References
- Welch and Bishop, "An Introduction to the Kalman Filter" (UNC TR 95-041): https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
- Särkkä, Bayesian Filtering and Smoothing (Cambridge University Press, 2013), author's online version: https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
- FilterPy KalmanFilter documentation: https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
