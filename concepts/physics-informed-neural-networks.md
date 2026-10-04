---
title: Physics-informed neural networks and scientific ML
category: concepts
tags: [pinn, scientific-ml, physics-informed, differential-equations, surrogate-models, neural-operators, fno, inverse-problems, autograd, pytorch]
use_cases:
  - "fit a model to sparse, noisy sensor data while respecting a known physical law"
  - "estimate an unknown physical parameter (damping, diffusivity, friction) from measurements"
  - "build a fast surrogate for a slow simulator (CFD, FEM, heat transfer)"
  - "extrapolate beyond the measured range where a plain neural net would guess"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.sciencedirect.com/science/article/pii/S0021999118307125
  - https://arxiv.org/abs/2010.08895
  - https://github.com/lululxvi/deepxde
  - https://docs.nvidia.com/physicsnemo/latest/index.html
---

# Physics-informed neural networks and scientific ML

## Summary
A physics-informed neural network (PINN) is a network `u(t, x)` trained on two
losses: fit the measured data, and make the residual of a known differential
equation zero at many "collocation" points, computed with autograd. The
physics acts as a strong regularizer, so a PINN learns from far less data
than a plain network and can recover unknown equation parameters (inverse
problems). Neural operators (FNO, DeepONet) are the related tool for learning
fast surrogates of whole simulators from simulation data.

## Key concepts
- **Residual loss:** for an ODE/PDE `N[u] = 0`, compute derivatives of the
  network output with `torch.autograd.grad(..., create_graph=True)` and penalize
  `N[u]^2` at collocation points; no labels needed there.
- **Boundary/initial conditions:** either extra loss terms or hard-coded into
  the network output (e.g. `u = u0 + t * net(t)`), which is more reliable.
- **Inverse problems:** make unknown coefficients learnable parameters; data
  plus physics pin them down.
- **Loss balancing:** data, residual and boundary terms have different scales;
  weights (manual, or adaptive schemes) decide success or failure.
- **Activations:** use smooth ones (`tanh`, `sin`, SiLU); ReLU has zero second
  derivative and breaks second-order equations.
- **Neural operators:** FNO and DeepONet learn a map from inputs (initial
  condition, geometry, coefficients) to the full solution field, trained on
  simulator outputs; one forward pass replaces a long solve.
- **Hybrid / grey-box models:** a known physics model plus a learned
  correction term (universal differential equations).

## When to use / scenarios
- Manufacturing/energy: heat transfer, vibration, battery or fluid models
  with few sensors ([[manufacturing-iot]]).
- Parameter identification: estimate material properties, friction, damping,
  reaction rates from experiments.
- Surrogates for design loops: replace a slow CFD/FEM solve inside an
  optimizer ([[black-box-and-evolutionary-optimization]]) with a neural operator.
- Biomedical and geoscience inverse problems (flow, diffusion, seismic).
- NOT a replacement for a classical solver on a well-posed forward problem:
  finite-element/finite-volume solvers are faster and more accurate there.
  NOT useful when you do not trust the equation; use plain ML
  ([[time-series-forecasting]]) or a grey-box model instead.

## Setup & code
```bash
pip install torch
```
Damped oscillator `u'' + 2ζωu' + ω²u = 0`; 10 noisy points on the first 40%
of the time range; learn `u(t)` on the whole range and the unknown damping ζ:
```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
w, z_true = 20.0, 0.1

def exact(t):
    wd = w * math.sqrt(1 - z_true**2)
    return torch.exp(-z_true * w * t) * (torch.cos(wd * t) + z_true * w / wd * torch.sin(wd * t))

t_obs = torch.linspace(0, 0.4, 10).view(-1, 1)          # data only on the first 40%
u_obs = exact(t_obs) + 0.01 * torch.randn_like(t_obs)
t_col = torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True)   # physics points
t_test = torch.linspace(0, 1, 500).view(-1, 1)

def mlp():
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

def train(use_physics):
    net = mlp()
    log_z = torch.zeros(1, requires_grad=True)           # learnable physics parameter
    opt = torch.optim.Adam(list(net.parameters()) + [log_z], lr=1e-3)
    for _ in range(15000):
        loss = ((net(t_obs) - u_obs) ** 2).mean()
        if use_physics:
            u = net(t_col)
            du = torch.autograd.grad(u, t_col, torch.ones_like(u), create_graph=True)[0]
            d2u = torch.autograd.grad(du, t_col, torch.ones_like(du), create_graph=True)[0]
            z = torch.exp(log_z)
            residual = d2u + 2 * z * w * du + w**2 * u
            loss = loss + 1e-4 * (residual**2).mean()      # scale: w^2 = 400
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        err = (net(t_test) - exact(t_test)).abs().mean().item()
    return err, torch.exp(log_z).item()

err_plain, _ = train(False)
err_pinn, z_hat = train(True)
print(f"plain MLP test MAE={err_plain:.3f}")
print(f"PINN      test MAE={err_pinn:.3f}  damping estimate={z_hat:.3f} (true {z_true})")
```
Output (torch 2.13.0 CPU, a few minutes): the plain network fits the 10
points but has mean absolute error 0.473 over the full range; the PINN cuts
it to 0.070 and estimates damping 0.143 against a true 0.1. The parameter is
in the right range but not exact from 10 points; more data, more steps or a
final L-BFGS phase (standard PINN practice) tighten it.

Libraries: DeepXDE (PINNs, many backends), NVIDIA PhysicsNeMo (PINNs and
neural operators at scale), `neuraloperator` (FNO).

## Choosing / trade-offs
- Known equation, sparse data, unknown parameters: PINN (or a classical
  solver + parameter fit with SciPy, which is often simpler and more accurate
  for small ODEs; try it first).
- Many repeated solves with varying inputs (design, control, uncertainty
  quantification): neural operator trained on simulator runs.
- Equation partly known: grey-box model, physics term plus a learned residual.
- Equation unknown, lots of data: ordinary ML/sequence models; a PINN adds
  nothing without trusted physics.

## Gotchas
- Plain PINNs struggle with high frequencies, sharp fronts and long time
  ranges (spectral bias); split the domain in time, use Fourier features, or
  add collocation points where the residual is large.
- Loss weights decide everything: an unscaled residual (here `w² = 400`
  times larger than the data term) drowns the data loss. Non-dimensionalize
  the equation first.
- Adam alone often stalls; finish with L-BFGS (`torch.optim.LBFGS`).
- Training is slow: second derivatives through autograd at every step cost
  several forward/backward passes. Benchmark against a classical solver before
  committing.
- A low residual on the collocation points does not guarantee a correct
  solution between them; validate against held-out data or a reference solve.

## Related
- [[neural-network-fundamentals]] - autograd and backprop used for the residual.
- [[math-for-machine-learning]] - derivatives and optimization background.
- [[hidden-markov-models-and-kalman-filters]] - classical state estimation with known dynamics.
- [[black-box-and-evolutionary-optimization]] - optimizing designs with a surrogate.
- [[bayesian-and-gaussian-processes]] - GP surrogates with uncertainty.
- [[manufacturing-iot]] - sensor and process scenarios.

## References
- Raissi, Perdikaris, Karniadakis, Physics-informed neural networks (J. Comput. Phys. 2019): https://www.sciencedirect.com/science/article/pii/S0021999118307125
- Li et al., Fourier Neural Operator for Parametric PDEs: https://arxiv.org/abs/2010.08895
- DeepXDE: https://github.com/lululxvi/deepxde
- NVIDIA PhysicsNeMo documentation: https://docs.nvidia.com/physicsnemo/latest/index.html
