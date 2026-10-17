---
title: 3D Gaussian Splatting and NeRF (scene reconstruction from photos)
category: cv
tags: [gaussian-splatting, 3dgs, nerf, 3d-reconstruction, novel-view-synthesis, colmap, nerfstudio, gsplat, photogrammetry]
use_cases:
  - "turn a phone video of a room or product into an interactive 3D scene for the web"
  - "build a digital twin of a site or facility from drone or walk-through footage"
  - "create 3D product views for e-commerce from a turntable photo set"
  - "capture real locations for VFX, games or virtual production backgrounds"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/graphdeco-inria/gaussian-splatting
  - https://github.com/nerfstudio-project/nerfstudio
  - https://github.com/nerfstudio-project/gsplat
  - https://github.com/colmap/colmap
  - https://github.com/ArthurBrussee/brush
  - https://github.com/playcanvas/supersplat
  - https://arxiv.org/abs/2308.04079
  - https://arxiv.org/abs/2003.08934
---

# 3D Gaussian Splatting and NeRF (scene reconstruction from photos)

## Summary
Given many overlapping photos (or video frames) of a static scene, NeRF and 3D
Gaussian Splatting (3DGS) learn a 3D representation that renders new viewpoints
photorealistically. NeRF stores the scene in a neural network queried by ray
marching; 3DGS stores millions of explicit 3D Gaussians that are rasterised in
real time. In 2026 3DGS is the default for most capture-and-view work, with
NeRF-style methods kept for research and some quality niches. Both start from
camera poses, usually computed with COLMAP.

## Key concepts
- **Pipeline**: capture -> camera poses via Structure-from-Motion (COLMAP;
  GLOMAP is its global SfM variant) -> optimise the scene against the photos ->
  view, edit, export.
- **NeRF** (Mildenhall et al. 2020): an MLP maps 3D position + view direction to
  density and colour; pixels come from volume rendering along rays
  (weights w_i = T_i * (1 - exp(-sigma_i * delta_i))). Slow to render unless
  accelerated (hash grids as in instant-ngp, nerfacto).
- **3D Gaussian Splatting** (Kerbl et al. 2023): each Gaussian has position,
  covariance (scale + rotation), opacity and view-dependent colour (spherical
  harmonics). Gaussians are projected to 2D ellipses, depth-sorted and
  alpha-composited front to back. Training adds/splits/prunes Gaussians
  ("densification"); the reference trainer defaults to 30,000 iterations.
- **Outputs**: splats are usually `.ply` files (large); compressed formats and
  web viewers exist. A splat is not a mesh: getting watertight meshes needs
  extra surface-reconstruction methods.
- **Feed-forward reconstruction**: newer models predict poses, depth or even
  Gaussians directly from a few images in one pass (for example Depth Anything 3
  lists 3D Gaussian estimation); faster but lower fidelity than per-scene
  optimisation. See [[depth-estimation]].
- **Static-scene assumption**: moving people, cars or changing light break the
  photo-consistency both methods rely on.

## When to use / scenarios
- Real estate, hospitality, museums: walk-through scenes in a browser.
- E-commerce: 3D product spins from a turntable shoot.
- Construction, facilities, inspection: visual digital twins from drone or
  handheld video (for measurement-grade geometry, prefer photogrammetry meshes
  or LiDAR).
- VFX, games, virtual production: photoreal backgrounds and props.
- Robotics and simulation: realistic environments for testing perception.
- Not for: dynamic scenes (needs 4D methods), precise CAD-grade dimensions,
  or a single photo (use [[depth-estimation]] for a 2.5D result).

## Setup & code
Core rendering idea, runnable with numpy: project 3D Gaussians through a pinhole
camera (EWA: Sigma2D = J W Sigma W^T J^T), sort by depth and alpha-composite.
```python
import numpy as np

H, W, f = 40, 60, 50.0
cx, cy = W / 2, H / 2

def quat_to_rot(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

# Three Gaussians in camera space: mean, scales, rotation, opacity, RGB.
gaussians = [
    (np.array([0.0, 0.0, 4.0]), np.array([0.6, 0.3, 0.3]), np.array([1, 0, 0, 0.3]), 0.9, (1.0, 0.2, 0.2)),
    (np.array([0.4, 0.2, 3.0]), np.array([0.2, 0.2, 0.2]), np.array([1, 0, 0, 0.0]), 0.8, (0.2, 0.9, 0.2)),
    (np.array([-0.6, -0.3, 5.0]), np.array([0.5, 0.5, 0.1]), np.array([1, 0.4, 0, 0]), 0.7, (0.2, 0.3, 1.0)),
]

def project(mean, scale, quat):
    R = quat_to_rot(np.asarray(quat, float))
    cov3d = R @ np.diag(scale**2) @ R.T
    x, y, z = mean
    J = np.array([[f / z, 0, -f * x / z**2],
                  [0, f / z, -f * y / z**2]])         # Jacobian of the projection
    cov2d = J @ cov3d @ J.T + 0.3 * np.eye(2)          # low-pass term, as in 3DGS
    return np.array([f * x / z + cx, f * y / z + cy]), cov2d, z

v, u = np.mgrid[0:H, 0:W] + 0.5
pix = np.stack([u, v], -1)
color = np.zeros((H, W, 3)); T = np.ones((H, W))      # transmittance per pixel
for mean, scale, quat, opac, rgb in sorted(gaussians, key=lambda g: g[0][2]):  # front to back
    mu, cov, _ = project(mean, scale, quat)
    d = pix - mu
    power = -0.5 * np.einsum("hwi,ij,hwj->hw", d, np.linalg.inv(cov), d)
    alpha = np.minimum(0.99, opac * np.exp(power))
    color += (T * alpha)[..., None] * np.array(rgb)
    T *= 1 - alpha

print("image", color.shape, "max", color.max().round(3), "min transmittance", T.min().round(3))

# NeRF-style volume rendering weights along one ray, for comparison.
sigma = np.array([0.0, 0.1, 2.0, 5.0, 0.5]); delta = np.full(5, 0.2)
alpha_r = 1 - np.exp(-sigma * delta)
T_r = np.concatenate([[1.0], np.cumprod(1 - alpha_r)[:-1]])
w = T_r * alpha_r
print("ray weights", w.round(3), "sum", w.sum().round(3))
assert 0 < color.max() <= 1.0 and w.sum() <= 1.0
```

Practical pipeline with nerfstudio (Apache-2.0; NVIDIA GPU with CUDA; install
steps in its README):
```bash
ns-process-data video --data walkthrough.mp4 --output-dir data/room   # runs COLMAP
ns-train splatfacto --data data/room          # or splatfacto-big for more Gaussians
ns-export gaussian-splat --load-config outputs/room/splatfacto/<run>/config.yml --output-dir exports/splat
ns-train nerfacto --data data/room            # NeRF baseline on the same data
```
Reference 3DGS code (graphdeco-inria): put photos in `<scene>/input`, run
`python convert.py -s <scene>` (COLMAP), then `python train.py -s <scene>`.
Library route: `pip install gsplat` (Apache-2.0 CUDA rasteriser with Python
bindings) to build your own trainer.

## Choosing / trade-offs
| Tool | Licence | Notes |
|---|---|---|
| graphdeco-inria/gaussian-splatting | non-commercial research/evaluation only; commercial licence from Inria | reference implementation; about 24 GB VRAM for paper quality, CUDA compute capability 7.0+ |
| nerfstudio (`splatfacto`, `nerfacto`) | Apache-2.0 | data processing, many methods, web viewer, export |
| gsplat | Apache-2.0 | rasteriser library, used by nerfstudio |
| Brush | Apache-2.0 | WebGPU trainer/viewer on macOS, Windows, Linux, Android and browser; AMD/Intel/Nvidia GPUs |
| SuperSplat | MIT | browser editor for cleaning, cropping and publishing splats |
| COLMAP | new BSD | camera poses (SfM) and classic multi-view stereo |

- **3DGS vs NeRF**: 3DGS renders in real time and trains fast but produces
  large files and floaters; NeRF is compact and smooth but slow to render.
- **Licence**: shipping a product on the reference 3DGS code needs Inria's
  commercial licence; nerfstudio/gsplat/Brush are permissive.
- **Hardware**: CUDA tools need an NVIDIA GPU; Brush runs on other GPUs; cloud
  GPUs for large scenes ([[gpu-cloud-options]]).
- **Quality vs size**: more Gaussians (splatfacto-big) means sharper results and
  bigger files; compress or prune for web delivery.
- **Per-scene optimisation vs feed-forward**: optimisation gives the best
  fidelity; feed-forward models give instant previews from few images.

## Gotchas
Capture tips (the part that decides quality):
- Static scene, fixed exposure and white balance; avoid auto-exposure jumps
  (the reference code has an exposure-compensation option for phone captures).
- Lots of overlap: move slowly, orbit objects at two or three heights, and
  avoid pure rotation in place (SfM needs parallax).
- Sharp frames: good light, fast shutter, drop blurry frames; reference trainers
  work around 1-1.6K pixel images, so huge photos only cost time.
- Textureless walls, reflections, glass and moving foliage cause floaters and
  failed poses; add texture or accept artefacts.
- Areas you never filmed cannot be rendered; plan the viewer's path.

Other traps:
- COLMAP failing silently on part of the set: check how many images registered
  before training.
- Splats are not meshes and do not give physics collisions or CAD measurements.
- People and licence plates in captured scenes are personal data; blur or get
  consent before publishing ([[ai-security-privacy-compliance]]).
- Scene scale is arbitrary unless you add a known-size reference or GPS/LiDAR.

## Related
- [[depth-estimation]] - single-image depth and multi-view depth/pose models.
- [[ransac-homography-from-scratch]] - robust geometry estimation underlying SfM.
- [[diffusion-models]] - generative priors used by few-view and text-to-3D methods.
- [[gpu-cloud-options]] - GPUs for training large scenes.
- [[video-analytics]] - frame extraction and video handling.
- [[3d-generation-models]] - generating 3D assets from text or images instead of capturing them.

## References
- 3D Gaussian Splatting paper (Kerbl et al., SIGGRAPH 2023): https://arxiv.org/abs/2308.04079
- NeRF paper (Mildenhall et al., ECCV 2020): https://arxiv.org/abs/2003.08934
- Reference implementation and licence: https://github.com/graphdeco-inria/gaussian-splatting
- nerfstudio: https://github.com/nerfstudio-project/nerfstudio
- gsplat: https://github.com/nerfstudio-project/gsplat
- COLMAP: https://github.com/colmap/colmap
- Brush: https://github.com/ArthurBrussee/brush
- SuperSplat: https://github.com/playcanvas/supersplat
