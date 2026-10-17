---
title: Monocular depth estimation (Depth Anything, Depth Pro, MoGe)
category: cv
tags: [depth-estimation, monocular-depth, metric-depth, depth-anything, depth-pro, moge, point-cloud, 3d]
use_cases:
  - "estimate how far objects are from a single phone or dashcam image"
  - "turn product photos into rough 3D point clouds for AR previews"
  - "add depth-aware blur, relighting or occlusion to a photo or video editing app"
  - "give a robot or drone a cheap depth signal when there is no LiDAR"
  - "measure room or object sizes from one image when a scale reference is available"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/DepthAnything/Depth-Anything-V2
  - https://github.com/ByteDance-Seed/Depth-Anything-3
  - https://github.com/apple/ml-depth-pro
  - https://github.com/microsoft/MoGe
  - https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation
---

# Monocular depth estimation (Depth Anything, Depth Pro, MoGe)

## Summary
Monocular depth estimation predicts a per-pixel depth map from one RGB image,
with no stereo pair or LiDAR. Foundation models trained on huge mixed datasets
(Depth Anything V2/3, Apple Depth Pro, Microsoft MoGe) now give sharp, stable
depth zero-shot. The key decision is whether you need **relative** depth
(ordering and shape, unknown scale) or **metric** depth (metres), because they
are different models with different failure modes and licences.

## Key concepts
- **Relative (affine-invariant) depth**: correct up to an unknown scale and
  shift; many models output inverse depth (disparity), where larger = closer.
  Good for effects, segmentation cues and ordering, not for measuring.
- **Metric depth**: absolute distance in metres. Single-image metric depth has
  to guess scale from scene cues and camera focal length, so error grows with
  unusual cameras, zoom and scenes unlike the training data.
- **Scale-and-shift alignment**: turn relative depth into metric with a few
  known depths (LiDAR points, a known object size, SfM points) by least squares;
  this is often more accurate than trusting a metric model blindly.
- **Intrinsics matter**: back-projecting depth to 3D needs focal length and
  principal point. Depth Pro predicts focal length; MoGe predicts intrinsics
  (field of view) and a point map directly.
- **Point maps** (MoGe, Depth Anything 3): predict 3D points per pixel instead of
  depth only, which avoids a separate intrinsics guess.
- **Multi-view**: Depth Anything 3 also handles several views, estimating camera
  pose and consistent depth, which bridges into
  [[gaussian-splatting-and-nerf]] style reconstruction.
- **Metrics**: AbsRel (mean |d - d*| / d*) and delta1 (share of pixels with
  max(d/d*, d*/d) < 1.25). Always say whether they were computed after
  scale/shift alignment.

Models as of 2026-10 (from each repo):

| Model | Output | Sizes | Licence |
|---|---|---|---|
| Depth Anything V2 | relative; metric variants (Small/Base) in `metric_depth/` | Small 24.8M, Base 97.5M, Large 335.3M | Small Apache-2.0; Base/Large CC-BY-NC-4.0 |
| Depth Anything 3 | depth, camera pose, ray maps, 3D Gaussians (main series); metric (DA3METRIC-LARGE); mono relative (DA3MONO-LARGE) | 0.08B-1.40B | SMALL, BASE, METRIC-LARGE, MONO-LARGE Apache 2.0; LARGE, GIANT, NESTED CC BY-NC 4.0 |
| Apple Depth Pro | metric depth in metres + focal length in pixels | one model | custom licence file in the repo; read it |
| MoGe-2 / MoGe-3 | metric point map, depth, intrinsics, mask, optional normals | 35M-1.25B | code MIT; weights licence not stated on the README |

## When to use / scenarios
- Photo/video apps: portrait blur, fog, relighting, 2D-to-3D parallax.
- AR and e-commerce: place a product in a room, rough room scale (with a scale
  reference).
- Robotics and drones: cheap depth prior or obstacle cue alongside a real
  sensor; never as the only safety sensor.
- Mapping/inspection: densify sparse LiDAR or SfM points with predicted depth.
- Dataset tooling: pseudo-depth for training smaller task models.
- Not for: precise measurement without a reference (use stereo, LiDAR or
  structured light), or full 3D scenes from many photos (use
  [[gaussian-splatting-and-nerf]]).

## Setup & code
Core idea, runnable with numpy: align relative depth to a few metric points,
back-project to a point cloud, and score it.
```python
import numpy as np

rng = np.random.default_rng(0)
H, W = 48, 64
fx = fy = 60.0; cx, cy = W / 2, H / 2

# Ground-truth metric depth: a tilted floor plane plus a box (metres).
v, u = np.mgrid[0:H, 0:W].astype(float)
depth_gt = 2.0 + 0.05 * v
depth_gt[15:30, 20:40] = 1.5

# A "relative" model output: unknown scale/shift, in inverse depth, with noise.
disp_rel = 3.0 * (1.0 / depth_gt) + 0.7 + rng.normal(0, 0.005, depth_gt.shape)

def align_scale_shift(pred, target, mask):
    """Least squares s, t minimising ||s*pred + t - target|| on mask."""
    A = np.stack([pred[mask], np.ones(mask.sum())], axis=1)
    s, t = np.linalg.lstsq(A, target[mask], rcond=None)[0]
    return s * pred + t

# 20 sparse metric anchors (e.g. LiDAR returns), aligned in disparity space.
mask = np.zeros_like(depth_gt, bool)
mask.flat[rng.choice(H * W, 20, replace=False)] = True
disp_metric = align_scale_shift(disp_rel, 1.0 / depth_gt, mask)
depth_pred = 1.0 / np.clip(disp_metric, 1e-6, None)

def absrel(p, g):
    return float(np.mean(np.abs(p - g) / g))

def delta1(p, g):
    return float(np.mean(np.maximum(p / g, g / p) < 1.25))

# Back-project to a 3D point cloud with pinhole intrinsics.
X = (u - cx) / fx * depth_pred
Y = (v - cy) / fy * depth_pred
points = np.stack([X, Y, depth_pred], axis=-1).reshape(-1, 3)

print(f"AbsRel={absrel(depth_pred, depth_gt):.4f} delta1={delta1(depth_pred, depth_gt):.3f}")
print("point cloud:", points.shape, "box depth ~", round(float(np.median(depth_pred[15:30, 20:40])), 3))
assert absrel(depth_pred, depth_gt) < 0.02
```

Pretrained model through Transformers (pipeline from the Depth Anything V2
README; Small is the Apache-2.0 checkpoint):
<!-- skip-check: downloads model weights from Hugging Face; needs torch -->
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
depth = pipe(Image.open("room.jpg"))["depth"]   # PIL image of relative depth
depth.save("room_depth.png")
```

Metric depth plus focal length with Depth Pro (from its README, after
`pip install -e .` and `source get_pretrained_models.sh`):
<!-- skip-check: needs the cloned ml-depth-pro repo, checkpoints and torch -->
```python
import depth_pro

model, transform = depth_pro.create_model_and_transforms()
model.eval()
image, _, f_px = depth_pro.load_rgb("room.jpg")
prediction = model.infer(transform(image), f_px=f_px)
print(prediction["depth"].shape, prediction["focallength_px"])   # metres, pixels
```

## Choosing / trade-offs
- **Relative vs metric**: relative models are sharper and more robust; metric
  models save you a reference but can be off by a global scale on unusual
  cameras. With any metric anchor, prefer relative + alignment.
- **Licence**: commercial products should start from the Apache-2.0
  checkpoints (Depth Anything V2 Small, DA3 SMALL/BASE/METRIC-LARGE/MONO-LARGE)
  or check Depth Pro's and MoGe's weight terms; the larger Depth Anything models
  are non-commercial ([[model-licenses]]).
- **Speed vs detail**: small ViT-S models run in real time on a GPU and are
  candidates for edge export ([[edge-on-device]]); giant models give finer
  edges for offline work.
- **Single image vs video**: per-frame depth flickers; use a video or
  multi-view model (DA3, DA3-Streaming) or temporal smoothing for video.
- **Depth vs point map**: if you need 3D geometry, a point-map model (MoGe,
  DA3) avoids guessing intrinsics separately.

## Gotchas
- Mixing up depth and disparity (inverse depth): aligning in the wrong space
  bends planes. Check whether "bigger" means nearer or farther.
- Visualised PNGs are normalised per image; never read metres from them.
- Sky, glass, mirrors and water get confident nonsense; mask them (DA3 lists
  sky segmentation for some models).
- Metric error grows with distance; report error by range bucket.
- Transformers pipeline and native repo outputs differ slightly (OpenCV vs PIL
  resizing, per the Depth Anything V2 README).
- Cropping or resizing changes intrinsics; scale fx, fy, cx, cy with the image.
- Benchmarks often report aligned metrics; a "metric" claim needs unaligned
  numbers on your own camera.

## Related
- [[gaussian-splatting-and-nerf]] - multi-view 3D reconstruction; depth priors help it.
- [[segmentation]] - combine masks with depth to measure objects.
- [[object-detection]] - distance to detected objects.
- [[video-analytics]] - temporal consistency on video.
- [[edge-on-device]] - running small depth models on devices.
- [[model-licenses]] - non-commercial vs Apache checkpoints.
- [[open-weight-model-landscape]] - where depth models sit among current open releases.

## References
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- Apple Depth Pro: https://github.com/apple/ml-depth-pro
- MoGe: https://github.com/microsoft/MoGe
- Transformers depth task guide: https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation
