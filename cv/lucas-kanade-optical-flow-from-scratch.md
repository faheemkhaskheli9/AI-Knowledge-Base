---
title: Lucas-Kanade optical flow from scratch (Shi-Tomasi corners, Gauss-Newton, image pyramids, aperture problem)
category: cv
tags: [optical-flow, lucas-kanade, klt-tracker, shi-tomasi, good-features-to-track, image-pyramid, structure-tensor, aperture-problem, motion-estimation, opencv, numpy, from-scratch]
use_cases:
  - "track feature points between video frames for stabilisation, visual odometry or motion counting"
  - "choose window size and pyramid levels for cv2.calcOpticalFlowPyrLK"
  - "understand why points on edges or flat regions track badly and how to pick good ones"
  - "estimate how fast objects move in a fixed camera feed without a deep model"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf
  - https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
  - https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
---

# Lucas-Kanade optical flow from scratch (Shi-Tomasi corners, Gauss-Newton, image pyramids, aperture problem)

## Summary
Lucas-Kanade (LK) estimates how a small image patch moved between two frames by assuming the patch keeps its brightness and moves as one rigid shift. Linearising the brightness with its gradient turns this into a 2×2 least-squares system per patch, solved iteratively (Gauss-Newton). Two additions make it a practical tracker (KLT, the algorithm behind `cv2.calcOpticalFlowPyrLK`): choose points where that 2×2 system is well-conditioned (Shi-Tomasi "good features to track"), and run LK coarse to fine on an image pyramid so it can follow motion larger than the window. This file implements all three in NumPy/SciPy and measures the errors on frames with known motion.

## Key concepts
- **Brightness constancy.** `I1(x + d) ≈ I0(x)`. A first-order Taylor expansion gives `∇I · d + I_t ≈ 0` per pixel: one equation, two unknowns. A window of pixels supplies enough equations.
- **The normal equations.** `A d = −b` with `A = Σ ∇I ∇Iᵀ` (the structure tensor) and `b = Σ ∇I · I_t`. Iterate: warp, recompute `I_t`, solve, add the step.
- **Shi-Tomasi corners.** Track only where the smaller eigenvalue of `A` is large. Both eigenvalues small means a flat region (no information). One large and one near zero means an edge.
- **Aperture problem.** On a straight edge, motion along the edge does not change the image, so only the component normal to the edge is observable. `A` is singular in exactly that direction.
- **Pyramids.** The linearisation holds only within about a window radius (and less on fine texture). Halving the image `L` times divides the motion by `2^L`. Solve on the coarsest level, double the estimate, refine on the next.

## When to use / scenarios
- Sparse point tracking in video: digital stabilisation, visual odometry and SLAM front ends, tracking points on a face or a moving part, estimating vehicle speed from a fixed camera.
- Real-time on CPUs and embedded boards, where a deep flow network is too slow. Tracking a few hundred points with `calcOpticalFlowPyrLK` costs milliseconds.
- Not for dense, per-pixel flow with large motion, occlusions or motion blur. Use a learned model (RAFT and its successors) or Farnebäck for a quick dense field. See [[video-analytics]].
- Not for re-identifying objects after they leave the frame or are occluded for long. LK assumes small changes between consecutive frames. Combine detection with a tracker instead, see [[hungarian-algorithm-tracking-from-scratch]].

## Setup & code
NumPy and SciPy (`gaussian_filter`, `map_coordinates` for sub-pixel sampling). Frame 2 is frame 1 shifted by a known sub-pixel flow, so the end-point error (EPE) is exact. Runs in about 2 s.

```python
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

rng = np.random.default_rng(0)
H = W = 160
I0 = gaussian_filter(rng.random((H, W)), 1.5)    # fine random texture
I0 = (I0 - I0.min()) / (I0.max() - I0.min())


def shift(img, u, v):
    """Frame 2: the scene moved by (u, v) pixels, i.e. I1(x, y) = I0(x - u, y - v)."""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]].astype(float)
    return map_coordinates(img, [yy - v, xx - u], order=3, mode="reflect")


def structure_tensor(img, sigma=2.0):
    Iy, Ix = np.gradient(img)
    return (gaussian_filter(Ix * Ix, sigma), gaussian_filter(Ix * Iy, sigma), gaussian_filter(Iy * Iy, sigma))


def good_features(img, n=40, border=24, min_dist=10):
    """Shi-Tomasi: corners are where the smaller eigenvalue of the structure tensor is large."""
    a, b, c = structure_tensor(img)
    lam_min = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2)
    lam_min[:border], lam_min[-border:], lam_min[:, :border], lam_min[:, -border:] = 0, 0, 0, 0
    pts = []
    for idx in np.argsort(lam_min, axis=None)[::-1]:
        y, x = divmod(idx, img.shape[1])
        if all((y - py) ** 2 + (x - px) ** 2 >= min_dist ** 2 for py, px in pts):
            pts.append((y, x))
        if len(pts) == n:
            break
    return np.array(pts, float)


def lk_point(I0, I1, y, x, d, half=7, iters=20):
    """Iterative Lucas-Kanade for one window: solve A d = -b by Gauss-Newton."""
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    Iy, Ix = np.gradient(I0)
    P = [yy + y, xx + x]
    gx, gy, T = map_coordinates(Ix, P, order=1), map_coordinates(Iy, P, order=1), map_coordinates(I0, P, order=1)
    A = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
    for _ in range(iters):
        It = map_coordinates(I1, [yy + y + d[1], xx + x + d[0]], order=1) - T
        # lstsq = solve when A is invertible, minimum-norm step when it is not
        step = np.linalg.lstsq(A, -np.array([(gx * It).sum(), (gy * It).sum()]), rcond=1e-6)[0]
        d = d + step
        if np.abs(step).max() < 1e-3:
            break
    return d, np.linalg.eigvalsh(A)


def pyramid_lk(I0, I1, pts, levels=3):
    pyr0, pyr1 = [I0], [I1]
    for _ in range(levels - 1):                                  # blur, then keep every 2nd pixel
        pyr0.append(gaussian_filter(pyr0[-1], 1.0)[::2, ::2])
        pyr1.append(gaussian_filter(pyr1[-1], 1.0)[::2, ::2])
    flows = []
    for y, x in pts:
        d = np.zeros(2)
        for L in range(levels - 1, -1, -1):                     # coarse to fine
            d, _ = lk_point(pyr0[L], pyr1[L], y / 2 ** L, x / 2 ** L, d)
            if L:
                d = 2 * d                                        # carry the estimate to the finer level
        flows.append(d)
    return np.array(flows)


pts = good_features(I0)
for u, v in [(0.4, -0.3), (2.0, 1.5), (5.0, -3.0), (12.0, 8.0)]:
    I1 = shift(I0, u, v)
    row = []
    for levels in [1, 2, 4]:
        epe = np.linalg.norm(pyramid_lk(I0, I1, pts, levels) - [u, v], axis=1)
        row.append(f"{levels} level(s): median EPE {np.median(epe):6.3f} px, {np.mean(epe < 0.1):4.0%} < 0.1 px")
    print(f"true flow ({u:4.1f}, {v:4.1f})  |  " + "  |  ".join(row))

# aperture problem: a window that only sees a straight vertical edge
stripes = np.tile(gaussian_filter((np.arange(W) % 32 < 16).astype(float), 2), (H, 1))
s1 = shift(stripes, 1.0, 1.0)
for name, img, img1 in [("textured window", I0, shift(I0, 1.0, 1.0)), ("vertical-edge window", stripes, s1)]:
    d, lam = lk_point(img, img1, 80.0, 80.0, np.zeros(2))
    print(f"{name:21s} eigenvalues {lam.round(4)}  flow estimate {d.round(3)}  (true [1. 1.])")
```

Output (Python 3.14, NumPy 2.5, SciPy 1.18):
```
true flow ( 0.4, -0.3)  |  1 level(s): median EPE  0.012 px, 100% < 0.1 px  |  2 level(s): median EPE  0.012 px, 100% < 0.1 px  |  4 level(s): median EPE  0.012 px,  90% < 0.1 px
true flow ( 2.0,  1.5)  |  1 level(s): median EPE  0.005 px, 100% < 0.1 px  |  2 level(s): median EPE  0.005 px, 100% < 0.1 px  |  4 level(s): median EPE  0.005 px,  92% < 0.1 px
true flow ( 5.0, -3.0)  |  1 level(s): median EPE  7.249 px,  38% < 0.1 px  |  2 level(s): median EPE  0.000 px, 100% < 0.1 px  |  4 level(s): median EPE  0.000 px,  70% < 0.1 px
true flow (12.0,  8.0)  |  1 level(s): median EPE 14.940 px,   0% < 0.1 px  |  2 level(s): median EPE 14.720 px,   2% < 0.1 px  |  4 level(s): median EPE  0.000 px,  70% < 0.1 px
textured window       eigenvalues [0.3829 1.0273]  flow estimate [1. 1.]  (true [1. 1.])
vertical-edge window  eigenvalues [0.     2.0517]  flow estimate [1. 0.]  (true [1. 1.])
```

How to read it:
- **Small motion** (up to about 2 px here): single-level LK is accurate to about 0.01 px. Sub-pixel accuracy is what makes LK useful for stabilisation and odometry.
- **Medium motion** (5 px): single-level LK converges to the wrong place for most points, because on this fine texture the linearisation breaks down beyond a few pixels. Two levels fix it.
- **Large motion** (14 px): only the 4-level pyramid recovers it.
- **Too many levels** also hurts. With 4 levels the coarsest image is 20×20 pixels, the 15×15 window covers most of it, and 8 to 10% of the points lock onto a wrong solution that finer levels cannot undo, even for small motion. Use only as many levels as the expected motion needs.
- **Aperture problem.** On the vertical edge, one eigenvalue is exactly 0: vertical motion leaves the image unchanged. LK returns the minimum-norm answer `[1, 0]`, the normal flow, and misses the vertical component entirely. The Shi-Tomasi score (the smaller eigenvalue) would reject this point before tracking.

## Choosing / trade-offs
- **Window size.** Larger windows average more pixels (robust to noise) but blur motion boundaries and assume one shift over a bigger area. Start around 15 to 21 pixels and adjust.
- **Pyramid levels.** Choose so that `max_motion / 2^levels` is within about a window radius, and the coarsest level is still several windows wide.
- **Which points.** Use `cv2.goodFeaturesToTrack` (Shi-Tomasi) or FAST corners, with a minimum distance between points so they cover the frame. Re-detect when too many are lost.
- **Sparse LK vs dense flow.** LK gives accurate flow at chosen points, fast. Dense methods (Farnebäck, RAFT-style networks) give a field over every pixel, which segmentation-by-motion or frame interpolation need, at much higher cost.
- **LK vs feature matching (ORB/SIFT + matching).** LK is faster and sub-pixel accurate between consecutive frames. Descriptor matching handles large viewpoint changes and re-acquiring points after loss.

## Gotchas
- Always run a **forward-backward check**: track `t → t+1`, then back `t+1 → t`, and drop points that do not return within about 1 px. This catches most occlusion and drift failures cheaply.
- Brightness constancy fails under auto-exposure, flicker or shadows. Normalise frames or track on gradients when lighting changes between frames.
- Points on moving objects and on the background are mixed together. For camera-motion estimation, fit a homography or essential matrix with RANSAC and treat the outliers as independent movers.
- Use `status` and the error output from `calcOpticalFlowPyrLK`: lost points still come back with coordinates.
- Points near the image border have windows that fall off the image. Keep a border margin or drop those tracks.
- OpenCV's LK needs 8-bit images and `float32` point arrays of shape `(N, 1, 2)`. Float images or `float64` points raise `cv2.error` (checked on OpenCV 5.0).

## Related
- [[video-analytics]] - building tracking and counting pipelines on live video.
- [[hungarian-algorithm-tracking-from-scratch]] - matching detections to tracks when LK alone is not enough.
- [[object-detection]] - detecting the objects whose points you track.
- [[convolution-layer-from-scratch-numpy]] - image gradients and filters as convolutions.
- [[kalman-filter-from-scratch]] - smoothing tracked positions over time.

## References
- Lucas & Kanade (1981), "An Iterative Image Registration Technique with an Application to Stereo Vision": https://www.ri.cmu.edu/pub_files/pub3/lucas_bruce_d_1981_2/lucas_bruce_d_1981_2.pdf
- Shi & Tomasi (1994), "Good Features to Track", CVPR 1994.
- OpenCV, "Optical Flow" tutorial: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- OpenCV, Object Tracking (video/track) module reference: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
