---
title: Harris corner detector from scratch (structure tensor, corner response, Shi-Tomasi, non-maximum suppression)
category: cv
tags: [harris-corner-detector, corner-detection, shi-tomasi, good-features-to-track, structure-tensor, keypoints, feature-detection, non-maximum-suppression, opencv, numpy, from-scratch]
use_cases:
  - "detect corners or keypoints in an image without a learned model"
  - "choose between cv2.cornerHarris and cv2.goodFeaturesToTrack and set their thresholds"
  - "understand why corners are good points to track or match and edges are not"
  - "find stable points for image registration, homography estimation or camera calibration"
status: stable
last_verified: 2026-10-05
sources:
  - https://doi.org/10.5244/C.2.23
  - https://doi.org/10.1109/CVPR.1994.323794
  - https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
---

# Harris corner detector from scratch (structure tensor, corner response, Shi-Tomasi, non-maximum suppression)

## Summary
A corner is a point where the image changes in every direction, so a small window around it cannot slide without its content changing. That makes it locatable in two dimensions, unlike a point on an edge, which can slide along the edge. Harris & Stephens (1988) measure this with the structure tensor, a 2×2 matrix of smoothed gradient products per pixel, and score it with `R = det(M) − k·trace(M)²`. Shi & Tomasi (1994) score the same matrix by its smaller eigenvalue. This file implements both in NumPy on synthetic shapes with known corner positions, then measures recall, precision and localisation error under noise and rotation. The main finding is practical: the two scores scale differently with contrast, so the same relative threshold means very different things for each.

## Key concepts
- **Structure tensor.** `M = G_σ * [[Ix², IxIy], [IxIy, Iy²]]`: products of image gradients, averaged over a Gaussian window. Its eigenvalues `λ1 ≥ λ2` give the gradient energy along the strongest and weakest directions.
- **Three cases.** Flat region: both eigenvalues near 0. Edge: `λ1` large, `λ2 ≈ 0`. Corner: both large.
- **Harris response.** `R = λ1λ2 − k(λ1 + λ2)² = det M − k·trace(M)²`, with `k` in 0.04–0.06. `R > 0` at corners, `R < 0` on edges, `R ≈ 0` in flat areas. It avoids computing eigenvalues.
- **Shi-Tomasi response.** `λ2 = min eigenvalue`. Directly asks "is the weakest direction still strong?", which is the condition for tracking a patch with Lucas-Kanade ([[lucas-kanade-optical-flow-from-scratch]]).
- **Non-maximum suppression (NMS).** The response is a blob a few pixels wide around each corner. Keep only local maxima within a window, then threshold or keep the top N.
- **Invariance.** Rotation-invariant (eigenvalues do not change under rotation). Not scale-invariant: a corner at one scale is an edge or a blob at another. Not contrast-invariant: `R` scales with contrast⁴, `λ2` with contrast².

## When to use / scenarios
- Classical pipelines on CPU: points to track for optical flow and video stabilisation, chessboard and marker corners for camera calibration, keypoints for image registration with RANSAC homographies.
- Industrial inspection with fixed scale and lighting: corners of parts, labels or PCBs, where a learned model is not needed.
- Teaching and debugging: the structure tensor also underlies Lucas-Kanade, edge orientation estimation and anisotropic filtering.
- Not when scale changes between images: use a scale-space detector with a descriptor (SIFT, ORB, AKAZE), or learned features (SuperPoint, DISK) with a matcher (LightGlue). Harris finds points; it does not describe them for matching.
- Not for object detection or semantic keypoints (faces, human pose): use a trained model ([[object-detection]], [[face-and-pose]]).

## Setup & code
NumPy only, runs in about a second. Four filled polygons (17 corners) are rasterised with anti-aliasing, so their true corner positions are known exactly. The scene is also rotated by 30° to check rotation invariance, and Gaussian noise is added at two levels (image range is 0–1).

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

H = W = 160


def render(polys, ss=4):
    """Rasterise filled polygons (even-odd rule) with ss x ss supersampling -> anti-aliased image in [0, 1]."""
    ys, xs = np.mgrid[0:H * ss, 0:W * ss]
    px, py = (xs + 0.5) / ss, (ys + 0.5) / ss
    img = np.zeros((H * ss, W * ss))
    for poly, val in polys:
        inside = np.zeros_like(img, bool)
        for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
            crosses = (y1 > py) != (y2 > py)
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            inside ^= crosses & (px < xint)
        img[inside] = val
    return img.reshape(H, ss, W, ss).mean((1, 3))


def scene(angle_deg):
    """A square, a triangle, an L-shape and a faint square, rotated about the centre. Returns image + true corners."""
    shapes = [(np.array([[30, 30], [70, 30], [70, 70], [30, 70]], float), 1.0),
              (np.array([[95, 35], [135, 45], [105, 75]], float), 0.6),
              (np.array([[35, 95], [60, 95], [60, 115], [80, 115], [80, 135], [35, 135]], float), 0.8),
              (np.array([[100, 100], [135, 100], [135, 135], [100, 135]], float), 0.35)]
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = [((p - 80) @ R.T + 80, v) for p, v in shapes]
    return render(rot), np.vstack([p for p, _ in rot])


def conv2(img, k):
    r = k.shape[0] // 2
    return np.einsum("ijkl,kl->ij", sliding_window_view(np.pad(img, r, mode="edge"), k.shape), k)


def gauss(sigma):
    r = int(3 * sigma)
    g = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


SOBEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0


def structure_tensor(img, sigma=1.5):
    ix, iy = conv2(img, SOBEL), conv2(img, SOBEL.T)
    w = gauss(sigma)
    return conv2(ix * ix, w), conv2(iy * iy, w), conv2(ix * iy, w)


def harris(img, k=0.05):
    a, b, c = structure_tensor(img)
    return a * b - c * c - k * (a + b) ** 2             # det - k trace^2


def shi_tomasi(img):
    a, b, c = structure_tensor(img)
    return (a + b) / 2 - np.sqrt(((a - b) / 2) ** 2 + c * c)   # smaller eigenvalue


def peaks(R, rel=0.01, nms=7, top=None):
    """Local maxima in an nms x nms window above rel * max(R), strongest first, optionally only the top N."""
    r = nms // 2
    local_max = sliding_window_view(np.pad(R, r, constant_values=-np.inf), (nms, nms)).max((2, 3))
    ys, xs = np.nonzero((R == local_max) & (R > rel * R.max()))
    order = np.argsort(-R[ys, xs])[:top]
    return np.c_[xs[order], ys[order]] + 0.5             # pixel centres, in the same frame as the polygons


def score(found, truth, tol=3.0):
    d = np.linalg.norm(found[:, None] - truth[None], axis=2)
    hit = d.min(0) <= tol
    tp = (d.min(1) <= tol).sum()
    return hit.mean(), tp / max(len(found), 1), d.min(1)[d.min(1) <= tol].mean()


rng = np.random.default_rng(0)
img, truth = scene(0)
R = harris(img)
print(f"{len(truth)} true corners")
print("R sign: flat {:+.1e}  edge {:+.1e}  corner {:+.1e}".format(R[10, 10], R[50, 30], R[30, 30]))

print("angle noise  detector    | 1% of max: found recall precision | top-17: recall loc.err(px)")
for angle in (0, 30):
    img, truth = scene(angle)
    for noise in (0.0, 0.05, 0.15):
        noisy = img + noise * rng.standard_normal(img.shape)
        for name, fn in (("Harris", harris), ("Shi-Tomasi", shi_tomasi)):
            R = fn(noisy)
            found = peaks(R)
            rc, pr, _ = score(found, truth)
            rc_top, _, err = score(peaks(R, top=len(truth)), truth)
            print(f"{angle:5d} {noise:5.2f}  {name:10s}  | {len(found):15d} {rc:6.2f} {pr:9.2f} | "
                  f"{rc_top:14.2f} {err:11.2f}")

# Contrast: R scales with contrast^4, so a fixed relative threshold drops faint corners
img, truth = scene(0)
for scale in (1.0, 0.25):
    R = harris(img * scale)
    print(f"contrast x{scale}: max R {R.max():.2e}")
```

Output (Python 3.14, NumPy 2.5):
```
17 true corners
R sign: flat +0.0e+00  edge -7.2e-04  corner +3.0e-03
angle noise  detector    | 1% of max: found recall precision | top-17: recall loc.err(px)
    0  0.00  Harris      |              17   1.00      1.00 |           1.00        0.89
    0  0.00  Shi-Tomasi  |              17   1.00      1.00 |           1.00        2.06
    0  0.05  Harris      |              17   1.00      1.00 |           1.00        1.28
    0  0.05  Shi-Tomasi  |             174   1.00      0.10 |           1.00        1.97
    0  0.15  Harris      |             139   0.82      0.10 |           0.76        1.57
    0  0.15  Shi-Tomasi  |             285   0.94      0.06 |           0.82        1.93
   30  0.00  Harris      |              17   1.00      1.00 |           1.00        1.67
   30  0.00  Shi-Tomasi  |              17   1.00      1.00 |           1.00        1.66
   30  0.05  Harris      |              17   1.00      1.00 |           1.00        1.61
   30  0.05  Shi-Tomasi  |             249   1.00      0.07 |           1.00        1.67
   30  0.15  Harris      |             216   0.88      0.07 |           0.76        1.61
   30  0.15  Shi-Tomasi  |             289   0.94      0.06 |           0.82        1.65
contrast x1.0: max R 2.98e-03
contrast x0.25: max R 1.16e-05
```

How to read it:
- The sign of `R` separates the three cases: exactly 0 in the flat background, negative on the square's edge, positive at its corner.
- On clean images both detectors find all 17 corners at both angles with no false positives. Rotation by 30° changes nothing in recall; the localisation error moves between 0.9 and 2 px because the anti-aliased corners land differently on the pixel grid.
- With a threshold at 1% of the maximum response, Shi-Tomasi already returns 174 points at noise 0.05, mostly noise, while Harris still returns exactly 17. This is not Harris being better: `R` grows like λ², so 1% of max `R` is a much stricter cut than 1% of max `λ2` (roughly 10% in eigenvalue terms). A relative threshold has to be tuned per detector.
- Keeping the strongest 17 instead (like `maxCorners` in OpenCV) removes the threshold question. Both are perfect at noise 0.05; at noise 0.15 Shi-Tomasi recovers 82% of corners and Harris 76%. The misses are mostly the four corners of the faint square (contrast 0.35), occasionally a triangle vertex: with top N, strong noise peaks crowd out weak real corners.
- Scaling the image contrast by 0.25 divides `R` by 256 (4⁴). An absolute threshold tuned on one image fails on a darker one; normalise the image or use a relative or top-N rule.

## Choosing / trade-offs
- **Harris vs Shi-Tomasi.** Shi-Tomasi's `λ2` has the units of gradient energy and maps directly onto trackability; it is the default in `cv2.goodFeaturesToTrack`. Harris (`useHarrisDetector=True`, or `cv2.cornerHarris`) avoids the square root and has the `k` knob. In practice they find nearly the same points; the threshold rule matters more than the score.
- **Window σ and derivative scale.** A larger integration σ suppresses noise but blurs nearby corners together and shifts the peak inside the corner (a few pixels of localisation bias). Match σ to the corner scale you care about, or detect at several scales (Harris-Laplace).
- **Thresholding.** Top N with a minimum distance spreads points over the image and is robust to contrast; a relative threshold keeps all strong corners but varies in count; an absolute threshold only works with controlled lighting.
- **Sub-pixel accuracy.** For calibration, refine each detection with `cv2.cornerSubPix` (an iterative gradient-orthogonality fit), which reaches small fractions of a pixel on clean chessboards. Peak picking alone is accurate to about a pixel.
- **Classical vs learned.** Harris is fast, deterministic and needs no training data. Learned detectors (SuperPoint and successors) are more repeatable under strong viewpoint and lighting changes and come with descriptors for matching.

## Gotchas
- `cv2.cornerHarris` takes a single-channel image (convert colour to grey first), and its `blockSize`, `ksize` and `k` correspond to the window, the Sobel aperture and the Harris constant. The response values depend on the input's dtype and range, so a threshold tuned on a `float32` image in [0, 1] does not transfer to `uint8` in [0, 255].
- Without NMS every corner produces a cluster of detections, inflating counts and skewing downstream RANSAC.
- Image borders: padding creates artificial gradients. Ignore detections within a few pixels of the edge, or pad with edge replication as above.
- JPEG block artefacts and text produce many strong corners. Mask regions you do not care about before taking the top N, or they crowd out the useful points.
- A corner is a 2D structure only at the scale of the window. Zoom in and a rounded corner becomes an edge; zoom out and a small square becomes a blob. Harris has no answer to scale change.

## Related
- [[lucas-kanade-optical-flow-from-scratch]] - tracks Shi-Tomasi points with the same structure tensor.
- [[hungarian-algorithm-tracking-from-scratch]] - frame-to-frame association for detections.
- [[convolution-layer-from-scratch-numpy]] - the convolution used for Sobel and Gaussian filtering.
- [[object-detection]] - learned detectors when "corner" is really "object".
- [[video-analytics]] - pipelines where classical tracking points still appear.

## References
- Harris & Stephens (1988), "A combined corner and edge detector", Alvey Vision Conference: https://doi.org/10.5244/C.2.23
- Shi & Tomasi (1994), "Good features to track", CVPR: https://doi.org/10.1109/CVPR.1994.323794
- OpenCV, Harris corner detection tutorial: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
