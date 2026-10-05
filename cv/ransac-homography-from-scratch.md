---
title: RANSAC homography from scratch (DLT, Hartley normalisation, adaptive iterations, inlier threshold)
category: cv
tags: [ransac, homography, dlt, robust-fitting, outliers, image-stitching, feature-matching, perspective-transform, opencv, numpy, from-scratch]
use_cases:
  - "estimate a homography from keypoint matches that include many wrong matches (stitching, document scan, AR)"
  - "choose the reprojection threshold and confidence for cv2.findHomography with RANSAC"
  - "work out how many RANSAC iterations a given outlier rate needs"
  - "understand why least squares on all matches fails and RANSAC does not"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1145/358669.358692
  - https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
---

# RANSAC homography from scratch (DLT, Hartley normalisation, adaptive iterations, inlier threshold)

## Summary
RANSAC (Random Sample Consensus, Fischler & Bolles 1981) fits a model to data with many outliers: repeatedly fit the model to a random minimal sample, count how many points agree with it within a threshold, keep the model with the most agreement, and refit it on those inliers. A homography (the 3×3 projective map between two views of a plane) needs 4 correspondences, and keypoint matching typically produces 20–60% wrong matches, so it is the textbook use. This file implements the normalised DLT solver and RANSAC with adaptive stopping in NumPy, and measures accuracy as the outlier rate and the threshold change.

## Key concepts
- **Minimal sample.** The smallest number of points that determines the model: 2 for a line, 4 for a homography, 7–8 for a fundamental matrix, 5 for an essential matrix. Smaller samples are more likely to be all-inlier.
- **Iterations needed.** With inlier ratio `w` and sample size `s`, one sample is all-inlier with probability `wˢ`. To succeed with probability `p`, run `N = log(1 − p) / log(1 − wˢ)` samples. N grows explosively with the outlier rate: for s = 4 at 99%, 9 samples at 20% outliers, 178 at 60%, 2,876 at 80%.
- **Adaptive stopping.** `w` is unknown, so start with a large N and lower it each time a model with more inliers is found.
- **DLT.** Each correspondence gives two linear equations in the 9 entries of H; stack them and take the singular vector with the smallest singular value. Hartley normalisation (centre the points, scale to mean distance √2) first, or the system is badly conditioned in pixel coordinates.
- **Refit and re-score.** The best 4-point model is noisy. Refit on all its inliers, then re-select inliers with the refined model; that recovers inliers the 4-point model missed at the edges of the image.
- **Threshold.** The inlier test `reprojection error < t` should match the matching noise: about 2–3σ of the keypoint localisation error.

## When to use / scenarios
- Panorama stitching and image mosaicking: homographies between overlapping photos from feature matches ([[harris-corner-detector-from-scratch]], ORB, SIFT).
- Document and screen capture: map the photographed page to a flat rectangle; AR marker and planar object tracking.
- Visual odometry and SfM: essential/fundamental matrix estimation from matches, PnP pose with `cv2.solvePnPRansac`.
- Any model fit where a minimal-sample solver exists and outliers are gross: lines, circles, planes in point clouds, affine transforms, robust regression (`sklearn.linear_model.RANSACRegressor`).
- Not when several models are present at once in a low-dimensional space (several lines in an edge map): the Hough transform votes for all of them in one pass ([[hough-transform-from-scratch]]).
- Not at very high outlier rates (above ~80–90%) with large minimal samples: the iteration count explodes; use guided sampling (PROSAC) or improve matching first (ratio test, cross-check).

## Setup & code
NumPy only, a few seconds. 300 points on a plane, a known homography, 1 px Gaussian matching noise, and a fraction of matches replaced by random points.

```python
import numpy as np

rng = np.random.default_rng(0)


def normalise(p):
    """Hartley normalisation: centre the points, scale mean distance to sqrt(2)."""
    c = p.mean(0)
    s = np.sqrt(2) / np.linalg.norm(p - c, axis=1).mean()
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def dlt_homography(src, dst):
    """Direct linear transform: least-squares H (dst ~ H src) from >= 4 correspondences via SVD."""
    Ts, Td = normalise(src), normalise(dst)
    s = np.c_[src, np.ones(len(src))] @ Ts.T
    d = np.c_[dst, np.ones(len(dst))] @ Td.T
    rows = []
    for (x, y, _), (u, v, _) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    Hn = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)   # null vector = smallest singular value
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def project(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:]


def ransac_homography(src, dst, thresh=3.0, p_success=0.99, max_iters=5000):
    """Fit H to random 4-point samples, keep the one with the most inliers, adapt the iteration count, refit."""
    best = np.zeros(len(src), bool)
    n_iters, i = max_iters, 0
    while i < n_iters:
        idx = rng.choice(len(src), 4, replace=False)
        H = dlt_homography(src[idx], dst[idx])
        err = np.linalg.norm(project(H, src) - dst, axis=1)
        inl = err < thresh
        if inl.sum() > best.sum():
            best = inl
            w = inl.mean()                                     # inlier ratio estimate so far
            n_iters = min(max_iters, int(np.ceil(np.log(1 - p_success) / np.log(1 - w ** 4 + 1e-12))))
        i += 1
    H = dlt_homography(src[best], dst[best])                  # refit on all inliers of the best sample
    best = np.linalg.norm(project(H, src) - dst, axis=1) < thresh   # re-score: a 4-point H misses some inliers
    return dlt_homography(src[best], dst[best]), best, i


H_true = np.array([[0.9, 0.15, 30.0], [-0.1, 1.05, 10.0], [4e-4, -2e-4, 1.0]])
n = 300
src = rng.uniform(0, 640, (n, 2))
grid = rng.uniform(0, 640, (20000, 2))                         # dense check of the fitted mapping
print("required iterations for 99% success, s = 4:", {f"{e:.0%}": int(np.ceil(np.log(0.01) / np.log(1 - (1 - e) ** 4)))
                                                         for e in (0.2, 0.4, 0.6, 0.8)})
for outlier_frac in (0.0, 0.3, 0.6):
    dst = project(H_true, src) + rng.normal(0, 1.0, (n, 2))   # 1 px matching noise
    is_out = rng.random(n) < outlier_frac
    dst[is_out] = rng.uniform(0, 640, (is_out.sum(), 2))       # wrong matches
    H_all = dlt_homography(src, dst)
    H_r, inl, iters = ransac_homography(src, dst)
    e_all = np.linalg.norm(project(H_all, grid) - project(H_true, grid), axis=1).mean()
    e_r = np.linalg.norm(project(H_r, grid) - project(H_true, grid), axis=1).mean()
    tp = (inl & ~is_out).sum()
    print(f"outliers {outlier_frac:4.0%}: DLT on all points {e_all:7.2f} px | RANSAC {e_r:5.2f} px after {iters:4d} iters, "
          f"inlier precision {tp / inl.sum():.3f} recall {tp / (~is_out).sum():.3f}")

# Threshold too tight or too loose at 30% outliers
dst = project(H_true, src) + rng.normal(0, 1.0, (n, 2))
is_out = rng.random(n) < 0.3
dst[is_out] = rng.uniform(0, 640, (is_out.sum(), 2))
for t in (0.5, 3.0, 30.0):
    H_r, inl, iters = ransac_homography(src, dst, thresh=t)
    e_r = np.linalg.norm(project(H_r, grid) - project(H_true, grid), axis=1).mean()
    tp = (inl & ~is_out).sum()
    print(f"threshold {t:4.1f} px: error {e_r:5.2f} px, {inl.sum():3d} inliers (precision {tp / inl.sum():.3f}, "
          f"recall {tp / (~is_out).sum():.3f}), {iters} iters")
```

Output (Python 3.14, NumPy 2.5):
```
required iterations for 99% success, s = 4: {'20%': 9, '40%': 34, '60%': 178, '80%': 2876}
outliers   0%: DLT on all points    0.12 px | RANSAC  0.13 px after    7 iters, inlier precision 1.000 recall 0.990
outliers  30%: DLT on all points   51.06 px | RANSAC  0.15 px after   48 iters, inlier precision 1.000 recall 0.991
outliers  60%: DLT on all points 4323.19 px | RANSAC  0.28 px after  680 iters, inlier precision 1.000 recall 0.991
threshold  0.5 px: error  0.45 px,  31 inliers (precision 1.000, recall 0.161), 5000 iters
threshold  3.0 px: error  0.06 px, 190 inliers (precision 1.000, recall 0.990), 36 iters
threshold 30.0 px: error  0.22 px, 193 inliers (precision 0.995, recall 1.000), 25 iters
```

How to read it:
- Error is the mean distance between the fitted and true mapping over the whole image, so it measures the homography, not just the matched points. With no outliers, plain DLT and RANSAC agree (0.12 vs 0.13 px). With 30% wrong matches, DLT on all points is off by 51 px and at 60% the result is meaningless (4,323 px); RANSAC stays within 0.3 px.
- Inlier precision is 1.000 at every outlier rate and recall about 0.99: the 1% missed are true matches whose 1 px noise happened to exceed 3 px, as expected for 2D Gaussian noise (1 − e^(−4.5) ≈ 0.989).
- At 60% outliers adaptive RANSAC ran 680 iterations against the 178 the formula needs with the true ratio. Early best models have few inliers, so the running estimate of `w` is pessimistic until a good sample appears. Set `max_iters` with that in mind.
- Threshold 0.5 px is below the noise: only 16% of true matches qualify, `w` stays tiny, the loop hits the 5,000-iteration cap and the fit is 7× worse. 30 px lets one outlier in and costs a little accuracy. Around 3σ of the matching noise is the sweet spot, and the result is not very sensitive within a factor of 2–3 of it.

## Choosing / trade-offs
- **OpenCV.** `cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.995)` does all of the above plus a Levenberg-Marquardt refinement. `cv2.USAC_MAGSAC` (OpenCV 4.5+) is less sensitive to the threshold and usually more accurate; prefer it when available.
- **Threshold.** Set it from the localisation noise of your keypoints at the working resolution, and scale it when images are resized. 1–3 px for sub-pixel detectors at full resolution, more for downscaled images or blurry matches.
- **Confidence vs time.** Raising confidence from 0.99 to 0.999 multiplies the iteration count by 1.5. Better matching (Lowe's ratio test, mutual nearest neighbours) cuts outliers and is worth far more than more iterations.
- **Variants.** PROSAC samples high-quality matches first (big speed-up when match scores are informative). LO-RANSAC adds a local optimisation of each good model. MLESAC/MSAC score by truncated error instead of inlier count, which is less sensitive to the threshold.
- **Degenerate samples.** Three collinear points of four give an unstable H. Production code checks sample geometry before solving; at worst a degenerate model wins because it fits a line of points, which the refit-and-rescore step usually exposes.

## Gotchas
- Without Hartley normalisation, DLT in pixel coordinates (values in the hundreds and their products in the tens of thousands) is badly conditioned and visibly less accurate.
- The minimal sample must be drawn without replacement; duplicated points produce a degenerate system.
- Count inliers with the forward error only for speed, but be aware the symmetric transfer error (forward + backward) is more robust when one image has much larger scale.
- A homography is valid only for a plane or a pure camera rotation. For general 3D scenes with translation, matches off the plane are "outliers" for the wrong reason; fit a fundamental/essential matrix instead.
- If RANSAC returns very few inliers relative to the matches, the model is wrong or the matches are bad; check the inlier count before warping, and reject results below a minimum.
- The random generator makes results vary run to run. Seed it in tests and compare inlier counts, not exact matrices.

## Related
- [[hough-transform-from-scratch]] - the voting alternative for several low-dimensional models.
- [[harris-corner-detector-from-scratch]] - keypoints whose matches feed RANSAC.
- [[lucas-kanade-optical-flow-from-scratch]] - tracked points that RANSAC filters for camera motion.
- [[hungarian-algorithm-tracking-from-scratch]] - another way to clean up correspondences, by assignment rather than consensus.
- [[ocr]] - document rectification before text recognition.

## References
- Fischler & Bolles (1981), "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography", Communications of the ACM: https://doi.org/10.1145/358669.358692
- Hartley (1997), "In Defense of the Eight-Point Algorithm", IEEE TPAMI: https://doi.org/10.1109/34.601246
- OpenCV, camera calibration and 3D reconstruction (`findHomography`, USAC methods): https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
