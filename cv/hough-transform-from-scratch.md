---
title: Hough transform for lines from scratch (voting accumulator, bin size, peak suppression, robustness to outliers)
category: cv
tags: [hough-transform, line-detection, accumulator, voting, robust-fitting, ransac, lane-detection, opencv, numpy, from-scratch]
use_cases:
  - "detect straight lines in an edge image (lanes, document borders, shelves, pipes) without a learned model"
  - "fit several lines at once to points with many outliers and gaps"
  - "choose rho/theta resolution and threshold for cv2.HoughLines or cv2.HoughLinesP"
  - "understand why least squares fails on cluttered points and voting does not"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1145/361237.361242
  - https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
---

# Hough transform for lines from scratch (voting accumulator, bin size, peak suppression, robustness to outliers)

## Summary
The Hough transform finds shapes by voting: each edge point votes for every line that could pass through it, and lines supported by many points collect many votes. With lines written in normal form `x·cosθ + y·sinθ = ρ` (Duda & Hart 1972), one point votes along a sinusoid in the (θ, ρ) accumulator, and collinear points' sinusoids cross in one cell. It finds several lines at once, tolerates gaps and heavy clutter, and needs no starting guess. This file implements it in NumPy on synthetic edge points with known lines, measures the effect of the bin size, and compares it with least-squares fitting as outliers increase.

## Key concepts
- **Normal form.** `ρ = x cosθ + y sinθ`, with θ in [0°, 180°) and ρ in [−diag, diag]. Unlike `y = mx + c`, it handles vertical lines and has a bounded parameter space.
- **Accumulator.** A 2D histogram over (θ, ρ). Each point adds 1 to one ρ-bin per θ-bin. Cost is `points × θ-bins`, independent of the number of lines.
- **Peaks.** True lines are local maxima. Votes from one line spread into a butterfly shape around its peak, so take the maximum, suppress a window around it, and repeat (non-maximum suppression).
- **Bin size.** Small bins: precise parameters, but jittered points split their votes across neighbouring bins and the peak shrinks. Large bins: tall peaks, coarser parameters, and nearby lines merge.
- **Robustness.** A point off the line votes for other cells, so it cannot pull the estimate. Its breakdown point is far higher than least squares, which every outlier pulls.
- **Generalisation.** Circles use (x, y, r) and other shapes use their own parameters. The generalised Hough transform votes for an arbitrary template's pose. The accumulator grows with each parameter dimension.

## When to use / scenarios
- Lane markings, road edges and parking lines in classical driver-assist or robotics pipelines on CPU.
- Document and screen detection (four border lines then a perspective warp), table and form line extraction, barcode orientation.
- Industrial inspection: straight edges of parts, conveyor belts, pipes, shelf rows. Circles for coins, holes, pupils (`cv2.HoughCircles`).
- Fitting several models at once to cluttered points, where RANSAC would need to be run once per model.
- Not for curved lanes or objects with no simple shape: use segmentation or a learned detector ([[segmentation]], [[object-detection]]).
- Not when there is exactly one model with moderate outliers in high dimensions (homographies, fundamental matrices): RANSAC is simpler and does not need an accumulator per parameter.

## Setup & code
NumPy only, under a second. Three lines in a 200 × 200 image, 120 jittered edge points each with an occluded segment, plus 300 uniformly scattered clutter points (45% of all points).

```python
import numpy as np

rng = np.random.default_rng(0)
H = W = 200
# Three true lines in normal form: x cos(theta) + y sin(theta) = rho
TRUE = [(np.deg2rad(30), 80.0), (np.deg2rad(100), 60.0), (np.deg2rad(150), -40.0)]


def edge_points(n_per_line=120, jitter=0.7, clutter=300, gap=0.4):
    """Points on each line inside the image (with a missing segment), Gaussian jitter, plus uniform clutter."""
    pts = []
    for th, rho in TRUE:
        c, s = np.cos(th), np.sin(th)
        t = rng.uniform(-300, 300, 20 * n_per_line)
        p = np.c_[rho * c - t * s, rho * s + t * c]
        keep = (p >= 0).all(1) & (p < [W, H]).all(1) & (np.abs(t - 20) > gap * 50)   # occluded segment
        p = p[keep][:n_per_line]
        pts.append(p + jitter * rng.standard_normal(p.shape))
    pts.append(rng.uniform(0, [W, H], (clutter, 2)))
    return np.vstack(pts)


def hough_lines(pts, d_theta=1.0, d_rho=1.0):
    """Every point votes for all (theta, rho) lines through it: one sinusoid per point in the accumulator."""
    thetas = np.deg2rad(np.arange(0, 180, d_theta))
    rho_max = np.hypot(H, W)
    rhos = np.arange(-rho_max, rho_max + d_rho, d_rho)
    acc = np.zeros((len(thetas), len(rhos)), int)
    r = pts[:, :1] * np.cos(thetas) + pts[:, 1:] * np.sin(thetas)      # (n_points, n_theta)
    idx = np.round((r + rho_max) / d_rho).astype(int)
    np.add.at(acc, (np.broadcast_to(np.arange(len(thetas)), idx.shape), idx), 1)
    return acc, thetas, rhos


def peaks(acc, k, nms=(10, 10)):
    """Take the global max, zero a window around it, repeat k times. Theta wraps with a rho sign flip; ignored here."""
    acc = acc.copy()
    out = []
    for _ in range(k):
        i, j = np.unravel_index(acc.argmax(), acc.shape)
        out.append((i, j, acc[i, j]))
        acc[max(i - nms[0], 0):i + nms[0] + 1, max(j - nms[1], 0):j + nms[1] + 1] = 0
    return out


def line_error(th, rho):
    """Distance to the nearest true line in (degrees, pixels)."""
    return min((abs(np.rad2deg(th - t)), abs(rho - r)) for t, r in TRUE)


pts = edge_points()
print(f"{len(pts)} edge points, {3 * 120} on lines, 300 clutter")
for d in (1.0, 0.5, 2.0):
    acc, thetas, rhos = hough_lines(pts, d_theta=d, d_rho=d)
    found = peaks(acc, 4)
    print(f"bin {d} deg/{d} px  accumulator {acc.shape}: " + "  ".join(
        f"[{np.rad2deg(thetas[i]):5.1f} deg {rhos[j]:6.1f} px  {v:3d} votes]" for i, j, v in found))
    errs = [line_error(thetas[i], rhos[j]) for i, j, _ in found[:3]]
    print(f"   top-3 max error: {max(e[0] for e in errs):.1f} deg, {max(e[1] for e in errs):.1f} px;"
          f"  4th peak / 3rd peak votes = {found[3][2] / found[2][2]:.2f}")

# Least squares on one line's points + clutter vs Hough: LS is dragged by outliers, Hough is not
th, rho = TRUE[0]
line_pts = edge_points(clutter=0)[:120]
for frac in (0.0, 0.2, 0.5):
    n_out = int(frac * len(line_pts) / (1 - frac))
    p = np.vstack([line_pts, rng.uniform(0, [W, H], (n_out, 2))])
    centred = p - p.mean(0)
    normal = np.linalg.svd(centred)[2][-1]                             # total least squares normal
    ls_th = np.arctan2(normal[1], normal[0]) % np.pi
    acc, thetas, rhos = hough_lines(p)
    i, j, _ = peaks(acc, 1)[0]
    print(f"outliers {frac:4.0%}: TLS angle error {abs(np.rad2deg(ls_th - th)):5.1f} deg | "
          f"Hough angle error {abs(np.rad2deg(thetas[i] - th)):4.1f} deg, rho error {abs(rhos[j] - rho):4.1f} px")
```

Output (Python 3.14, NumPy 2.5):
```
660 edge points, 360 on lines, 300 clutter
bin 1.0 deg/1.0 px  accumulator (180, 567): [ 30.0 deg   80.2 px   59 votes]  [150.0 deg  -39.8 px   59 votes]  [100.0 deg   60.2 px   57 votes]  [153.0 deg  -51.8 px   22 votes]
   top-3 max error: 0.0 deg, 0.2 px;  4th peak / 3rd peak votes = 0.39
bin 0.5 deg/0.5 px  accumulator (360, 1133): [100.5 deg   59.2 px   40 votes]  [ 30.0 deg   79.7 px   37 votes]  [150.5 deg  -41.3 px   31 votes]  [148.5 deg  -34.8 px   18 votes]
   top-3 max error: 0.5 deg, 1.3 px;  4th peak / 3rd peak votes = 0.58
bin 2.0 deg/2.0 px  accumulator (90, 284): [100.0 deg   59.2 px   83 votes]  [150.0 deg  -40.8 px   79 votes]  [ 30.0 deg   79.2 px   78 votes]  [144.0 deg  -16.8 px   25 votes]
   top-3 max error: 0.0 deg, 0.8 px;  4th peak / 3rd peak votes = 0.32
outliers   0%: TLS angle error   0.1 deg | Hough angle error  0.0 deg, rho error  0.2 px
outliers  20%: TLS angle error  10.8 deg | Hough angle error  0.0 deg, rho error  0.2 px
outliers  50%: TLS angle error  15.7 deg | Hough angle error  0.0 deg, rho error  0.2 px
```

How to read it:
- With 1° / 1 px bins the three strongest peaks are the three true lines, each within 0.2 px and 0° of the truth, despite 45% clutter and a missing segment on each line. Each line collects 57–59 of its 120 points: the rest of its votes fall into neighbouring bins because of the 0.7 px jitter.
- The 4th peak (22 votes, 39% of the 3rd) is not a line. It sits just outside the suppression window of the 150° line: part of that line's own butterfly spread, plus a few clutter votes. Setting the threshold or the number of lines is the part that needs tuning; a vote ratio like 0.39 between the last real and the first false peak is the margin to look at.
- Halving the bin size to 0.5 lowers the true peaks to 31–40 votes and raises the false/true ratio to 0.58: the jitter now spans several bins, so votes split and the peaks are harder to separate. Doubling it to 2 gives the cleanest separation (0.32) at 1 px of extra quantisation. Bin size should match the noise in the edge points, not the precision you would like.
- One line plus outliers: total least squares is within 0.1° on clean points but off by 10.8° at 20% outliers and 15.7° at 50%. The Hough peak does not move. Points that do not lie on the line vote elsewhere.

## Choosing / trade-offs
- **Standard vs probabilistic.** `cv2.HoughLines` returns infinite (ρ, θ) lines. `cv2.HoughLinesP` samples a subset of points and returns segments with endpoints, controlled by `minLineLength` and `maxLineGap`. Use P when you need segments (lane dashes, table cells), standard when you need the line's parameters.
- **Bin size and threshold.** OpenCV's `rho`, `theta` are the bin sizes and `threshold` is the minimum votes. Votes scale with edge density and line length in pixels, so a threshold that works at one resolution fails at another. Rescale the threshold with image size or use top-k.
- **Hough vs RANSAC.** Hough handles several models and very high outlier rates in low-dimensional parameter spaces (2D lines, 3D circles). RANSAC handles one model at a time in any dimension and needs a minimal-sample solver. A common pattern is Hough to find candidates, then a least-squares refit on each line's inliers for sub-bin precision.
- **Edge input.** Quality depends on the edge map. Canny with sensible thresholds, plus a region-of-interest mask (the road ahead, the page), removes most clutter before voting and is cheaper than a better peak finder.
- **Learned alternatives.** Deep line and lane detectors handle curves, occlusion and lighting better and run on GPU. Hough remains the choice on small CPUs, for simple scenes, and where behaviour must be explainable.

## Gotchas
- θ wraps around: a line at θ = 179° and one at θ = 1° with ρ negated are nearly the same line. Suppression that does not wrap (as above) can report it twice; merge peaks across the θ boundary.
- Long lines get more votes than short ones simply because they have more pixels. A threshold tuned for long borders misses short segments, and a threshold for short ones lets texture through.
- Textured regions (grass, gravel, text) create many weak lines that together produce false peaks. Mask or blur them before edge detection.
- Bin quantisation limits precision to about half a bin. For measurement (calibration, metrology), refit each line on its inliers.
- `np.add.at` is needed for voting; `acc[i, j] += 1` with repeated index pairs counts each pair only once.
- Circle Hough with an unknown radius has a 3D accumulator and gets slow and noisy. Restrict the radius range (`minRadius`, `maxRadius` in OpenCV).

## Related
- [[harris-corner-detector-from-scratch]] - another classical feature detector on image gradients.
- [[nms-iou-and-map-from-scratch]] - the same suppress-around-the-maximum idea for detection boxes.
- [[convolution-layer-from-scratch-numpy]] - the filtering used to compute the edge map.
- [[object-detection]] - learned detectors when the target is not a simple shape.
- [[video-analytics]] - pipelines where classical lane and line detection still run.

## References
- Duda & Hart (1972), "Use of the Hough transformation to detect lines and curves in pictures", Communications of the ACM: https://doi.org/10.1145/361237.361242
- OpenCV, Hough line transform tutorial: https://docs.opencv.org/4.x/d6/d10/tutorial_py_houghlines.html
