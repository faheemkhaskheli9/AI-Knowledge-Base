---
title: Canny edge detector from scratch (Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis)
category: cv
tags: [canny, edge-detection, sobel, gradient, non-maximum-suppression, hysteresis, gaussian-blur, opencv, scikit-image, numpy, from-scratch]
use_cases:
  - "understand what cv2.Canny's two thresholds and the blur before it actually do"
  - "pick the smoothing sigma and low/high thresholds for edge detection on noisy images"
  - "build a classical edge pre-processing step for Hough lines, document boundary detection or contour measurement"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1109/TPAMI.1986.4767851
  - https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
---

# Canny edge detector from scratch (Gaussian smoothing, Sobel gradients, non-maximum suppression, hysteresis)

## Summary
The Canny detector (Canny 1986) is still the default classical edge detector. Four steps turn an image into thin, connected edges: smooth with a Gaussian to suppress noise, compute the gradient with Sobel filters, keep only pixels that are the local maximum along the gradient direction (non-maximum suppression, which makes edges one pixel wide), then link edges with two thresholds: strong pixels start an edge and weak pixels are kept only if they connect to a strong one (hysteresis). This file implements all four in NumPy on a synthetic image with known boundaries, and measures precision and recall against plain thresholded Sobel as noise, sigma and the thresholds change.

## Key concepts
- **Smoothing scale (sigma).** Differentiation amplifies noise, so blur first. Larger sigma suppresses more noise but rounds corners, merges nearby edges and loses fine detail. It is the main knob for noisy images.
- **Gradient.** Sobel `gx`, `gy`; magnitude `hypot(gx, gy)` and direction `atan2(gy, gx)`. Edges are where the magnitude is high.
- **Non-maximum suppression (NMS).** A blurred step gives a ridge of high magnitude several pixels wide. Keep a pixel only if it is at least as large as its two neighbours along the gradient direction (rounded to 0, 45, 90 or 135 degrees). This thins ridges to about one pixel.
- **Hysteresis thresholds.** One threshold either breaks weak edges into dashes (high) or lets noise in (low). With two, pixels above `high` seed edges and pixels above `low` survive only when 8-connected to a seed. Canny suggested a high:low ratio of 2:1 to 3:1.
- **Relative thresholds.** Absolute thresholds depend on exposure and bit depth. Setting them as fractions of a robust maximum gradient (here the 99.5th percentile) keeps one setting usable across images.

## When to use / scenarios
- Pre-processing for geometric fitting: Hough lines and circles ([[hough-transform-from-scratch]]), document and card boundary detection before a perspective warp ([[ransac-homography-from-scratch]], [[ocr]]).
- Industrial measurement on controlled images: part outlines, gap widths, crack detection where lighting is fixed.
- Cheap, explainable features on edge devices where a CNN is not an option.
- As a conditioning signal: Canny edge maps are a common ControlNet input for diffusion models ([[diffusion-models]]).
- Not for semantic boundaries ("where the person ends") in cluttered photos: Canny finds every intensity step, including texture and shadows. Use segmentation ([[segmentation]]) or a learned edge model there.

## Setup & code
NumPy only, under a second. A 256x256 image has a rectangle, a disc and a faint triangle (contrast 0.08) on an illumination ramp. The true boundary mask comes from the region labels. A predicted edge counts as correct within 1 pixel.

```python
import numpy as np

rng = np.random.default_rng(0)


def scene(n=256):
    """Rectangle, disc and a faint triangle on a gradient background; returns image and true boundary mask."""
    yy, xx = np.mgrid[:n, :n]
    regions = [((yy > 40) & (yy < 120) & (xx > 30) & (xx < 140), 0.45),
               ((yy - 170) ** 2 + (xx - 170) ** 2 < 55 ** 2, 0.35),
               ((yy > 150) & (xx > 20) & (xx < 20 + (yy - 150)) & (yy < 240), 0.08)]   # low-contrast object
    img = 0.2 + 0.2 * xx / n                                                         # smooth illumination ramp
    label = np.zeros((n, n), int)
    for i, (mask, c) in enumerate(regions, 1):
        img = np.where(mask, img + c, img)
        label[mask] = i
    edge = np.zeros((n, n), bool)                                    # true boundary: label changes to a neighbour
    edge[:-1] |= label[:-1] != label[1:]
    edge[:, :-1] |= label[:, :-1] != label[:, 1:]
    return img, edge


def conv2(img, k):
    p = k.shape[0] // 2
    x = np.pad(img, p, mode="reflect")
    out = np.zeros_like(img)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            out += k[i, j] * x[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def gaussian(sigma):
    r = int(3 * sigma + 0.5)
    g = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)


def gradients(img, sigma):
    s = conv2(img, gaussian(sigma)) if sigma > 0 else img
    gx, gy = conv2(s, SX), conv2(s, SX.T)                       # conv2 correlates, so these are the Sobel x/y
    return np.hypot(gx, gy), np.arctan2(gy, gx)


def non_max_suppression(mag, ang):
    """Keep a pixel only if it is a local max along the gradient direction (rounded to 0/45/90/135 degrees)."""
    q = (np.round(ang / (np.pi / 4)) % 4).astype(int)
    offs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    p = np.pad(mag, 1)
    H, W = mag.shape
    keep = np.zeros_like(mag, bool)
    for d, (dy, dx) in offs.items():
        a = p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        b = p[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        keep |= (q == d) & (mag >= a) & (mag >= b)
    return np.where(keep, mag, 0)


def hysteresis(nms, low, high):
    """Strong pixels seed edges; weak pixels survive only if 8-connected to a strong one."""
    strong, weak = nms >= high, nms >= low
    edges = strong.copy()
    while True:
        p = np.pad(edges, 1)
        grown = np.zeros_like(edges)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= p[1 + dy:1 + dy + edges.shape[0], 1 + dx:1 + dx + edges.shape[1]]
        new = grown & weak
        if (new == edges).all():
            return edges
        edges = new


def canny(img, sigma=1.4, low=0.1, high=0.2):
    """Thresholds are fractions of a robust maximum gradient (99.5th percentile), so they survive exposure changes."""
    mag, ang = gradients(img, sigma)
    nms = non_max_suppression(mag, ang)
    ref = np.percentile(mag, 99.5)
    return hysteresis(nms, low * ref, high * ref)


def sobel_threshold(img, t=0.2):
    mag, _ = gradients(img, 0)
    return mag > t * np.percentile(mag, 99.5)


def dilate(m, r=1):
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= p[r + dy:r + dy + m.shape[0], r + dx:r + dx + m.shape[1]]
    return out


def score(pred, truth, tol=1):
    """Precision/recall with a small localisation tolerance (a predicted pixel within tol of a true edge counts)."""
    p = (pred & dilate(truth, tol)).sum() / max(pred.sum(), 1)
    r = (truth & dilate(pred, tol)).sum() / truth.sum()
    return p, r, 2 * p * r / max(p + r, 1e-9)


img, truth = scene()
print(f"{truth.sum()} true edge pixels")
for noise in (0.02, 0.04, 0.08):
    x = img + noise * rng.standard_normal(img.shape)
    rows = [("Sobel > threshold", sobel_threshold(x))] + [
        (f"Canny sigma={s}", canny(x, sigma=s)) for s in (0.5, 1.4, 3.0)]
    print(f"noise {noise:.2f}: " + " | ".join(
        f"{name} P {p:.2f} R {r:.2f} F1 {f:.2f} ({pred.sum()} px)"
        for name, pred in rows for p, r, f in [score(pred, truth)]))

x = img + 0.04 * rng.standard_normal(img.shape)
mag, ang = gradients(x, 1.4)
ref = np.percentile(mag, 99.5)
print("pixels marked per true edge pixel: threshold only", round((mag > 0.2 * ref).sum() / truth.sum(), 2),
      "-> after NMS + hysteresis", round(canny(x).sum() / truth.sum(), 2))
for lo, hi in ((0.2, 0.2), (0.1, 0.2), (0.05, 0.2), (0.1, 0.35)):
    p, r, f = score(canny(x, 1.4, lo, hi), truth)
    print(f"noise 0.04, sigma 1.4, low={lo} high={hi}: P {p:.2f} R {r:.2f} F1 {f:.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
1099 true edge pixels
noise 0.02: Sobel > threshold P 1.00 R 0.93 F1 0.96 (1794 px) | Canny sigma=0.5 P 1.00 R 1.00 F1 1.00 (1102 px) | Canny sigma=1.4 P 1.00 R 1.00 F1 1.00 (1123 px) | Canny sigma=3.0 P 0.99 R 0.99 F1 0.99 (1118 px)
noise 0.04: Sobel > threshold P 0.49 R 0.97 F1 0.65 (3763 px) | Canny sigma=0.5 P 0.26 R 1.00 F1 0.42 (4358 px) | Canny sigma=1.4 P 1.00 R 1.00 F1 1.00 (1129 px) | Canny sigma=3.0 P 0.99 R 0.98 F1 0.99 (1118 px)
noise 0.08: Sobel > threshold P 0.09 R 0.99 F1 0.17 (26416 px) | Canny sigma=0.5 P 0.05 R 1.00 F1 0.10 (23598 px) | Canny sigma=1.4 P 0.34 R 0.99 F1 0.51 (3327 px) | Canny sigma=3.0 P 0.98 R 0.98 F1 0.98 (1119 px)
pixels marked per true edge pixel: threshold only 3.75 -> after NMS + hysteresis 1.02
noise 0.04, sigma 1.4, low=0.2 high=0.2: P 1.00 R 0.83 F1 0.91
noise 0.04, sigma 1.4, low=0.1 high=0.2: P 1.00 R 1.00 F1 1.00
noise 0.04, sigma 1.4, low=0.05 high=0.2: P 1.00 R 1.00 F1 1.00
noise 0.04, sigma 1.4, low=0.1 high=0.35: P 1.00 R 0.68 F1 0.81
```

How to read it:
- At low noise, thresholded Sobel finds the edges but marks 1794 pixels for 1099 true ones (thick edges) and misses part of the faint triangle (R 0.93). Canny marks about one pixel per true edge pixel. The NMS line shows the same thing directly: a threshold alone marks 3.75 pixels per edge pixel, NMS plus hysteresis 1.02.
- Sigma has to match the noise. At noise 0.04, sigma 0.5 lets noise ridges through (P 0.26) while sigma 1.4 is clean. At 0.08, only sigma 3.0 holds (P 0.98). The shapes here are large and have few corners, so the cost of a big sigma (rounded corners, merged nearby edges) does not show in this score.
- Hysteresis is what keeps the faint triangle. With a single threshold (low = high = 0.2), recall falls to 0.83 because parts of its edges drop below the cut. Lowering `low` to 0.1 recovers them with no precision loss, because weak pixels count only when connected to strong ones. Raising `high` to 0.35 removes the seeds on the faint edges entirely (R 0.68): `high` decides which objects are found, `low` how completely their outlines are traced.

## Choosing / trade-offs
- **Library calls.** `cv2.Canny(img, threshold1, threshold2)` takes absolute low/high thresholds on the gradient of an 8-bit image and does not blur for you: call `cv2.GaussianBlur` first. `skimage.feature.canny(image, sigma=...)` blurs internally and accepts thresholds as quantiles (`use_quantiles=True`). Write your own only to learn or to change a step.
- **Thresholds.** A ratio of 2:1 to 3:1 is the usual start. A common automatic rule sets them at about 0.66x and 1.33x the median pixel intensity; it works for natural photos and fails on mostly flat images. Thresholds relative to a high gradient percentile, as here, are more stable.
- **Canny vs learned edges.** Canny is fast, needs no training and finds every step edge. Learned detectors (HED and its successors) find object boundaries and ignore texture, which is usually what a downstream task means by "edge".
- **Scale.** One sigma gives one scale. Fine and coarse structures in the same image may need two passes with different sigmas.

## Gotchas
- `cv2.Canny` thresholds are on the gradient magnitude of the 8-bit image, so the same values behave differently on 16-bit or float images, and differently with `L2gradient=True` (true magnitude) than the default L1 approximation.
- JPEG block artifacts and sensor noise create short edge fragments; blur more or remove connected components smaller than a few pixels.
- Edges are found on intensity only. Colour boundaries with equal brightness (red vs green of the same luminance) vanish when converted to grayscale; run per channel or on a different colour space.
- NMS quantises direction to 4 bins, so diagonal and curved edges can come out slightly jagged or with 1-pixel gaps. Hysteresis bridges most of them.
- Illumination gradients are low-frequency and do not create edges, but strong shadows and specular highlights do. Normalise lighting or mask them out before trusting edge counts as measurements.

## Related
- [[hough-transform-from-scratch]] - fits lines and circles to the edge map Canny produces.
- [[harris-corner-detector-from-scratch]] - the same image gradients, used to find corners instead of edges.
- [[convolution-layer-from-scratch-numpy]] - the convolution that Gaussian and Sobel filtering are built on.
- [[ransac-homography-from-scratch]] - document boundary to perspective warp, often after edges and contours.
- [[segmentation]] - the learned alternative when you need object boundaries, not intensity steps.

## References
- Canny (1986), "A Computational Approach to Edge Detection", IEEE TPAMI: https://doi.org/10.1109/TPAMI.1986.4767851
- OpenCV Canny tutorial (minVal/maxVal thresholds, L2gradient): https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
