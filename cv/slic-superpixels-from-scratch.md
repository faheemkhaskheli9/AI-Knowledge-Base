---
title: SLIC superpixels from scratch (local k-means in Lab+xy, compactness, connectivity, boundary recall)
category: cv
tags: [slic, superpixels, k-means, cielab, oversegmentation, compactness, boundary-recall, undersegmentation-error, scikit-image, numpy, from-scratch]
use_cases:
  - "split an image into a few hundred colour-coherent regions to speed up segmentation, labelling or graph-based processing"
  - "understand the compactness parameter and why superpixels collapse or fragment on noisy images"
  - "evaluate an oversegmentation with boundary recall and undersegmentation error"
status: draft
last_verified: 2026-10-05
sources:
  - https://doi.org/10.1109/TPAMI.2012.120
  - https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic
---

# SLIC superpixels from scratch (local k-means in Lab+xy, compactness, connectivity, boundary recall)

## Summary
Superpixels group pixels into a few hundred compact regions of similar colour, so later steps work on 200 regions instead of 200,000 pixels. SLIC (Simple Linear Iterative Clustering, Achanta et al. 2012) is k-means on 5-D points (L, a, b, y, x). Centres start on a regular grid, and each centre only looks at a `2S x 2S` window, so one iteration is O(N) regardless of how many superpixels you ask for. A compactness weight `m` trades colour adherence against regular shape. This file implements SLIC with an sRGB→Lab conversion and a connectivity pass. It scores the result with boundary recall and undersegmentation error against a synthetic scene with known regions, and compares with `skimage.segmentation.slic` as the noise level rises.

## Key concepts
- **Distance.** `D² = d_lab² + (m/S)² · d_xy²`, where `S = sqrt(N / K)` is the grid step for `K` superpixels. Dividing the spatial distance by `S` makes `m` independent of image size. Lab values span about 0–100, so `m` of 1–40 is the usable range.
- **Local search.** Each centre assigns only pixels within `±S`. That is what makes SLIC linear-time and gives roughly equal-sized regions.
- **Seed perturbation.** Seeds are moved to the lowest-gradient pixel in their 3x3 neighbourhood so no centre starts on an edge.
- **Connectivity enforcement.** k-means labels are not guaranteed to be connected. A final pass relabels 4-connected components and merges components smaller than a threshold (here `S²/4`) into a neighbour.
- **Metrics.** *Boundary recall*: the share of true boundary pixels with a superpixel boundary within 2 px. *Undersegmentation error (UE)*: the share of pixels that sit in a superpixel whose majority belongs to another true region (leakage across object edges). Good superpixels have high recall and low UE at a given count.

## When to use / scenarios
- Interactive and weak labelling: annotators click superpixels instead of painting pixels, then a model refines the mask ([[segmentation]]).
- Graph-based methods: superpixels become nodes for CRFs, graph cuts or GNNs ([[gcn-from-scratch-numpy]]), which is cheaper than a pixel graph.
- Remote sensing, medical and microscopy images where colour regions matter and labelled data is scarce. Superpixels give per-region features for a classical classifier.
- Explainability: LIME-style image explanations perturb superpixels ([[integrated-gradients-from-scratch-numpy]] covers the gradient-based alternative).
- Not for: semantic segmentation in production when you have labels. A trained segmentation network or a promptable model (SAM, the Segment Anything model) gives object-level masks directly.

## Setup & code
NumPy only for the algorithm, about 2 seconds for all runs. `scikit-image` is an optional reference. The scene is 120x160 with a background and three shapes in different colours, plus Gaussian pixel noise at three levels. The target is 130 superpixels (`S = 12`).

```python
import numpy as np
from collections import deque


def rgb_to_lab(img):
    """sRGB in [0, 1] -> CIELAB (D65). Lab distances roughly match perceived colour differences."""
    c = np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def enforce_connectivity(labels, min_size):
    """Relabel 4-connected components; merge components smaller than min_size into a neighbour."""
    h, w = labels.shape
    out = -np.ones_like(labels)
    nxt = 0
    for sy in range(h):
        for sx in range(w):
            if out[sy, sx] >= 0:
                continue
            comp, q, adj = [(sy, sx)], deque([(sy, sx)]), -1
            out[sy, sx] = nxt
            while q:
                y, x = q.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w:
                        if out[ny, nx] < 0 and labels[ny, nx] == labels[sy, sx]:
                            out[ny, nx] = nxt
                            comp.append((ny, nx))
                            q.append((ny, nx))
                        elif out[ny, nx] >= 0 and out[ny, nx] != nxt:
                            adj = out[ny, nx]                      # an already-labelled neighbour
            if len(comp) < min_size and adj >= 0:
                for y, x in comp:
                    out[y, x] = adj
            else:
                nxt += 1
    return out


def slic(img, n_segments=100, m=10.0, iters=10):
    lab = rgb_to_lab(img)
    h, w = lab.shape[:2]
    S = int(np.sqrt(h * w / n_segments))                         # grid step = expected superpixel width
    ys, xs = np.meshgrid(np.arange(S // 2, h, S), np.arange(S // 2, w, S), indexing="ij")
    centers = np.c_[lab[ys.ravel(), xs.ravel()], ys.ravel(), xs.ravel()].astype(float)
    # Move each seed to the lowest-gradient pixel in its 3x3 neighbourhood (avoid seeding on an edge)
    gy, gx = np.gradient(lab, axis=(0, 1))
    grad = (gy ** 2 + gx ** 2).sum(-1)
    for c in centers:
        y, x = int(c[3]), int(c[4])
        y0, x0 = max(y - 1, 0), max(x - 1, 0)
        win = grad[y0:y + 2, x0:x + 2]
        dy, dx = np.unravel_index(win.argmin(), win.shape)
        c[3:] = y0 + dy, x0 + dx
        c[:3] = lab[int(c[3]), int(c[4])]
    yy, xx = np.mgrid[:h, :w]
    for _ in range(iters):
        dist = np.full((h, w), np.inf)
        labels = -np.ones((h, w), int)
        for k, (L, a, b, cy, cx) in enumerate(centers):         # each centre only searches a 2S x 2S window
            y0, y1 = max(int(cy) - S, 0), min(int(cy) + S + 1, h)
            x0, x1 = max(int(cx) - S, 0), min(int(cx) + S + 1, w)
            dc = ((lab[y0:y1, x0:x1] - (L, a, b)) ** 2).sum(-1)
            ds = (yy[y0:y1, x0:x1] - cy) ** 2 + (xx[y0:y1, x0:x1] - cx) ** 2
            D = dc + (m / S) ** 2 * ds                               # D^2 = dc^2 + (m/S)^2 ds^2
            better = D < dist[y0:y1, x0:x1]
            dist[y0:y1, x0:x1][better] = D[better]
            labels[y0:y1, x0:x1][better] = k
        for k in range(len(centers)):
            mask = labels == k
            if mask.any():
                centers[k] = np.r_[lab[mask].mean(0), yy[mask].mean(), xx[mask].mean()]
    return enforce_connectivity(labels, min_size=S * S // 4)


def make_scene(rng, h=120, w=160, noise=0.05):
    """Flat background with a disc, a rectangle and a triangle; gt holds the region index of every pixel."""
    yy, xx = np.mgrid[:h, :w]
    gt = np.zeros((h, w), int)
    gt[(yy - 45) ** 2 + (xx - 45) ** 2 < 30 ** 2] = 1
    gt[(yy > 60) & (yy < 110) & (xx > 70) & (xx < 140)] = 2
    gt[(yy < 55) & (xx - 100 > 55 - yy) & (xx - 100 < yy - 10)] = 3
    colors = np.array([[0.55, 0.6, 0.55], [0.85, 0.3, 0.25], [0.25, 0.45, 0.8], [0.6, 0.65, 0.5]])
    img = colors[gt] + noise * rng.standard_normal((h, w, 3))
    return np.clip(img, 0, 1), gt


def boundaries(lab):
    b = np.zeros(lab.shape, bool)
    b[:-1] |= lab[:-1] != lab[1:]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    return b


def boundary_recall(sp, gt, tol=2):
    """Share of ground-truth boundary pixels with a superpixel boundary within `tol` pixels."""
    bs = boundaries(sp)
    near = np.zeros_like(bs)
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            near |= np.roll(np.roll(bs, dy, 0), dx, 1)
    bg = boundaries(gt)
    return near[bg].mean()


def undersegmentation_error(sp, gt):
    """Corrected UE: for each superpixel, the pixels outside its majority ground-truth region, over all pixels."""
    leak = 0
    for s in np.unique(sp):
        counts = np.bincount(gt[sp == s])
        leak += counts.sum() - counts.max()
    return leak / sp.size


white = rgb_to_lab(np.ones((1, 1, 3)))[0, 0]
assert np.allclose(white, [100, 0, 0], atol=0.05), white
print("rgb_to_lab(white) =", np.round(white, 2))


def report(name, sp, gt):
    print(f"{name:<26} {len(np.unique(sp)):4d} segs  recall {boundary_recall(sp, gt):.3f}  UE {undersegmentation_error(sp, gt):.3f}")


try:
    from skimage.segmentation import slic as sk_slic          # optional reference implementation
except ImportError:
    sk_slic = None

_, gt = make_scene(np.random.default_rng(0))
yy, xx = np.mgrid[:120, :160]
report("square grid 12x12", (yy // 12) * 100 + xx // 12, gt)
for noise in (0.0, 0.02, 0.05):
    img, gt = make_scene(np.random.default_rng(0), noise=noise)
    print(f"-- pixel noise std {noise}")
    for m in (1, 10, 40):
        report(f"   SLIC m={m}", slic(img, n_segments=130, m=m), gt)
        if sk_slic is not None:
            report(f"   skimage compactness={m}", sk_slic(img, n_segments=130, compactness=m, start_label=0), gt)
```

Output (Python 3.14, NumPy 2.5, scikit-image 0.26):
```
rgb_to_lab(white) = [ 1.e+02  1.e-02 -1.e-02]
square grid 12x12           140 segs  recall 0.734  UE 0.069
-- pixel noise std 0.0
   SLIC m=1                 130 segs  recall 1.000  UE 0.000
   skimage compactness=1     96 segs  recall 0.998  UE 0.001
   SLIC m=10                130 segs  recall 1.000  UE 0.000
   skimage compactness=10   109 segs  recall 1.000  UE 0.000
   SLIC m=40                130 segs  recall 0.926  UE 0.015
   skimage compactness=40   124 segs  recall 0.950  UE 0.009
-- pixel noise std 0.02
   SLIC m=1                  39 segs  recall 0.779  UE 0.167
   skimage compactness=1      1 segs  recall 0.000  UE 0.346
   SLIC m=10                128 segs  recall 1.000  UE 0.005
   skimage compactness=10    94 segs  recall 0.959  UE 0.041
   SLIC m=40                130 segs  recall 0.934  UE 0.015
   skimage compactness=40   129 segs  recall 0.937  UE 0.011
-- pixel noise std 0.05
   SLIC m=1                  36 segs  recall 0.760  UE 0.181
   skimage compactness=1      1 segs  recall 0.000  UE 0.346
   SLIC m=10                 74 segs  recall 0.946  UE 0.067
   skimage compactness=10    18 segs  recall 0.592  UE 0.222
   SLIC m=40                130 segs  recall 0.954  UE 0.012
   skimage compactness=40   130 segs  recall 0.934  UE 0.013
```

How to read it:
- A plain square grid of the same size catches only 73% of the object boundaries and leaks 6.9% of pixels across them. That is the baseline SLIC has to beat.
- On a clean image, SLIC with `m` 1–10 snaps to every boundary (recall 1.0, UE 0). Raising `m` to 40 makes the regions more square, so they cut across edges a little (recall 0.93, UE 1.5%).
- Noise reverses the ranking. With noise std 0.02–0.05 (mild sensor noise), `m = 1` lets colour noise dominate. Clusters break into speckled, disconnected fragments, and the connectivity pass merges them into a few giant regions (36–39 left, UE 17–18%). scikit-image merges more aggressively and collapses to a single region.
- `m = 40` is the most robust setting under noise: 130 regions, recall 0.95, UE about 1%. The default `compactness=10` sits in between and degrades at noise std 0.05 (74 and 18 segments).
- The two implementations agree on the trend but not the exact numbers. They differ in connectivity merging, Lab scaling details and seed placement, so do not expect identical labels.

## Choosing / trade-offs
- **Compactness `m`.** Low `m` (1–5) for clean, high-contrast images where boundary adherence matters most. Higher `m` (20–40) for noisy, textured or compressed images, or when downstream code wants evenly sized nodes. Check a few values visually with `skimage.segmentation.mark_boundaries`.
- **Number of superpixels.** More superpixels give higher boundary recall and lower UE but less speed-up. A few hundred per megapixel is common. Pick it from the smallest object you must not merge.
- **Smooth first or not.** A light Gaussian blur before SLIC (scikit-image's `sigma` argument) reduces the noise fragmentation shown above, at a small cost in boundary sharpness.
- **SLIC vs alternatives.** SLIC is the fast default. Felzenszwalb graph segmentation adapts region size to content (no fixed count). Watershed on gradients follows edges more tightly. SLICO (zero-parameter SLIC) adapts `m` per cluster. Learned superpixels exist, but rarely justify a model for a preprocessing step.

## Gotchas
- `m` depends on the colour space and its scale. Lab values span 0–100; if you run SLIC on RGB in [0, 1], the same `m` is about 100x too strong. scikit-image converts to Lab by default for 3-channel input (`convert2lab`), and its `compactness` has no fixed scale on other channels.
- The requested count is a target. The grid step rounds, and the connectivity pass deletes small regions, so the output count can be far lower, which is how the noisy runs end up with 1–74 segments. Check `len(np.unique(labels))`.
- Leave connectivity enforcement on. Without it a "superpixel" can be several disconnected pieces, which breaks region-adjacency graphs and per-region features.
- Superpixels inherit the colour model's blind spots. Two objects with the same colour (a grey car on a grey road) are merged regardless of `m`. Add texture or depth channels if that matters.
- The Python connectivity loop is slow for large images. For real images use scikit-image or OpenCV's `cv2.ximgproc.createSuperpixelSLIC` (in opencv-contrib).

## Related
- [[k-means-from-scratch]] - SLIC is k-means restricted to local windows in a 5-D feature space.
- [[segmentation]] - superpixels as a preprocessing step for labelling and classical segmentation.
- [[canny-edge-detector-from-scratch]] - another low-level edge-respecting step; boundary recall compares against edges like these.
- [[mean-shift-clustering-from-scratch]] - the mode-seeking alternative that also clusters in joint colour-space features.
- [[gcn-from-scratch-numpy]] - superpixel adjacency graphs as GNN input.

## References
- Achanta et al. (2012), "SLIC Superpixels Compared to State-of-the-Art Superpixel Methods", IEEE TPAMI: https://doi.org/10.1109/TPAMI.2012.120
- scikit-image `segmentation.slic` documentation: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic
