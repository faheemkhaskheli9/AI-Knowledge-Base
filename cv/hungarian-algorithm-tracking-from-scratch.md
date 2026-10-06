---
title: Hungarian algorithm for detection-to-track matching from scratch (optimal assignment, IoU cost, SORT)
category: cv
tags: [hungarian-algorithm, linear-assignment, kuhn-munkres, multi-object-tracking, sort, iou, detr, bipartite-matching, numpy, from-scratch]
use_cases:
  - "match this frame's detections to existing tracks in a multi-object tracker"
  - "solve a min-cost one-to-one assignment (workers to jobs, predictions to ground truth)"
  - "understand the bipartite matching step in DETR / set-prediction losses"
  - "see why greedy matching swaps IDs and when it is good enough"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1002/nav.3800020109
  - https://arxiv.org/abs/1602.00763
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
---

# Hungarian algorithm for detection-to-track matching from scratch (optimal assignment, IoU cost, SORT)

## Summary
Given an `n × m` cost matrix, the assignment problem asks for a one-to-one matching of rows to columns with minimum total cost. Trackers such as SORT and ByteTrack face it every frame: rows are existing tracks, columns are new detections, and the cost is `1 − IoU` or an appearance distance. DETR solves the same problem to pair predictions with ground-truth boxes before computing its loss. The Hungarian (Kuhn-Munkres) algorithm solves it exactly in `O(n³)`. Below is a roughly 40-line shortest-augmenting-path version with dual potentials. It matches brute force on 200 random matrices, including rectangular ones. On random 8×8 costs, greedy "take the cheapest pair first" is suboptimal 265 times in 300, with 29.6% excess cost on average. In a two-pedestrian IoU example, greedy pays 1.26 where optimal pays 0.80, which shows up as an ID swap.

## Key concepts
- **Assignment as an LP.** Minimize `Σ c_ij x_ij` with each row and each column used at most once. The constraint matrix is totally unimodular, so the optimum is integral and the dual is clean.
- **Potentials (duals).** Keep `u_i + v_j ≤ c_ij`. The *reduced cost* `c_ij − u_i − v_j ≥ 0`, and edges at zero reduced cost are "tight". An assignment using only tight edges is optimal, by complementary slackness.
- **Augmenting path.** Add one row at a time. Grow a tree of alternating tight edges until a free column is reached, shifting potentials by the minimum slack `delta` whenever the tree gets stuck. Then flip the path. Each row costs `O(nm)`, so `O(n²m)` in total.
- **Rectangular and unmatched.** With `n ≤ m`, every row is matched and extra columns stay free (transpose if `n > m`). In tracking, add **gating**: after solving, reject pairs whose IoU is below a threshold (often 0.3). The leftover detections start new tracks and the leftover tracks age out.
- **Greedy is myopic.** Taking the globally cheapest pair first can force a terrible second pair. Optimal matching gives up a slightly worse first pair to avoid that.

## When to use / scenarios
- Practice: the association step of tracking-by-detection (SORT, DeepSORT, ByteTrack, OC-SORT) for people, vehicles or retail shoppers in [[video-analytics]]. Also matching predicted boxes to ground truth for evaluation (MOTA/IDF1) or for DETR's Hungarian loss ([[object-detection]]), assigning drivers to riders, matching keypoints or cells across microscopy frames, and record linkage with a similarity matrix.
- Learning: a compact example of primal-dual optimization.
- Interviews: "how does SORT associate detections", "why does DETR need bipartite matching", "greedy vs optimal matching".
- Not for: thousands of rows per frame where `O(n³)` hurts (use sparse or auction algorithms, or gate to a sparse graph first); many-to-one assignments with capacities (use min-cost flow); or when greedy is already near-optimal because costs are well separated (crowd-free scenes, high-FPS video). Greedy is simpler and faster there.

## Setup & code
NumPy plus `itertools` from the stdlib. Runs in under a second. In production, call `scipy.optimize.linear_sum_assignment(cost)` instead, which runs the same algorithm in C.

```python
import itertools
import numpy as np

rng = np.random.default_rng(0)


def hungarian(cost):
    """Min-cost assignment for an n x m cost matrix (n <= m), O(n^2 m).
    Shortest augmenting path with row/column potentials u, v. Returns col index per row."""
    n, m = cost.shape
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)              # p[j] = row (1-based) matched to column j; column 0 is a dummy
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while p[j0] != 0:                       # grow the alternating tree until a free column is reached
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]   # reduced cost
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):              # shift potentials so a new zero-reduced-cost edge appears
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
        while j0:                               # flip the augmenting path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assign = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if p[j]:
            assign[p[j] - 1] = j - 1
    return assign


def greedy(cost):
    """Repeatedly take the cheapest remaining (row, col) pair."""
    c, assign = cost.astype(float).copy(), np.empty(cost.shape[0], dtype=int)
    for _ in range(cost.shape[0]):
        i, j = np.unravel_index(np.argmin(c), c.shape)
        assign[i] = j
        c[i, :], c[:, j] = np.inf, np.inf
    return assign


total = lambda c, a: c[np.arange(len(a)), a].sum()

for trial in range(200):                        # check against brute force on small matrices
    n = int(rng.integers(1, 6))
    m = n + int(rng.integers(0, 3))
    c = rng.integers(0, 20, (n, m)).astype(float)
    best = min(sum(c[i, perm[i]] for i in range(n)) for perm in itertools.permutations(range(m), n))
    assert total(c, hungarian(c)) == best
print("matches brute force on 200 random matrices (n<=5, rectangular included)")

worse, gap = 0, []
for _ in range(300):
    c = rng.uniform(0, 1, (8, 8))
    h, g = total(c, hungarian(c)), total(c, greedy(c))
    worse += g > h + 1e-9
    gap.append(g / h - 1)
print(f"8x8 uniform costs: greedy worse than optimal in {worse}/300, mean excess cost {np.mean(gap):.1%}")


def iou(a, b):                                  # boxes as x1, y1, x2, y2
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# Two pedestrians walking side by side: predicted track boxes vs this frame's detections.
tracks = np.array([[7, 0, 27, 50], [17, 0, 37, 50]])
dets = np.array([[12, 0, 32, 50], [2, 0, 22, 50]])
cost = np.array([[1 - iou(t, d) for d in dets] for t in tracks])
print("\n1 - IoU cost (rows = tracks, cols = detections):\n", np.round(cost, 3))
print("greedy   :", greedy(cost), f"total cost {total(cost, greedy(cost)):.3f}")
print("hungarian:", hungarian(cost), f"total cost {total(cost, hungarian(cost)):.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
matches brute force on 200 random matrices (n<=5, rectangular included)
8x8 uniform costs: greedy worse than optimal in 265/300, mean excess cost 29.6%

1 - IoU cost (rows = tracks, cols = detections):
 [[0.4   0.4  ]
 [0.4   0.857]]
greedy   : [0 1] total cost 1.257
hungarian: [1 0] total cost 0.800
```

Track 0 overlaps both detections equally (IoU 0.6 each), and greedy, breaking the tie, gives it detection 0. That leaves track 1 with detection 1, at IoU 0.14. A 0.3 gate would then reject that pair, killing track 1 and spawning a new ID for the same person. The optimal matching gives track 0 the other detection, and both pairs get IoU 0.6. Uniform random costs are the worst case for greedy. In real tracking most cost matrices are nearly diagonal, so greedy usually agrees with the optimum and differs only in crowded crossings, which is exactly where ID switches come from.

## Choosing / trade-offs
- **Hungarian vs greedy.** Greedy is `O(nm log nm)`, trivial to write, and fine when objects are well separated. Hungarian is optimal and costs little at tracking scales (tens of objects). Prefer Hungarian unless profiling says otherwise.
- **Cost choice.** `1 − IoU` alone fails under occlusion and fast motion. DeepSORT and its successors mix in appearance-embedding distance and Mahalanobis distance from a Kalman filter prediction ([[kalman-filter-from-scratch]]). ByteTrack runs two matching rounds: high-confidence detections first, then low-confidence ones.
- **Gate before or after.** Setting gated entries to a large cost before solving keeps the solver from "trading" one good match for two bad ones that later get rejected. Rejecting after solving is simpler. SORT rejects after.
- **Library.** `scipy.optimize.linear_sum_assignment` handles rectangular matrices, `maximize=True`, and inputs into the thousands. `lap` / `lapx` (Jonker-Volgenant) are faster for large dense problems.

## Gotchas
- Never put `inf` in the cost matrix for forbidden pairs with SciPy unless every row still has a feasible column. Otherwise it raises "cost matrix is infeasible". Use a large finite cost, then gate.
- Maximizing IoU means minimizing `1 − IoU` (or passing `maximize=True`). Feeding raw IoU to a minimizer silently produces the worst matching.
- If `n > m`, transpose, solve, and swap the result back. The code here assumes `n ≤ m`.
- Ties and floating point mean many equally optimal matchings can exist. Do not assert a specific assignment in tests; assert the total cost.
- An optimal assignment does not mean every pair is good. Always gate, or the matcher will happily pair a track with a detection across the frame.
- The per-frame cost is `O(n³)`. With hundreds of objects (crowds, cells), gate first to a sparse graph, or split the frame into connected components and solve each separately.

## Related
- [[video-analytics]] - tracking-by-detection pipelines where this is the association step.
- [[object-detection]] - DETR's set-prediction loss and the detectors whose boxes are matched.
- [[kalman-filter-from-scratch]] - the motion model that predicts track boxes before matching.
- [[classification-metrics-and-cross-validation-from-scratch]] - matching predictions to ground truth before scoring is the same step in detection mAP.
- [[sinkhorn-optimal-transport-from-scratch]] - Sinkhorn: the soft, differentiable version of this hard assignment.

## References
- Kuhn (1955), "The Hungarian method for the assignment problem", Naval Research Logistics Quarterly: https://doi.org/10.1002/nav.3800020109
- Bewley et al. (2016), "Simple Online and Realtime Tracking" (SORT): https://arxiv.org/abs/1602.00763
- SciPy `linear_sum_assignment`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
