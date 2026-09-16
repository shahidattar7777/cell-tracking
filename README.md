# Cell Tracking During Development — A Diagnostic Rebuild

> Reverse-engineering a 3D+time cell-tracking pipeline to find out *why* it
> fails, rather than tuning it until it stops.

[![score](https://img.shields.io/badge/leaderboard-0.866-blue)]()
[![validation](https://img.shields.io/badge/local%20val%20(20%20videos)-0.879-green)]()
[![detection](https://img.shields.io/badge/node%20recall-0.993-brightgreen)]()

---

## TL;DR

The baseline is **not detection-limited**. It finds 99.3% of annotated nuclei
and then wires them together wrongly. Of the edge errors that survive global
optimisation, **62 of 67 are cases where the model linked a cell to a nearer
neighbour instead of its true partner** — often at high confidence, and often
when the true partner had moved further than the metric's own 7 µm matching
cutoff.

Two structural findings came out of reading the code rather than sweeping it:

1. A `softmax(dim=0)` over candidate parents, combined with a 0.5 threshold,
   makes in-degree ≥ 2 **arithmetically impossible** — so the ILP downstream
   had no alternatives to arbitrate and was acting as a filter, not a solver.
2. Two parameters that materially affect the result (`cfg.threshold`,
   `pool_kernel_um`) have **no CLI flags**, and are therefore unreachable by
   the environment-variable sweeps most forks run.

Acting on the first of these, plus an ILP cost adjustment, produced a
**+0.025 mean edge-Jaccard gain on the three worst videos** — 3/3, no cherry-picking.

---

## The problem

```
  100 frames  ×  64 × 256 × 256 voxels  ×  uint16
  ┌─────────────────────────────────────────────────┐
  │  t=0        t=1        t=2       ...    t=99    │
  │   ●  ●       ●  ●       ●   ●            ●  ●   │
  │  ●    ●  →  ●    ●  →  ●     ●   →  ...  ●   ●  │
  │    ●  ●       ● ●        ●  ●             ● ●   │
  └─────────────────────────────────────────────────┘
       detect ──────► link ──────► lineage graph

  voxels are ANISOTROPIC:  z = 1.625 µm,  y = x = 0.40625 µm   (4:1)
  labels are SPARSE:       ~2–5% of cells annotated per video
```

**Metric.** `score = adjusted_edge_jaccard + 0.1 × division_jaccard`.
Predictions are matched to ground truth by bipartite assignment on scaled
centroid distance, max 7 µm. Unmatched predictions are largely *not*
penalised — but this is narrower than it sounds (see [Precision is not
free](#precision-is-not-free)).

**Data.** 199 labelled training videos. Estimated node counts span
**3,783 → 78,644** — a 21× range, with a standard deviation 82% of the mean.
Only two stem prefixes (`6bba`: 128, `44b6`: 71), which almost certainly means
two embryos, making cross-embryo leakage a live concern for any split.

---

## Pipeline as built

```
   raw zarr
      │
      ▼
 ┌─────────────┐
 │TemporalUNet3D│  window_size=2, downsample=[1,4,4]
 │  detector   │  ── anisotropy handled here: y/x ÷4 makes
 └─────────────┘     the data isotropic before convolution
      │
      ▼  local-max peaks   (pool_kernel_um — config says 5.0, code runs 3.0)
 ┌─────────────┐
 │ transformer │  scores candidate t → t+1 links → edge_prob
 │ edge scorer │
 └─────────────┘
      │
      ▼  softmax(dim=0) + threshold  ◄── THE BOTTLENECK
 ┌─────────────┐
 │ ILP  /  greedy │  global assignment over the candidate graph
 └─────────────┘
      │
      ▼
   .geff lineage graph
```

---

## Findings

### 1. Detection is solved; linking is not

Across the 20-video validation set:

| metric | mean | min |
|---|---|---|
| node recall | **0.9930** | 0.9484 |
| edge Jaccard | 0.8850 | 0.5942 |

A *perfect* detector would recover at most ~3.5% of edge mass on a typical
video (each missed node destroys at most two edges). The observed loss is
9–13%. **Replacing the detector cannot reach most of the gap.**

On `44b6_d78e09d9`, of 33 missed edges: 13 had a missing endpoint
(detection), **20 had both endpoints correctly detected** (linking).

### 2. Greedy assignment was discarding usable scores

Switching one flag, same weights, same scores:

```
                greedy     ILP      Δ
 6bba_0c7fa718   0.898  →  0.917   +0.019
 44b6_d78e09d9   0.816  →  0.874   +0.058
 division FPs    36     →  0
```

The transformer's output was fine. The thing consuming it was not.

### 3. The softmax caps in-degree at 1 — by arithmetic

```python
probs = torch.softmax(raw, dim=0)     # raw is (n_src, n_tgt)
...
if probs[i, j] > cfg.threshold        # 0.5
```

`dim=0` normalises across **sources**, so for any target the probabilities sum
to 1. At a 0.5 threshold, **at most one source can ever clear it**.
`max_parents_per_node` never binds; the constraint is mathematical.

Two consequences:

- The ILP receives exactly one candidate parent per node. It can keep or
  delete, never re-parent. That is the opposite of what a solver is for.
- A target whose softmax is *diffuse* — 0.45 / 0.35 / 0.20 across three
  plausible parents — clears nothing and gets **no parent at all**. Roughly
  **1,950 nodes per video** are orphaned because the model was uncertain
  rather than wrong.

Measured effect of lowering the threshold to 0.3 on one video: TP +5, FP +6,
score flat — *but* the ILP weights became live levers for the first time.

### 4. The dominant failure is a distance prior, not noise

For every FN edge where both endpoints were detected, we asked what the model
linked instead, and how far away it was.

**`6bba_207c6aaf`** (worst video, edge-Jaccard 0.657), 69 linking failures:

```
  displaced (model linked elsewhere)  67
  orphaned  (no link at all)           2

  chose a NEARER node                 62 / 67
  median TRUE distance              6.08 µm
  median CHOSEN distance            2.24 µm
  median confidence                  0.612
  max true displacement            19.02 µm   ◄── beyond the metric's
                                                  own 7 µm match cutoff
```

**`44b6_d78e09d9`** (good video, 0.877), same query, 19 of 20 failures follow
the identical pattern — but at **much higher confidence** (7/10 above p=0.8).
On good videos the model is *confidently* wrong; on bad videos it is
*uncertain* and wrong.

Representative cases:

```
  true = 10.41 µm   chosen = 1.41   p = 0.940
  true = 10.28 µm   chosen = 2.24   p = 0.846
  true =  6.08 µm   chosen = 1.00   p = 0.897
```

### 5. The anisotropy hypothesis — refuted

`edge_dist` looked like it might be computed in raw voxels, which would
compress z by 4× and bias the model toward lateral neighbours. Tested by
recomputing every distance two ways:

```
  ratio (µm / stored) across 300 sampled edges
  min 1.625 │ 25% 1.625 │ 50% 1.625 │ 75% 1.625 │ max 1.625
```

Exactly constant. Distances are **isotropic**, expressed in downsampled units
where 1 unit = 1.625 µm. No bug. The distance prior is a genuine modelling
failure, not an implementation error.

*A negative result, kept because it rules out an attractive wrong answer.*

### 6. ILP continuity costs recover displaced links

`--ilp-appearance-weight` and `--ilp-disappearance-weight` both default to
**0.1** — nearly free. The solver has little reason to prefer maintaining a
track over breaking it and starting another.

Sweep on the three worst videos:

| config | 207c6aaf | 57b7cc1e | 6feb10f0 | mean Δ |
|---|---|---|---|---|
| baseline (0.1) | 0.657 | 0.662 | 0.807 | — |
| app = 0.5 | 0.692 | **0.682** | 0.814 | +0.021 |
| app = 1.0 | **0.711** | 0.675 | 0.814 | **+0.025** |

Improved on **3/3 videos at both settings**. The mechanism is visible in the
counts — on `207c6aaf`, TP 367 → 392 and FN 83 → 58, while FP actually *fell*
(109 → 101). Recall is recovered without paying in precision.

**Cost:** node recall declines with the weight (0.981 → 0.964). The solver
drops nodes it cannot fit into a continuous track. This is the eventual
ceiling on the lever.

**Runtime cliff:** at weight 5.0 the ILP does not converge in usable time
(5 min → 2+ hours per video). The tightened constraints appear to break
branch-and-bound. Usable range is bounded above by ~2.0.

### 7. Error mass is concentrated

Three of twenty videos carry **63% of all false positives and 64% of all
false negatives**:

```
  6bba_207c6aaf   TP  367   FP 109   FN  83    edgeJ 0.657
  6bba_57b7cc1e   TP 1299   FP 371   FN 293    edgeJ 0.662
  6bba_6feb10f0   TP 1184   FP 125   FN 159    edgeJ 0.807
  ────────────────────────────────────────────────────────
  other 17                                     0.90 – 1.00
```

Density correlates with score, but weakly — `corr(n_total, edge_jaccard) =
−0.47`. Notably `207c6aaf` is *not* dense (12k estimated nodes) yet scores
worst, so crowding does not explain the outliers.

### Precision is not free

A widely repeated reading of this metric is that unmatched predictions carry
no penalty. Reading the reference implementation shows this is narrower than
it sounds: an edge from a *matched* node to an unmatched one is counted as a
full false positive. Only edges between two entirely unmatched nodes are
ignored. Every spurious link hanging off a real cell costs you.

---

## Validation methodology

- **20-video subset**, stratified by estimated node count, spanning
  3,783 → 65,511. Prefix ratio 13:7, matching the population's 128:71.
  Frozen before any experiments were run.
- **Local mean (micro-averaged) 0.879 vs leaderboard 0.866** — a 0.013 gap.
  Validation tracks, so local iteration is meaningful.
- Two-video validation was **optimistic by ~0.03** and gave no visibility into
  the density range. Most published single-video results on this baseline
  should be read with that in mind.
- **MPS (Apple) and CUDA agree to 16 significant digits** on identical inputs.
  Device is not a confound.

---

## Reproducibility notes

<details>
<summary><b>Offline Kaggle submission — dependency resolution</b></summary>

Code competitions run with internet disabled. Installing the project's wheel
set naively breaks the environment: replacing NumPy or SciPy with versions
from the wheel folder causes `ImportError: cannot import name '_center' from
'numpy._core.umath'`, because Kaggle's preinstalled SciPy is compiled against
its own NumPy.

Working install:

```python
import subprocess, glob
wheels = [w for w in glob.glob(f"{WHEELS}/*.whl")
          if not any(k in w.lower() for k in ("numpy", "scipy"))]
subprocess.run(["pip", "install", "--no-index", "--no-deps",
                f"--find-links={WHEELS}"] + wheels, check=True)
```

Leave NumPy and SciPy alone. Everything else installs cleanly with
`--no-deps`.
</details>

<details>
<summary><b>Guards worth having</b></summary>

Six separate path bugs in this project produced *valid-looking output* rather
than errors — including scoring ground truth against itself, which returned a
perfect 1.000 and an `adj_edge_jaccard` of 1.098.

```python
assert gt_path.exists()
assert pred_path.exists()
assert pred_path.stem == stem
assert "predictions" in str(pred_path)
```

The mathematical invariant is the cheapest and catches the most:

```python
assert row["adj_edge_jaccard"] <= 1.0   # except on genuine under-prediction
```

(One video legitimately exceeds 1.0 — the metric awards a bonus for predicting
fewer nodes than estimated. Treat it as a warning, not an assertion.)
</details>

---

## Related work

**HOCT** (Bragantini, Theodoro & Royer, 2026) reaches the same structural
conclusion from the opposite direction: that candidate-graph topology carries
little usable signal because edges sharing a node have near-random label
agreement (adjusted homophily 0.01 ± 0.04 on the line graph). Their answer is
an edge-centric transformer where candidate links attend to one another under
a 3D geometric prior.

Their softmax formulation is the principled version of the bottleneck
described in §3 — normalised per target *and* per temporal gap, with a
constant in the denominator acting as an implicit "no parent" option, so a
link need not exceed 0.5 to survive. The ILP decides instead of the threshold.

---

## Status & open questions

**Done**
- Verified metric harness, independent of the reference implementation
- 20-video stratified validation set, tracking the leaderboard to 0.013
- Failure mechanism identified and measured on two videos at opposite ends of
  the quality range
- ILP continuity costs: +0.025 mean on the worst three videos, 3/3

**Open**
- Does the ILP gain hold across all 20, or only the worst?
- `pool_kernel_um` is untuned and unreachable from the CLI — the weights ship
  with 5.0 and the code runs 3.0
- The 2 of 69 failures where the true link was *nearer* and still lost are
  unexplained by the distance prior
- Zero-distance edges (`chosen_dist = 0.00`) suggest duplicate detections
- Line-graph homophily has not been measured on this dataset

---

*Built as a learning exercise: every function typed and understood rather than
pasted. The diagnosis is the deliverable.*
