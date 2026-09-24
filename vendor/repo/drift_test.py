"""Is there a population-level tissue drift, and does it discriminate?

Run from vendor/repo:   uv run python drift_test.py [stem ...]

CPU only. Ground truth geometry plus the prediction graph -- no model, no GPU.

WHY THIS IS NOT THE MOTION TEST AGAIN
-------------------------------------
motion_test.py and motion_rescore_test.py used each cell's OWN velocity history
and both came back at chance (47-50%). But a zebrafish embryo is not a bag of
independently wandering cells: epiboly, convergent extension and tissue flow
produce COHERENT directional motion shared across a whole region.

A shared drift is a different signal from per-cell velocity. Per-cell velocity
is noisy -- GT centroids are integer voxels and median displacement is ~3 um, so
half-voxel rounding is a large fraction of it. A drift shared by hundreds of
cells survives that averaging while an individual cell's velocity does not.

So this is genuinely untested, not a rerun.

It may also connect an observation left hanging much earlier: on 44b6 the
failures clustered after frame 90, with nothing before frame 67. If drift grows
during development, that would explain the timing.

WHAT IT REPORTS
---------------
1. GLOBAL DRIFT       mean displacement vector over all GT edges, and the
                      coherence ratio |mean| / mean|.|. Near 0 means cells move
                      in random directions; near 1 means they all move together.

2. DRIFT OVER TIME    per-frame drift magnitude and coherence against frame
                      index, plus a first-half / second-half split.

3. DISCRIMINATION     for each STOLEN failure, whether the TRUE partner sits
                      closer to the drift-predicted position than the claimant
                      the model actually chose. Same FLIPPABLE structure as the
                      motion tests, so the number is directly comparable to the
                      47-50% that killed them.

Drift is estimated per frame from ALL GT edges at that frame, which includes the
failing edge itself. With hundreds of edges per frame the self-contribution is
negligible, but it does make (3) a mild upper bound.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td

# --- paths: EDIT THESE ---------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
PRED_DIR = Path(r"predictions\shahi\ilp_app2.0\split_0")
DEFAULT_STEMS = ["6bba_207c6aaf", "44b6_d78e09d9"]
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset, compute_metric  # noqa: E402


def gt_geometry(stem):
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    scale = np.array(ds.scale[-3:], dtype=float)
    na = ds.tracks.node_attrs()
    pos, frame = {}, {}
    for r in na.iter_rows(named=True):
        i = r["node_id"]
        pos[i] = np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        frame[i] = int(r["t"])
    edges = [(int(s), int(t)) for s, t in ds.tracks.edge_list()]
    return ds, pos, frame, edges


def report_global(pos, edges):
    disp = np.array([pos[t] - pos[s] for s, t in edges])
    mean_vec = disp.mean(axis=0)
    coherence = np.linalg.norm(mean_vec) / np.linalg.norm(disp, axis=1).mean()
    print("\n  GLOBAL DRIFT")
    print(f"    mean displacement (z,y,x)  "
          f"[{mean_vec[0]:+.3f} {mean_vec[1]:+.3f} {mean_vec[2]:+.3f}] um")
    print(f"    |mean|                     {np.linalg.norm(mean_vec):.3f} um")
    print(f"    mean |displacement|        {np.linalg.norm(disp,axis=1).mean():.3f} um")
    print(f"    coherence |mean|/mean|.|   {coherence:.3f}")
    print("      (0 = random directions, 1 = every cell moving identically)")
    return coherence


def per_frame_drift(pos, frame, edges, min_n=5):
    """drift[t] = mean displacement of GT edges leaving frame t."""
    by_t = {}
    for s, t in edges:
        by_t.setdefault(frame[s], []).append(pos[t] - pos[s])
    drift, coh = {}, {}
    for t, v in by_t.items():
        if len(v) < min_n:
            continue
        a = np.array(v)
        m = a.mean(axis=0)
        drift[t] = m
        coh[t] = np.linalg.norm(m) / max(np.linalg.norm(a, axis=1).mean(), 1e-9)
    return drift, coh


def report_time(drift, coh):
    ts = sorted(drift)
    if not ts:
        print("\n  DRIFT OVER TIME: too few edges per frame")
        return
    mags = np.array([np.linalg.norm(drift[t]) for t in ts])
    cohs = np.array([coh[t] for t in ts])
    mid = len(ts) // 2
    print("\n  DRIFT OVER TIME")
    print(f"    frames with >=5 edges      {len(ts)}  "
          f"(t {ts[0]}..{ts[-1]})")
    print(f"    magnitude  first half {mags[:mid].mean():.3f}   "
          f"second half {mags[mid:].mean():.3f} um")
    print(f"    coherence  first half {cohs[:mid].mean():.3f}   "
          f"second half {cohs[mid:].mean():.3f}")
    r = np.corrcoef(np.array(ts, dtype=float), mags)[0, 1]
    print(f"    corr(frame, drift magnitude)  {r:+.3f}")


def report_discrimination(stem, ds, pos, frame, drift):
    pred = td.graph.IndexedRXGraph.from_geff(PRED_DIR / f"{stem}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred
    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    scale = np.array(ds.scale[-3:], dtype=float)

    na, ea = pred.node_attrs(), pred.edge_attrs()
    p_pos = {r["node_id"]: np.array([r["z"], r["y"], r["x"]], float) * scale
             for r in na.iter_rows(named=True)}
    matched = na.filter(pl.col("match_node_id") != -1)
    gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
    pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))

    tp = ea.filter(pl.col("matched_edge_mask"))
    tp_pairs = {(pred_to_gt[s], pred_to_gt[t])
                for s, t in zip(tp["source_id"], tp["target_id"])}
    fn = [(s, t) for s, t in ds.tracks.edge_list() if (s, t) not in tp_pairs]
    both = [(s, t) for s, t in fn if s in gt_to_pred and t in gt_to_pred]

    rt, rc = [], []
    for gs, gtt in both:
        t0 = frame[gs]
        if t0 not in drift:
            continue
        ps, pt = gt_to_pred[gs], gt_to_pred[gtt]
        out = ea.filter(pl.col("source_id") == ps)
        if len(out) == 0:
            continue
        pc = int(out["target_id"][0])
        if pc not in p_pos:
            continue
        predicted = p_pos[ps] + drift[t0]
        rt.append(np.linalg.norm(p_pos[pt] - predicted))
        rc.append(np.linalg.norm(p_pos[pc] - predicted))

    print(f"\n  DISCRIMINATION   TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}")
    if not rt:
        print("    no usable failure cases")
        return
    rt, rc = np.array(rt), np.array(rc)
    flip = int((rt < rc).sum())
    print(f"    n = {len(rt)}")
    print(f"    residual TRUE partner    median {np.median(rt):5.2f} um")
    print(f"    residual CHOSEN decoy    median {np.median(rc):5.2f} um")
    print(f"    median margin            {np.median(rc - rt):+5.2f} um")
    print(f"    FLIPPABLE (true nearer)  {flip}/{len(rt)}  "
          f"({100*flip/len(rt):.0f}%)")


def main(stems):
    for stem in stems:
        print("=" * 74)
        print(stem)
        ds, pos, frame, edges = gt_geometry(stem)
        print(f"  GT nodes {len(pos)}   GT edges {len(edges)}")
        report_global(pos, edges)
        drift, coh = per_frame_drift(pos, frame, edges)
        report_time(drift, coh)
        try:
            report_discrimination(stem, ds, pos, frame, drift)
        except FileNotFoundError:
            print(f"\n  (no prediction at {PRED_DIR/(stem+'.geff')})")

    print("\n" + "=" * 74)
    print("coherence < 0.2   -> no meaningful tissue drift; hypothesis dead")
    print("coherence > 0.4   -> real coherent flow worth encoding")
    print("FLIPPABLE ~50%    -> drift does not separate the decoys either,")
    print("                     same verdict as per-cell velocity")
    print("FLIPPABLE > 70%   -> a drift term in the ILP objective is justified")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_STEMS)