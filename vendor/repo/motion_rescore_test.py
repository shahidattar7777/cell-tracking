"""Would a motion bias flip the failures? (offline upper bound)

Run from vendor/repo:   uv run python motion_rescore_test.py [stem]

THE QUESTION
------------
There is no spatial candidate gate -- predict_video scores the full cartesian
product of source x target and filters only on probability. So the true partner
was always scored. It lost on SCORE, to a nearer competitor.

A motion bias would add a term to the edge logits before the softmax:

    raw_adj[i,j] = raw[i,j] + alpha * (-residual[i,j] / sigma)

where residual[i,j] is the distance from target j to source i's velocity-
predicted position. This only helps if, for the edges the model got wrong,

    residual(TRUE partner)  <  residual(CHOSEN competitor)

If the chosen competitor is ALSO near the predicted position -- or nearer --
then no value of alpha flips the decision and the idea is dead.

WHAT THIS REPORTS
-----------------
For each failure with both endpoints detected, using k-frame GT velocity:

    d_true      raw distance source -> true partner
    d_chosen    raw distance source -> the node the model actually linked
    r_true      residual of the true partner vs predicted position
    r_chosen    residual of the chosen competitor vs predicted position

    FLIPPABLE   r_true < r_chosen   (motion bias points the right way)

This is an UPPER BOUND: it uses ground-truth velocity. The real pipeline would
compute velocity from its own -- sometimes wrong -- prior assignments, so the
achievable number is lower. If the bound is not comfortably positive, stop.
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
PRED_DIR = Path(r"predictions\shahi\unet_transformer\split_0")
DEFAULT_STEM = "6bba_207c6aaf"
WINDOWS = [1, 2, 3, 5]
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset, compute_metric  # noqa: E402


def ancestor(node, parent, k):
    cur = node
    for _ in range(k):
        if cur not in parent:
            return None
        cur = parent[cur]
    return cur


def main(stem):
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    pred = td.graph.IndexedRXGraph.from_geff(PRED_DIR / f"{stem}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred
    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    print(f"{stem}: TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}")

    scale = np.array(ds.scale[-3:], dtype=float)

    # GT geometry + lineage
    gna = ds.tracks.node_attrs()
    gt_pos = {
        r["node_id"]: np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        for r in gna.iter_rows(named=True)
    }
    gt_edges = [(int(s), int(t)) for s, t in ds.tracks.edge_list()]
    gt_parent = {t: s for s, t in gt_edges}

    # predicted geometry + matching
    na = pred.node_attrs()
    ea = pred.edge_attrs()
    p_pos = {
        r["node_id"]: np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        for r in na.iter_rows(named=True)
    }
    matched = na.filter(pl.col("match_node_id") != -1)
    gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
    pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))

    tp_pairs = {
        (pred_to_gt[s], pred_to_gt[t])
        for s, t in zip(
            ea.filter(pl.col("matched_edge_mask"))["source_id"],
            ea.filter(pl.col("matched_edge_mask"))["target_id"],
        )
    }
    fn = [(s, t) for s, t in gt_edges if (s, t) not in tp_pairs]
    both = [(s, t) for s, t in fn if s in gt_to_pred and t in gt_to_pred]
    print(f"FN edges {len(fn)}   both endpoints detected {len(both)}")

    for k in WINDOWS:
        rows = []
        for gs, gtt in both:
            anc = ancestor(gs, gt_parent, k)
            if anc is None:
                continue
            ps, pt = gt_to_pred[gs], gt_to_pred[gtt]

            v = (gt_pos[gs] - gt_pos[anc]) / k
            predicted = gt_pos[gs] + v

            # what the model linked this source to instead
            out = ea.filter(pl.col("source_id") == ps)
            if len(out) == 0:
                continue
            pc = int(out["target_id"][0])
            if pc not in p_pos:
                continue

            rows.append({
                "d_true": float(np.linalg.norm(p_pos[pt] - p_pos[ps])),
                "d_chosen": float(np.linalg.norm(p_pos[pc] - p_pos[ps])),
                "r_true": float(np.linalg.norm(p_pos[pt] - predicted)),
                "r_chosen": float(np.linalg.norm(p_pos[pc] - predicted)),
            })

        if not rows:
            print(f"  k={k}: no usable cases")
            continue

        dt = np.array([r["d_true"] for r in rows])
        dc = np.array([r["d_chosen"] for r in rows])
        rt = np.array([r["r_true"] for r in rows])
        rc = np.array([r["r_chosen"] for r in rows])

        nearer_raw = int((dc < dt).sum())          # what the model saw
        flippable = int((rt < rc).sum())           # what motion would see
        margin = np.median(rc - rt)

        print(f"\n  k={k}  n={len(rows)}")
        print(f"    by RAW distance     chosen nearer than true : {nearer_raw:3d}/{len(rows)}"
              f"  ({100*nearer_raw/len(rows):3.0f}%)   <- the failure")
        print(f"    by MOTION residual  true nearer than chosen : {flippable:3d}/{len(rows)}"
              f"  ({100*flippable/len(rows):3.0f}%)   <- FLIPPABLE")
        print(f"    median d_true {np.median(dt):5.2f}  d_chosen {np.median(dc):5.2f}"
              f"   |  r_true {np.median(rt):5.2f}  r_chosen {np.median(rc):5.2f}"
              f"   median margin {margin:+5.2f}")

    print("\n" + "=" * 74)
    print("FLIPPABLE near 50% -> motion points the right way no more often than")
    print("chance; no alpha rescues it. Comfortably above 70% -> worth building.")
    print("Remember this uses GT velocity, so it is an optimistic ceiling.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM)