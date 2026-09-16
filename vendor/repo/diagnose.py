"""edge_dist scaling check + FN edge classification.

Run from vendor/repo:   uv run python diagnose.py <stem>
Defaults to the worst video in the 20-video baseline if no stem given.

Two things it answers:

1. Is edge_dist in voxels or micrometres? If voxels, z is compressed 4x
   relative to physical space, so the model's notion of "near" is wrong
   along the axis where voxels are coarsest.

2. For every FN edge: was it a detection failure (endpoint missing), or a
   linking failure (both endpoints found)? And for linking failures, what
   did the model link instead, and how far away was it?
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
DEFAULT_STEM = "6bba_207c6aaf"  # worst video: edgeJ 0.657
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset, compute_metric  # noqa: E402


def load(stem):
    gt_path = DATA_DIR / f"{stem}.geff"
    pred_path = PRED_DIR / f"{stem}.geff"
    assert gt_path.exists(), f"GT not found: {gt_path}"
    assert pred_path.exists(), f"prediction not found: {pred_path}"

    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    pred = td.graph.IndexedRXGraph.from_geff(pred_path)
    pred = pred[0] if isinstance(pred, tuple) else pred

    er = compute_metric(pred, ds.tracks, scale=ds.scale)  # writes match_node_id
    print(f"{stem}: TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}")
    return pred, ds, er


def check_scaling(pred, ds, n=300):
    """Recompute edge_dist two ways and see which the stored value matches."""
    na = pred.node_attrs()
    ea = pred.edge_attrs()
    coords = {r["node_id"]: (r["z"], r["y"], r["x"]) for r in na.iter_rows(named=True)}
    scale = np.array(ds.scale[-3:], dtype=float)

    rows = []
    for r in ea.head(n).iter_rows(named=True):
        s, t = r["source_id"], r["target_id"]
        if s in coords and t in coords:
            d = np.array(coords[t]) - np.array(coords[s])
            rows.append({
                "stored": r["edge_dist"],
                "voxel": float(np.linalg.norm(d)),
                "um": float(np.linalg.norm(d * scale)),
            })

    df = pl.DataFrame(rows)
    ev = (df["stored"] - df["voxel"]).abs().mean()
    eu = (df["stored"] - df["um"]).abs().mean()

    print("\n=== edge_dist scaling ===")
    print(f"scale (z,y,x) = {scale}")
    print(df.head(6))
    print(f"mean |stored - voxel| = {ev:.4f}")
    print(f"mean |stored - um|    = {eu:.4f}")
    df = pl.DataFrame(rows)   # from the scaling check
    print((df["um"] / df["stored"]).describe())
    if ev < eu:
        print("VERDICT: VOXELS -- unscaled, z compressed 4x. This is the bug.")
    else:
        print("VERDICT: MICROMETRES -- correctly scaled.")
    return "voxel" if ev < eu else "um"


def classify_fn(pred, ds):
    """Split FN edges into detection vs linking failures, and for linking
    failures compare the true partner's distance against what was chosen."""
    na = pred.node_attrs()
    ea = pred.edge_attrs()
    scale = np.array(ds.scale[-3:], dtype=float)

    matched = na.filter(pl.col("match_node_id") != -1)
    gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
    pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))
    coords = {r["node_id"]: (r["z"], r["y"], r["x"]) for r in na.iter_rows(named=True)}

    tp = ea.filter(pl.col("matched_edge_mask"))
    tp_pairs = {
        (pred_to_gt[s], pred_to_gt[t])
        for s, t in zip(tp["source_id"], tp["target_id"])
    }
    fn = [(s, t) for s, t in ds.tracks.edge_list() if (s, t) not in tp_pairs]

    both = [(s, t) for s, t in fn if s in gt_to_pred and t in gt_to_pred]
    print("\n=== FN classification ===")
    print(f"total FN edges        : {len(fn)}")
    print(f"  endpoint missing    : {len(fn) - len(both)}  (detection)")
    print(f"  both detected       : {len(both)}  (linking)")

    out = []
    for gs, gtt in both:
        ps, pt = gt_to_pred[gs], gt_to_pred[gtt]
        true_d = float(np.linalg.norm((np.array(coords[pt]) - np.array(coords[ps])) * scale))

        src_out = ea.filter(pl.col("source_id") == ps)
        tgt_in = ea.filter(pl.col("target_id") == pt)

        if len(src_out) == 0 and len(tgt_in) == 0:
            kind, chosen_d, chosen_p = "orphaned", None, None
        else:
            side = src_out if len(src_out) else tgt_in
            kind = "displaced"
            chosen_d = float(side["edge_dist"][0])
            chosen_p = float(side["edge_prob"][0])

        out.append({
            "gt_src": gs, "gt_tgt": gtt, "kind": kind,
            "true_um": round(true_d, 2),
            "chosen_dist": None if chosen_d is None else round(chosen_d, 2),
            "chosen_prob": None if chosen_p is None else round(chosen_p, 3),
        })

    df = pl.DataFrame(out)
    if len(df) == 0:
        return df

    disp = df.filter(pl.col("kind") == "displaced")
    print(f"\n  displaced (model linked elsewhere): {len(disp)}")
    print(f"  orphaned  (no link at all)        : {len(df) - len(disp)}")

    if len(disp):
        print(f"\n  median true distance   : {disp['true_um'].median():.2f}")
        print(f"  median chosen distance : {disp['chosen_dist'].median():.2f}")
        print(f"  median chosen prob     : {disp['chosen_prob'].median():.3f}")
        nearer = (disp["chosen_dist"] < disp["true_um"]).sum()
        print(f"  chose a NEARER node    : {nearer}/{len(disp)}")
        print(f"  high confidence (>0.8) : {(disp['chosen_prob'] > 0.8).sum()}/{len(disp)}")

    print(df.head(25))
    df.write_csv(f"fn_{STEM}.csv")
    print(f"\nwrote fn_{STEM}.csv")
    return df


if __name__ == "__main__":
    STEM = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM
    pred, ds, er = load(STEM)
    check_scaling(pred, ds)
    classify_fn(pred, ds)