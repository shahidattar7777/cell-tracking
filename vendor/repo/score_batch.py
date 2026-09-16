"""Score the 20-video validation batch and inspect edge_dist scaling.

Run from vendor/repo:   uv run python score_batch.py

Edit the three paths below to match your machine.
"""

import json
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
VAL_JSON = Path("val20.json")
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import (  # noqa: E402
    open_dataset,
    compute_metric,
    node_recall,
    _read_estimated_n_total,
)
from biohub_tracking.metrics import per_sample_metrics  # noqa: E402


def score_run(pred_path, stem, data_dir, label=None):
    """One experiment -> one row. Guards against the path bugs that have
    silently produced valid-looking rows before."""
    pred_path = Path(pred_path)
    data_dir = Path(data_dir)

    gt_path = data_dir / f"{stem}.geff"
    assert gt_path.exists(), f"GT not found: {gt_path}"
    assert pred_path.exists(), f"prediction not found: {pred_path}"
    assert pred_path.stem == stem, f"stem mismatch: {pred_path.stem} != {stem}"
    assert "predictions" in str(pred_path), f"looks like GT, not a prediction: {pred_path}"

    ds = open_dataset(data_dir / stem, require_tracks=True)

    pred = td.graph.IndexedRXGraph.from_geff(pred_path)
    pred = pred[0] if isinstance(pred, tuple) else pred

    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    recall = node_recall(pred, ds.tracks)
    n_total = _read_estimated_n_total(gt_path)

    row = per_sample_metrics(er, n_total, recall)

    # mathematical invariant: catches GT-scored-against-itself and similar
    if row["adj_edge_jaccard"] > 1.0:
        print(f"WARNING: adj > 1 for {stem} (ratio {row['total_node_ratio']:.3f})")

    row["stem"] = stem
    row["label"] = label or pred_path.parent.name
    row["gt_nodes"] = ds.tracks.num_nodes()
    row["gt_edges"] = ds.tracks.num_edges()
    row["n_total"] = float(n_total)
    return row, pred, ds


def check_edge_dist_scaling(pred, ds, stem):
    """Is edge_dist in voxels or micrometres?

    Recompute distance from coordinates two ways and see which matches.
    If edge_dist is unscaled, z is compressed 4x relative to physical space
    -- which would bias the model toward lateral neighbours over cells that
    genuinely moved in z.
    """
    na = pred.node_attrs()
    ea = pred.edge_attrs()

    coords = {
        r["node_id"]: (r["z"], r["y"], r["x"]) for r in na.iter_rows(named=True)
    }
    scale = np.array(ds.scale[-3:], dtype=float)

    sample = ea.head(200)
    rows = []
    for r in sample.iter_rows(named=True):
        s, t = r["source_id"], r["target_id"]
        if s not in coords or t not in coords:
            continue
        d = np.array(coords[t]) - np.array(coords[s])
        rows.append(
            {
                "stored": r["edge_dist"],
                "voxel": float(np.linalg.norm(d)),
                "um": float(np.linalg.norm(d * scale)),
            }
        )

    df = pl.DataFrame(rows)
    err_voxel = (df["stored"] - df["voxel"]).abs().mean()
    err_um = (df["stored"] - df["um"]).abs().mean()

    print(f"\n--- edge_dist scaling check ({stem}) ---")
    print(f"scale (z,y,x): {scale}")
    print(df.head(8))
    print(f"mean |stored - voxel_dist| = {err_voxel:.4f}")
    print(f"mean |stored - um_dist|    = {err_um:.4f}")
    verdict = "VOXELS (unscaled -- z compressed 4x)" if err_voxel < err_um else "MICROMETRES (scaled)"
    print(f"VERDICT: edge_dist is in {verdict}")


def main():
    stems = json.loads(VAL_JSON.read_text())
    print(f"Scoring {len(stems)} videos...\n")

    rows = []
    first = None
    for i, stem in enumerate(stems, 1):
        pred_path = PRED_DIR / f"{stem}.geff"
        if not pred_path.exists():
            print(f"[{i}/{len(stems)}] {stem}: MISSING prediction, skipping")
            continue
        row, pred, ds = score_run(pred_path, stem, DATA_DIR)
        rows.append(row)
        if first is None:
            first = (pred, ds, stem)
        print(f"[{i}/{len(stems)}] {stem}: score={row['adj_edge_jaccard']:.4f}")

    df = pl.DataFrame(rows).sort("n_total")
    df.write_csv("baseline_20.csv")

    print("\n" + "=" * 70)
    print(
        df.select(
            "stem", "n_total", "gt_nodes", "gt_edges",
            "edge_tp", "edge_fp", "edge_fn",
            "node_recall", "edge_jaccard", "adj_edge_jaccard",
        )
    )
    print("=" * 70)
    print(f"\nmean adj_edge_jaccard : {df['adj_edge_jaccard'].mean():.4f}")
    print(f"min / max             : {df['adj_edge_jaccard'].min():.4f} / {df['adj_edge_jaccard'].max():.4f}")
    print(f"mean node_recall      : {df['node_recall'].mean():.4f}")
    print("\nLeaderboard was 0.866 -- does the mean above track it?")

    # does score degrade with density?
    corr = np.corrcoef(df["n_total"].to_numpy(), df["adj_edge_jaccard"].to_numpy())[0, 1]
    print(f"corr(n_total, score)  : {corr:.3f}   (negative => denser videos score worse)")

    if first is not None:
        check_edge_dist_scaling(*first)


if __name__ == "__main__":
    main()