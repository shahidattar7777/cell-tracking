"""Score the cfg.threshold sweep against the app=2.0 baseline.

Run from vendor/repo:  uv run python score_threshold.py

Prints per-video deltas plus the TP/FP trade, which is the thing that decides
this experiment: a lower threshold should RAISE TP (orphaned true edges become
available) and will also raise the candidate count. Whether FP follows depends
on how well the ILP at weight 2.0 filters the extra candidates.
"""

import sys
from pathlib import Path

import polars as pl
import tracksdata as td

# --- paths: EDIT THESE ---------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
PRED_ROOT = Path("predictions/shahi")
STEMS = ["6bba_207c6aaf", "6bba_6feb10f0"]

# threshold 0.3, app 2.0
BASELINE = {
    "6bba_207c6aaf": {"edgeJ": 0.7351, "tp": 394, "fp": 86, "fn": 56},
    "6bba_6feb10f0": {"edgeJ": 0.8262, "tp": 1203, "fp": 113, "fn": 140},
}
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import (  # noqa: E402
    open_dataset, compute_metric, node_recall, _read_estimated_n_total,
)
from biohub_tracking.metrics import per_sample_metrics  # noqa: E402


def score(pred_path, stem):
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    pred = td.graph.IndexedRXGraph.from_geff(pred_path)
    pred = pred[0] if isinstance(pred, tuple) else pred
    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    recall = node_recall(pred, ds.tracks)
    n_total = _read_estimated_n_total(DATA_DIR / f"{stem}.geff")
    row = per_sample_metrics(er, n_total, recall)
    assert row["adj_edge_jaccard"] <= 1.05, f"implausible score for {stem}"
    return row


rows = []
for method_dir in sorted(PRED_ROOT.iterdir()):
    if not method_dir.is_dir():
        continue
    for stem in STEMS:
        hits = list(method_dir.rglob(f"{stem}.geff"))
        if not hits:
            continue
        r = score(hits[0], stem)
        b = BASELINE[stem]
        r["method"] = method_dir.name
        r["stem"] = stem
        r["delta"] = r["edge_jaccard"] - b["edgeJ"]
        r["d_tp"] = r["edge_tp"] - b["tp"]
        r["d_fp"] = r["edge_fp"] - b["fp"]
        r["d_fn"] = r["edge_fn"] - b["fn"]
        rows.append(r)

if not rows:
    raise SystemExit(f"no predictions found under {PRED_ROOT}")

df = pl.DataFrame(rows)

print(f"{'method':16}{'stem':16}{'nodes':>8}{'TP':>7}{'FP':>6}{'FN':>6}"
      f"{'dTP':>6}{'dFP':>6}{'dFN':>6}{'recall':>9}{'edgeJ':>9}{'delta':>9}")
for r in df.sort(["stem", "method"]).iter_rows(named=True):
    print(f"{r['method']:16}{r['stem']:16}{r['num_pred_nodes']:8.0f}"
          f"{r['edge_tp']:7.0f}{r['edge_fp']:6.0f}{r['edge_fn']:6.0f}"
          f"{r['d_tp']:+6.0f}{r['d_fp']:+6.0f}{r['d_fn']:+6.0f}"
          f"{r['node_recall']:9.4f}{r['edge_jaccard']:9.4f}{r['delta']:+9.4f}")

print()
print(df.group_by("method").agg(
    pl.col("delta").mean().alias("mean_delta"),
    pl.col("d_tp").sum().alias("tot_dTP"),
    pl.col("d_fp").sum().alias("tot_dFP"),
    pl.col("node_recall").mean().alias("mean_recall"),
).sort("mean_delta", descending=True))

df.write_csv("threshold_results.csv")
print("\nwrote threshold_results.csv")
print("\nTP up and FP flat/down -> the ILP is filtering the extra candidates.")
print("TP up and FP up more    -> the threshold was doing useful work; stop.")