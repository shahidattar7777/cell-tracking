"""Score the ILP weight sweep and compare against baseline.

Run from vendor/repo:  uv run python score_sweep.py
"""

import sys
from pathlib import Path

import polars as pl
import tracksdata as td

DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
PRED_ROOT = Path("predictions/shahi")
STEMS = ["6bba_207c6aaf", "6bba_57b7cc1e", "6bba_6feb10f0"]

# from baseline_20.csv
BASELINE = {
    "6bba_207c6aaf": 0.6565,
    "6bba_57b7cc1e": 0.6617,
    "6bba_6feb10f0": 0.8065,
}

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
    return per_sample_metrics(er, n_total, recall)


rows = []
for method_dir in sorted(PRED_ROOT.iterdir()):
    if not method_dir.is_dir():
        continue
    for stem in STEMS:
        matches = list(method_dir.rglob(f"{stem}.geff"))
        if not matches:
            continue
        r = score(matches[0], stem)
        r["method"] = method_dir.name
        r["stem"] = stem
        r["baseline"] = BASELINE[stem]
        r["delta"] = r["edge_jaccard"] - BASELINE[stem]
        rows.append(r)
        print(f"{method_dir.name:16} {stem:16} "
              f"edgeJ={r['edge_jaccard']:.4f}  "
              f"delta={r['delta']:+.4f}  "
              f"TP/FP/FN={r['edge_tp']}/{r['edge_fp']}/{r['edge_fn']}")

df = pl.DataFrame(rows)
print()
print(df.select("method", "stem", "edge_tp", "edge_fp", "edge_fn",
                "node_recall", "edge_jaccard", "baseline", "delta"))
print()
print(df.group_by("method").agg(
    pl.col("delta").mean().alias("mean_delta"),
    pl.col("edge_jaccard").mean().alias("mean_edgeJ"),
).sort("mean_delta", descending=True))
df.write_csv("sweep_results.csv")