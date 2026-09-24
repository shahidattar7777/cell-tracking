"""Does source competition explain BOTH failure subpopulations?

Run from vendor/repo:   uv run python competition_test.py [stem]

THE HYPOTHESIS
--------------
`softmax(dim=0)` normalises across SOURCES for each target, so sources compete
for targets. Probability mass flowing into a target is divided among every cell
that might be its parent. If a nearer cell claims most of that mass, the true
parent scores low on its own true target and settles for whatever is left.

That predicts the two subpopulations are one mechanism, not two:

    NEARER group (50/67)  source grabs a close decoy instead of its true,
                          more distant partner
    FARTHER group (17/67) source LOSES its true, close partner to an even
                          closer competitor and is pushed onto a farther
                          leftover

Both reduce to "the target goes to the nearest claimant".

Consistent with the confidence gap already measured: the farther group sits at
median p=0.530 vs 0.642 for the nearer group -- what you would expect from a
source that lost a competition.

WHAT THIS REPORTS
-----------------
For every failure with both endpoints detected, it asks who claimed the TRUE
target and whether that claimant was closer to it than the true parent was.

    CLAIMED      the true target has an incoming edge from a different source
    STOLEN       ...and that source is nearer to it than the true parent is
    orphaned     nothing claimed the true target at all

If STOLEN dominates in BOTH groups, competition is confirmed and the two
symptoms collapse into one mechanism.
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
DEFAULT_STEM = "6bba_207c6aaf"
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset, compute_metric  # noqa: E402


def main(stem):
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    pred = td.graph.IndexedRXGraph.from_geff(PRED_DIR / f"{stem}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred
    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    print(f"{stem}: TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}\n")

    scale = np.array(ds.scale[-3:], dtype=float)
    na, ea = pred.node_attrs(), pred.edge_attrs()
    pos = {
        r["node_id"]: np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        for r in na.iter_rows(named=True)
    }
    matched = na.filter(pl.col("match_node_id") != -1)
    gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
    pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))

    tp = ea.filter(pl.col("matched_edge_mask"))
    tp_pairs = {(pred_to_gt[s], pred_to_gt[t])
                for s, t in zip(tp["source_id"], tp["target_id"])}
    gt_edges = [(int(s), int(t)) for s, t in ds.tracks.edge_list()]
    fn = [(s, t) for s, t in gt_edges if (s, t) not in tp_pairs]
    both = [(s, t) for s, t in fn if s in gt_to_pred and t in gt_to_pred]

    rows = []
    for gs, gtt in both:
        ps, pt = gt_to_pred[gs], gt_to_pred[gtt]
        d_true = float(np.linalg.norm(pos[pt] - pos[ps]))

        # what did this source link to instead?
        out = ea.filter(pl.col("source_id") == ps)
        if len(out):
            pc = int(out["target_id"][0])
            d_chosen = float(np.linalg.norm(pos[pc] - pos[ps]))
            p_chosen = float(out["edge_prob"][0])
            group = "nearer" if d_chosen < d_true else "farther"
        else:
            d_chosen, p_chosen, group = float("nan"), float("nan"), "unlinked"

        # who claimed the TRUE target?
        inn = ea.filter(pl.col("target_id") == pt)
        if len(inn) == 0:
            status, d_claim, p_claim = "orphaned", float("nan"), float("nan")
        else:
            claimant = int(inn["source_id"][0])
            p_claim = float(inn["edge_prob"][0])
            d_claim = float(np.linalg.norm(pos[pt] - pos[claimant]))
            status = "STOLEN" if d_claim < d_true else "claimed-farther"

        rows.append({"group": group, "status": status,
                     "d_true": d_true, "d_chosen": d_chosen,
                     "d_claim": d_claim, "p_chosen": p_chosen, "p_claim": p_claim})

    df = pl.DataFrame(rows)
    print(f"failures with both endpoints detected: {len(df)}\n")

    print("who claimed the TRUE target, split by failure group")
    print(df.group_by(["group", "status"]).len().sort(["group", "status"]))

    for g in ("nearer", "farther"):
        sub = df.filter(pl.col("group") == g)
        if len(sub) == 0:
            continue
        stolen = sub.filter(pl.col("status") == "STOLEN")
        print(f"\n  {g.upper()} group  (n={len(sub)})")
        print(f"    STOLEN (a nearer source took the true target): "
              f"{len(stolen)}/{len(sub)}  ({100*len(stolen)/len(sub):.0f}%)")
        if len(stolen):
            print(f"    median d_true {stolen['d_true'].median():.2f}  "
                  f"vs claimant's d {stolen['d_claim'].median():.2f}")
            print(f"    median p(claimant) {stolen['p_claim'].median():.3f}  "
                  f"vs p(what our source got) {stolen['p_chosen'].median():.3f}")

    print("\n" + "=" * 72)
    print("STOLEN high in BOTH groups -> one mechanism: the target goes to the")
    print("nearest claimant, and everything else is downstream of that.")
    print("STOLEN low in the farther group -> the two groups are genuinely")
    print("different failures and need separate explanations.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM)