"""Is cell motion smooth enough for a velocity-shifted candidate gate?

Run from vendor/repo:   uv run python motion_test.py [stem ...]

THE QUESTION
------------
The pipeline proposes candidate links inside a fixed radius around each cell's
CURRENT position. On 6bba_207c6aaf, 29 of 67 linking failures had their true
partner further than 7 um away -- outside any reasonable gate, so the correct
edge was never proposed at all.

A velocity-shifted gate would search around where the cell is GOING rather than
where it IS:

        predicted = position(t) + [position(t) - position(t-1)]

Same radius, different centre. This only works if motion is SMOOTH -- if
velocity at t predicts displacement at t+1. If cells jitter, the shifted gate
points at empty space and makes things worse.

WHAT THIS MEASURES
------------------
Purely on ground truth (no predictions involved). For every GT edge where the
source cell has a parent:

    raw      = |pos(target) - pos(source)|            what a FIXED gate must cover
    residual = |pos(target) - predicted|              what a SHIFTED gate must cover

If residual << raw, motion is predictable. The decisive number is how many
edges have raw > gate but residual < gate -- those are recoverable by shifting
and unreachable otherwise.

Reported for ALL GT edges and, separately, for the subset that the model
actually got wrong (read from fn_<stem>.csv if present).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# --- paths: EDIT THESE ---------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
DEFAULT_STEMS = ["6bba_207c6aaf", "44b6_d78e09d9"]
GATES_UM = [5.0, 7.0, 10.0]
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset  # noqa: E402


def gt_geometry(stem):
    """Pull GT node positions (in um) and the edge list."""
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    g = ds.tracks
    scale = np.array(ds.scale[-3:], dtype=float)

    na = g.node_attrs()
    pos = {
        r["node_id"]: np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        for r in na.iter_rows(named=True)
    }
    edges = [(int(s), int(t)) for s, t in g.edge_list()]
    return pos, edges


def measure(stem):
    pos, edges = gt_geometry(stem)

    # parent lookup: a node's predecessor in the GT lineage
    parent = {}
    for s, t in edges:
        parent[t] = s   # divisions give a daughter one parent, which is what we want

    rows = []
    no_parent = 0
    for s, t in edges:
        if s not in parent:
            no_parent += 1
            continue
        p = parent[s]
        v = pos[s] - pos[p]                 # velocity into the current frame
        predicted = pos[s] + v              # constant-velocity extrapolation
        rows.append({
            "src": s,
            "tgt": t,
            "raw": float(np.linalg.norm(pos[t] - pos[s])),
            "residual": float(np.linalg.norm(pos[t] - predicted)),
            "speed": float(np.linalg.norm(v)),
        })

    return rows, no_parent, len(edges)


def summarise(rows, label):
    if not rows:
        print(f"  {label}: no rows")
        return
    raw = np.array([r["raw"] for r in rows])
    res = np.array([r["residual"] for r in rows])

    print(f"\n  {label}  (n = {len(rows)})")
    print(f"    raw displacement   median {np.median(raw):6.2f}   mean {raw.mean():6.2f}   p90 {np.percentile(raw,90):6.2f}")
    print(f"    velocity residual  median {np.median(res):6.2f}   mean {res.mean():6.2f}   p90 {np.percentile(res,90):6.2f}")
    better = (res < raw).sum()
    print(f"    residual < raw     {better}/{len(rows)}  ({100*better/len(rows):.0f}%)")

    print(f"    {'gate':>6}  {'fixed covers':>13}  {'shifted covers':>15}  {'RECOVERABLE':>12}  {'lost':>6}")
    for gate in GATES_UM:
        fixed_in = raw <= gate
        shift_in = res <= gate
        recoverable = int((~fixed_in & shift_in).sum())   # gained by shifting
        lost = int((fixed_in & ~shift_in).sum())          # given up by shifting
        print(f"    {gate:6.1f}  {fixed_in.sum():6d} ({100*fixed_in.mean():3.0f}%)  "
              f"{shift_in.sum():8d} ({100*shift_in.mean():3.0f}%)  "
              f"{recoverable:12d}  {lost:6d}")


def main(stems):
    for stem in stems:
        print("=" * 78)
        print(stem)
        rows, no_parent, n_edges = measure(stem)
        print(f"  GT edges {n_edges}   usable {len(rows)}   "
              f"skipped (source has no parent) {no_parent}")

        summarise(rows, "ALL GT edges")

        # restrict to the edges the model actually missed, if we have them
        fn_path = Path(f"fn_{stem}.csv")
        if fn_path.exists():
            fn_pairs = set()
            for r in csv.DictReader(open(fn_path)):
                fn_pairs.add((int(r["gt_src"]), int(r["gt_tgt"])))
            sub = [r for r in rows if (r["src"], r["tgt"]) in fn_pairs]
            summarise(sub, f"FAILED edges only (from {fn_path.name})")
        else:
            print(f"\n  ({fn_path.name} not found -- run diagnose.py for the "
                  f"failure-only breakdown)")

    print("\n" + "=" * 78)
    print("READ IT LIKE THIS")
    print("  RECOVERABLE  = true partner outside the fixed gate but inside the")
    print("                 shifted one. These are edges only a motion-aware")
    print("                 gate can propose.")
    print("  lost         = inside the fixed gate but outside the shifted one.")
    print("                 Shifting gives these up. Needs to be much smaller.")
    print("  If 'residual < raw' is near 50%, motion is NOT predictable and the")
    print("  whole idea is dead -- stop here rather than building it.")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_STEMS)