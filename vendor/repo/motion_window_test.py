"""Does motion become predictable over LONGER windows?

Run from vendor/repo:   uv run python motion_window_test.py [stem ...]

BACKGROUND
----------
motion_test.py showed that one-frame velocity barely beats chance on the
failing video (residual < raw for only 54% of GT edges) and that shifting the
candidate gate by it loses more edges than it gains.

Two reasons that might understate the real signal:

  1. QUANTISATION. GT centroids are integer voxels. Median displacement is
     ~3 um -- about 7 voxels laterally but under 2 in z. Half-voxel rounding
     (0.8 um in z) is a large fraction of that, so a single-frame difference is
     partly noise.

  2. BIOLOGY. A cell jostled by dividing neighbours moves erratically frame to
     frame while still drifting consistently over several frames.

Averaging over k frames suppresses both.

WHAT CHANGES
------------
Only the velocity estimate:

    k = 1   v = pos(t) - pos(t-1)                (what motion_test.py did)
    k > 1   v = [pos(t) - pos(t-k)] / k          averaged over the chain

Everything downstream -- predicted position, residual, gate comparison -- is
identical, so the numbers are directly comparable to the k=1 run.

Longer windows need longer history, so the usable sample shrinks with k. The
script reports n per window; if it collapses, the comparison is not meaningful.

HOW TO READ IT
--------------
If 'residual < raw' climbs toward 70%+ as k grows, motion IS predictable and
the gate idea is revived at the right window. If it sits near 54% at every k,
motion is not the signal here -- close the direction and move on.
"""

import csv
import sys
from pathlib import Path

import numpy as np

# --- paths: EDIT THESE ---------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
DEFAULT_STEMS = ["6bba_207c6aaf", "44b6_d78e09d9"]
WINDOWS = [1, 2, 3, 5]
GATE_UM = 7.0
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset  # noqa: E402


def gt_geometry(stem):
    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    g = ds.tracks
    scale = np.array(ds.scale[-3:], dtype=float)
    na = g.node_attrs()
    pos = {
        r["node_id"]: np.array([r["z"], r["y"], r["x"]], dtype=float) * scale
        for r in na.iter_rows(named=True)
    }
    edges = [(int(s), int(t)) for s, t in g.edge_list()]
    parent = {t: s for s, t in edges}
    return pos, edges, parent


def ancestor(node, parent, k):
    """Walk back k steps up the lineage. None if the chain runs out."""
    cur = node
    for _ in range(k):
        if cur not in parent:
            return None
        cur = parent[cur]
    return cur


def measure(pos, edges, parent, k):
    rows = []
    for s, t in edges:
        anc = ancestor(s, parent, k)
        if anc is None:
            continue
        v = (pos[s] - pos[anc]) / k          # averaged velocity
        predicted = pos[s] + v
        rows.append({
            "src": s, "tgt": t,
            "raw": float(np.linalg.norm(pos[t] - pos[s])),
            "residual": float(np.linalg.norm(pos[t] - predicted)),
        })
    return rows


def line(rows, k, n_edges, label_width=6):
    if not rows:
        return f"  k={k:<2}  (no usable edges)"
    raw = np.array([r["raw"] for r in rows])
    res = np.array([r["residual"] for r in rows])
    better = int((res < raw).sum())
    fixed_in = raw <= GATE_UM
    shift_in = res <= GATE_UM
    recovered = int((~fixed_in & shift_in).sum())
    lost = int((fixed_in & ~shift_in).sum())
    return (f"  k={k:<2}  n={len(rows):4d}/{n_edges:<4d}  "
            f"median raw {np.median(raw):5.2f}  median resid {np.median(res):5.2f}  "
            f"resid<raw {better:4d} ({100*better/len(rows):3.0f}%)  "
            f"gate{GATE_UM:.0f}: +{recovered:<3d} -{lost:<3d}  net {recovered-lost:+d}")


def main(stems):
    for stem in stems:
        print("=" * 100)
        print(stem)
        pos, edges, parent = gt_geometry(stem)

        print("\n  ALL GT edges")
        for k in WINDOWS:
            print(line(measure(pos, edges, parent, k), k, len(edges)))

        fn_path = Path(f"fn_{stem}.csv")
        if fn_path.exists():
            fn_pairs = {
                (int(r["gt_src"]), int(r["gt_tgt"]))
                for r in csv.DictReader(open(fn_path))
            }
            print(f"\n  FAILED edges only ({fn_path.name})")
            for k in WINDOWS:
                rows = [r for r in measure(pos, edges, parent, k)
                        if (r["src"], r["tgt"]) in fn_pairs]
                print(line(rows, k, len(fn_pairs)))
        else:
            print(f"\n  ({fn_path.name} not found -- run diagnose.py {stem} first)")

    print("\n" + "=" * 100)
    print("resid<raw near 50%  ->  velocity is no better than chance at that window")
    print("net negative        ->  shifting the gate costs more edges than it gains")
    print("watch n: longer windows need longer lineages, so the sample shrinks")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_STEMS)