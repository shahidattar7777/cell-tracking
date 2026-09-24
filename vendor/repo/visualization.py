"""Animated GIF: a selected set of predicted tracks over a max-intensity projection.

Run from vendor/repo:   uv run python make_gif.py [stem]

CPU only. Reads the raw zarr and the prediction .geff, projects each 3D frame
along z, and overlays a chosen subset of tracks with their recent history.

Showing every detection is unreadable -- a few hundred dots per frame with no
way to follow any of them. Instead this keeps the N longest-surviving lineages,
drawn boldly with long trails. Those are the tracks that actually have history
to show, and with hundreds of trails reduced to a few dozen the coherent tissue
drift (coherence 0.74 on this video) becomes legible rather than a blur.

Each lineage keeps one colour across all frames, so a trail reads as a
continuous path.

SELECTION MODES
---------------
  "longest"  N lineages surviving the most rendered frames. Best for showing
             flow -- these are the clean, long tracks.
  "gt"       only lineages containing a GT-matched node. Much sparser, and it
             is the exact subset every number in the README refers to.
  "all"      no filtering (the original behaviour).

GitHub renders GIFs inline in READMEs but not MP4. Keep the file under a few MB
or it will not load; the knobs are FRAMES, STRIDE, DOWNSCALE and FPS.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tracksdata as td
import zarr
from matplotlib.animation import FuncAnimation, PillowWriter

# --- paths and knobs: EDIT THESE -----------------------------------------
DATA_DIR = Path(
    r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data"
    r"\biohub-cell-tracking-during-development\train"
)
PRED_DIR = Path(r"predictions\shahi\ilp_app2.0\split_0")
DEFAULT_STEM = "6bba_207c6aaf"

SELECT = "longest"   # "longest" | "gt" | "all"
MAX_TRACKS = 5      # None = no cap
MIN_LIFETIME = 20    # a lineage must appear in this many rendered frames

START = 30
FRAMES = 50
STRIDE = 1
TRAIL = 25           # frames of history behind each cell
LINEWIDTH = 1.6
DOTSIZE = 22
DIM_BG = 0.55        # unselected detections drawn faintly; 0 hides them
DOWNSCALE = 1
FPS = 10
CONTRAST = (17, 1133)     # from the zarr's own image_statistics quantiles
OUT = "tracks.gif"
# -------------------------------------------------------------------------


def load(stem):
    group = zarr.open(DATA_DIR / f"{stem}.zarr", mode="r")
    arr = group["0"]

    pred = td.graph.IndexedRXGraph.from_geff(PRED_DIR / f"{stem}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred

    na = pred.node_attrs()
    has_match = "match_node_id" in na.columns

    pos, matched = {}, set()
    by_frame = defaultdict(list)
    for r in na.iter_rows(named=True):
        i = r["node_id"]
        pos[i] = (int(r["t"]), float(r["y"]), float(r["x"]))
        by_frame[int(r["t"])].append(i)
        if has_match and r["match_node_id"] != -1:
            matched.add(i)

    parent = {int(t): int(s) for s, t in pred.edge_list()}
    return arr, pos, by_frame, parent, matched


def lineage_roots(pos, parent):
    """Map every node to its lineage root, following parents back."""
    root_of = {}
    for i in pos:
        chain, cur = [], i
        while cur in parent and cur not in root_of:
            chain.append(cur)
            cur = parent[cur]
        r = root_of.get(cur, cur)
        for c in chain:
            root_of[c] = r
        root_of[i] = r
    return root_of


def choose_tracks(pos, root_of, matched, frames):
    """Which nodes to draw boldly."""
    window = set(frames)
    counts = defaultdict(int)
    for i, (t, _, _) in pos.items():
        if t in window:
            counts[root_of[i]] += 1

    if SELECT == "all":
        return set(pos), counts

    if SELECT == "gt":
        roots = {root_of[i] for i in matched}
        if not roots:
            print("  no GT-matched nodes found -- falling back to 'longest'")
            roots = {r for r, n in counts.items() if n >= MIN_LIFETIME}
    else:
        roots = {r for r, n in counts.items() if n >= MIN_LIFETIME}

    if MAX_TRACKS:
        roots = set(sorted(roots, key=lambda r: -counts[r])[:MAX_TRACKS])

    return {i for i in pos if root_of[i] in roots}, counts


def trail_points(i, pos, parent, n):
    pts, cur = [], i
    for _ in range(n):
        if cur not in pos:
            break
        _, y, x = pos[cur]
        pts.append((x, y))
        if cur not in parent:
            break
        cur = parent[cur]
    return np.array(pts) if len(pts) > 1 else None


def main(stem):
    arr, pos, by_frame, parent, matched = load(stem)
    frames = list(range(START, min(START + FRAMES * STRIDE, arr.shape[0]), STRIDE))
    root_of = lineage_roots(pos, parent)
    keep, counts = choose_tracks(pos, root_of, matched, frames)

    kept_roots = {root_of[i] for i in keep}
    print(f"{stem}: {arr.shape}")
    print(f"  frames {frames[0]}..{frames[-1]} ({len(frames)})")
    print(f"  select={SELECT}  lineages kept {len(kept_roots)} "
          f"of {len(set(root_of.values()))}")
    if kept_roots:
        lens = sorted((counts[r] for r in kept_roots), reverse=True)
        print(f"  lifetimes: longest {lens[0]}, shortest {lens[-1]} frames")

    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("turbo")
    colour = {r: cmap(rng.random()) for r in sorted(kept_roots)}

    h, w = arr.shape[2] // DOWNSCALE, arr.shape[3] // DOWNSCALE
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    def mip(t):
        vol = arr[t]
        if DOWNSCALE > 1:
            vol = vol[:, ::DOWNSCALE, ::DOWNSCALE]
        return vol.max(axis=0)

    im = ax.imshow(mip(frames[0]), cmap="gray",
                   vmin=CONTRAST[0], vmax=CONTRAST[1], interpolation="nearest")
    bg = ax.scatter([], [], s=4, c="white", alpha=DIM_BG, linewidths=0)
    fg = ax.scatter([], [], s=DOTSIZE, linewidths=0.4, edgecolors="white")
    lines = []
    label = ax.text(0.015, 0.975, "", transform=ax.transAxes, color="white",
                    fontsize=9, va="top", family="monospace")

    def update(t):
        im.set_data(mip(t))
        ids = by_frame.get(t, [])
        sel = [i for i in ids if i in keep]
        rest = [i for i in ids if i not in keep]

        def xy(lst):
            if not lst:
                return np.empty((0, 2))
            return np.array([[pos[i][2] / DOWNSCALE, pos[i][1] / DOWNSCALE]
                             for i in lst])

        bg.set_offsets(xy(rest) if DIM_BG > 0 else np.empty((0, 2)))
        fg.set_offsets(xy(sel))
        if sel:
            fg.set_facecolors([colour[root_of[i]] for i in sel])

        for ln in lines:
            ln.remove()
        lines.clear()
        for i in sel:
            pts = trail_points(i, pos, parent, TRAIL)
            if pts is None:
                continue
            ln, = ax.plot(pts[:, 0] / DOWNSCALE, pts[:, 1] / DOWNSCALE,
                          color=colour[root_of[i]], lw=LINEWIDTH, alpha=0.85,
                          solid_capstyle="round")
            lines.append(ln)

        label.set_text(f"t = {t:3d}   tracked = {len(sel)}   "
                       f"detections = {len(ids)}")
        return [im, bg, fg, label, *lines]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=FPS))
    plt.close()

    size = Path(OUT).stat().st_size / 1e6
    print(f"wrote {OUT}  ({size:.1f} MB)")
    if size > 5:
        print("  >5 MB: GitHub may not render it inline. Raise STRIDE or")
        print("  DOWNSCALE, or lower FRAMES.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM)