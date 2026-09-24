"""Do the UNet features separate true partners from decoys?

THE QUESTION
------------
HOCT's 19 input features are segmentation-derived: position, equivalent
diameter, intensity statistics, the 3x3 inertia tensor (9 of the 19), and
distance to the field-of-view border. Roughly ten of those describe SHAPE.

Our detector produces bare local-max centroids -- no mask, no diameter, no
inertia tensor. So half of HOCT's input dimensionality does not exist in this
pipeline. The substitute on hand is the 32-channel UNet feature vector already
gathered per detection by `_index_features` and handed to `predict_edges`.

If those features separate the true partner from the decoy on the STOLEN cases,
substitution is viable and an edge-centric model can be built on this pipeline.
If they do not, the appearance signal is genuinely absent and a segmentation
stage has to come first -- a different and much larger project.

Note this is a test of SEPARABILITY, not of whether the current model uses the
signal. `predict_edges` already receives these vectors and still picks the
decoy, so a positive result here means the information is present and
underweighted; a negative result means it is not there at all.

-------------------------------------------------------------------------------
STEP 1 -- PATCH predict_video TO DUMP FEATURES
-------------------------------------------------------------------------------
`predict_video` computes unet_feat_src / unet_feat_tgt per frame pair and throws
them away. Accumulate them into one array aligned with the global node index.

(a) Near the other running registries, add:

        node_feats: dict[int, np.ndarray] = {}

(b) Immediately AFTER the two `model._index_features(...)` calls, add:

        f_src = unet_feat_src[0].detach().cpu().numpy()   # (n_src, C)
        f_tgt = unet_feat_tgt[0].detach().cpu().numpy()   # (n_tgt, C)
        for k, gk in enumerate(idx_src):
            node_feats.setdefault(int(gk), f_src[k])
        for k, gk in enumerate(idx_tgt):
            node_feats.setdefault(int(gk), f_tgt[k])

    `setdefault` matters: with W=2 and stride 1 each frame appears in two
    windows, once as source and once as target, so features get computed twice
    from different windows. Keep the first and stay consistent.

(c) Just before `return coords, all_edges`, add:

        import numpy as _np
        n_nodes = len(coords)
        C = len(next(iter(node_feats.values())))
        feat_arr = _np.zeros((n_nodes, C), dtype=_np.float32)
        for gi, v in node_feats.items():
            feat_arr[gi] = v
        _np.savez_compressed(
            Path(ds_path).name + "_feats.npz", feats=feat_arr, coords=coords,
        )

    A node with no feature row (never in a scored pair) stays zero; the
    analysis below skips those.

I am NOT certain `_index_features` returns shape (1, n, C) -- that is inferred
from how it is called. If the reshape fails, print the shape and adjust.

-------------------------------------------------------------------------------
STEP 2 -- RERUN ONE VIDEO, THEN RUN THIS SCRIPT
-------------------------------------------------------------------------------
    bash: rerun 6bba_207c6aaf with the app=2.0 config
    uv run python feature_test.py 6bba_207c6aaf

WHAT IT REPORTS
---------------
For each STOLEN / displaced failure, cosine similarity in feature space:

    s_true      source vs the TRUE partner
    s_chosen    source vs the node the model actually linked
    FLIPPABLE   s_true > s_chosen  (appearance points the right way)

Same structure as the motion test, so the numbers are directly comparable.
Near 50% means chance, and the appearance hypothesis dies with the other four.
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
FEATS = Path("6bba_207c6aaf_feats.npz")       # written by the patch
DEFAULT_STEM = "6bba_207c6aaf"
# -------------------------------------------------------------------------

sys.path.insert(0, ".")
from scripts.evaluate import open_dataset, compute_metric  # noqa: E402


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(a @ b / (na * nb))


def main(stem):
    z = np.load(FEATS)
    feats = z["feats"]
    print(f"features {feats.shape}   "
          f"all-zero rows {int((np.abs(feats).sum(1) == 0).sum())}")

    ds = open_dataset(DATA_DIR / stem, require_tracks=True)
    pred = td.graph.IndexedRXGraph.from_geff(PRED_DIR / f"{stem}.geff")
    pred = pred[0] if isinstance(pred, tuple) else pred
    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    print(f"{stem}: TP={er.edge_tp} FP={er.edge_fp} FN={er.edge_fn}")

    scale = np.array(ds.scale[-3:], dtype=float)
    na, ea = pred.node_attrs(), pred.edge_attrs()
    pos = {r["node_id"]: np.array([r["z"], r["y"], r["x"]], float) * scale
           for r in na.iter_rows(named=True)}
    matched = na.filter(pl.col("match_node_id") != -1)
    gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
    pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))

    tp = ea.filter(pl.col("matched_edge_mask"))
    tp_pairs = {(pred_to_gt[s], pred_to_gt[t])
                for s, t in zip(tp["source_id"], tp["target_id"])}
    fn = [(s, t) for s, t in ds.tracks.edge_list() if (s, t) not in tp_pairs]
    both = [(s, t) for s, t in fn if s in gt_to_pred and t in gt_to_pred]
    print(f"FN {len(fn)}   both endpoints detected {len(both)}")

    rows = []
    for gs, gtt in both:
        ps, pt = gt_to_pred[gs], gt_to_pred[gtt]
        out = ea.filter(pl.col("source_id") == ps)
        if len(out) == 0:
            continue
        pc = int(out["target_id"][0])
        if max(ps, pt, pc) >= len(feats):
            continue
        fs, ft, fc = feats[ps], feats[pt], feats[pc]
        if min(np.abs(fs).sum(), np.abs(ft).sum(), np.abs(fc).sum()) == 0:
            continue
        rows.append({
            "s_true": cosine(fs, ft),
            "s_chosen": cosine(fs, fc),
            "d_true": float(np.linalg.norm(pos[pt] - pos[ps])),
            "d_chosen": float(np.linalg.norm(pos[pc] - pos[ps])),
        })

    if not rows:
        raise SystemExit("no usable cases -- check the feature dump aligned "
                         "with node ids")

    df = pl.DataFrame(rows)
    st = df["s_true"].to_numpy()
    sc = df["s_chosen"].to_numpy()
    flip = int((st > sc).sum())

    print(f"\n=== appearance separability  (n={len(df)}) ===")
    print(f"  cosine(source, TRUE partner)    median {np.median(st):.4f}")
    print(f"  cosine(source, CHOSEN decoy)    median {np.median(sc):.4f}")
    print(f"  median margin (true - chosen)   {np.median(st - sc):+.4f}")
    print(f"  FLIPPABLE (s_true > s_chosen)   {flip}/{len(df)}  "
          f"({100*flip/len(df):.0f}%)")

    # is similarity just distance in disguise?
    d = np.concatenate([df["d_true"].to_numpy(), df["d_chosen"].to_numpy()])
    s = np.concatenate([st, sc])
    print(f"  corr(cosine, distance)          {np.corrcoef(d, s)[0,1]:+.3f}")

    print("\n" + "=" * 70)
    print("FLIPPABLE near 50% -> the UNet features do not separate these pairs.")
    print("  Appearance is absent, not underweighted; segmentation-derived")
    print("  shape features would be needed, which is a separate project.")
    print("Comfortably above 70% -> substitution is viable; an edge-centric")
    print("  model can be built on this pipeline's features.")
    print("A strong corr(cosine, distance) means the features encode position")
    print("  rather than appearance -- check before trusting a positive result.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM)