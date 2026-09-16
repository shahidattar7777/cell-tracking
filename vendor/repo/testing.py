from scripts.evaluate import (
    open_dataset, compute_metric, node_recall,
    _read_estimated_n_total, DATA_DIR,
)
from src.biohub_tracking.metrics import per_sample_metrics, summarise
import tracksdata as td
import polars as pl
from pathlib import Path
import json
import numpy as np

# name = "6bba_0c7fa718"
# pred_path = "/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/6bba_0c7fa718.geff"
# pred_path_2 = "/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/44b6_d78e09d9.geff"
# pred_path_2 = "/c/users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train/44b6_d78e09d9.geff"

pred_path = r"C:\Users\shahi\OneDrive\Documents\cell_tracking\cell-tracking\vendor\repo\predictions\shahi\unet_transformer\split_0\6bba_0c7fa718.geff"
pred_path_2 = r"C:\Users\shahi\OneDrive\Documents\cell_tracking\cell-tracking\vendor\repo\predictions\shahi\unet_transformer\split_0\44b6_d78e09d9.geff"

# # # ds = open_dataset(DATA_DIR / name, require_tracks=True)
# # ds = open_dataset("/Users/shahidattar/cell_tracking/data/train/6bba_0c7fa718", require_tracks=True) 
# # pred = td.graph.IndexedRXGraph.from_geff(pred_path)
# # pred = pred[0] if isinstance(pred, tuple) else pred

# # er = compute_metric(pred, ds.tracks, scale=ds.scale)
# # recall = node_recall(pred, ds.tracks)
# # n_total = _read_estimated_n_total("/Users/shahidattar/cell_tracking/data/train/6bba_0c7fa718.geff")
# # print(er.edge_tp, er.edge_fp, er.edge_fn, er.num_pred_nodes)
# # print(summarise([per_sample_metrics(er, n_total, recall)]))

def score_run(pred_path, stem, data_dir, label=None):
    import tracksdata as td
    from pathlib import Path
    from scripts.evaluate import open_dataset, compute_metric, node_recall, _read_estimated_n_total
    from src.biohub_tracking.metrics import per_sample_metrics

    data_dir = Path(data_dir)
    # ds = open_dataset(data_dir / stem, require_tracks=True) #Path for Linux
    ds = open_dataset(f"{data_dir}\\{stem}", require_tracks=True) #Path for Windows

    pred = td.graph.IndexedRXGraph.from_geff(pred_path)
    pred = pred[0] if isinstance(pred, tuple) else pred

    er = compute_metric(pred, ds.tracks, scale=ds.scale)
    recall = node_recall(pred, ds.tracks)
    n_total = _read_estimated_n_total(data_dir / f"{stem}.geff")

    row = per_sample_metrics(er, n_total, recall)
    row["stem"] = stem
    row["label"] = label or Path(pred_path).stem
    row["pred_path"] = str(pred_path)
    row["gt_nodes"] = ds.tracks.num_nodes()
    row["gt_edges"] = ds.tracks.num_edges()
    return row



# p = "/Users/shahidattar/cell_tracking/data/train/"
# p = "/c/users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train/"
# p = r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data\biohub-cell-tracking-during-development\train"

# rows = [
#     score_run(pred_path, "6bba_0c7fa718", p, label = "ilp" ),
#     score_run(pred_path_2, "44b6_d78e09d9", p, label = "ilp" )
# ]
# df = pl.DataFrame(rows)
# # print(score_run(pred_path_2, "44b6_d78e09d9", p, label = "ilp" ))

# p2 = td.graph.IndexedRXGraph.from_geff("/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/44b6_d78e09d9.geff")
# p2 = td.graph.IndexedRXGraph.from_geff("/c/users/shahi/OneDrive/Documents/cell_tracking/cell-tracking/vendor/repo/predictions/shahi/unet_transformer/split_0/44b6_d78e09d9.geff")

p2 = td.graph.IndexedRXGraph.from_geff(r"C:\Users\shahi\OneDrive\Documents\cell_tracking\cell-tracking\vendor\repo\predictions\shahi\unet_transformer\split_0\44b6_d78e09d9.geff")
pred = p2[0]
# print([m for m in dir(pred) if not m.startswith("_")])

ds = open_dataset(r"C:\Users\shahi\OneDrive\Documents\cell_tracking\data\biohub-cell-tracking-during-development\train\44b6_d78e09d9", require_tracks=True)
# p2 = p2[0] if isinstance(p2, tuple) else p2
# print(p2.num_nodes(), p2.num_edges())
print(compute_metric(pred, ds.tracks, scale=ds.scale))
# print(pred.node_attr_keys())
# print(pred.edge_attr_keys())
# print(type(pred.node_attrs(attr_keys=["t", "z", "y", "x"])))
# print(pred.edge_list()[:5])

ea = pred.edge_attrs()
print(ea.columns, len(ea))
print(ea["matched_edge_mask"].sum())

na = pred.node_attrs()
# print(na.head())
# print(na["match_node_id"].is_not_null().sum())

# print((na["match_node_id"] != -1).sum())
ea = pred.edge_attrs()
# print(ea.columns, len(ea))
# print(ea["matched_edge_mask"].sum())
fp = ea.filter(~pl.col("matched_edge_mask"))
print(len(fp))
matched = na.filter(pl.col("match_node_id") != -1)
gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))
gt_edges = ds.tracks.edge_list()
print(len(gt_to_pred), len(gt_edges))

pred_to_gt = dict(zip(matched["node_id"], matched["match_node_id"]))

tp_edges = ea.filter(pl.col("matched_edge_mask"))
tp_gt_pairs = {
    (pred_to_gt[s], pred_to_gt[t])
    for s, t in zip(tp_edges["source_id"], tp_edges["target_id"])
}

fn_edges = [(s, t) for s, t in gt_edges if (s, t) not in tp_gt_pairs]
# print(len(fn_edges), len(tp_gt_pairs))

gt_to_pred = dict(zip(matched["match_node_id"], matched["node_id"]))

both_found = [(s, t) for s, t in fn_edges if s in gt_to_pred and t in gt_to_pred]
# print(len(both_found))

# for gs, gt_ in both_found:
#     ps, pt = gt_to_pred[gs], gt_to_pred[gt_]
#     out = ea.filter(pl.col("source_id") == ps)
#     inn = ea.filter(pl.col("target_id") == pt)
#     print(gs, gt_, "| src linked to:", out["target_id"].to_list(),
#           "| tgt parent:", inn["source_id"].to_list())


scale = np.array(ds.scale[-3:])   # (z, y, x) µm per voxel

coords = {
    r["node_id"]: (r["z"], r["y"], r["x"])
    for r in na.iter_rows(named=True)
}

# for gs, gt_ in both_found:
#     ps, pt = gt_to_pred[gs], gt_to_pred[gt_]

#     out = ea.filter(pl.col("source_id") == ps)
#     if len(out) == 0:
#         continue   # source unlinked — nothing to compare

#     true_d = np.linalg.norm(
#         (np.array(coords[ps]) - np.array(coords[pt])) * scale
#     )
#     chosen_d = out["edge_dist"].to_list()
#     chosen_p = out["edge_prob"].to_list()
#     print(f"{gs}: true={true_d:.2f}  chosen={[f'{d:.2f}' for d in chosen_d]}  prob={[f'{p:.3f}' for p in chosen_p]}")


for gs, gt_ in both_found:
    ps, pt = gt_to_pred[gs], gt_to_pred[gt_]

    inn = ea.filter(pl.col("target_id") == pt)
    if len(inn) == 0:
        continue   # target orphaned — already covered by the first query

    true_d = np.linalg.norm(
        (np.array(coords[ps]) - np.array(coords[pt])) * scale
    )
    chosen_d = inn["edge_dist"].to_list()
    chosen_p = inn["edge_prob"].to_list()
    print(f"{gs}: true={true_d:.2f}  chosen={[f'{d:.2f}' for d in chosen_d]}  prob={[f'{p:.3f}' for p in chosen_p]}")


# .........................................
# this part is to figure out the stems for 20 of the files according to their size


# stems = sorted(p.name[:-5] for p in Path(f"{DATA_DIR}").iterdir() if p.name.endswith(".geff"))
# print(len(stems), stems)
# rows = []
# for stem in stems:
#     n = _read_estimated_n_total(f"{DATA_DIR}/{stem}.geff")
#     rows.append({"stem": stem, "prefix": stem.split("_")[0], "n_total": n})


# df = pl.DataFrame(rows)
# bad = df.filter(pl.col("n_total").is_nan())
# print(len(bad), bad["stem"].to_list()[:10])
# print(df.filter(~pl.col("n_total").is_nan())["n_total"].describe())

# sub = df.sort("n_total")[::10]
# print(len(sub), sub["n_total"].to_list())
# print(sub.group_by("prefix").len())


# Path(r"C:\Users\shahi\OneDrive\Documents\cell_tracking\cell-tracking\vendor\repo\val20.json").write_text(json.dumps(sub["stem"].to_list()))

# print(Path("/storage"))
# .............................................

#.......................................................
#Processing and scoring 20 batch predictions using score run