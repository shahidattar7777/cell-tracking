from repo.scripts.evaluate import (
    open_dataset, compute_metric, node_recall,
    _read_estimated_n_total, DATA_DIR,
)
from src.biohub_tracking.metrics import per_sample_metrics, summarise
import tracksdata as td
import polars as pl
from pathlib import Path

# name = "6bba_0c7fa718"
# pred_path = "/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/6bba_0c7fa718.geff"
# pred_path_2 = "/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/44b6_d78e09d9.geff"

# # # ds = open_dataset(DATA_DIR / name, require_tracks=True)
# # ds = open_dataset("/Users/shahidattar/cell_tracking/data/train/6bba_0c7fa718", require_tracks=True) 
# # pred = td.graph.IndexedRXGraph.from_geff(pred_path)
# # pred = pred[0] if isinstance(pred, tuple) else pred

# # er = compute_metric(pred, ds.tracks, scale=ds.scale)
# # recall = node_recall(pred, ds.tracks)
# # n_total = _read_estimated_n_total("/Users/shahidattar/cell_tracking/data/train/6bba_0c7fa718.geff")
# # print(er.edge_tp, er.edge_fp, er.edge_fn, er.num_pred_nodes)
# # print(summarise([per_sample_metrics(er, n_total, recall)]))

# def score_run(pred_path, stem, data_dir, label=None):
#     import tracksdata as td
#     from pathlib import Path
#     from repo.scripts.evaluate import open_dataset, compute_metric, node_recall, _read_estimated_n_total
#     from src.biohub_tracking.metrics import per_sample_metrics

#     data_dir = Path(data_dir)
#     ds = open_dataset(data_dir / stem, require_tracks=True)

#     pred = td.graph.IndexedRXGraph.from_geff(pred_path)
#     pred = pred[0] if isinstance(pred, tuple) else pred

#     er = compute_metric(pred, ds.tracks, scale=ds.scale)
#     recall = node_recall(pred, ds.tracks)
#     n_total = _read_estimated_n_total(data_dir / f"{stem}.geff")

#     row = per_sample_metrics(er, n_total, recall)
#     row["stem"] = stem
#     row["label"] = label or Path(pred_path).stem
#     row["pred_path"] = str(pred_path)
#     row["gt_nodes"] = ds.tracks.num_nodes()
#     row["gt_edges"] = ds.tracks.num_edges()
#     return row



# p = "/Users/shahidattar/cell_tracking/data/train/"

# rows = [
#     # score_run(pred_path, "6bba_0c7fa718", p, label = "ilp" ),
#     score_run(pred_path_2, "44b6_d78e09d9", p, label = "ilp" )
# ]
# df = pl.DataFrame(rows)
# print(score_run(pred_path_2, "44b6_d78e09d9", p, label = "ilp" ))

# p2 = td.graph.IndexedRXGraph.from_geff("/Users/shahidattar/Downloads/archive/repo/predictions/shahidattar/unet_transformer/split_0/44b6_d78e09d9.geff")
# p2 = p2[0] if isinstance(p2, tuple) else p2
# print(p2.num_nodes(), p2.num_edges())



stems = sorted(p.name[:-5] for p in Path("/Users/shahidattar/cell_tracking/data/train").iterdir() if p.name.endswith(".geff"))
# print(len(stems), stems)


rows = []
for stem in stems:
    n = _read_estimated_n_total(f"/Users/shahidattar/cell_tracking/data/train/{stem}.geff")
    rows.append({"stem": stem, "prefix": stem.split("_")[0], "n_total": n})


df = pl.DataFrame(rows)
bad = df.filter(pl.col("n_total").is_nan())
# print(len(bad), bad["stem"].to_list()[:10])
# print(df.filter(~pl.col("n_total").is_nan())["n_total"].describe())

sub = df.sort("n_total")[::10]
print(len(sub), sub["n_total"].to_list())
print(sub.group_by("prefix").len())

