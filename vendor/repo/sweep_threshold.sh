#!/usr/bin/env bash
set -uo pipefail
# no -e: one config failing shouldn't kill the sweep

# cfg.threshold sweep, on top of the validated app/disapp=2.0 config.
#
# WHY: 18 of 69 linking failures on 6bba_207c6aaf had the true target
# ORPHANED -- nothing claimed it, nothing competed for it, and the true source
# still had a free child slot. The only way that happens is the true edge
# scoring below cfg.threshold. Lowering it admits those edges.
#
# 0.5 -> 0.3 was tested before and came out break-even (+5 TP / +6 FP), but
# that was at ILP weight 0.1, when the solver had no reason to reject spurious
# links. At weight 2.0 it cut FP while raising TP across 20 videos, so the
# extra candidates may now get filtered rather than accepted. Different regime.
#
# Baselines to beat (threshold 0.3, app 2.0):
#   6bba_207c6aaf   TP  394  FP  86  FN  56   edgeJ 0.7351
#   6bba_6feb10f0   TP 1203  FP 113  FN 140   edgeJ 0.8262
#
# WATCH: a lower threshold means MORE candidate edges and a LARGER ILP.
# 6bba_57b7cc1e (65k nodes) already exhausted 16 GB at threshold 0.3, so it is
# deliberately excluded here. Add it back only on a bigger machine.
#
# Run from vendor/repo:  bash sweep_threshold.sh 2>&1 | tee sweep_threshold.log

DATA_DIR="C:/Users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train"
WEIGHTS="weights/unet_transformer/split_0/edge_predictor_best.pth"

STEMS="6bba_207c6aaf "
THRESH_SWEEP="0.3"

APP=2.0

[ -d "$DATA_DIR" ] || { echo "ERROR: data dir not found: $DATA_DIR"; exit 1; }
[ -f "$WEIGHTS" ]  || { echo "ERROR: weights not found: $WEIGHTS"; exit 1; }

echo "Started: $(date)"

for th in $THRESH_SWEEP; do
  METHOD="th${th}_app${APP}"
  echo ""
  echo "############ cfg.threshold=$th  app=disapp=$APP ############"
  for stem in $STEMS; do
    echo "=== $stem  ($(date +%H:%M:%S)) ==="
    uv run python scripts/predict_unet_transformer.py \
      --weights "$WEIGHTS" \
      --method "$METHOD" \
      --debug-video "$DATA_DIR/$stem" \
      --use-ilp \
      --ilp-appearance-weight "$APP" \
      --ilp-disappearance-weight "$APP" \
      --threshold "$th" \
      --pool-kernel-um 2.0 \
      --unet-batch-size 1
  done
done

echo ""
echo "Finished: $(date)"
echo "Check each startup line reports the swept threshold, and watch the"
echo "'build_graph: N coords, M edges' line -- M should RISE as threshold falls."