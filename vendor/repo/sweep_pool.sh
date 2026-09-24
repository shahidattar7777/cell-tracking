#!/usr/bin/env bash
set -uo pipefail
# no -e: one config failing shouldn't kill the sweep

# pool_kernel_um sweep, on top of the validated app/disapp=2.0 config.
#
# pool_kernel_um is the non-maximum-suppression radius for detection peaks.
# Larger -> fewer, more separated detections. The weights ship with
# config.json saying 5.0; the code has always run 3.0 (no CLI flag existed
# until now), so this parameter is unreachable by env-var sweeps.
#
# Baselines to beat (app=2.0, pool=3.0):
#   6bba_207c6aaf   TP  394  FP  86  FN  56   edgeJ 0.7351
#   6bba_57b7cc1e   TP 1325  FP 409  FN 267   edgeJ 0.6622
#   6bba_6feb10f0   TP 1203  FP 113  FN 140   edgeJ 0.8262
#
# Watch node_recall: it was 0.985 mean at app=2.0, so there is little
# headroom before suppression starts costing real cells.
#
# Run from vendor/repo:  bash sweep_pool.sh 2>&1 | tee sweep_pool.log

DATA_DIR="C:/Users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train"
WEIGHTS="weights/unet_transformer/split_0/edge_predictor_best.pth"

STEMS="6bba_207c6aaf 6bba_6feb10f0"
POOL_SWEEP="4.0 5.0 6.0"

APP=2.0

[ -d "$DATA_DIR" ] || { echo "ERROR: data dir not found: $DATA_DIR"; exit 1; }
[ -f "$WEIGHTS" ]  || { echo "ERROR: weights not found: $WEIGHTS"; exit 1; }

echo "Started: $(date)"

for pool in $POOL_SWEEP; do
  METHOD="pool${pool}_app${APP}"
  echo ""
  echo "############ pool_kernel_um=$pool  app=disapp=$APP ############"
  for stem in $STEMS; do
    echo "=== $stem  ($(date +%H:%M:%S)) ==="
    uv run python scripts/predict_unet_transformer.py \
      --weights "$WEIGHTS" \
      --method "$METHOD" \
      --debug-video "$DATA_DIR/$stem" \
      --use-ilp \
      --ilp-appearance-weight "$APP" \
      --ilp-disappearance-weight "$APP" \
      --pool-kernel-um "$pool"
  done
done

echo ""
echo "Finished: $(date)"
echo "Check the startup line of each run says pool_kernel_um=<the swept value>."