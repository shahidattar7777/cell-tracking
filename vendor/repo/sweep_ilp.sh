#!/usr/bin/env bash
set -uo pipefail
# no -e: one config failing shouldn't kill the sweep

# ILP appearance/disappearance weight sweep on the 3 worst videos.
# Baseline (app=0.1, disapp=0.1) edge_jaccard:
#   6bba_207c6aaf  0.657
#   6bba_57b7cc1e  0.662
#   6bba_6feb10f0  0.807
#
# Run from vendor/repo:  bash sweep_ilp.sh 2>&1 | tee sweep.log

DATA_DIR="C:/Users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train"
WEIGHTS="weights/unet_transformer/split_0/edge_predictor_best.pth"

STEMS="6bba_207c6aaf 6bba_57b7cc1e 6bba_6feb10f0"
WEIGHTS_SWEEP="2.0"

for w in $WEIGHTS_SWEEP; do
  METHOD="ilp_app${w}"
  echo ""
  echo "############ appearance=disappearance=$w ############"
  for stem in $STEMS; do
    echo "=== $stem  ($(date +%H:%M:%S)) ==="
    python scripts/predict_unet_transformer.py \
      --weights "$WEIGHTS" \
      --method "$METHOD" \
      --debug-video "$DATA_DIR/$stem" \
      --use-ilp \
      --ilp-appearance-weight "$w" \
      --ilp-disappearance-weight "$w"
  done
done

echo ""
echo "Done: $(date)"