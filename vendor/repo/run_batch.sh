#!/usr/bin/env bash
set -euo pipefail

# Batch inference over the 20-video validation set.
# Run from vendor/repo:   bash run_batch.sh 2>&1 | tee batch.log

DATA_DIR="C:/Users/shahi/OneDrive/Documents/cell_tracking/data/biohub-cell-tracking-during-development/train"
WEIGHTS="weights/unet_transformer/split_0/edge_predictor_best.pth"
METHOD="unet_transformer"
VAL_JSON="val20.json"

# --- sanity checks: fail loudly before burning hours ---
[ -d "$DATA_DIR" ] || { echo "ERROR: data dir not found: $DATA_DIR"; exit 1; }
[ -f "$WEIGHTS" ]  || { echo "ERROR: weights not found: $WEIGHTS"; exit 1; }
[ -f "$VAL_JSON" ] || { echo "ERROR: $VAL_JSON not found"; exit 1; }

STEMS=$(uv run python -c "import json;print(' '.join(json.load(open('$VAL_JSON'))))")

# verify every video exists before starting
for stem in $STEMS; do
  [ -d "$DATA_DIR/$stem.zarr" ] || { echo "ERROR: missing $DATA_DIR/$stem.zarr"; exit 1; }
  [ -d "$DATA_DIR/$stem.geff" ] || { echo "ERROR: missing $DATA_DIR/$stem.geff"; exit 1; }
done

TOTAL=$(echo "$STEMS" | wc -w)
echo "All checks passed. Running $TOTAL videos."
echo "Started: $(date)"

i=0
for stem in $STEMS; do
  i=$((i + 1))
  echo ""
  echo "=== [$i/$TOTAL] $stem  ($(date +%H:%M:%S)) ==="
  uv run python scripts/predict_unet_transformer.py \
    --weights "$WEIGHTS" \
    --method "$METHOD" \
    --debug-video "$DATA_DIR/$stem" \
    --use-ilp
done

echo ""
echo "Finished: $(date)"