#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-geo0}"
DATA_DIR="${2:-Dataset}"
EXTERNAL_TEST_DIR="${3:-Dataset_Test_Enriched}"

case "$VARIANT" in
  geo0)
    ROOT_OUT="Outputs/gear_tuning04_gap12h_opfilter_1to12_geo0_multiseed"
    USE_LOCATION=1
    ;;
  motion_only)
    ROOT_OUT="Outputs/gear_tuning05_gap12h_opfilter_1to12_geo0_motiononly_multiseed"
    USE_LOCATION=0
    ;;
  *)
    echo "Usage: bash run_gear_gap12_segmented_ablation.sh {geo0|motion_only} [data_dir] [external_test_dir]"
    exit 2
    ;;
esac

SEEDS="${SEEDS:-42 43 44 45 46}" \
DEVICE="${DEVICE:-cuda}" \
EPOCHS="${EPOCHS:-50}" \
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}" \
RESUME_COMPLETED_SEEDS="${RESUME_COMPLETED_SEEDS:-1}" \
COMMAND_RETRIES="${COMMAND_RETRIES:-2}" \
GAP_SECONDS=43200 \
GEAR_USE_OPERATIONAL_FILTER=1 \
GEAR_OP_SPEED_MIN=1 \
GEAR_OP_SPEED_MAX=12 \
GEAR_USE_LOCATION_FEATURES="$USE_LOCATION" \
GEO_AUX_WEIGHT=0 \
bash run_gear_external_multiseed.sh \
  "$DATA_DIR" \
  "$EXTERNAL_TEST_DIR" \
  "$ROOT_OUT"
