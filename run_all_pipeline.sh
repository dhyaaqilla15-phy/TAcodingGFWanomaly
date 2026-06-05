#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$ROOT_DIR/Dataset}"
RUN_DIR="${2:-$ROOT_DIR/output/run01}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LIMIT_ROWS="${LIMIT_ROWS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_SIZE="${VAL_SIZE:-0.15}"
GEAR_VAL_SIZE="${GEAR_VAL_SIZE:-0.20}"
GEAR_MIN_WINDOWS_PER_VESSEL="${GEAR_MIN_WINDOWS_PER_VESSEL:-0}"

usage() {
  cat <<EOF
Usage:
  bash run_all_pipeline.sh [DATA_DIR] [RUN_DIR]

Default:
  DATA_DIR = $ROOT_DIR/Dataset
  RUN_DIR  = $ROOT_DIR/output/run01

Optional environment variables:
  DEVICE=auto|cpu|cuda
  SEED=42
  LIMIT_ROWS=300000
  BATCH_SIZE=128
  VAL_SIZE=0.15
  GEAR_VAL_SIZE=0.20
  GEAR_MIN_WINDOWS_PER_VESSEL=0

Example:
  bash run_all_pipeline.sh
  bash run_all_pipeline.sh "$ROOT_DIR/Dataset" "$ROOT_DIR/output/run02"
  DEVICE=cuda SEED=43 bash run_all_pipeline.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

MAIN_PY="$ROOT_DIR/main.py"
OUT_GEAR="$RUN_DIR/gear"
OUT_SPOOF="$RUN_DIR/spoofing"
OUT_GODARK="$RUN_DIR/godark"

mkdir -p "$OUT_GEAR" "$OUT_SPOOF" "$OUT_GODARK"

run_step() {
  echo
  echo "============================================================"
  echo "$1"
  shift
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

echo "[pipeline] root     = $ROOT_DIR"
echo "[pipeline] data_dir = $DATA_DIR"
echo "[pipeline] run_dir  = $RUN_DIR"
echo "[pipeline] device   = $DEVICE"
echo "[pipeline] seed     = $SEED"
echo "[pipeline] limit    = $LIMIT_ROWS"
echo "[pipeline] val_size = $VAL_SIZE"
echo "[pipeline] gear_val_size = $GEAR_VAL_SIZE"
echo "[pipeline] gear_min_windows_per_vessel = $GEAR_MIN_WINDOWS_PER_VESSEL"

run_step "[1/15] Gear preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_GEAR" \
  --task gear \
  --exclude_labels unknown \
  --min_windows_per_vessel "$GEAR_MIN_WINDOWS_PER_VESSEL"

run_step "[2/15] Gear train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_GEAR/processed_gear.npz" \
  --out_dir "$OUT_GEAR/model_gear" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$GEAR_VAL_SIZE"

run_step "[3/15] Gear eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_GEAR/processed_gear.npz" \
  --model_path "$OUT_GEAR/model_gear/model.pt" \
  --out_dir "$OUT_GEAR/model_gear" \
  --device "$DEVICE" \
  --random_state "$SEED"

run_step "[4/15] Spoofing generate" \
  python3 "$MAIN_PY" make_spoofing \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_SPOOF" \
  --attacks gradual_drift location_jump replay meaconing ghost mirroring \
  --limit_rows "$LIMIT_ROWS" \
  --normal_keep_frac 0.25 \
  --max_vessels_per_file 20 \
  --points_per_attack 120 \
  --combine_outputs \
  --seed "$SEED"

run_step "[5/15] Spoofing preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_SPOOF" \
  --out_dir "$OUT_SPOOF" \
  --task spoofing

run_step "[6/15] Spoofing plot preprocessed trajectory" \
  python3 "$MAIN_PY" plot_preprocessed \
  --npz_path "$OUT_SPOOF/processed_spoofing.npz" \
  --out_dir "$OUT_SPOOF/plots/preprocessed" \
  --task spoofing \
  --max_windows 12

run_step "[7/15] Spoofing plot attack examples (6 types)" \
  python3 "$MAIN_PY" plot_spoofing_examples \
  --csv_path "$OUT_SPOOF/spoofed_all.csv" \
  --out_dir "$OUT_SPOOF/plots/attacks" \
  --attacks gradual_drift location_jump replay meaconing ghost mirroring

run_step "[8/15] Spoofing train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_SPOOF/processed_spoofing.npz" \
  --out_dir "$OUT_SPOOF/model_spoofing" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$VAL_SIZE"

run_step "[9/15] Spoofing eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_SPOOF/processed_spoofing.npz" \
  --model_path "$OUT_SPOOF/model_spoofing/model.pt" \
  --out_dir "$OUT_SPOOF/model_spoofing" \
  --device "$DEVICE" \
  --random_state "$SEED"

run_step "[10/15] Go-dark generate" \
  python3 "$MAIN_PY" make_godark \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_GODARK" \
  --limit_rows "$LIMIT_ROWS" \
  --max_vessels_per_file 20 \
  --min_points_per_vessel 120 \
  --events_per_vessel 6 \
  --min_hidden_points 20 \
  --max_hidden_points 120 \
  --min_dark_seconds 3600 \
  --max_dark_seconds 604800 \
  --min_hidden_distance_km 0.5 \
  --label_before_points 5 \
  --label_after_points 60 \
  --combine_outputs \
  --seed "$SEED"

run_step "[11/15] Go-dark preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_GODARK" \
  --out_dir "$OUT_GODARK" \
  --task godark \
  --seq_len 120 \
  --stride 6 \
  --gap_seconds 86400 \
  --max_implied_knots 1000 \
  --min_points_per_vessel 80 \
  --spoofing_window_threshold 0.05

run_step "[12/15] Go-dark plot preprocessed trajectory" \
  python3 "$MAIN_PY" plot_preprocessed \
  --npz_path "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/plots/preprocessed" \
  --task godark \
  --max_windows 12

run_step "[13/15] Go-dark plot event examples (6 events)" \
  python3 "$MAIN_PY" plot_go_dark_examples \
  --csv_path "$OUT_GODARK/godark_all.csv" \
  --out_dir "$OUT_GODARK/plots/events" \
  --num_examples 6

run_step "[14/15] Go-dark train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/model_godark" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$VAL_SIZE"

run_step "[15/15] Go-dark eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --model_path "$OUT_GODARK/model_godark/model.pt" \
  --out_dir "$OUT_GODARK/model_godark" \
  --device "$DEVICE" \
  --random_state "$SEED"

echo
echo "============================================================"
echo "[15/15] Pipeline selesai"
echo "Gear output     : $OUT_GEAR"
echo "Spoofing output : $OUT_SPOOF"
echo "Go-dark output  : $OUT_GODARK"
