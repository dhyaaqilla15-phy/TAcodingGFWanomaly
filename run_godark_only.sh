#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$ROOT_DIR/Dataset}"
RUN_DIR="${2:-$ROOT_DIR/output/run04}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LIMIT_ROWS="${LIMIT_ROWS:-300000}"
SOURCE_EXCLUDE_LABELS="${SOURCE_EXCLUDE_LABELS:-pole_and_line trollers}"

usage() {
  cat <<EOF
Usage:
  bash run_godark_only.sh [DATA_DIR] [RUN_DIR]

Default:
  DATA_DIR = $ROOT_DIR/Dataset
  RUN_DIR  = $ROOT_DIR/output/run04

Optional environment variables:
  DEVICE=auto|cpu|cuda
  SEED=42
  LIMIT_ROWS=300000
  SOURCE_EXCLUDE_LABELS="pole_and_line trollers"

Example:
  bash run_godark_only.sh
  DEVICE=cuda bash run_godark_only.sh "$ROOT_DIR/Dataset" "$ROOT_DIR/output/run04"
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
OUT_GODARK="$RUN_DIR/godark"

mkdir -p "$OUT_GODARK"

read -r -a source_exclude_labels <<< "$SOURCE_EXCLUDE_LABELS"

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
echo "[pipeline] source_exclude_labels = $SOURCE_EXCLUDE_LABELS"

run_step "[1/4] Go-dark generate (AGGRESSIVE: 5x events, 100 label_after)" \
  python3 "$MAIN_PY" make_godark \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_GODARK" \
  --limit_rows "$LIMIT_ROWS" \
  --exclude_labels "${source_exclude_labels[@]}" \
  --max_vessels_per_file 20 \
  --min_points_per_vessel 120 \
  --events_per_vessel 5 \
  --min_hidden_points 10 \
  --max_hidden_points 80 \
  --min_dark_seconds 1800 \
  --max_dark_seconds 604800 \
  --min_hidden_distance_km 0.2 \
  --label_before_points 10 \
  --label_after_points 100 \
  --combine_outputs \
  --seed "$SEED"

run_step "[2/4] Go-dark preprocess (lenient gap detection)" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_GODARK" \
  --out_dir "$OUT_GODARK" \
  --task godark \
  --seq_len 120 \
  --stride 3 \
  --gap_seconds 43200 \
  --max_implied_knots 1000 \
  --min_points_per_vessel 80 \
  --spoofing_window_threshold 0.05

run_step "[2b/4] Go-dark plot preprocessed trajectory" \
  python3 "$MAIN_PY" plot_preprocessed \
  --npz_path "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/plots/preprocessed" \
  --task godark \
  --max_windows 12

run_step "[3/4] Go-dark train (AGGRESSIVE: dropout=0.4, lr=1e-4)" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/model_godark" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --dropout 0.4 \
  --batch_size 32 \
  --lr 1e-4 \
  --epochs 400 \
  --early_stop_patience 120

run_step "[4/4] Go-dark eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --model_path "$OUT_GODARK/model_godark/model.pt" \
  --out_dir "$OUT_GODARK/model_godark" \
  --device "$DEVICE" \
  --random_state "$SEED"

echo
echo "============================================================"
echo "[4/4] Go-dark pipeline selesai"
echo "Output: $OUT_GODARK"
echo "Results: $OUT_GODARK/model_godark/eval_summary.json"
