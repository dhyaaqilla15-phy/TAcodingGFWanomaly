#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-help}"
INTERNAL_DIR="${TRANSSHIPMENT_INTERNAL_DIR:-$ROOT_DIR/Dataset}"
EXTERNAL_DIR="${TRANSSHIPMENT_EXTERNAL_DIR:-$ROOT_DIR/Dataset_Test_Enriched}"
OUT_ROOT="${TRANSSHIPMENT_OUT_ROOT:-$ROOT_DIR/Outputs/transshipment_external01}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
VAL_FRACTION="${TRANS_VAL_FRACTION:-0.20}"
LIMIT_ROWS="${TRANS_LIMIT_ROWS:-0}"
SYNTHETIC_EVENTS="${TRANS_SYNTHETIC_EVENTS:-250}"

MANIFEST_DIR="$OUT_ROOT/manifests"
TRAIN_CSV_DIR="$OUT_ROOT/internal_train_candidates"
VAL_CSV_DIR="$OUT_ROOT/internal_validation_real_candidates"
EXTERNAL_CSV_DIR="$OUT_ROOT/external_test_real_candidates"
TRAIN_NPZ_DIR="$OUT_ROOT/data_internal_train"
VAL_NPZ_DIR="$OUT_ROOT/data_internal_validation"
EXTERNAL_NPZ_DIR="$OUT_ROOT/data_external_test"
TRAINVAL_DIR="$OUT_ROOT/data_internal_trainval"
MODEL_DIR="$OUT_ROOT/model_transshipment"
SMOKE_MODEL_DIR="$OUT_ROOT/model_transshipment_smoke"

ALLOWED_SOURCES=(drifting_longlines fixed_gear purse_seines trawlers)

usage() {
  cat <<EOF
Usage:
  bash run_transshipment_external_test_pipeline.sh prepare
  DEVICE=cuda bash run_transshipment_external_test_pipeline.sh smoke
  DEVICE=cuda bash run_transshipment_external_test_pipeline.sh train
  DEVICE=cuda bash run_transshipment_external_test_pipeline.sh evaluate
  DEVICE=cuda bash run_transshipment_external_test_pipeline.sh all

Protocol:
  internal train      = real candidates + synthetic encounter
  internal validation = real candidates only, vessel-disjoint from train
  external test       = Dataset_Test_Enriched real candidates only
  allowed sources     = drifting_longlines fixed_gear purse_seines trawlers

Training is never started by prepare. Smoke runs 5 epochs and validation only.
Full train runs exactly 50 epochs, then validation and external evaluation.
EOF
}

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

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] required file not found: $1" >&2
    exit 1
  fi
}

generate_candidates() {
  local input_dir="$1"
  local output_dir="$2"
  local synthetic_count="$3"
  local mmsi_manifest="${4:-}"
  local mmsi_args=()
  if [[ -n "$mmsi_manifest" ]]; then
    mmsi_args+=(--include_mmsi_path "$mmsi_manifest")
  fi
  python "$ROOT_DIR/main.py" make_transshipment \
    --input_path "$input_dir" \
    --out_dir "$output_dir" \
    --mode both \
    --seed "$SEED" \
    --limit_rows "$LIMIT_ROWS" \
    --include_labels "${ALLOWED_SOURCES[@]}" \
    "${mmsi_args[@]}" \
    --max_vessels_per_file 0 \
    --min_points_per_vessel 40 \
    --grid_minutes 10 \
    --max_interp_gap_minutes 90 \
    --encounter_distance_km 0.5 \
    --encounter_candidate_distance_km 2.0 \
    --encounter_min_hours 2 \
    --encounter_max_speed_knots 2 \
    --encounter_min_port_km 10 \
    --loitering_min_hours 8 \
    --loitering_max_speed_knots 2 \
    --loitering_min_shore_nm 20 \
    --synthetic_encounters_per_file "$synthetic_count" \
    --synthetic_min_distance_km 0.05 \
    --synthetic_max_distance_km 0.48 \
    --synthetic_min_duration_hours 2 \
    --synthetic_max_duration_hours 6 \
    --synthetic_min_speed_knots 0.2 \
    --synthetic_max_speed_knots 1.9 \
    --synthetic_course_jitter_deg 30 \
    --max_normal_events_per_file 1500 \
    --combine_outputs
}

preprocess_candidates() {
  local data_dir="$1"
  local output_dir="$2"
  python "$ROOT_DIR/main.py" preprocess \
    --data_dir "$data_dir" \
    --out_dir "$output_dir" \
    --task transshipment \
    --transshipment_target any \
    --transshipment_feature_mode fair \
    --seq_len 24 \
    --stride 3 \
    --min_points_per_vessel 3 \
    --max_windows_per_vessel 2000 \
    --max_windows_per_file 0 \
    --no_jump_filter
}

prepare() {
  mkdir -p "$MANIFEST_DIR" "$TRAIN_CSV_DIR" "$VAL_CSV_DIR" "$EXTERNAL_CSV_DIR" \
    "$TRAIN_NPZ_DIR" "$VAL_NPZ_DIR" "$EXTERNAL_NPZ_DIR" "$TRAINVAL_DIR"

  run_step "[1/8] Source-stratified vessel-disjoint MMSI manifests" \
    python "$ROOT_DIR/prepare_transshipment_protocol.py" split-mmsi \
      --internal_dir "$INTERNAL_DIR" \
      --external_dir "$EXTERNAL_DIR" \
      --out_dir "$MANIFEST_DIR" \
      --val_fraction "$VAL_FRACTION" \
      --seed "$SEED"

  run_step "[2/8] Internal TRAIN candidates: real + synthetic" \
    generate_candidates "$INTERNAL_DIR" "$TRAIN_CSV_DIR" "$SYNTHETIC_EVENTS" \
      "$MANIFEST_DIR/train_mmsi.csv"

  run_step "[3/8] Internal VALIDATION candidates: real only" \
    generate_candidates "$INTERNAL_DIR" "$VAL_CSV_DIR" 0 \
      "$MANIFEST_DIR/validation_mmsi.csv"

  run_step "[4/8] PURE EXTERNAL candidates: real only" \
    generate_candidates "$EXTERNAL_DIR" "$EXTERNAL_CSV_DIR" 0 ""

  run_step "[5/8] Preprocess internal train" \
    preprocess_candidates "$TRAIN_CSV_DIR" "$TRAIN_NPZ_DIR"
  run_step "[6/8] Preprocess real internal validation" \
    preprocess_candidates "$VAL_CSV_DIR" "$VAL_NPZ_DIR"
  run_step "[7/8] Preprocess pure external test" \
    preprocess_candidates "$EXTERNAL_CSV_DIR" "$EXTERNAL_NPZ_DIR"

  run_step "[8/8] Combine explicit train/validation and enforce leakage guards" \
    python "$ROOT_DIR/prepare_transshipment_protocol.py" combine-audit \
      --train_npz "$TRAIN_NPZ_DIR/processed_transshipment.npz" \
      --validation_npz "$VAL_NPZ_DIR/processed_transshipment.npz" \
      --external_npz "$EXTERNAL_NPZ_DIR/processed_transshipment.npz" \
      --out_dir "$TRAINVAL_DIR"

  echo "[transshipment] prepare complete; no training was run."
}

train_model() {
  local epochs="$1"
  local model_dir="$2"
  local include_external="$3"
  require_file "$TRAINVAL_DIR/processed_transshipment_trainval.npz"
  require_file "$TRAINVAL_DIR/split_indices.npz"
  require_file "$EXTERNAL_NPZ_DIR/processed_transshipment.npz"
  local external_args=()
  if [[ "$include_external" == "1" ]]; then
    external_args+=(
      --external_test_npz "$EXTERNAL_NPZ_DIR/processed_transshipment.npz"
      --external_eval_out "$OUT_ROOT/external_test_eval"
    )
  fi
  python "$ROOT_DIR/main.py" train \
    --data_npz "$TRAINVAL_DIR/processed_transshipment_trainval.npz" \
    --split_indices_path "$TRAINVAL_DIR/split_indices.npz" \
    --out_dir "$model_dir" \
    --device "$DEVICE" \
    --random_state "$SEED" \
    --split_random_state "$SEED" \
    --train_random_state "$SEED" \
    --test_size 0 \
    --val_size "$VAL_FRACTION" \
    --epochs "$epochs" \
    --batch_size 128 \
    --lr 0.00025 \
    --hidden_size 128 \
    --num_layers 1 \
    --input_proj_dim 96 \
    --embed_dim 128 \
    --dropout 0.40 \
    --attention_heads 4 \
    --attention_layers 1 \
    --optimizer adamw \
    --weight_decay 0.0013 \
    --geo_aux_weight 0 \
    --disable_early_stopping \
    --eval_after_train \
    --validation_eval_out "$OUT_ROOT/validation_eval" \
    "${external_args[@]}"
}

evaluate_only() {
  require_file "$MODEL_DIR/model.pt"
  run_step "Evaluate real internal validation" \
    python "$ROOT_DIR/main.py" eval \
      --data_npz "$TRAINVAL_DIR/processed_transshipment_trainval.npz" \
      --model_path "$MODEL_DIR/model.pt" \
      --out_dir "$OUT_ROOT/validation_eval" \
      --device "$DEVICE" \
      --eval_split val
  run_step "Evaluate PURE EXTERNAL test" \
    python "$ROOT_DIR/main.py" eval \
      --data_npz "$EXTERNAL_NPZ_DIR/processed_transshipment.npz" \
      --model_path "$MODEL_DIR/model.pt" \
      --out_dir "$OUT_ROOT/external_test_eval" \
      --device "$DEVICE" \
      --eval_split all
}

case "$MODE" in
  prepare) prepare ;;
  smoke)
    run_step "Smoke training: 5 epochs, validation only" train_model 5 "$SMOKE_MODEL_DIR" 0
    ;;
  train)
    run_step "Full training: 50 epochs + automatic validation/external eval" \
      train_model 50 "$MODEL_DIR" 1
    ;;
  evaluate) evaluate_only ;;
  all)
    prepare
    run_step "Full training: 50 epochs + automatic validation/external eval" \
      train_model 50 "$MODEL_DIR" 1
    ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
