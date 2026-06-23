#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-help}"

INTERNAL_SOURCE="${INTERNAL_SOURCE:-$ROOT_DIR/Dataset}"
EXTERNAL_SOURCE="${EXTERNAL_SOURCE:-$ROOT_DIR/Dataset_Test_Enriched}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/Outputs/godark_external01}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
EXTERNAL_SEED="${EXTERNAL_SEED:-42042}"
LIMIT_ROWS="${LIMIT_ROWS:-0}"
SOURCE_INCLUDE_LABELS="${SOURCE_INCLUDE_LABELS:-drifting_longlines fixed_gear purse_seines trawlers}"
REBUILD_EXTERNAL_ENRICHED="${REBUILD_EXTERNAL_ENRICHED:-0}"
GODARK_MIN_DISTANCE_FROM_SHORE_NM="${GODARK_MIN_DISTANCE_FROM_SHORE_NM:-5}"

INTERNAL_DATA="$RUN_DIR/data_internal_trainval"
EXTERNAL_DATA="$RUN_DIR/data_external_test"
MODEL_DIR="$RUN_DIR/model_godark"
VAL_EVAL_DIR="$RUN_DIR/validation_eval"
EXTERNAL_EVAL_DIR="$RUN_DIR/external_test_eval"
AUDIT_PATH="$RUN_DIR/external_protocol_audit.json"

usage() {
  cat <<EOF
Usage:
  bash run_godark_external_test_pipeline.sh prepare
  bash run_godark_external_test_pipeline.sh train      # train + internal validation only
  bash run_godark_external_test_pipeline.sh evaluate   # internal validation only
  bash run_godark_external_test_pipeline.sh all
  bash run_godark_external_test_pipeline.sh final-external

Protocol:
  Internal Dataset               -> synthetic Go-Dark -> train + validation only
  Dataset_Test_Enriched external -> synthetic Go-Dark -> locked pure external test
  Training always runs exactly 50 epochs with early stopping disabled.
  External metrics require an internal-only tuning winner manifest.

Important environment variables:
  DEVICE=auto|cpu|cuda
  SEED=42
  EXTERNAL_SEED=42042
  RUN_DIR=Outputs/godark_external01
  SOURCE_INCLUDE_LABELS="drifting_longlines fixed_gear purse_seines trawlers"
  GODARK_MIN_DISTANCE_FROM_SHORE_NM=5
  REBUILD_EXTERNAL_ENRICHED=1  # optional: rerun enrich_dataset_test.py first
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

generate_domain() {
  local source_dir="$1"
  local out_dir="$2"
  local seed="$3"
  shift 3
  local -a include_args=("$@")

  run_step "Generate Go-Dark candidates from $source_dir" \
    "$PYTHON_BIN" "$ROOT_DIR/main.py" make_godark \
    --input_path "$source_dir" \
    --out_dir "$out_dir" \
    --limit_rows "$LIMIT_ROWS" \
    --include_labels "${include_args[@]}" \
    --max_vessels_per_file 0 \
    --min_points_per_vessel 180 \
    --events_per_vessel 3 \
    --min_hidden_points 20 \
    --max_hidden_points 120 \
    --min_dark_seconds 43200 \
    --max_dark_seconds 604800 \
    --min_hidden_distance_km 0.5 \
    --max_source_gap_seconds 10800 \
    --context_points_each_side 60 \
    --min_distance_from_shore_nm "$GODARK_MIN_DISTANCE_FROM_SHORE_NM" \
    --shore_distance_unit meters \
    --ping_window_seconds 43200 \
    --min_ping_count_prev_window 14 \
    --combine_outputs \
    --seed "$seed"

  run_step "Preprocess independent Go-Dark domain $out_dir" \
    "$PYTHON_BIN" "$ROOT_DIR/main.py" preprocess \
    --data_dir "$out_dir" \
    --out_dir "$out_dir" \
    --task godark \
    --seq_len 120 \
    --stride 3 \
    --gap_seconds 43200 \
    --max_implied_knots 1000 \
    --min_points_per_vessel 80 \
    --max_windows_per_vessel 200 \
    --godark_min_distance_from_shore_nm "$GODARK_MIN_DISTANCE_FROM_SHORE_NM" \
    --godark_ping_window_seconds 43200 \
    --godark_min_ping_count_prev_window 14
}

prepare() {
  [[ -d "$INTERNAL_SOURCE" ]] || { echo "[error] missing $INTERNAL_SOURCE" >&2; exit 1; }

  if [[ "$REBUILD_EXTERNAL_ENRICHED" == "1" ]]; then
    run_step "Rebuild Dataset_Test_Enriched" \
      "$PYTHON_BIN" "$ROOT_DIR/enrich_dataset_test.py"
  fi
  [[ -d "$EXTERNAL_SOURCE" ]] || { echo "[error] missing $EXTERNAL_SOURCE" >&2; exit 1; }

  mkdir -p "$RUN_DIR" "$INTERNAL_DATA" "$EXTERNAL_DATA"
  read -r -a include_args <<< "$SOURCE_INCLUDE_LABELS"

  generate_domain "$INTERNAL_SOURCE" "$INTERNAL_DATA" "$SEED" "${include_args[@]}"
  generate_domain "$EXTERNAL_SOURCE" "$EXTERNAL_DATA" "$EXTERNAL_SEED" "${include_args[@]}"

  require_file "$INTERNAL_DATA/processed_godark.npz"
  require_file "$EXTERNAL_DATA/processed_godark.npz"

  run_step "Audit internal/external domain separation before training" \
    "$PYTHON_BIN" "$ROOT_DIR/audit_godark_external.py" \
    --internal_npz "$INTERNAL_DATA/processed_godark.npz" \
    --external_npz "$EXTERNAL_DATA/processed_godark.npz" \
    --internal_source_dir "$INTERNAL_SOURCE" \
    --external_source_dir "$EXTERNAL_SOURCE" \
    --allowed_labels "${include_args[@]}" \
    --expected_shore_nm "$GODARK_MIN_DISTANCE_FROM_SHORE_NM" \
    --out_path "$AUDIT_PATH"

}

ensure_prepared() {
  if [[ ! -f "$INTERNAL_DATA/processed_godark.npz" ]] \
     || [[ ! -f "$EXTERNAL_DATA/processed_godark.npz" ]]; then
    echo "[pipeline] Prepared Go-Dark NPZ not found; running prepare automatically."
    prepare
  else
    read -r -a include_args <<< "$SOURCE_INCLUDE_LABELS"
    if "$PYTHON_BIN" "$ROOT_DIR/audit_godark_external.py" \
      --internal_npz "$INTERNAL_DATA/processed_godark.npz" \
      --external_npz "$EXTERNAL_DATA/processed_godark.npz" \
      --internal_source_dir "$INTERNAL_SOURCE" \
      --external_source_dir "$EXTERNAL_SOURCE" \
      --allowed_labels "${include_args[@]}" \
      --expected_shore_nm "$GODARK_MIN_DISTANCE_FROM_SHORE_NM" \
      --out_path "$AUDIT_PATH"; then
      echo "[pipeline] Reusing protocol-compatible internal/external Go-Dark NPZ files."
    else
      echo "[pipeline] Existing NPZ is stale/incompatible; rebuilding prepare outputs."
      prepare
    fi
  fi
}

train_internal() {
  require_file "$INTERNAL_DATA/processed_godark.npz"
  mkdir -p "$MODEL_DIR"

  set +e
  run_step "Train Go-Dark on INTERNAL train/validation only (no internal test)" \
    "$PYTHON_BIN" "$ROOT_DIR/main.py" train \
    --data_npz "$INTERNAL_DATA/processed_godark.npz" \
    --out_dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --random_state "$SEED" \
    --split_random_state "$SEED" \
    --train_random_state "$SEED" \
    --test_size 0 \
    --val_size 0.20 \
    --hidden_size 128 \
    --num_layers 1 \
    --input_proj_dim 96 \
    --embed_dim 128 \
    --dropout 0.4 \
    --weight_decay 0.003 \
    --batch_size 32 \
    --lr 3e-4 \
    --geo_aux_weight 0 \
    --epochs 50 \
    --disable_early_stopping \
    --eval_after_train \
    --validation_eval_out "$VAL_EVAL_DIR"
  local train_status=$?
  set -e

  # A completed checkpoint is authoritative. On some Windows/Git-Bash runs
  # Python can return nonzero after flushing progress-bar output even though
  # every required training artifact was saved successfully.
  local -a required_training=(
    "$MODEL_DIR/model.pt"
    "$MODEL_DIR/scaler.joblib"
    "$MODEL_DIR/split_indices.npz"
    "$MODEL_DIR/best_epoch.json"
    "$MODEL_DIR/train_config.json"
    "$MODEL_DIR/godark_hardnegative_hybrid.joblib"
  )
  for artifact in "${required_training[@]}"; do
    if [[ ! -f "$artifact" ]]; then
      echo "[error] training incomplete (exit=$train_status), missing: $artifact" >&2
      return "${train_status:-1}"
    fi
  done
  if [[ "$train_status" -ne 0 ]]; then
    echo "[pipeline] WARNING: train exited $train_status but all required artifacts are complete; continuing to eval."
  fi
}

audit_internal() {
  require_file "$INTERNAL_DATA/processed_godark.npz"
  require_file "$EXTERNAL_DATA/processed_godark.npz"
  require_file "$MODEL_DIR/model.pt"
  require_file "$MODEL_DIR/split_indices.npz"
  require_file "$VAL_EVAL_DIR/confusion_matrix.png"
  read -r -a include_args <<< "$SOURCE_INCLUDE_LABELS"

  run_step "Audit internal train/validation protocol (external metrics locked)" \
    "$PYTHON_BIN" "$ROOT_DIR/audit_godark_external.py" \
    --internal_npz "$INTERNAL_DATA/processed_godark.npz" \
    --external_npz "$EXTERNAL_DATA/processed_godark.npz" \
    --internal_source_dir "$INTERNAL_SOURCE" \
    --external_source_dir "$EXTERNAL_SOURCE" \
    --allowed_labels "${include_args[@]}" \
    --expected_shore_nm "$GODARK_MIN_DISTANCE_FROM_SHORE_NM" \
    --split_indices "$MODEL_DIR/split_indices.npz" \
    --out_path "$AUDIT_PATH"

  echo
  echo "Internal validation: $VAL_EVAL_DIR/eval_summary.json"
  echo "Internal matrix: $VAL_EVAL_DIR/confusion_matrix.png"
  echo "External metrics remain locked until the internal tuning winner exists."
  echo "Protocol audit: $AUDIT_PATH"
}

evaluate_internal() {
  require_file "$INTERNAL_DATA/processed_godark.npz"
  require_file "$MODEL_DIR/model.pt"
  require_file "$MODEL_DIR/split_indices.npz"

  run_step "Evaluate INTERNAL validation" \
    "$PYTHON_BIN" "$ROOT_DIR/main.py" eval \
    --data_npz "$INTERNAL_DATA/processed_godark.npz" \
    --model_path "$MODEL_DIR/model.pt" \
    --out_dir "$VAL_EVAL_DIR" \
    --device "$DEVICE" \
    --random_state "$SEED" \
    --eval_split val

  audit_internal
}

final_external() {
  local winner="$ROOT_DIR/Outputs/godark_tuning01_internal_oof/winner_internal_only.json"
  require_file "$winner"
  run_step "Evaluate locked internal winner on PURE EXTERNAL test" \
    "$PYTHON_BIN" "$ROOT_DIR/run_godark_hparam_tuning.py" external
}

case "$MODE" in
  prepare) prepare ;;
  train) ensure_prepared; train_internal; audit_internal ;;
  evaluate) evaluate_internal ;;
  all) prepare; train_internal; audit_internal ;;
  final-external) final_external ;;
  help|-h|--help) usage ;;
  *) echo "[error] unknown mode: $MODE" >&2; usage; exit 2 ;;
esac
