#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-Dataset}"
EXTERNAL_TEST_DIR="${2:-Dataset_Test_Enriched}"
ROOT_OUT="${3:-Outputs/gear_external_multiseed}"

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
RESUME_COMPLETED_SEEDS="${RESUME_COMPLETED_SEEDS:-1}"
COMMAND_RETRIES="${COMMAND_RETRIES:-2}"

SEQ_LEN="${SEQ_LEN:-120}"
STRIDE="${STRIDE:-6}"
GAP_SECONDS="${GAP_SECONDS:-10800}"
MIN_POINTS_PER_VESSEL="${MIN_POINTS_PER_VESSEL:-80}"
MIN_WINDOWS_PER_VESSEL="${MIN_WINDOWS_PER_VESSEL:-0}"
MAX_WINDOWS_PER_VESSEL="${MAX_WINDOWS_PER_VESSEL:-1200}"
MAX_WINDOWS_PER_FILE="${MAX_WINDOWS_PER_FILE:-20000}"
EXCLUDE_LABELS="${EXCLUDE_LABELS:-unknown pole_and_line trollers}"

GEAR_USE_OPERATIONAL_FILTER="${GEAR_USE_OPERATIONAL_FILTER:-0}"
GEAR_OP_SPEED_MIN="${GEAR_OP_SPEED_MIN:-1.0}"
GEAR_OP_SPEED_MAX="${GEAR_OP_SPEED_MAX:-12.0}"
GEAR_USE_LOCATION_FEATURES="${GEAR_USE_LOCATION_FEATURES:-1}"
GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.03}"

COMMON_DATA_OUT="$ROOT_OUT/_data_internal_trainval"
COMMON_EXTERNAL_OUT="$ROOT_OUT/_data_external_test"

mkdir -p "$ROOT_OUT"
read -r -a EXCLUDE_ARGS <<< "$EXCLUDE_LABELS"
echo "[external-multiseed] seeds=$SEEDS device=$DEVICE root=$ROOT_OUT"

run_with_retry() {
  local label="$1"
  shift
  local attempt=1
  local max_attempts=$((COMMAND_RETRIES + 1))
  while true; do
    echo "[external-multiseed] $label attempt=$attempt/$max_attempts"
    local status=0
    if "$@"; then
      return 0
    else
      status=$?
    fi
    if (( attempt >= max_attempts )); then
      echo "[external-multiseed] ERROR: $label failed after $attempt attempts (exit=$status)"
      return "$status"
    fi
    echo "[external-multiseed] WARNING: $label failed (exit=$status); retrying in 10 seconds"
    sleep 10
    attempt=$((attempt + 1))
  done
}

training_complete() {
  local model_out="$1"
  python3 - "$model_out" "$EPOCHS" <<'PY'
import json
import sys
from pathlib import Path

model_out = Path(sys.argv[1])
expected_epochs = int(sys.argv[2])
required = [
    "model.pt",
    "history.json",
    "best_epoch.json",
    "train_config.json",
    "training_curves.png",
    "split_indices.npz",
    "scaler.joblib",
]
if not all((model_out / name).is_file() for name in required):
    raise SystemExit(1)
try:
    history = json.loads((model_out / "history.json").read_text(encoding="utf-8"))
    config = json.loads((model_out / "train_config.json").read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
last_epoch = int(history[-1].get("epoch", 0)) if history else 0
configured_epochs = int(config.get("epochs", 0))
raise SystemExit(0 if last_epoch >= expected_epochs and configured_epochs == expected_epochs else 1)
PY
}

eval_complete() {
  local eval_out="$1"
  [[ -f "$eval_out/eval_summary.json" \
     && -f "$eval_out/confusion_matrix.png" \
     && -f "$eval_out/confusion_matrix_normalized.png" ]]
}

OP_FILTER_ARGS=()
if [[ "$GEAR_USE_OPERATIONAL_FILTER" == "1" ]]; then
  OP_FILTER_ARGS=(
    --use_operational_filter
    --op_speed_min "$GEAR_OP_SPEED_MIN"
    --op_speed_max "$GEAR_OP_SPEED_MAX"
  )
fi

LOCATION_ARGS=()
if [[ "$GEAR_USE_LOCATION_FEATURES" == "0" ]]; then
  LOCATION_ARGS=(--exclude_location_features)
fi

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$COMMON_DATA_OUT/processed_gear.npz" ]]; then
  python3 main.py preprocess \
    --data_dir "$DATA_DIR" \
    --out_dir "$COMMON_DATA_OUT" \
    --task gear \
    --exclude_labels "${EXCLUDE_ARGS[@]}" \
	    --seq_len "$SEQ_LEN" \
	    --stride "$STRIDE" \
	    --gap_seconds "$GAP_SECONDS" \
	    --min_points_per_vessel "$MIN_POINTS_PER_VESSEL" \
	    --min_windows_per_vessel "$MIN_WINDOWS_PER_VESSEL" \
	    --max_windows_per_vessel "$MAX_WINDOWS_PER_VESSEL" \
	    --max_windows_per_file "$MAX_WINDOWS_PER_FILE" \
	    "${OP_FILTER_ARGS[@]}" \
	    "${LOCATION_ARGS[@]}"
else
  echo "[external-multiseed] reuse internal NPZ: $COMMON_DATA_OUT/processed_gear.npz"
fi

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$COMMON_EXTERNAL_OUT/processed_gear.npz" ]]; then
  python3 main.py preprocess \
    --data_dir "$EXTERNAL_TEST_DIR" \
    --out_dir "$COMMON_EXTERNAL_OUT" \
    --task gear \
    --exclude_labels "${EXCLUDE_ARGS[@]}" \
	    --seq_len "$SEQ_LEN" \
	    --stride "$STRIDE" \
	    --gap_seconds "$GAP_SECONDS" \
	    --min_points_per_vessel "$MIN_POINTS_PER_VESSEL" \
	    --min_windows_per_vessel "$MIN_WINDOWS_PER_VESSEL" \
	    --max_windows_per_vessel "$MAX_WINDOWS_PER_VESSEL" \
	    --max_windows_per_file "$MAX_WINDOWS_PER_FILE" \
	    --no_jump_filter \
    "${OP_FILTER_ARGS[@]}" \
    "${LOCATION_ARGS[@]}"
else
  echo "[external-multiseed] reuse external NPZ: $COMMON_EXTERNAL_OUT/processed_gear.npz"
fi

for SEED in $SEEDS; do
  RUN_OUT="$ROOT_OUT/seed_${SEED}"
  MODEL_OUT="$RUN_OUT/model_gear"
  VAL_EVAL_OUT="$RUN_OUT/validation_eval"
  EXTERNAL_EVAL_OUT="$RUN_OUT/external_test_eval"

  echo "[external-multiseed] seed=$SEED model_out=$MODEL_OUT"
  if [[ "$RESUME_COMPLETED_SEEDS" == "1" ]] && training_complete "$MODEL_OUT"; then
    echo "[external-multiseed] seed=$SEED training already complete; skip train"
  else
    train_status=0
    python3 main.py train \
      --data_npz "$COMMON_DATA_OUT/processed_gear.npz" \
      --out_dir "$MODEL_OUT" \
      --device "$DEVICE" \
      --random_state "$SEED" \
      --split_random_state "$SEED" \
      --train_random_state "$SEED" \
      --test_size 0 \
      --val_size 0.20 \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --hidden_size 384 \
      --num_layers 2 \
      --input_proj_dim 256 \
      --embed_dim 512 \
      --dropout 0.30 \
      --optimizer adamw \
      --weight_decay 0.0013 \
      --early_stop_patience 90 \
      --geo_aux_weight "$GEO_AUX_WEIGHT" \
      --gear_minority_f1_weight 0.03 \
      --gear_class_weight_power 1.0 \
      --gear_class_weight_max 10.0 \
      --gear_tau_max 0.6 || train_status=$?

    if training_complete "$MODEL_OUT"; then
      echo "[external-multiseed] seed=$SEED training files verified complete"
    else
      if (( train_status == 0 )); then
        train_status=1
      fi
      echo "[external-multiseed] ERROR: seed=$SEED training incomplete (exit=$train_status)"
      exit "$train_status"
    fi
  fi

  if [[ "$RESUME_COMPLETED_SEEDS" == "1" ]] && eval_complete "$VAL_EVAL_OUT"; then
    echo "[external-multiseed] seed=$SEED validation eval already complete; skip"
  else
    run_with_retry "seed=$SEED validation-eval" python3 main.py eval \
      --data_npz "$COMMON_DATA_OUT/processed_gear.npz" \
      --model_path "$MODEL_OUT/model.pt" \
      --out_dir "$VAL_EVAL_OUT" \
      --device "$DEVICE" \
      --batch_size 256 \
      --eval_split val
  fi

  if [[ "$RESUME_COMPLETED_SEEDS" == "1" ]] && eval_complete "$EXTERNAL_EVAL_OUT"; then
    echo "[external-multiseed] seed=$SEED external eval already complete; skip"
  else
    run_with_retry "seed=$SEED external-eval" python3 main.py eval \
      --data_npz "$COMMON_EXTERNAL_OUT/processed_gear.npz" \
      --model_path "$MODEL_OUT/model.pt" \
      --out_dir "$EXTERNAL_EVAL_OUT" \
      --device "$DEVICE" \
      --batch_size 256 \
      --eval_split all
  fi
  echo "[external-multiseed] seed=$SEED COMPLETE"
done

echo "[external-multiseed] ALL SEEDS COMPLETE: $SEEDS"
