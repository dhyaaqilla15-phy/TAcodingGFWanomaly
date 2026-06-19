#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-Dataset}"
EXTERNAL_TEST_DIR="${2:-Dataset_Test_Enriched}"
OUT="${3:-Outputs/gear_trainval_external_test}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"

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

TRAIN_DATA_OUT="$OUT/data_internal_trainval"
MODEL_OUT="$OUT/model_gear"
EXTERNAL_DATA_OUT="$OUT/data_external_test"
VAL_EVAL_OUT="$OUT/validation_eval"
EXTERNAL_EVAL_OUT="$OUT/external_test_eval"

mkdir -p "$OUT"

read -r -a EXCLUDE_ARGS <<< "$EXCLUDE_LABELS"

OP_FILTER_ARGS=()
if [[ "$GEAR_USE_OPERATIONAL_FILTER" == "1" ]]; then
  OP_FILTER_ARGS=(
    --use_operational_filter
    --op_speed_min "$GEAR_OP_SPEED_MIN"
    --op_speed_max "$GEAR_OP_SPEED_MAX"
  )
fi

python3 main.py preprocess \
  --data_dir "$DATA_DIR" \
  --out_dir "$TRAIN_DATA_OUT" \
  --task gear \
  --exclude_labels "${EXCLUDE_ARGS[@]}" \
	  --seq_len "$SEQ_LEN" \
	  --stride "$STRIDE" \
	  --gap_seconds "$GAP_SECONDS" \
	  --min_points_per_vessel "$MIN_POINTS_PER_VESSEL" \
	  --min_windows_per_vessel "$MIN_WINDOWS_PER_VESSEL" \
	  --max_windows_per_vessel "$MAX_WINDOWS_PER_VESSEL" \
	  --max_windows_per_file "$MAX_WINDOWS_PER_FILE" \
	  "${OP_FILTER_ARGS[@]}"

python3 main.py train \
  --data_npz "$TRAIN_DATA_OUT/processed_gear.npz" \
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
  --geo_aux_weight 0.03 \
  --gear_minority_f1_weight 0.03 \
  --gear_class_weight_power 1.0 \
  --gear_class_weight_max 10.0 \
  --gear_tau_max 0.6

python3 main.py eval \
  --data_npz "$TRAIN_DATA_OUT/processed_gear.npz" \
  --model_path "$MODEL_OUT/model.pt" \
  --out_dir "$VAL_EVAL_OUT" \
  --device "$DEVICE" \
  --batch_size 256 \
  --eval_split val

python3 main.py preprocess \
  --data_dir "$EXTERNAL_TEST_DIR" \
  --out_dir "$EXTERNAL_DATA_OUT" \
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
  "${OP_FILTER_ARGS[@]}"

python3 main.py eval \
  --data_npz "$EXTERNAL_DATA_OUT/processed_gear.npz" \
  --model_path "$MODEL_OUT/model.pt" \
  --out_dir "$EXTERNAL_EVAL_OUT" \
  --device "$DEVICE" \
  --batch_size 256 \
  --eval_split all
