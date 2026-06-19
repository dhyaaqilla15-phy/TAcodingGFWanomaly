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

COMMON_DATA_OUT="$ROOT_OUT/_data_internal_trainval"
COMMON_EXTERNAL_OUT="$ROOT_OUT/_data_external_test"

mkdir -p "$ROOT_OUT"
read -r -a EXCLUDE_ARGS <<< "$EXCLUDE_LABELS"

OP_FILTER_ARGS=()
if [[ "$GEAR_USE_OPERATIONAL_FILTER" == "1" ]]; then
  OP_FILTER_ARGS=(
    --use_operational_filter
    --op_speed_min "$GEAR_OP_SPEED_MIN"
    --op_speed_max "$GEAR_OP_SPEED_MAX"
  )
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
	    "${OP_FILTER_ARGS[@]}"
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
    "${OP_FILTER_ARGS[@]}"
else
  echo "[external-multiseed] reuse external NPZ: $COMMON_EXTERNAL_OUT/processed_gear.npz"
fi

for SEED in $SEEDS; do
  RUN_OUT="$ROOT_OUT/seed_${SEED}"
  MODEL_OUT="$RUN_OUT/model_gear"
  VAL_EVAL_OUT="$RUN_OUT/validation_eval"
  EXTERNAL_EVAL_OUT="$RUN_OUT/external_test_eval"

  echo "[external-multiseed] seed=$SEED model_out=$MODEL_OUT"
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
    --geo_aux_weight 0.03 \
    --gear_minority_f1_weight 0.03 \
    --gear_class_weight_power 1.0 \
    --gear_class_weight_max 10.0 \
    --gear_tau_max 0.6

  python3 main.py eval \
    --data_npz "$COMMON_DATA_OUT/processed_gear.npz" \
    --model_path "$MODEL_OUT/model.pt" \
    --out_dir "$VAL_EVAL_OUT" \
    --device "$DEVICE" \
    --batch_size 256 \
    --eval_split val

  python3 main.py eval \
    --data_npz "$COMMON_EXTERNAL_OUT/processed_gear.npz" \
    --model_path "$MODEL_OUT/model.pt" \
    --out_dir "$EXTERNAL_EVAL_OUT" \
    --device "$DEVICE" \
    --batch_size 256 \
    --eval_split all
done
