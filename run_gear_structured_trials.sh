#!/usr/bin/env bash
set -euo pipefail

ROOT_OUT="${ROOT_OUT:-output/gear_structured_trials}"
DATA_DIR="${DATA_DIR:-Dataset}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44 45 46}"
EPOCHS_LIST="${EPOCHS_LIST:-50}"
GEO_AUX_WEIGHTS="${GEO_AUX_WEIGHTS:-0.03}"
GEAR_MINORITY_WEIGHTS="${GEAR_MINORITY_WEIGHTS:-0.03}"
GEAR_CLASS_WEIGHT_POWERS="${GEAR_CLASS_WEIGHT_POWERS:-1.0}"
GEAR_CLASS_WEIGHT_MAX="${GEAR_CLASS_WEIGHT_MAX:-10.0}"
GEAR_TAU_MAX="${GEAR_TAU_MAX:-0.6}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
NON_DETERMINISTIC="${NON_DETERMINISTIC:-0}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"

deterministic_args=()
if [[ "$NON_DETERMINISTIC" == "1" ]]; then
  deterministic_args+=(--non_deterministic)
fi
if [[ "$STRICT_DETERMINISTIC" == "1" ]]; then
  deterministic_args+=(--strict_deterministic)
fi

DATA_OUT="$ROOT_OUT/data"
NPZ="$DATA_OUT/processed_gear.npz"
mkdir -p "$DATA_OUT"

if [[ "$FORCE_PREPROCESS" == "1" || ! -f "$NPZ" ]]; then
  echo "[gear-trials] preprocess -> $NPZ"
  python3 main.py preprocess \
    --data_dir "$DATA_DIR" \
    --out_dir "$DATA_OUT" \
    --task gear \
    --exclude_labels unknown \
    --max_windows_per_file 20000 \
    --min_windows_per_vessel 0
else
  echo "[gear-trials] reuse preprocess -> $NPZ"
fi

run_one() {
  local seed="$1"
  local epochs="$2"
  local geo="$3"
  local minority="$4"
  local weight_power="$5"
  local run_name="seed${seed}_ep${epochs}_geo${geo}_mw${minority}_wp${weight_power}_tau${GEAR_TAU_MAX}"
  run_name="${run_name//./p}"
  local model_dir="$ROOT_OUT/runs/$run_name/model_gear"
  local model_path="$model_dir/model.pt"
  mkdir -p "$model_dir"

  echo "[gear-trials] train $run_name"
  PYTHONHASHSEED="$seed" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" python3 main.py train \
    --data_npz "$NPZ" \
    --out_dir "$model_dir" \
    --device "$DEVICE" \
    --random_state "$seed" \
    --test_size 0.20 \
    --val_size 0.20 \
    --epochs "$epochs" \
    --batch_size 128 \
    --hidden_size 384 \
    --num_layers 2 \
    --input_proj_dim 256 \
    --embed_dim 512 \
    --dropout 0.30 \
    --optimizer adamw \
    --weight_decay 0.0013 \
    --early_stop_patience 90 \
    "${deterministic_args[@]}" \
    --geo_aux_weight "$geo" \
    --gear_minority_f1_weight "$minority" \
    --gear_class_weight_power "$weight_power" \
    --gear_class_weight_max "$GEAR_CLASS_WEIGHT_MAX" \
    --gear_tau_max "$GEAR_TAU_MAX"

  if [[ ! -f "$model_path" ]]; then
    echo "[gear-trials] ERROR: model missing for $run_name" >&2
    return 1
  fi

  echo "[gear-trials] eval $run_name"
  python3 main.py eval \
    --data_npz "$NPZ" \
    --model_path "$model_path" \
    --out_dir "$model_dir" \
    --device "$DEVICE" \
    --random_state "$seed"
}

for seed in $SEEDS; do
  for epochs in $EPOCHS_LIST; do
    for geo in $GEO_AUX_WEIGHTS; do
      for minority in $GEAR_MINORITY_WEIGHTS; do
        for weight_power in $GEAR_CLASS_WEIGHT_POWERS; do
          run_one "$seed" "$epochs" "$geo" "$minority" "$weight_power"
        done
      done
    done
  done
done

python3 summarize_gear_trials.py "$ROOT_OUT/runs" "$ROOT_OUT/summary.csv"
echo "[gear-trials] summary -> $ROOT_OUT/summary.csv"
