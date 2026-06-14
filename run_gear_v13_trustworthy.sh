#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-output/gear_v16_lowtau}"
DATA_DIR="${DATA_DIR:-Dataset}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
GEAR_MINORITY_F1_WEIGHT="${GEAR_MINORITY_F1_WEIGHT:-0.03}"
GEAR_CLASS_WEIGHT_POWER="${GEAR_CLASS_WEIGHT_POWER:-1.0}"
GEAR_CLASS_WEIGHT_MAX="${GEAR_CLASS_WEIGHT_MAX:-10.0}"
GEAR_TAU_MAX="${GEAR_TAU_MAX:-0.6}"
GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.03}"
NON_DETERMINISTIC="${NON_DETERMINISTIC:-0}"
STRICT_DETERMINISTIC="${STRICT_DETERMINISTIC:-0}"

deterministic_args=()
if [[ "$NON_DETERMINISTIC" == "1" ]]; then
  deterministic_args+=(--non_deterministic)
fi
if [[ "$STRICT_DETERMINISTIC" == "1" ]]; then
  deterministic_args+=(--strict_deterministic)
fi

OUT_GEAR="$OUT_ROOT/gear"
MODEL_DIR="$OUT_GEAR/model_gear"
NPZ="$OUT_GEAR/processed_gear.npz"
MODEL="$MODEL_DIR/model.pt"

mkdir -p "$OUT_GEAR" "$MODEL_DIR"

echo "[gear-stable] preprocess -> $NPZ"
python3 main.py preprocess \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_GEAR" \
  --task gear \
  --exclude_labels unknown \
  --max_windows_per_file 20000 \
  --min_windows_per_vessel 0

echo "[gear-stable] train -> $MODEL"
train_args=(
  main.py train
  --data_npz "$NPZ"
  --out_dir "$MODEL_DIR"
  --device "$DEVICE"
  --random_state "$SEED"
  --test_size 0.20
  --val_size 0.20
  --epochs "$EPOCHS"
  --batch_size 128
  --hidden_size 384
  --num_layers 2
  --input_proj_dim 256
  --embed_dim 512
  --dropout 0.30
  --optimizer adamw
  --weight_decay 0.0013
  --early_stop_patience 90
  "${deterministic_args[@]}"
  --geo_aux_weight "$GEO_AUX_WEIGHT"
  --gear_minority_f1_weight "$GEAR_MINORITY_F1_WEIGHT"
  --gear_class_weight_power "$GEAR_CLASS_WEIGHT_POWER"
  --gear_class_weight_max "$GEAR_CLASS_WEIGHT_MAX"
  --gear_tau_max "$GEAR_TAU_MAX"
)
PYTHONHASHSEED="$SEED" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" python3 "${train_args[@]}"

if [[ ! -f "$MODEL" ]]; then
  echo "[gear-stable] ERROR: model not found after train: $MODEL" >&2
  exit 1
fi

echo "[gear-stable] eval -> $MODEL_DIR/eval_summary.json"
python3 main.py eval \
  --data_npz "$NPZ" \
  --model_path "$MODEL" \
  --out_dir "$MODEL_DIR" \
  --device "$DEVICE" \
  --random_state "$SEED"

echo "[gear-stable] done"
echo "[gear-stable] summary: $MODEL_DIR/eval_summary.json"
echo "[gear-stable] per-class: $MODEL_DIR/per_class_metrics.csv"
echo "[gear-stable] wrong-high-confidence: $MODEL_DIR/wrong_high_confidence_predictions.csv"
