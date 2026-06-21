#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$ROOT_DIR/Dataset}"
RUN_DIR="${2:-$ROOT_DIR/output/run01}"

DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
LIMIT_ROWS="${LIMIT_ROWS:-300000}"
GEAR_LIMIT_ROWS="${GEAR_LIMIT_ROWS:-0}"
SOURCE_LIMIT_ROWS="${SOURCE_LIMIT_ROWS:-$LIMIT_ROWS}"
BATCH_SIZE="${BATCH_SIZE:-128}"
VAL_SIZE="${VAL_SIZE:-0.15}"
GEAR_VAL_SIZE="${GEAR_VAL_SIZE:-0.20}"
GEAR_MIN_WINDOWS_PER_VESSEL="${GEAR_MIN_WINDOWS_PER_VESSEL:-0}"
GEAR_EXCLUDE_LABELS="${GEAR_EXCLUDE_LABELS:-unknown pole_and_line trollers}"
SOURCE_EXCLUDE_LABELS="${SOURCE_EXCLUDE_LABELS:-pole_and_line trollers}"
GEAR_USE_OPERATIONAL_FILTER="${GEAR_USE_OPERATIONAL_FILTER:-0}"
GEAR_OP_SPEED_MIN="${GEAR_OP_SPEED_MIN:-1.0}"
GEAR_OP_SPEED_MAX="${GEAR_OP_SPEED_MAX:-12.0}"
PREPROCESS_MAX_WINDOWS_PER_FILE="${PREPROCESS_MAX_WINDOWS_PER_FILE:-20000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
TRAIN_HIDDEN_SIZE="${TRAIN_HIDDEN_SIZE:-384}"
TRAIN_NUM_LAYERS="${TRAIN_NUM_LAYERS:-2}"
TRAIN_INPUT_PROJ_DIM="${TRAIN_INPUT_PROJ_DIM:-256}"
TRAIN_EMBED_DIM="${TRAIN_EMBED_DIM:-512}"
TRAIN_ATTENTION_HEADS="${TRAIN_ATTENTION_HEADS:-4}"
TRAIN_ATTENTION_LAYERS="${TRAIN_ATTENTION_LAYERS:-1}"
TRAIN_EARLY_STOP_PATIENCE="${TRAIN_EARLY_STOP_PATIENCE:-90}"
TRAIN_GEO_AUX_WEIGHT="${TRAIN_GEO_AUX_WEIGHT:-0.03}"
TRANS_TARGET="${TRANS_TARGET:-any}"
TRANS_FEATURE_MODE="${TRANS_FEATURE_MODE:-fair}"
TRANS_GEO_AUX_WEIGHT="${TRANS_GEO_AUX_WEIGHT:-0}"
TRANS_SYNTHETIC_ENCOUNTERS_PER_FILE="${TRANS_SYNTHETIC_ENCOUNTERS_PER_FILE:-250}"

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
  GEAR_LIMIT_ROWS=0
  SOURCE_LIMIT_ROWS=\$LIMIT_ROWS
  BATCH_SIZE=128
  VAL_SIZE=0.15
  GEAR_VAL_SIZE=0.20
  GEAR_MIN_WINDOWS_PER_VESSEL=0
  GEAR_EXCLUDE_LABELS="unknown pole_and_line trollers"
  SOURCE_EXCLUDE_LABELS="pole_and_line trollers"
  GEAR_USE_OPERATIONAL_FILTER=0
  GEAR_OP_SPEED_MIN=1.0
  GEAR_OP_SPEED_MAX=12.0
  PREPROCESS_MAX_WINDOWS_PER_FILE=20000
  TRAIN_EPOCHS=50
  TRAIN_HIDDEN_SIZE=384
  TRAIN_NUM_LAYERS=2
  TRAIN_INPUT_PROJ_DIM=256
  TRAIN_EMBED_DIM=512
  TRAIN_ATTENTION_HEADS=4
  TRAIN_ATTENTION_LAYERS=1
  TRAIN_EARLY_STOP_PATIENCE=90
  TRAIN_GEO_AUX_WEIGHT=0.03
  TRANS_TARGET=any|encounter|loitering|multiclass|auto
  TRANS_FEATURE_MODE=fair|full
  TRANS_GEO_AUX_WEIGHT=0
  TRANS_SYNTHETIC_ENCOUNTERS_PER_FILE=250

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
OUT_TRANS="$RUN_DIR/transshipment"

mkdir -p "$OUT_GEAR" "$OUT_SPOOF" "$OUT_GODARK" "$OUT_TRANS"

read -r -a gear_exclude_labels <<< "$GEAR_EXCLUDE_LABELS"
read -r -a source_exclude_labels <<< "$SOURCE_EXCLUDE_LABELS"
gear_operational_filter_args=()
if [[ "$GEAR_USE_OPERATIONAL_FILTER" == "1" ]]; then
  gear_operational_filter_args+=(
    --use_operational_filter
    --op_speed_min "$GEAR_OP_SPEED_MIN"
    --op_speed_max "$GEAR_OP_SPEED_MAX"
  )
fi

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
echo "[pipeline] legacy_limit_rows = $LIMIT_ROWS"
echo "[pipeline] gear_limit_rows = $GEAR_LIMIT_ROWS"
echo "[pipeline] source_limit_rows = $SOURCE_LIMIT_ROWS"
echo "[pipeline] val_size = $VAL_SIZE"
echo "[pipeline] gear_val_size = $GEAR_VAL_SIZE"
echo "[pipeline] gear_min_windows_per_vessel = $GEAR_MIN_WINDOWS_PER_VESSEL"
echo "[pipeline] gear_exclude_labels = $GEAR_EXCLUDE_LABELS"
echo "[pipeline] gear_use_operational_filter = $GEAR_USE_OPERATIONAL_FILTER"
echo "[pipeline] gear_op_speed = $GEAR_OP_SPEED_MIN..$GEAR_OP_SPEED_MAX"
echo "[pipeline] source_exclude_labels = $SOURCE_EXCLUDE_LABELS"
echo "[pipeline] preprocess_max_windows_per_file = $PREPROCESS_MAX_WINDOWS_PER_FILE"
echo "[pipeline] train_epochs = $TRAIN_EPOCHS"
echo "[pipeline] train_hidden_size = $TRAIN_HIDDEN_SIZE"
echo "[pipeline] train_num_layers = $TRAIN_NUM_LAYERS"
echo "[pipeline] train_attention_heads = $TRAIN_ATTENTION_HEADS"
echo "[pipeline] train_attention_layers = $TRAIN_ATTENTION_LAYERS"
echo "[pipeline] train_geo_aux_weight = $TRAIN_GEO_AUX_WEIGHT"
echo "[pipeline] trans_target = $TRANS_TARGET"
echo "[pipeline] trans_feature_mode = $TRANS_FEATURE_MODE"
echo "[pipeline] trans_geo_aux_weight = $TRANS_GEO_AUX_WEIGHT"
echo "[pipeline] trans_synthetic_encounters_per_file = $TRANS_SYNTHETIC_ENCOUNTERS_PER_FILE"

run_step "[1/20] Gear preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_GEAR" \
  --task gear \
  --limit_rows "$GEAR_LIMIT_ROWS" \
  --max_windows_per_file "$PREPROCESS_MAX_WINDOWS_PER_FILE" \
  --exclude_labels "${gear_exclude_labels[@]}" \
  --min_windows_per_vessel "$GEAR_MIN_WINDOWS_PER_VESSEL" \
  "${gear_operational_filter_args[@]}"

run_step "[2/20] Gear train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_GEAR/processed_gear.npz" \
  --out_dir "$OUT_GEAR/model_gear" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$GEAR_VAL_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --hidden_size "$TRAIN_HIDDEN_SIZE" \
  --num_layers "$TRAIN_NUM_LAYERS" \
  --input_proj_dim "$TRAIN_INPUT_PROJ_DIM" \
  --embed_dim "$TRAIN_EMBED_DIM" \
  --attention_heads "$TRAIN_ATTENTION_HEADS" \
  --attention_layers "$TRAIN_ATTENTION_LAYERS" \
  --early_stop_patience "$TRAIN_EARLY_STOP_PATIENCE" \
  --geo_aux_weight "$TRAIN_GEO_AUX_WEIGHT"

run_step "[3/20] Gear eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_GEAR/processed_gear.npz" \
  --model_path "$OUT_GEAR/model_gear/model.pt" \
  --out_dir "$OUT_GEAR/model_gear" \
  --device "$DEVICE" \
  --random_state "$SEED"

run_step "[4/20] Spoofing generate" \
  python3 "$MAIN_PY" make_spoofing \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_SPOOF" \
  --attacks gradual_drift location_jump \
  --limit_rows "$SOURCE_LIMIT_ROWS" \
  --exclude_labels "${source_exclude_labels[@]}" \
  --normal_keep_frac 0.25 \
  --max_vessels_per_file 20 \
  --points_per_attack 120 \
  --combine_outputs \
  --seed "$SEED"

run_step "[5/20] Spoofing preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_SPOOF" \
  --out_dir "$OUT_SPOOF" \
  --task spoofing \
  --max_windows_per_file "$PREPROCESS_MAX_WINDOWS_PER_FILE"

run_step "[6/20] Spoofing plot preprocessed trajectory" \
  python3 "$MAIN_PY" plot_preprocessed \
  --npz_path "$OUT_SPOOF/processed_spoofing.npz" \
  --out_dir "$OUT_SPOOF/plots/preprocessed" \
  --task spoofing \
  --max_windows 12

run_step "[7/20] Spoofing plot identifiable attack examples" \
  python3 "$MAIN_PY" plot_spoofing_examples \
  --csv_path "$OUT_SPOOF/spoofed_all.csv" \
  --out_dir "$OUT_SPOOF/plots/attacks" \
  --attacks gradual_drift location_jump

run_step "[8/20] Spoofing train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_SPOOF/processed_spoofing.npz" \
  --out_dir "$OUT_SPOOF/model_spoofing" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$VAL_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --hidden_size "$TRAIN_HIDDEN_SIZE" \
  --num_layers "$TRAIN_NUM_LAYERS" \
  --input_proj_dim "$TRAIN_INPUT_PROJ_DIM" \
  --embed_dim "$TRAIN_EMBED_DIM" \
  --attention_heads "$TRAIN_ATTENTION_HEADS" \
  --attention_layers "$TRAIN_ATTENTION_LAYERS" \
  --early_stop_patience "$TRAIN_EARLY_STOP_PATIENCE" \
  --geo_aux_weight 0

run_step "[9/20] Spoofing eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_SPOOF/processed_spoofing.npz" \
  --model_path "$OUT_SPOOF/model_spoofing/model.pt" \
  --out_dir "$OUT_SPOOF/model_spoofing" \
  --device "$DEVICE" \
  --random_state "$SEED"

run_step "[10/20] Go-dark generate" \
  python3 "$MAIN_PY" make_godark \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_GODARK" \
  --limit_rows "$SOURCE_LIMIT_ROWS" \
  --exclude_labels "${source_exclude_labels[@]}" \
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

run_step "[11/20] Go-dark preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_GODARK" \
  --out_dir "$OUT_GODARK" \
  --task godark \
  --seq_len 120 \
  --stride 6 \
  --gap_seconds 86400 \
  --max_implied_knots 1000 \
  --min_points_per_vessel 80 \
  --spoofing_window_threshold 0.05 \
  --max_windows_per_file "$PREPROCESS_MAX_WINDOWS_PER_FILE"

run_step "[12/20] Go-dark plot preprocessed trajectory" \
  python3 "$MAIN_PY" plot_preprocessed \
  --npz_path "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/plots/preprocessed" \
  --task godark \
  --max_windows 12

run_step "[13/20] Go-dark plot event examples (6 events)" \
  python3 "$MAIN_PY" plot_go_dark_examples \
  --csv_path "$OUT_GODARK/godark_all.csv" \
  --out_dir "$OUT_GODARK/plots/events" \
  --num_examples 6

run_step "[14/20] Go-dark train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --out_dir "$OUT_GODARK/model_godark" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$VAL_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --hidden_size "$TRAIN_HIDDEN_SIZE" \
  --num_layers "$TRAIN_NUM_LAYERS" \
  --input_proj_dim "$TRAIN_INPUT_PROJ_DIM" \
  --embed_dim "$TRAIN_EMBED_DIM" \
  --attention_heads "$TRAIN_ATTENTION_HEADS" \
  --attention_layers "$TRAIN_ATTENTION_LAYERS" \
  --early_stop_patience "$TRAIN_EARLY_STOP_PATIENCE" \
  --geo_aux_weight "$TRAIN_GEO_AUX_WEIGHT"

run_step "[15/20] Go-dark eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_GODARK/processed_godark.npz" \
  --model_path "$OUT_GODARK/model_godark/model.pt" \
  --out_dir "$OUT_GODARK/model_godark" \
  --device "$DEVICE" \
  --random_state "$SEED"

run_step "[16/20] Transshipment candidate generate" \
  python3 "$MAIN_PY" make_transshipment \
  --input_path "$DATA_DIR" \
  --out_dir "$OUT_TRANS" \
  --mode both \
  --limit_rows "$SOURCE_LIMIT_ROWS" \
  --exclude_labels "${source_exclude_labels[@]}" \
  --max_vessels_per_file 60 \
  --min_points_per_vessel 40 \
  --grid_minutes 10 \
  --encounter_distance_km 0.5 \
  --encounter_min_hours 2 \
  --encounter_max_speed_knots 2 \
  --loitering_min_hours 8 \
  --loitering_max_speed_knots 2 \
  --loitering_min_shore_nm 20 \
  --synthetic_encounters_per_file "$TRANS_SYNTHETIC_ENCOUNTERS_PER_FILE" \
  --max_normal_events_per_file 500 \
  --combine_outputs \
  --seed "$SEED"

run_step "[17/20] Transshipment preprocess" \
  python3 "$MAIN_PY" preprocess \
  --data_dir "$OUT_TRANS" \
  --out_dir "$OUT_TRANS" \
  --task transshipment \
  --transshipment_target "$TRANS_TARGET" \
  --transshipment_feature_mode "$TRANS_FEATURE_MODE" \
  --seq_len 24 \
  --stride 3 \
  --min_points_per_vessel 3 \
  --max_windows_per_vessel 2000 \
  --max_windows_per_file "$PREPROCESS_MAX_WINDOWS_PER_FILE"

run_step "[18/20] Transshipment plot examples" \
  python3 "$MAIN_PY" plot_transshipment_examples \
  --csv_path "$OUT_TRANS/transshipment_all.csv" \
  --out_dir "$OUT_TRANS/plots/events" \
  --num_examples 6

run_step "[19/20] Transshipment train" \
  python3 "$MAIN_PY" train \
  --data_npz "$OUT_TRANS/processed_transshipment.npz" \
  --out_dir "$OUT_TRANS/model_transshipment" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --random_state "$SEED" \
  --val_size "$VAL_SIZE" \
  --epochs "$TRAIN_EPOCHS" \
  --hidden_size "$TRAIN_HIDDEN_SIZE" \
  --num_layers "$TRAIN_NUM_LAYERS" \
  --input_proj_dim "$TRAIN_INPUT_PROJ_DIM" \
  --embed_dim "$TRAIN_EMBED_DIM" \
  --attention_heads "$TRAIN_ATTENTION_HEADS" \
  --attention_layers "$TRAIN_ATTENTION_LAYERS" \
  --early_stop_patience "$TRAIN_EARLY_STOP_PATIENCE" \
  --geo_aux_weight "$TRANS_GEO_AUX_WEIGHT"

run_step "[20/20] Transshipment eval" \
  python3 "$MAIN_PY" eval \
  --data_npz "$OUT_TRANS/processed_transshipment.npz" \
  --model_path "$OUT_TRANS/model_transshipment/model.pt" \
  --out_dir "$OUT_TRANS/model_transshipment" \
  --device "$DEVICE" \
  --random_state "$SEED"

echo
echo "============================================================"
echo "[20/20] Pipeline selesai"
echo "Gear output     : $OUT_GEAR"
echo "Spoofing output : $OUT_SPOOF"
echo "Go-dark output  : $OUT_GODARK"
echo "Transshipment output : $OUT_TRANS"
