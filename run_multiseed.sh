#!/usr/bin/env bash
set -e

CODE="$HOME/newtrain_GFW_suspectedship"
DATA="$CODE"
BASE_RUN="$CODE/outputs/runs/run_multiseed"
NPZ="$CODE/outputs/runs/run15/processed_gear.npz"   # <-- ganti kalau NPZ kamu ada di run lain

mkdir -p "$BASE_RUN"

# seeds yang mau dicoba (bisa nambah)
SEEDS=(41 42 43 44 45 46 47 48)

best_seed=""
best_f1="0"

for s in "${SEEDS[@]}"; do
  R="$BASE_RUN/seed_$s"
  mkdir -p "$R"

  echo "=============================="
  echo "[SEED $s] TRAIN -> $R"
  python3 "$CODE/main.py" train --data_npz "$NPZ" --out_dir "$R" --device cuda --random_state "$s"

  echo "[SEED $s] EVAL -> $R"
  python3 "$CODE/main.py" eval  --data_npz "$NPZ" --model_path "$R/model.pt" --out_dir "$R" --device cuda --random_state "$s"

  # ambil macro_f1 vessel dari eval_summary.json
  f1=$(python3 - <<PY
import json
p="$R/eval_summary.json"
d=json.load(open(p,"r",encoding="utf-8"))
print(d["metrics_vessel"]["macro_f1"])
PY
)

  echo "[SEED $s] vessel_macro_f1 = $f1"

  # compare float
  better=$(python3 - <<PY
a=float("$f1"); b=float("$best_f1")
print(1 if a>b else 0)
PY
)
  if [ "$better" = "1" ]; then
    best_f1="$f1"
    best_seed="$s"
  fi
done

echo "=============================="
echo "BEST SEED = $best_seed   BEST vessel_macro_f1 = $best_f1"
echo "Best run folder: $BASE_RUN/seed_$best_seed"
