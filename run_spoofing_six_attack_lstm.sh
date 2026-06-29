#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# New protocol only. The locked two-attack baseline remains untouched.
export SPOOFING_OUTPUT_ROOT="${SPOOFING_OUTPUT_ROOT:-$ROOT_DIR/Outputs/spoofing_hybrid02_context_oof_four_gear}"
export SPOOFING_ATTACKS="gradual_drift location_jump replay meaconing ghost mirroring"
export SPOOFING_INCLUDE_LABELS="drifting_longlines fixed_gear purse_seines trawlers"
export SPOOFING_STRICT_FOUR_GEAR=1
export SPOOFING_SEEDS="${SPOOFING_SEEDS:-42,43,44}"
export SPOOFING_EPOCHS="${SPOOFING_EPOCHS:-50}"
export SPOOFING_DISABLE_EARLY_STOPPING=1

if [[ $# -ne 1 ]]; then
  echo "Usage:"
  echo "  bash run_spoofing_six_attack_lstm.sh prepare"
  echo "  DEVICE=cuda bash run_spoofing_six_attack_lstm.sh run"
  echo "  bash run_spoofing_six_attack_lstm.sh status"
  exit 1
fi

exec python "$ROOT_DIR/run_spoofing_multiseed.py" "$1"
