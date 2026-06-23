#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat <<'EOF'
[godark] run_godark_only.sh sekarang memakai protokol permanen:
  internal Dataset = train + validation only
  Dataset_Test_Enriched = pure external test
EOF

exec bash "$ROOT_DIR/run_godark_external_test_pipeline.sh" "${1:-all}"
