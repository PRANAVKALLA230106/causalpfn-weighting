#!/usr/bin/env bash
# Build cfm-bundle.zip for Kaggle: this repo's code + CausalPFN source + checkpoint.
# Usage: bash scripts/build_kaggle_bundle.sh /path/to/causalpfn_v0.pt [/path/to/CausalPFN-clone]
# The CausalPFN source and weights are NOT part of this repository (they carry their own
# licence); this script copies them into the bundle at build time.
set -euo pipefail
CKPT=${1:?path to causalpfn_v0.pt}
CPFN=${2:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
if [ -z "$CPFN" ]; then git clone -q --depth 1 https://github.com/vdblm/CausalPFN "$TMP/CausalPFN"; CPFN="$TMP/CausalPFN"; fi
B="$TMP/bundle"; mkdir -p "$B/causalpfn_src"
cp -r "$ROOT/cfm" "$ROOT/checks" "$B/"
cp "$ROOT/scripts/run_kaggle.py" "$ROOT/scripts/analyze_full.py" "$ROOT/docs/KAGGLE.md" "$B/"
# analyze_full.py inside the bundle sits next to cfm/, so point its repo-root lookup there
sed -i 's/parents\[1\]/parents[0]/' "$B/analyze_full.py"
cp -r "$CPFN/src/causalpfn" "$B/causalpfn_src/"
cp "$CPFN/LICENSE" "$B/causalpfn_src/LICENSE_causalpfn"
cp "$CKPT" "$B/causalpfn_v0.pt"
find "$B" -name __pycache__ -prune -exec rm -rf {} +
(cd "$B" && zip -qr "$ROOT/cfm-bundle.zip" .)
rm -rf "$TMP"
echo "wrote $ROOT/cfm-bundle.zip"
