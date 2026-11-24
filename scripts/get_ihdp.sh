#!/usr/bin/env bash
# Fetch the IHDP covariates/treatment (CEVAE's copy of the NPCI files) into data/IHDP/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
git clone -q --depth 1 https://github.com/AMLab-Amsterdam/CEVAE "$TMP/CEVAE"
mkdir -p "$ROOT/data/IHDP"
cp -r "$TMP/CEVAE/datasets/IHDP/csv" "$ROOT/data/IHDP/"
rm -rf "$TMP"
echo "IHDP csv files in $ROOT/data/IHDP/csv"
