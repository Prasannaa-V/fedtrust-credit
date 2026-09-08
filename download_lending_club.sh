#!/usr/bin/env bash
# download_lending_club.sh — Download Lending Club dataset via Kaggle API
# FedTrust-Credit | Day 1 setup
#
# Prerequisites:
#   1. Your kaggle.json API token placed at ~/.kaggle/kaggle.json (chmod 600)
#      OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables.
#   2. The venv python with kaggle package: .venv/bin/pip install kaggle
#
# Usage:
#   bash download_lending_club.sh
#
# Output:
#   data/lending_club/loan.csv  (the main dataset file, ~1.5 GB unzipped)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/lending_club"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_KAGGLE="$PROJECT_ROOT/.venv/bin/kaggle"

mkdir -p "$DATA_DIR"

echo "=== FedTrust-Credit: Lending Club Dataset Download ==="
echo "Target directory: $DATA_DIR"
echo ""

# Check kaggle credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        echo "ERROR: Kaggle credentials not found."
        echo ""
        echo "Option 1: Place your kaggle.json at ~/.kaggle/kaggle.json"
        echo "  1. Go to https://www.kaggle.com/settings/account"
        echo "  2. Under 'API', click 'Create New API Token' — this downloads kaggle.json"
        echo "  3. Run:  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "Option 2: Export environment variables:"
        echo "  export KAGGLE_USERNAME=your_username"
        echo "  export KAGGLE_KEY=your_api_key"
        echo "  Then re-run this script."
        exit 1
    fi
fi

# Download using kaggle CLI in venv
echo "Downloading wordsforthewise/lending-club dataset..."
"$VENV_KAGGLE" datasets download \
    --dataset wordsforthewise/lending-club \
    --path "$DATA_DIR" \
    --unzip

echo ""
echo "Download complete. Contents of $DATA_DIR:"
ls -lh "$DATA_DIR/"

# Identify the main CSV file
LOAN_CSV=$(find "$DATA_DIR" -name "*.csv" | sort | head -1)
if [ -z "$LOAN_CSV" ]; then
    echo "ERROR: No CSV found in $DATA_DIR after download."
    exit 1
fi

echo ""
echo "Main CSV file: $LOAN_CSV"
echo "Size: $(du -h "$LOAN_CSV" | cut -f1)"
echo ""
echo "=== Next step ==="
echo "Run the Day 1 partition script:"
echo "  .venv/bin/python src/data_partition.py \"$LOAN_CSV\""
