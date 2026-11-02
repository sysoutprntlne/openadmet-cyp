#!/bin/bash -l
# Run head-search with its current default config (greedy, protein granularity,
# 3 seeds, 30 epochs, data/union_train.csv). Assumes union_train.csv already
# has a pinned 'split' column (add-split-column) -- head-search refuses to run
# without one.
#
# Usage: scripts/run_head_search.sh [extra head-search args...]
# Anything passed on the command line is appended, e.g.:
#   scripts/run_head_search.sh --strategy single --seeds 5

set -euo pipefail

DATA_PATH="${DATA_PATH:-data/union_train.csv}"

if [ ! -f "$DATA_PATH" ]; then
    echo "Missing $DATA_PATH. It needs a union table with a pinned split column." >&2
    exit 1
fi

head-search \
    --data-path "$DATA_PATH" \
    --strategy greedy \
    --granularity protein \
    --seeds 3 \
    --epochs 30 \
    --results-path results/head_search.json \
    "$@"
