#!/usr/bin/env bash
# Week 3 (notes v2): with notes vs without notes on every metric.
# 1. validation sweeps (main notes per perfume, then note weights at the best k)
# 2. the 3 frozen configurations x 3 seeds + the 3 cold-start models (same config, same seed)
# 3. the ablation table on validation, then ONCE on test.
#
# Usage:  scripts/notes_ablation.sh [sweeps|train|report]   (default: all three)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
STEP="${1:-all}"

if [[ $STEP == all || $STEP == sweeps ]]; then
  python -m scentmatch.train.sweep mainnotes                    # k = 3, 5, 8 and no notes (seed 3)
  python -m scentmatch.train.sweep mainnotes2 --jobs reports/sweeps/jobs/mainnotes2.json
fi
if [[ $STEP == all || $STEP == train ]]; then
  cp checkpoints/mainnotes2_k3_w0.3_ncl0.pt checkpoints/final_nlgcl_mainnotes_s3.pt
  cp checkpoints/mainnotes_none_nonotes.pt checkpoints/final_nlgcl_rescued_s3.pt
  python -m scentmatch.train.sweep mainnotes_final --jobs reports/sweeps/jobs/mainnotes_final.json
fi
if [[ $STEP == all || $STEP == report ]]; then
  seeds() { echo "$1_s1.pt,$1_s2.pt,$1_s3.pt"; }
  for split in val test; do
    python -m scentmatch.eval.innovation --split "$split" \
      --warm "No notes=$(seeds final_nlgcl_rescued)" \
      --warm "All notes (Week 3 served)=$(seeds final_nlgcl_notes_rescued)" \
      --warm "Main notes (k=3, notes v2)=$(seeds final_nlgcl_mainnotes)" \
      --cold "No notes=cold_nlgcl_rescued.pt" \
      --cold "All notes (Week 3 served)=cold_nlgcl_notes.pt" \
      --cold "Main notes (k=3, notes v2)=cold_nlgcl_mainnotes.pt"
  done
fi
