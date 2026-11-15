#!/usr/bin/env bash
# Download the public V1 data (about 4.2 GB). Resumable: rerun if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/vocab
HF=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main
curl -L -C - -o data/raw/meta_Beauty_and_Personal_Care.jsonl "$HF/raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl"
curl -L -C - -o data/raw/Beauty_and_Personal_Care.csv "$HF/benchmark/0core/rating_only/Beauty_and_Personal_Care.csv"
curl -L -o data/vocab/parfumo_data_clean.csv \
  https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-12-10/parfumo_data_clean.csv
echo "done"
