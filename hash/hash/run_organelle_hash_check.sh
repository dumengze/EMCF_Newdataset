#!/bin/bash
set -euo pipefail

python3 train_test_hash_leakage.py \
  --train "/Users/jason/Desktop/OrganelleClassifyDataset/train" \
  --test "/Users/jason/Desktop/OrganelleClassifyDataset/test" \
  --output "/Users/jason/Desktop/OrganelleClassifyDataset/hash_leakage_results" \
  --method phash \
  --threshold 0 \
  --top-n 200

python3 plot_hash_results.py \
  --input "/Users/jason/Desktop/OrganelleClassifyDataset/hash_leakage_results"
