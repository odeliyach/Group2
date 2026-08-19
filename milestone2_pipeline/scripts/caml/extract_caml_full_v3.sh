#!/bin/bash
#SBATCH --job-name=extract-broad-v3
#SBATCH --output=logs/broad_v3_%j.out
#SBATCH --error=logs/broad_v3_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
# Submit from the workshop ROOT directory:  sbatch scripts/caml/extract_caml_full_v3.sh

mkdir -p logs combined
echo "Start: $(date)"

/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python \
    scripts/caml/extract_caml_full_v3.py \
    --benign_ratio 2.5 \
    --min_events 2 \
    --out combined/raw_labeled_logs_v3.csv

echo "Exit: $?  Finished: $(date)"
ls -lh combined/raw_labeled_logs_v3.csv
