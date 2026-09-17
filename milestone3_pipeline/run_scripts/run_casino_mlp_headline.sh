#!/bin/bash
#SBATCH --job-name=m3-casino-mlp
#SBATCH --output=../logs/m3_casino_mlp_%j.out
#SBATCH --error=../logs/m3_casino_mlp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"

$PYTHON src/run_pipeline.py --dataset casino --model mlp --data-dir ../data \
    --out results/current/outputs_headline_original/
