#!/bin/bash
#SBATCH --job-name=m3-rf-verify
#SBATCH --output=logs/m3_rf_verify_%j.out
#SBATCH --error=logs/m3_rf_verify_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
mkdir -p logs milestone3_pipeline/results/current/hyperparameter_verification

$PYTHON milestone3_pipeline/src/verify_hyperparameter_candidate.py \
    --model rf --dataset camlds --data-dir data/v3 \
    --candidate-params '{"max_depth": 8}' \
    --out milestone3_pipeline/results/current/hyperparameter_verification/
