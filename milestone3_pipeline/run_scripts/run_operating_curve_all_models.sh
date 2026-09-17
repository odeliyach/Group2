#!/bin/bash
#SBATCH --job-name=milestone3-opcurve-all
#SBATCH --output=logs/m3_opcurve_all_%A_%a.out
#SBATCH --error=logs/m3_opcurve_all_%A_%a.err
#SBATCH --array=0,1,2,5,6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=23:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- MAX_FPR validation follow-up: config.MAX_FPR=0.05 was
# derived from an RF-only sweep (run_operating_curve.sh) and applied
# pipeline-wide without checking XGBoost/MLP/CNN/Isolation Forest's own
# F1-vs-FPR curves. This runs operating_curve.py (already supports all 5
# models via --model) for the 4 NOT already covered, across both datasets.
# RF is intentionally excluded here -- already run, already in the report.
#
# One array task per (dataset, model) pair (4 models x 2 datasets = 8).
# CNN/MLP get the same --repeats=5 exploratory count as the existing RF
# sweep, not the 15-repeat headline protocol -- this is a directional
# check of whether 0.05 holds per-model, not a replacement for headline CV.
#
# After all 8 tasks finish, run summarize_operating_curves.py to collect
# the per-model best-cap comparison into one table for the report.
#
# Submit from the milestone3_pipeline directory: sbatch run_scripts/run_operating_curve_all_models.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/"
DATA_DIR="data"
OUT_DIR="milestone3_pipeline/results/current/operating_curve_results"
REPEATS=5

mkdir -p logs "$OUT_DIR"

DATASETS=(camlds casino)
MODELS=(xgb mlp cnn iforest)

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
DATASET=${DATASETS[$DATASET_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "============================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID   Task: $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Dataset      : $DATASET"
echo "Model        : $MODEL"
echo "Repeats      : $REPEATS"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/operating_curve.py" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --repeats "$REPEATS" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
