#!/bin/bash
#SBATCH --job-name=milestone3-casino-reduced
#SBATCH --output=logs/m3_reduced_%A_%a.out
#SBATCH --error=logs/m3_reduced_%A_%a.err
#SBATCH --array=0-3%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
#
# Milestone 3 -- reduced-feature-policy rerun, CasinoLimit ONLY.
#
# Only CasinoLimit is rerun here. NOTE: M2's ablation suggested this
# reduced 1-feature set matched the full set for CasinoLimit -- re-tested
# under M3's leak-free protocol and it collapses (F1 0.919->0.137,
# AUROC->0.600); the M2 result was very likely a duplication artifact,
# not a genuine finding (see feature_selection.py). Kept for the record,
# not used in production. CAM-LDS was never a candidate for reduction
# (M2 found it costs real power there: F1 -0.1875, AUROC -0.2304) and is
# NOT rerun here.
#
# 1 dataset x 4 models = 4 independent array tasks, same CV_CONFIG
# (repeats=15, n_splits=5) as the full-policy headline run.
#
# Submit from the workshop ROOT directory.

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/"

DATA_DIR="data"

# Separate output dir so this run does NOT overwrite the existing
# full-policy results -- both are needed side by side for the
# full-vs-reduced comparison in the report.
OUT_DIR="milestone3_pipeline/results/current/outputs_casino_reduced"

mkdir -p logs "$OUT_DIR"

MODELS=(rf xgb cnn iforest)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=casino

echo "============================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID   Task: $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Dataset      : $DATASET"
echo "Model        : $MODEL"
echo "Feature pol. : reduced"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/run_pipeline.py" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR" \
    --feature-policy reduced
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
