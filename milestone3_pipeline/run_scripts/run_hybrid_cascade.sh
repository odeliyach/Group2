#!/bin/bash
#SBATCH --job-name=milestone3-hybrid-cascade
#SBATCH --output=logs/m3_cascade_%A_%a.out
#SBATCH --error=logs/m3_cascade_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 capstone: Hybrid Behavioural Cascading.
# XGBoost (stage 1, all rows) -> CNN (stage 2, only rows where XGBoost's
# probability falls in the uncertain band). See hybrid_cascade.py's
# module docstring for the pairwise-disagreement evidence behind this
# design, and why it deviates from the brief's IsolationForest-first
# example architecture.
#
# One array task per dataset. CAM-LDS uses the v3-regenerated variant
# (data/v3/) to match the current best-config headline results; edit
# DATA_DIR below if you want the original 54%-attack CAM-LDS instead.
#
# Submit from the repo ROOT directory:  sbatch milestone3_pipeline/run_scripts/run_hybrid_cascade.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_DIR="milestone3_pipeline/results/current/hybrid_cascade"

DATASETS=(camlds casino)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# CAM-LDS -> v3 regenerated variant; Casino has no variant, always "data"
if [ "$DATASET" == "camlds" ]; then
    DATA_DIR="data/v3"
else
    DATA_DIR="data"
fi

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (dataset=$DATASET)"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/hybrid_cascade.py" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
