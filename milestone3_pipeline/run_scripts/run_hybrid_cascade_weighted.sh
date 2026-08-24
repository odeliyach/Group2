#!/bin/bash
#SBATCH --job-name=milestone3-cascade-weighted
#SBATCH --output=logs/m3_cascade_weighted_%A_%a.out
#SBATCH --error=logs/m3_cascade_weighted_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- confidence-WEIGHTED cascade blend (see hybrid_cascade_weighted.py
# docstring for the full rationale). Tests whether scaling stage-2 influence by
# distance from the uncertain band's center -- instead of a flat 50/50 -- fixes
# the routing-volume-driven instability found when swapping CNN for RF didn't
# help (hybrid_cascade_rf_xgb.py's result: identical routing pattern, identical
# bad folds, regardless of stage-2 model).
#
# 5 combinations in one array:
#   0: camlds (original) + cnn      3: casino + cnn
#   1: camlds (original) + rf       4: casino + rf
#   2: camlds (v3)        + cnn
# (camlds v3 + rf omitted from this array -- add a 6th task manually if wanted,
#  same pattern as the others.)
#
# Submit from the repo ROOT directory:  sbatch milestone3_pipeline/run_scripts/run_hybrid_cascade_weighted.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"

case $SLURM_ARRAY_TASK_ID in
    0) DATASET="camlds"; STAGE2="cnn"; DATA_DIR="data";    OUT="milestone3_pipeline/results/current/hybrid_cascade_weighted_cnn" ;;
    1) DATASET="camlds"; STAGE2="rf";  DATA_DIR="data";    OUT="milestone3_pipeline/results/current/hybrid_cascade_weighted_rf" ;;
    2) DATASET="camlds"; STAGE2="cnn"; DATA_DIR="data/v3"; OUT="milestone3_pipeline/results/current/hybrid_cascade_weighted_cnn_v3" ;;
    3) DATASET="casino";  STAGE2="cnn"; DATA_DIR="data";   OUT="milestone3_pipeline/results/current/hybrid_cascade_weighted_cnn" ;;
    4) DATASET="casino";  STAGE2="rf";  DATA_DIR="data";   OUT="milestone3_pipeline/results/current/hybrid_cascade_weighted_rf" ;;
esac

mkdir -p logs "$OUT"

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (dataset=$DATASET, stage2=$STAGE2)"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/hybrid_cascade_weighted.py" \
    --dataset "$DATASET" \
    --stage2 "$STAGE2" \
    --data-dir "$DATA_DIR" \
    --out "$OUT"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
