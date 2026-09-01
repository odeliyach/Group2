#!/bin/bash
#SBATCH --job-name=m3-cascade-thresh-sweep
#SBATCH --output=logs/m3_cascade_thresh_%A_%a.out
#SBATCH --error=logs/m3_cascade_thresh_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- cascade confidence-band threshold sweep (see
# cascade_threshold_sweep.py docstring). Tests 5 half-widths x 3 repeats x
# 5 folds = 75 CNN fits per dataset -- budget 6h given CNN training cost
# and this session's observed pace (~75min per 3-repeat sensitivity config
# elsewhere in this project); reduce --repeats or the grid if this proves
# too slow, don't just extend time indefinitely.
#
# Submit from the repo ROOT directory:  sbatch milestone3_pipeline/run_scripts/run_cascade_threshold_sweep.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_DIR="milestone3_pipeline/results/current/cascade_threshold_sweep"

DATASETS=(camlds casino)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DATA_DIR="data"   # original CAM-LDS + Casino, matching the primary-reported cascade

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (dataset=$DATASET)"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/cascade_threshold_sweep.py" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR" \
    --repeats 3
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
