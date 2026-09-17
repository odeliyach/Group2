#!/bin/bash
#SBATCH --job-name=m3-cascade-fnr
#SBATCH --output=logs/m3_cascade_fnr_%A_%a.out
#SBATCH --error=logs/m3_cascade_fnr_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"

DATASETS=(camlds casino)
DATA_DIRS=(data data)
OUT_DIRS=("milestone3_pipeline/results/hybrid_cascade/camlds" \
          "milestone3_pipeline/results/hybrid_cascade/casino")

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
DATA_DIR=${DATA_DIRS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=${OUT_DIRS[$SLURM_ARRAY_TASK_ID]}

mkdir -p logs "$OUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET data_dir=$DATA_DIR out=$OUT_DIR"
echo "Start: $(date)"

$PYTHON "$SCRIPTS_DIR/compute_cascade_fnr.py" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR"

echo "Finished: $(date)"
