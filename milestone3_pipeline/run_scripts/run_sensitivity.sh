#!/bin/bash
#SBATCH --job-name=milestone3-sensitivity
#SBATCH --output=logs/m3_sens_%A_%a.out
#SBATCH --error=logs/m3_sens_%A_%a.err
#SBATCH --array=0-7%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 7 deliverable #5: hyperparameter sensitivity analysis.
#
# One array task per (dataset, model) pair, sweeping the grids defined in
# config.py's SENSITIVITY_GRIDS. Uses a cheaper --repeats than the headline
# run on purpose -- exploratory sweep, not the final reported number
# (re-run the winning config through run_milestone3.sbatch's full 15-repeat
# CV once you've picked it).
#
# Independent of run_milestone3.sbatch -- submit before, after, or in
# parallel with it.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_sensitivity.sbatch

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/" 

DATA_DIR="data"                # matches run_milestone3.sh's real data path
OUT_DIR="milestone3_pipeline/results/current/sensitivity_results"
REPEATS=3                            # exploratory sweep repeat count (cheap)

mkdir -p logs "$OUT_DIR"

DATASETS=(camlds casino)
MODELS=(rf xgb cnn iforest)

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

$PYTHON "$SCRIPTS_DIR/sensitivity_analysis.py" \
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
