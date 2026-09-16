#!/bin/bash
#SBATCH --job-name=milestone3-synth-missingness
#SBATCH --output=logs/m3_synth_missing_%A_%a.out
#SBATCH --error=logs/m3_synth_missing_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- M2 reviewer point 1 follow-up (synthetic_missingness_ablation.py):
# verify_unified_schema_missingness.py's cross-dataset test used an
# availability indicator that is a dataset-level CONSTANT (1 throughout
# CAM-LDS, 0 throughout Casino), so a model trained on CAM-LDS alone has
# zero within-training variance to learn a split on -- its null result was
# guaranteed by that design, not evidence about the signal itself. This
# reruns the question within CAM-LDS alone, synthetically masking a random
# row subset so the indicator has real row-level variance in every fold.
#
# One array task per model (rf, xgb) -- both tree-based, matching the
# encoding-tolerant model choice in the original missingness test.
#
# Submit from the milestone3_pipeline directory: sbatch run_scripts/run_synthetic_missingness_ablation.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/"
DATA_DIR="data"
OUT_DIR="milestone3_pipeline/results/current/synthetic_missingness_ablation"
MASK_FRAC=0.5
REPEATS=15   # matches this project's headline protocol -- cheap: RF/XGBoost, one dataset, no LLM

mkdir -p logs "$OUT_DIR"

MODELS=(rf xgb)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID   Task: $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Model        : $MODEL"
echo "Mask frac    : $MASK_FRAC"
echo "Repeats      : $REPEATS"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/synthetic_missingness_ablation.py" \
    --model "$MODEL" \
    --mask-frac "$MASK_FRAC" \
    --repeats "$REPEATS" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
