#!/bin/bash
#SBATCH --job-name=milestone3-crossds
#SBATCH --output=logs/m3_crossds_%A_%a.out
#SBATCH --error=logs/m3_crossds_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#
# Milestone 3 -- Step 8 cross-dataset generalization: train fully on ONE
# dataset, evaluate on the ENTIRETY of the other (both directions), per
# model. This is the actual "train on the primary set, test on a
# different environment" question Step 8 asks -- distinct from
# run_milestone3.sbatch's within-dataset CV.
#
# One array task per model (both train/test directions run inside a
# single cross_dataset_eval.py call).
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_cross_dataset.sbatch

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="scripts/milestone3"
DATA_DIR="combined"
OUT_DIR="$SCRIPTS_DIR/outputs/cross_dataset"
REPEATS=5   # confirmed necessary for ALL models, not just CNN -- a single
            # fit's cross-dataset AUROC has been shown to swing widely
            # (see run_cross_dataset_repeats.sh's CNN stability check, and
            # XGBoost's hist-method sensitivity to the log1p preprocessing
            # change). Mean+/-std across repeats is the number to report.

mkdir -p logs "$OUT_DIR"

MODELS=(rf xgb cnn iforest)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (model=$MODEL)"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/cross_dataset_eval.py" \
    --model "$MODEL" \
    --repeats "$REPEATS" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "Finished: $(date)  Exit: $EXIT_CODE"
exit $EXIT_CODE
