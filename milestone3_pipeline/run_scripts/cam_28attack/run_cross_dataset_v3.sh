#!/bin/bash
#SBATCH --job-name=milestone3-crossds-v3
#SBATCH --output=logs/m3_crossds_v3_%A_%a.out
#SBATCH --error=logs/m3_crossds_v3_%A_%a.err
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
# Submit from the workshop ROOT directory:  sbatch milestone3_pipeline/run_scripts/cam_28attack/run_cross_dataset.sbatch

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/"

DATA_DIR="data/v3"  
OUT_DIR="milestone3_pipeline/results/current/outputs_camlds_v3/cross_dataset"
CALIBRATION_FRAC=0.05   # NEW: with v3's AUROC now near-perfect for RF/XGB/CNN
                        # (0.90-0.96), the remaining F1 gap is very likely pure
                        # threshold miscalibration, not a ranking failure --
                        # recalibrating on a small target-domain sample should
                        # recover most of the F1. IsoForest (AUROC still ~0.015)
                        # is the negative control -- expect no recovery there.
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
    --calibration-frac "$CALIBRATION_FRAC" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "Finished: $(date)  Exit: $EXIT_CODE"
exit $EXIT_CODE
