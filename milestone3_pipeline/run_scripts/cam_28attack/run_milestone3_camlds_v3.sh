#!/bin/bash
#SBATCH --job-name=milestone3-camlds-v3
#SBATCH --output=logs/m3_camldsv3_%A_%a.out
#SBATCH --error=logs/m3_camldsv3_%A_%a.err
#SBATCH --array=0-3%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Runs the SAME run_pipeline.py used for all other headline results, now
# pointed at the full-corpus regenerated CAM-LDS variant (data/v3/,
# from run_camlds_full_corpus.sh). Output goes to a SEPARATE
# outputs_camlds_v3/ folder -- never touches the original outputs/ or
# outputs_camlds_v2/.
#
# Run run_camlds_full_corpus.sh FIRST and confirm it finished (check the
# achieved row count / attack rate) before submitting this.
#
# Submit from the workshop ROOT directory:  sbatch milestone3_pipeline/run_scripts/cam_28attack/run_milestone3_camlds_v3.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src/" 
DATA_DIR="data/v3"
OUT_DIR="milestone3_pipeline/results/current/outputs_camlds_v3"

mkdir -p logs "$OUT_DIR"

MODELS=(rf xgb cnn iforest)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID   Task: $SLURM_ARRAY_TASK_ID"
echo "Model        : $MODEL"
echo "Data dir     : $DATA_DIR  (full-corpus CAM-LDS variant)"
echo "Output dir   : $OUT_DIR"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/run_pipeline.py" \
    --dataset camlds \
    --model "$MODEL" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR" \
    --feature-policy full
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
