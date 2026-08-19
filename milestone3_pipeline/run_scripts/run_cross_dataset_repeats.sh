#!/bin/bash
#SBATCH --job-name=milestone3-crossds-repeats
#SBATCH --output=logs/m3_crossds_repeats_%j.out
#SBATCH --error=logs/m3_crossds_repeats_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- ad-hoc single-model repeated cross-dataset run. Unlike
# run_cross_dataset.sh (array over all 4 models, 1 fit each), this repeats
# ONE model's train-once/test-elsewhere fit several times with different
# seeds -- needed specifically for CNN, which has no fixed random_state on
# its weight init, so a single striking result (e.g. AUROC=0.18, below
# chance) can't be trusted as stable without repeating it.
#
# Edit MODEL/REPEATS below and resubmit for other combinations -- this is
# meant to be a quick, disposable ad-hoc job, not a permanent array.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_cross_dataset_repeats.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="scripts/milestone3"
DATA_DIR="combined"
OUT_DIR="$SCRIPTS_DIR/outputs/cross_dataset"

MODEL="cnn"
REPEATS=5

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Model        : $MODEL"
echo "Repeats      : $REPEATS"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/cross_dataset_eval.py" \
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
