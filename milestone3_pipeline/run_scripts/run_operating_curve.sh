#!/bin/bash
#SBATCH --job-name=milestone3-opcurve
#SBATCH --output=logs/m3_opcurve_%A_%a.out
#SBATCH --error=logs/m3_opcurve_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --partition=studentkillable
#
# Milestone 3 -- FPR operating-point sweep for RF (the fast, genuinely
# discriminative model -- see operating_curve.py's module docstring for
# why this sweep is meaningless for a non-discriminative model like
# Isolation Forest on CAM-LDS).
#
# One array task per dataset. Look at the resulting PNG's elbow point to
# pick and justify config.py's MAX_FPR -- then update MAX_FPR and rerun
# run_milestone3.sbatch for the final reported numbers.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_operating_curve.sbatch

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="scripts/milestone3"
DATA_DIR="combined"
OUT_DIR="$SCRIPTS_DIR/operating_curve_results"

mkdir -p logs "$OUT_DIR"

DATASETS=(camlds casino)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  ($DATASET / rf)"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/operating_curve.py" \
    --dataset "$DATASET" \
    --model rf \
    --repeats 5 \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "Finished: $(date)  Exit: $EXIT_CODE"
exit $EXIT_CODE
