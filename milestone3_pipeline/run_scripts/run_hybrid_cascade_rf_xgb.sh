#!/bin/bash
#SBATCH --job-name=milestone3-cascade-rf-xgb
#SBATCH --output=logs/m3_cascade_rf_xgb_%A_%a.out
#SBATCH --error=logs/m3_cascade_rf_xgb_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Hybrid cascade variant: XGB (stage 1) -> RF (stage 2),
# instead of the CNN-based hybrid_cascade.py. See hybrid_cascade_rf_xgb.py's
# module docstring for the pairwise-disagreement evidence behind this
# variant and why IForest is excluded from both cascades.
#
# Both tree models are fast -- 30min should be generous (the CNN cascade
# needed 2h mainly for CNN training on the uncertain band; this variant
# has no CNN at all).
#
# Output goes to results/current/hybrid_cascade_rf_xgb/ -- a SEPARATE
# directory from hybrid_cascade.py's results/current/hybrid_cascade/, so
# running this never overwrites the CNN-based cascade's results. Compare
# the two afterward, don't treat one as replacing the other.
#
# Submit from the repo ROOT directory:  sbatch milestone3_pipeline/run_scripts/run_hybrid_cascade_rf_xgb.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_DIR="milestone3_pipeline/results/current/hybrid_cascade_rf_xgb"

DATASETS=(camlds casino)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Same convention as run_hybrid_cascade.sh: CAM-LDS defaults to the v3
# regenerated variant here too (this is the experiment-section pairing --
# for the ORIGINAL-CAM-LDS comparison, resubmit with DATA_DIR="data"
# for the camlds task, matching how hybrid_cascade_original/ was produced).
if [ "$DATASET" == "camlds" ]; then
    DATA_DIR="data"   # original, to match the primary-reported cascade comparison
else
    DATA_DIR="data"
fi

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (dataset=$DATASET)"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/hybrid_cascade_rf_xgb.py" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
