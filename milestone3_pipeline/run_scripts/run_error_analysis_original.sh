#!/bin/bash
#SBATCH --job-name=milestone3-error-analysis-orig
#SBATCH --output=logs/m3_erroranalysis_orig_%A_%a.out
#SBATCH --error=logs/m3_erroranalysis_orig_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 sample-level error analysis (both datasets).
# One array task per dataset. Produces, per dataset:
#   <dataset>_error_rows.csv                          -- every row + all 4
#                                                         models' predictions
#   <dataset>_<model>_technique_breakdown.csv          -- FN rate per MITRE
#                                                         technique
#   <dataset>_<model>_feature_comparison.csv           -- FN vs TP features
#                                                         (missed vs caught attacks)
#   <dataset>_<model>_fp_vs_tn_feature_comparison.csv  -- FP vs TN features
#                                                         (flagged vs cleared benign)
#   <dataset>_fp_summary.csv                           -- FP rate per model
#   <dataset>_<A>_vs_<B>_pairwise.csv                  -- cross-model
#                                                         disagreement, per pair
#   <dataset>_technique_fn_rates.png                   -- chart
#   <dataset>_fp_feature_comparison.png                -- chart
#
# NOTE: this is a SINGLE 5-fold split (not 15-repeat CV) since the goal is
# per-row inspection, not a stable aggregate number -- much cheaper than
# the headline run, but CNN is still the long pole.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_error_analysis.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="scripts/milestone3"
DATA_DIR="combined"   # the ORIGINAL (54% attack) CAM-LDS data
OUT_BASE="$SCRIPTS_DIR/error_analysis_results_original"

mkdir -p logs "$OUT_BASE"

DATASETS=(camlds casino)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$OUT_BASE/$DATASET"

echo "============================================"
echo "Task         : $SLURM_ARRAY_TASK_ID  (dataset=$DATASET)"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "Start        : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/error_analysis.py" \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --out "$OUT_DIR"
EXIT_CODE=$?

echo "============================================"
echo "Finished     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
