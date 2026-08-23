#!/bin/bash
#SBATCH --job-name=eda-milestone2
#SBATCH --output=logs/eda_%j.out
#SBATCH --error=logs/eda_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=studentkillable
# Submit from the workshop ROOT directory:  sbatch scripts/eda/eda.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python"
SCRIPT="scripts/eda/run_eda.py"
FEATURES="data/dataset2_lidds_EPS36165.csv"
FIXED="data/dataset1_fixed.csv"
OUT="eda_output/"

mkdir -p "$OUT" logs

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Features   : $FEATURES"
echo "Output dir : $OUT"
echo "============================================"

$PYTHON "$SCRIPT" \
    --features "$FEATURES" \
    --fixed    "$FIXED" \
    --out      "$OUT"

EXIT_CODE=$?
echo "============================================"
echo "Finished   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
