#!/bin/bash
#SBATCH --job-name=prepare-dataset
#SBATCH --output=logs/prepare_dataset_%j.out
#SBATCH --error=logs/prepare_dataset_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=studentkillable
# Submit from the workshop ROOT directory:  sbatch scripts/dataset/prep_ds.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python"
SCRIPT="scripts/dataset/prepare_dataset.py"
CSV="combined/raw_labeled_logs_v3.csv"
OUT="combined/"

mkdir -p "$OUT" logs

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Input CSV  : $CSV"
echo "Output dir : $OUT"
echo "============================================"

$PYTHON "$SCRIPT" --csv "$CSV" --out "$OUT" --max_seq_len 50

EXIT_CODE=$?
echo "============================================"
echo "Finished   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
