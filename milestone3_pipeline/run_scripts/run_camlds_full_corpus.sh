#!/bin/bash
#SBATCH --job-name=camlds-full-corpus
#SBATCH --output=logs/camlds_fullcorpus_%j.out
#SBATCH --error=logs/camlds_fullcorpus_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Regenerates CAM-LDS using the FULL available corpus: the original 28
# scenario_* directories (98 audit.log files, ~90.8k rows) PLUS
# manifestations_filtered/sequences/ (~2,252 additional audit.log files --
# confirmed via md5sum to be genuinely independent repeated trial
# executions of the same scenarios, NOT duplicates -- see CHANGELOG.md
# "9d: full corpus discovery").
#
# With ~23x more raw data available, both matching the original ~90.8k
# row count AND getting substantially closer to a realistic attack rate
# may be jointly achievable now -- unlike the scenario_*-only corpus,
# where the benign pool ceiling made that mathematically impossible.
#
# --time=04:00:00 is a bigger budget than the scenario_*-only regen
# (02:00:00) since this scans ~23x more files -- not a measured number,
# raise it if this doesn't finish.
#
# manifestations_raw/ (5,753 files, different internal organization) is
# NOT included yet -- not explored for overlap with manifestations_filtered.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_camlds_full_corpus.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
TARGET_TOTAL_SIZE=90805
MAX_BENIGN_PER_SCENARIO=100000
OUT_DIR="combined_v3"

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Job ID                : $SLURM_JOB_ID"
echo "Start                  : $(date)"
echo "Target total size       : $TARGET_TOTAL_SIZE"
echo "Sources                 : scenario_* + manifestations_filtered"
echo "Out dir                 : $OUT_DIR"
echo "============================================"

echo ""
echo "--- Step 1/2: extract from the full corpus ---"
$PYTHON scripts/caml/extract_caml_full_corpus.py \
    --target_total_size "$TARGET_TOTAL_SIZE" \
    --max_benign_per_scenario "$MAX_BENIGN_PER_SCENARIO" \
    --min_events 2 \
    --sources scenario,manifestations \
    --out "$OUT_DIR/raw_labeled_logs_full_corpus.csv"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Step 1 FAILED (exit $EXIT_CODE)"; exit 1; }

echo ""
echo "--- Step 2/2: build process-level features ---"
$PYTHON scripts/dataset/prepare_dataset.py \
    --csv "$OUT_DIR/raw_labeled_logs_full_corpus.csv" \
    --out "$OUT_DIR/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Step 2 FAILED (exit $EXIT_CODE)"; exit 1; }

echo ""
echo "--- Verification ---"
$PYTHON -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/dataset1_features.csv')
print(f'Rows: {len(df):,}  (target was {$TARGET_TOTAL_SIZE:,})')
print(f'Attack rate: {df.label.mean():.1%}  (original CAM-LDS: 54.0%, CasinoLimit: 17.1%)')
"

echo "============================================"
echo "Finished     : $(date)"
echo "============================================"
