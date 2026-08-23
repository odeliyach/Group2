#!/bin/bash
#SBATCH --job-name=milestone2-pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=studentkillable
#
# Full Ch3/4/5 pipeline: check Casino file present -> unified EDA (both
# datasets) -> unified validation (both datasets) -> cross-dataset
# harmonization. One job, all stages sequential -- matches eda.sh's
# single-job style, just with more stages in the body.
#
# NOTE on --time: this runs several times the work of eda.sh's single
# run_eda.py call, including two repeated-CV validation passes (default
# 15x5=75 folds each) plus a 20-feature ablation sweep per dataset.
# 06:00:00 is a starting guess, not a measured number -- if it's not
# enough on your cluster, either raise --time or split validation into
# its own sbatch job (see the commented-out alternative at the bottom).
#
# Submit from the workshop ROOT directory:  sbatch scripts/pipeline/run_pipeline.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python"
SCRIPTS_DIR="milestone2_pipeline/scripts/eda"     # directory containing the 3 unified_*.py / cross_dataset_*.py scripts

REPEATS=15                    # CV repeats -- SAME value for both datasets, on purpose

# Dataset 1 (CAM-LDS) -- already-extracted process-level feature CSV.
DS1_FEATURES="data/dataset1_features.csv"
DS1_NAME="CAM-LDS"
DS1_LABEL_COL="label"
DS1_TECH_COL="technique"

# Dataset 2 (CasinoLimit) — you already have the final harmonized file.
# No raw/resolve/rollup regeneration in this script; if you ever need
# to rebuild casino_process_level_ds.csv from scratch, run
# resolve_dataset_casino_only.py then process_level_rollup_final.py
# manually first (both have their I/O paths hardcoded inside the
# scripts themselves, not CLI flags).
DS2_ROLLUP="data/casino_process_level_ds.csv"
DS2_NAME="CasinoLimit"
DS2_LABEL_COL="is_privesc"
DS2_TECH_COL="technique"
DS2_EXCLUDE_TECHNIQUES="T1003,T1078"

OUT="milestone2_pipeline/results/pipeline_output/"
mkdir -p "$OUT" logs "$OUT/eda/$DS1_NAME" "$OUT/eda/$DS2_NAME" \
         "$OUT/validation/$DS1_NAME" "$OUT/validation/$DS2_NAME" "$OUT/cross_dataset"

echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Dataset 1    : $DS1_FEATURES"
echo "Dataset 2    : $DS2_ROLLUP"
echo "Repeats      : $REPEATS"
echo "Output dir   : $OUT"
echo "============================================"

# ── Stage 1-2/7 — Casino prep (not needed — you already have the file) ──
echo; echo "--- Stage 1-2/7: Casino prep ---"; echo "Start: $(date)"
if [ -f "$DS2_ROLLUP" ]; then
    echo "$DS2_ROLLUP found — skipping resolve/rollup regeneration."
else
    echo "ERROR: $DS2_ROLLUP not found, and this script does not regenerate it"
    echo "from raw. Run resolve_dataset_casino_only.py then"
    echo "process_level_rollup_final.py manually first, or fix DS2_ROLLUP's"
    echo "path in the CONFIG block above."
    exit 1
fi

# ── Stage 3/7 — Unified EDA: Dataset 1 ──────────────────────────────────
echo; echo "--- Stage 3/7: unified_eda.py ($DS1_NAME) ---"; echo "Start: $(date)"
$PYTHON "$SCRIPTS_DIR/unified_eda.py" \
    --input "$DS1_FEATURES" --label-col "$DS1_LABEL_COL" --technique-col "$DS1_TECH_COL" \
    --dataset-name "$DS1_NAME" --out "$OUT/eda/$DS1_NAME/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Stage 3 FAILED (exit $EXIT_CODE) — aborting pipeline."; exit 1; }

# ── Stage 4/7 — Unified EDA: Dataset 2 ──────────────────────────────────
echo; echo "--- Stage 4/7: unified_eda.py ($DS2_NAME) ---"; echo "Start: $(date)"
$PYTHON "$SCRIPTS_DIR/unified_eda.py" \
    --input "$DS2_ROLLUP" --label-col "$DS2_LABEL_COL" --technique-col "$DS2_TECH_COL" \
    --exclude-techniques "$DS2_EXCLUDE_TECHNIQUES" \
    --dataset-name "$DS2_NAME" --out "$OUT/eda/$DS2_NAME/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Stage 4 FAILED (exit $EXIT_CODE) — aborting pipeline."; exit 1; }

# ── Stage 5/7 — Unified validation: Dataset 1 ───────────────────────────
echo; echo "--- Stage 5/7: unified_validation.py ($DS1_NAME, ${REPEATS}x CV) ---"; echo "Start: $(date)"
$PYTHON "$SCRIPTS_DIR/unified_validation.py" \
    --input "$DS1_FEATURES" --label-col "$DS1_LABEL_COL" --technique-col "$DS1_TECH_COL" \
    --dataset-name "$DS1_NAME" --repeats "$REPEATS" --out "$OUT/validation/$DS1_NAME/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Stage 5 FAILED (exit $EXIT_CODE) — aborting pipeline."; exit 1; }

# ── Stage 6/7 — Unified validation: Dataset 2 ───────────────────────────
echo; echo "--- Stage 6/7: unified_validation.py ($DS2_NAME, ${REPEATS}x CV) ---"; echo "Start: $(date)"
$PYTHON "$SCRIPTS_DIR/unified_validation.py" \
    --input "$DS2_ROLLUP" --label-col "$DS2_LABEL_COL" --technique-col "$DS2_TECH_COL" \
    --exclude-techniques "$DS2_EXCLUDE_TECHNIQUES" \
    --dataset-name "$DS2_NAME" --repeats "$REPEATS" --out "$OUT/validation/$DS2_NAME/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Stage 6 FAILED (exit $EXIT_CODE) — aborting pipeline."; exit 1; }

# ── Stage 7/7 — Cross-dataset harmonization ─────────────────────────────
echo; echo "--- Stage 7/7: cross_dataset_harmonization.py ---"; echo "Start: $(date)"
$PYTHON "$SCRIPTS_DIR/cross_dataset_harmonization.py" \
    --input-a "$DS1_FEATURES" --name-a "$DS1_NAME" --label-col-a "$DS1_LABEL_COL" \
    --input-b "$DS2_ROLLUP"   --name-b "$DS2_NAME" --label-col-b "$DS2_LABEL_COL" \
    --technique-col-b "$DS2_TECH_COL" --exclude-techniques-b "$DS2_EXCLUDE_TECHNIQUES" \
    --out "$OUT/cross_dataset/"
EXIT_CODE=$?
[ $EXIT_CODE -ne 0 ] && { echo "Stage 7 FAILED (exit $EXIT_CODE) — aborting pipeline."; exit 1; }

echo "============================================"
echo "Finished     : $(date)"
echo "All 7 stages completed successfully."
echo "Outputs      : $OUT"
echo "============================================"
exit 0

# ── Alternative: split into dependent jobs instead of one long job ─────
# If 06:00:00 isn't enough / studentkillable preempts you mid-run, split
# the EDA (fast) and validation (slow, repeated CV) stages into separate
# sbatch scripts and chain them:
#
#   jid1=$(sbatch --parsable scripts/pipeline/01_casino_prep.sh)
#   jid2=$(sbatch --parsable --dependency=afterok:$jid1 scripts/pipeline/02_eda.sh)
#   jid3=$(sbatch --parsable --dependency=afterok:$jid2 scripts/pipeline/03_validation.sh)
#   sbatch --dependency=afterok:$jid3 scripts/pipeline/04_cross_dataset.sh
#
# Each stage then gets its own --time budget and can be resubmitted
# individually if it's the one that gets killed.
