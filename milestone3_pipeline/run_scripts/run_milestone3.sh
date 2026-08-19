#!/bin/bash
#SBATCH --job-name=milestone3-pipeline
#SBATCH --output=logs/m3_%A_%a.out
#SBATCH --error=logs/m3_%A_%a.err
#SBATCH --array=0-7%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=20:00:00
#SBATCH --partition=studentkillable
#
# Milestone 3 -- Step 7 headline run.
#
# 2 datasets x 4 models = 8 independent array tasks. Each task runs
# run_pipeline.py for exactly ONE (dataset, model) pair, using the full
# CV_CONFIG in config.py (repeats=15, n_splits=5 -- same repeat count as
# Milestone 2's unified_validation.py).
#
# Resource sizing: --cpus-per-task=4 --mem=12G is right-sized for a
# ~90k-row, 20-feature dataset (RF/XGBoost/IsoForest don't need more;
# the CNN is the only borderline case and should still fit). This is
# intentionally smaller than Milestone 2's 8-cpu/32G budget, which ran a
# much heavier multi-stage job (EDA + validation + harmonization all in
# one) -- smaller requests here queue faster on a loaded partition.
#
# --array=0-7%3 caps concurrency at 3 simultaneous tasks, so the scheduler
# only needs to find 3 small slots at a time instead of 8 -- much easier
# to fit into partially-allocated ("mix") nodes. Raise the %N if the
# partition frees up and you want more parallelism.
#
# NOTE on --time: RF / XGBoost / Isolation Forest finish well under 30 min
# each at this repeat count on CPU. The 1D-CNN is the long pole -- budget
# roughly 40min-2hrs per dataset depending on epochs (config.py). 02:00:00
# gives the CNN task margin; if it still doesn't finish, split it into its
# own sbatch with a longer --time rather than raising it for all 8 tasks.
#
# Submit from the workshop ROOT directory:  sbatch scripts/milestone3/run_milestone3.sbatch

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="scripts/milestone3"   # directory containing config.py, ingestion.py, run_pipeline.py, etc.

# Both CSVs must live in the SAME directory (unlike Milestone 2's split
# combined/ vs root layout -- ingestion.py expects one --data-dir).
DATA_DIR="combined"    # dataset1_features.csv and casino_process_level_ds.csv both live here

OUT_DIR="$SCRIPTS_DIR/outputs"

mkdir -p logs "$OUT_DIR"

# Index -> (dataset, model) mapping: index = dataset_idx * 4 + model_idx
DATASETS=(camlds casino)
MODELS=(rf xgb cnn iforest)

DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
DATASET=${DATASETS[$DATASET_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "============================================"
echo "Array Job ID : $SLURM_ARRAY_JOB_ID   Task: $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start        : $(date)"
echo "Dataset      : $DATASET"
echo "Model        : $MODEL"
echo "Data dir     : $DATA_DIR"
echo "Output dir   : $OUT_DIR"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/run_pipeline.py" \
    --dataset "$DATASET" \
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
