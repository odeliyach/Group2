#!/bin/bash
#SBATCH --job-name=milestone3-llm-triage
#SBATCH --output=logs/m3_llm_triage_%A_%a.out
#SBATCH --error=logs/m3_llm_triage_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 capstone: LLM-Driven Triage and Contextual Arbitration.
# One array task per (dataset x model). Each task: dump the hybrid-cascade
# contradiction rows, run the LLM arbitrator over them, then evaluate.
#
#   task 0: camlds  x  fdtn-ai/Foundation-Sec-8B-Instruct
#   task 1: camlds  x  meta-llama/Llama-3.1-8B-Instruct
#   task 2: casino  x  fdtn-ai/Foundation-Sec-8B-Instruct
#   task 3: casino  x  meta-llama/Llama-3.1-8B-Instruct
#
# Requires src/requirements-llm.txt installed on the GPU node. If no GPU
# partition is available, set BACKEND=ollama below and serve a GGUF locally
# (ollama pull the model first); then --gres=gpu:1 can be dropped.
#
# Submit from the repo ROOT:  sbatch milestone3_pipeline/run_scripts/run_llm_triage.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_ROOT="milestone3_pipeline/results/current/llm_triage"
BACKEND="vllm"

DATASETS=(camlds camlds casino casino)
MODELS=(fdtn-ai/Foundation-Sec-8B-Instruct meta-llama/Llama-3.1-8B-Instruct \
        fdtn-ai/Foundation-Sec-8B-Instruct meta-llama/Llama-3.1-8B-Instruct)
# Local Ollama tags, parallel to MODELS -- adjust to your pulled tags.
OLLAMA_MODELS=(foundation-sec:8b-instruct llama3.1:8b \
               foundation-sec:8b-instruct llama3.1:8b)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
VLLM_MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
if [ "$BACKEND" == "ollama" ]; then
    MODEL=${OLLAMA_MODELS[$SLURM_ARRAY_TASK_ID]}
else
    MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
fi

if [ "$DATASET" == "camlds" ]; then DATA_DIR="data/v3"; else DATA_DIR="data"; fi
OUT_DIR="$OUT_ROOT/$DATASET"
# Derived from the vllm MODELS entry so output filenames are stable across backends.
MODEL_TAG=$(echo "$VLLM_MODEL" | sed 's#.*/##; s#[^A-Za-z0-9]#-#g' | tr 'A-Z' 'a-z')

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Task       : $SLURM_ARRAY_TASK_ID  ($DATASET x $MODEL)"
echo "Data dir   : $DATA_DIR"
echo "Backend    : $BACKEND"
echo "Out dir    : $OUT_DIR"
echo "Start      : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/llm_triage_dump.py" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR"
RC=$?
if [ $RC -ne 0 ]; then echo "dump failed"; fi

if [ $RC -eq 0 ]; then
    $PYTHON "$SCRIPTS_DIR/llm_triage_arbitrate.py" \
        --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR" \
        --model "$MODEL" --backend "$BACKEND"
    RC=$?
fi

if [ $RC -eq 0 ]; then
    $PYTHON "$SCRIPTS_DIR/llm_triage_evaluate.py" \
        --dataset "$DATASET" \
        --arbitration "$OUT_DIR/${DATASET}_${MODEL_TAG}_llm_arbitration.csv" \
        --oof "$OUT_DIR/${DATASET}_cascade_oof.csv" \
        --out "$OUT_DIR"
    RC=$?
fi

echo "============================================"
echo "Finished   : $(date)"
echo "Exit code  : $RC"
echo "============================================"
exit $RC
