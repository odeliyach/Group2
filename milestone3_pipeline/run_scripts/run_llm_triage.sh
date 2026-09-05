#!/bin/bash
#SBATCH --job-name=milestone3-llm-triage
#SBATCH --output=logs/m3_llm_triage_%A_%a.out
#SBATCH --error=logs/m3_llm_triage_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:titan:1
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 capstone: LLM-Driven Triage and Contextual Arbitration.
# One array task per (dataset x model): dump hybrid-cascade contradiction rows,
# run the LLM arbitrator, then evaluate.
#
#   task 0: camlds  x  Foundation-Sec-8B-Instruct  (Q4 GGUF via Ollama)
#   task 1: camlds  x  Llama-3.1-8B-Instruct       (Q4 GGUF via Ollama)
#   task 2: casino  x  Foundation-Sec-8B-Instruct
#   task 3: casino  x  Llama-3.1-8B-Instruct
#
# studentkillable GPUs are titan (12 GB) / rtx_2080 (8 GB) -> no room for an 8B
# model in bf16, so this runs 4-bit GGUF through a user-space Ollama server
# started per job. Both models are Q4 so the security-vs-general comparison is
# at matched precision -- state this in the write-up.
#
# One-time, on a login node:
#   export PATH=$LAB/ollama-bin/bin:$PATH ; export OLLAMA_MODELS=$LAB/ollama-models
#   ollama serve & sleep 5
#   ollama pull llama3.1:8b
#   ollama pull hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M   # verify repo/tag on HF
#
# Submit from the repo ROOT:  sbatch milestone3_pipeline/run_scripts/run_llm_triage.sh

LAB="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl"
PYTHON="$LAB/envs/llmtriage/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_ROOT="milestone3_pipeline/results/current/llm_triage"
BACKEND="vllm"

export TMPDIR="$LAB/tmp"; mkdir -p "$TMPDIR"
export HF_HOME="$LAB/hf_cache"          # unused by the ollama path; kept for a vllm switch-back

DATASETS=(camlds camlds casino casino)
# vllm ids (only used if BACKEND=vllm)
MODELS=(fdtn-ai/Foundation-Sec-8B-Instruct NousResearch/Meta-Llama-3.1-8B-Instruct \
        fdtn-ai/Foundation-Sec-8B-Instruct NousResearch/Meta-Llama-3.1-8B-Instruct)
# Ollama tags, parallel to MODELS -- must match what you `ollama pull`ed.
OLLAMA_MODELS=(hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M llama3.1:8b \
               hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M llama3.1:8b)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
VLLM_MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
if [ "$BACKEND" == "ollama" ]; then
    MODEL=${OLLAMA_MODELS[$SLURM_ARRAY_TASK_ID]}
else
    MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
fi

if [ "$DATASET" == "camlds" ]; then DATA_DIR="data/v3"; else DATA_DIR="data"; fi
OUT_DIR="$OUT_ROOT/$DATASET"
# MODEL_TAG from the selected $MODEL, so the evaluate step finds arbitrate's file
MODEL_TAG=$(echo "$MODEL" | sed 's#.*/##' | tr 'A-Z' 'a-z' | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##')

mkdir -p logs "$OUT_DIR"

# --- Ollama backend: user-space server on this node ---
if [ "$BACKEND" == "ollama" ]; then
    export OLLAMA_MODELS="$LAB/ollama-models"
    export PATH="$LAB/ollama-bin/bin:$PATH"
    ollama serve > "logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
    OLLAMA_PID=$!
    trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
    sleep 8
fi

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

