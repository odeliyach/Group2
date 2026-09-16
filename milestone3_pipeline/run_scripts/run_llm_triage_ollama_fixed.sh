#!/bin/bash
#SBATCH --job-name=milestone3-llm-triage
#SBATCH --output=logs/m3_llm_triage_%A_%a.out
#SBATCH --error=logs/m3_llm_triage_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 capstone: LLM-Driven Triage and Contextual Arbitration,
# Ollama/CPU backend. Runs all three stages (dump -> arbitrate -> evaluate) so
# the 5bb79a4 prompt fix (system-role message, balancing few-shot, trimmed
# feature table) gets exercised end-to-end, not just re-scored against a
# stale cascade_oof.csv.
#
# This is a de-personalized variant of run_llm_triage_ollama.sh: that script
# hardcodes another user's ($HOME=odeliyac) absolute paths for logs/scripts/
# output and permanently disables the dump stage (RC=0 short-circuit). Use
# THIS script instead; run_llm_triage_ollama.sh is left as-is for that user.
#
#   task 0: camlds  x  Foundation-Sec-8B  (security-tuned)
#   task 1: camlds  x  Llama-3.1-8B       (general-purpose control)
#   task 2: casino  x  Foundation-Sec-8B
#   task 3: casino  x  Llama-3.1-8B
#
# One-time setup: pull the two tags this script expects --
#   ollama pull llama3.1:8b
#   ollama pull hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M
# (adjust OLLAMA_MODELS below to match whatever tags you actually pulled).
#
# See docs/milestone3/llm_triage_runbook.md for setup, config and troubleshooting.
# Submit from the repo ROOT:  sbatch milestone3_pipeline/run_scripts/run_llm_triage_ollama_fixed.sh

LAB="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl"
PYTHON="$LAB/envs/llmtriage/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_ROOT="milestone3_pipeline/results/current/llm_triage"
BACKEND="ollama"

export HOME="$LAB"
export TMPDIR="$LAB/tmp";           mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$LAB/tmp/mpl"; mkdir -p "$MPLCONFIGDIR"   # matplotlib writes here (evaluate step)
export XDG_CACHE_HOME="$LAB/tmp/cache"

DATASETS=(camlds camlds casino casino)
MODELS=(fdtn-ai/Foundation-Sec-8B-Instruct NousResearch/Meta-Llama-3.1-8B-Instruct \
        fdtn-ai/Foundation-Sec-8B-Instruct NousResearch/Meta-Llama-3.1-8B-Instruct)
# Ollama tags, parallel to MODELS -- adjust to whatever tags you pulled.
OLLAMA_MODELS=(hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M llama3.1:8b \
               hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M llama3.1:8b)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OLLAMA_TAG=${OLLAMA_MODELS[$SLURM_ARRAY_TASK_ID]}

if [ "$DATASET" == "camlds" ]; then DATA_DIR="data/v3"; else DATA_DIR="data"; fi
OUT_DIR="$OUT_ROOT/$DATASET"
# MODEL_TAG from the Ollama tag actually sent to arbitrate, so evaluate finds
# the right *_llm_arbitration.csv (arbitrate names it after --model).
MODEL_TAG=$(echo "$OLLAMA_TAG" | sed 's#.*/##' | tr 'A-Z' 'a-z' | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##')

mkdir -p logs "$OUT_DIR"

# --- user-space Ollama server on this node, one port per array task so a
# node running several tasks doesn't collide on the default 11434 ---
export OLLAMA_MODELS="$LAB/ollama-models"
export PATH="$LAB/ollama-bin/bin:$PATH"
OLLAMA_PORT=$(( 11434 + ${SLURM_ARRAY_TASK_ID:-0} ))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_NUM_PARALLEL=1
export OMP_NUM_THREADS=16

ollama serve > "logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

echo "Waiting for Ollama server on port ${OLLAMA_PORT}..."
for i in {1..30}; do
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama server crashed during startup!"
        cat "logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
        exit 1
    fi
    if curl -s "http://${OLLAMA_HOST}/api/tags" > /dev/null; then
        echo "Ollama server is up."
        break
    fi
    sleep 2
done

echo "============================================"
echo "Task       : $SLURM_ARRAY_TASK_ID  ($DATASET x $OLLAMA_TAG)"
echo "Data dir   : $DATA_DIR"
echo "Backend    : $BACKEND (host=$OLLAMA_HOST)"
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
        --model "$OLLAMA_TAG" --backend "$BACKEND"
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
