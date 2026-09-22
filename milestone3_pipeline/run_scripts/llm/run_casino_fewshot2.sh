#!/bin/bash
#SBATCH --job-name=casino-fewshot2
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs/casino_fewshot2_%A_%a.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs/casino_fewshot2_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate

LAB="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl"
PYTHON="$LAB/envs/llmtriage/bin/python -u"
SCRIPTS_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/scripts/src"
OUT_ROOT="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results_fewshot2"
BACKEND="ollama"

export HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac"
export MPLCONFIGDIR="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/tmp/matplotlib_cache"
export XDG_CACHE_HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/tmp/cache"
export TMPDIR="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/tmp"; mkdir -p "$TMPDIR"

DATASET="casino"
DATA_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl/Group2/data"
OUT_DIR="$OUT_ROOT/$DATASET"

MODELS=(fdtn-ai/Foundation-Sec-8B-Instruct NousResearch/Meta-Llama-3.1-8B-Instruct)
OLLAMA_MODELS=(foundation-sec-8b:latest llama3.1:8b)

if [ "$BACKEND" == "ollama" ]; then
    MODEL=${OLLAMA_MODELS[$SLURM_ARRAY_TASK_ID]}
else
    MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
fi

MODEL_TAG=$(echo "$MODEL" | sed 's#.*/##' | tr 'A-Z' 'a-z' | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##')

LOGS_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs"
mkdir -p "$LOGS_DIR" "$OUT_DIR"

if [ "$BACKEND" == "ollama" ]; then
    export OLLAMA_MODELS="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/ollama-models"
    export PATH="$LAB/ollama-bin/bin:$PATH"

    OLLAMA_PORT=$(( 13434 + ${SLURM_ARRAY_TASK_ID:-0} ))
    export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
    export OLLAMA_NUM_PARALLEL=1
    export OMP_NUM_THREADS=16
    ollama serve > "$LOGS_DIR/ollama_casino_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
    OLLAMA_PID=$!
    trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

    echo "Waiting for Ollama server to start on port ${OLLAMA_PORT}..."
    for i in {1..30}; do
        if ! kill -0 $OLLAMA_PID 2>/dev/null; then
            echo "ERROR: Ollama server crashed during startup!"
            cat "$LOGS_DIR/ollama_casino_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
            exit 1
        fi
        if curl -s "http://${OLLAMA_HOST}/api/tags" > /dev/null; then
            echo "Ollama server is up and running."
            break
        fi
        sleep 2
    done
fi

echo "============================================"
echo "Task       : $SLURM_ARRAY_TASK_ID  ($DATASET x $MODEL)"
echo "Data dir   : $DATA_DIR"
echo "Backend    : $BACKEND"
echo "Out dir    : $OUT_DIR"
echo "Start      : $(date)"
echo "============================================"

RC=0

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
