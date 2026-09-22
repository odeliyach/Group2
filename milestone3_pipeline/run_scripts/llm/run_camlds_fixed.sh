#!/bin/bash
#SBATCH --job-name=camlds-fixed
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/scripts/logs/camlds_fixed_%A_%a.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/scripts/logs/camlds_fixed_%A_%a.err
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate

LAB="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl"
MY_DIR="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac"
PYTHON="$LAB/envs/llmtriage/bin/python -u"
SCRIPTS_DIR="$MY_DIR/scripts"
OUT_DIR="$MY_DIR/llm_triage_results/camlds"
DATA_DIR="data/v3"
BACKEND="ollama"

# תיקון קריטי: HOME מצביע לתיקייה שלך שיש בה הרשאות כתיבה
export HOME="$MY_DIR"
export TMPDIR="$MY_DIR/tmp";           mkdir -p "$TMPDIR"
export MPLCONFIGDIR="$MY_DIR/tmp/mpl"; mkdir -p "$MPLCONFIGDIR"
export XDG_CACHE_HOME="$MY_DIR/tmp/cache"

DATASET="camlds"
OLLAMA_MODELS=(hf.co/bartowski/Foundation-Sec-8B-Instruct-GGUF:Q4_K_M llama3.1:8b)
OLLAMA_TAG=${OLLAMA_MODELS[$SLURM_ARRAY_TASK_ID]}

MODEL_TAG=$(echo "$OLLAMA_TAG" | sed 's#.*/##' | tr 'A-Z' 'a-z' | sed -E 's#[^a-z0-9]+#-#g; s#^-+##; s#-+$##')

mkdir -p "$SCRIPTS_DIR/logs" "$OUT_DIR"

# Ollama קורא את קובצי המודל מהתיקייה המשותפת של חברתך, אך כותב קונפיגורציה ל-HOME שלך
export OLLAMA_MODELS="$LAB/ollama-models"
export PATH="$LAB/ollama-bin/bin:$PATH"
OLLAMA_PORT=$(( 11434 + ${SLURM_ARRAY_TASK_ID:-0} ))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_NUM_PARALLEL=1
export OMP_NUM_THREADS=16

ollama serve > "$SCRIPTS_DIR/logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null' EXIT

echo "Waiting for Ollama server on port ${OLLAMA_PORT}..."
for i in {1..30}; do
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama server crashed during startup!"
        cat "$SCRIPTS_DIR/logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
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
echo "Backend    : $BACKEND (host=$OLLAMA_HOST)"
echo "Out dir    : $OUT_DIR"
echo "Start      : $(date)"
echo "============================================"

echo "Skipping dump, running arbitration on existing contradictions..."
$PYTHON "$SCRIPTS_DIR/llm_triage_arbitrate.py" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR" \
    --model "$OLLAMA_TAG" --backend "$BACKEND"
RC=$?

if [ $RC -eq 0 ]; then
    echo "Arbitration succeeded. Running evaluation..."
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
