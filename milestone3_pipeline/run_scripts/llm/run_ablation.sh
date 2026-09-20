#!/bin/bash
#SBATCH --job-name=llm_ablation
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs/ablation_%j.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs/ablation_%j.err
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --open-mode=truncate

set -e
export HOME="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac"
export PYTHONDONTWRITEBYTECODE=1

LAB="/vol/joberant_nobck/data/NLP_368307701_2526a/alinl"
export PATH="$LAB/ollama-bin/bin:$PATH"
export OLLAMA_MODELS="/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/ollama-models"

export OLLAMA_PORT=$((12434 + SLURM_JOB_ID % 1000))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_NUM_PARALLEL=1
export OMP_NUM_THREADS=16

echo "=== Starting Ollama on ${OLLAMA_HOST} (Job ${SLURM_JOB_ID}) ==="
ollama serve > "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/logs/ollama_ablation_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill $OLLAMA_PID 2>/dev/null || true' EXIT

# Wait for Ollama service
for i in {1..30}; do
    if curl -s "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
done

PYTHON="$LAB/envs/llmtriage/bin/python -u"

echo "=== Running 3 Ablation Variants (100 rows each) ==="
$PYTHON /vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/scripts/llm_triage_ablation.py

echo "=== Finished Ablation. ==="
