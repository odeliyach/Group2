#!/bin/bash
#SBATCH --job-name=m3-cross-dataset-v3-15rep
#SBATCH --output=../logs/m3_crossdataset_v3_15rep_%A_%a.out
#SBATCH --error=../logs/m3_crossdataset_v3_15rep_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --partition=studentkillable

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"

MODELS=(rf xgb cnn mlp iforest)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

mkdir -p ../logs results/current/outputs_camlds_v3/cross_dataset

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL"
echo "Start: $(date)"

$PYTHON src/cross_dataset_eval.py --model "$MODEL" --data-dir ../data/v3 --repeats 15 \
    --out results/current/outputs_camlds_v3/cross_dataset/

echo "Finished: $(date)"
