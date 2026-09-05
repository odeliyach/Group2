# LLM Triage Layer — Run & Configure

Operational reference for the Step 8 capstone (`milestone3_pipeline/src/llm_triage_*.py`).
Design: `docs/superpowers/specs/2026-08-31-llm-triage-contextual-arbitration-design.md`.

The layer runs **offline in three stages, per dataset per model**:

```
llm_triage_dump.py       5-fold OOF pass -> rows where XGBoost's call != CNN's call
                         (both routed: 0.3 <= p_xgb <= 0.7), capped at 400/dataset
        |
llm_triage_arbitrate.py  render each row as a compact context window -> LLM verdict
                         (Foundation-Sec-8B-Instruct, control: Llama-3.1-8B-Instruct)
        |
llm_triage_evaluate.py   LLM accuracy vs 5 baselines, per-fold + pooled dF1/dFPR,
                         ECE, key-feature overlap -> CSVs, charts, report_section.md
```

`run_scripts/run_llm_triage.sh` chains all three as a 4-task SLURM array
(`{camlds, casino} x {foundation-sec, llama-3.1}`). Always submit from the repo root.

---

## 1. One-time cluster setup

Everything lives on the lab volume — the home directory quota is tiny and *will*
break matplotlib and pip mid-run.

```bash
export LAB=/vol/joberant_nobck/data/NLP_368307701_2526a/$USER
mkdir -p $LAB/{envs,pkgs,hf_cache,tmp,tmp/mpl}

# conda + pip + HF caches -> lab volume
export CONDA_ENVS_DIRS=$LAB/envs CONDA_PKGS_DIRS=$LAB/pkgs
export PIP_CACHE_DIR=$LAB/.pipcache TMPDIR=$LAB/tmp
export HF_HOME=$LAB/hf_cache MPLCONFIGDIR=$LAB/tmp/mpl

conda create -y -p $LAB/envs/llmtriage python=3.11
conda activate $LAB/envs/llmtriage
pip install -r milestone3_pipeline/src/requirements.txt        # CPU: dump + evaluate
pip install -r milestone3_pipeline/src/requirements-llm.txt    # GPU: vllm, transformers, accelerate
```

Run the `pip install` for `requirements-llm.txt` in an **interactive GPU-node shell**
(`srun --partition=studentkillable --gres=gpu:titan:1 --mem=24G --time=1:00:00 --pty bash`);
vLLM probes CUDA on install.

### Models (from HuggingFace — this cluster's network allows HF, but **not**
`github.com` / `ollama.com`, so the Ollama fallback can't be installed here)

```bash
export HF_HOME=$LAB/hf_cache
hf auth login                                                        # read token from hf.co/settings/tokens

hf download fdtn-ai/Foundation-Sec-8B-Instruct                       # open, ~16 GB
hf download NousResearch/Meta-Llama-3.1-8B-Instruct --exclude "original/*"   # ungated Llama-3.1 mirror
```

`meta-llama/Llama-3.1-8B-Instruct` is gated and was not approved for this account;
`NousResearch/Meta-Llama-3.1-8B-Instruct` is byte-identical and ungated. Give
`--exclude` **one** pattern at a time (a second bare pattern is parsed as a
filename by the `hf` CLI).

Point `run_llm_triage.sh`'s `PYTHON=` and `LAB=` lines at your paths.

---

## 2. Configuration

### `milestone3_pipeline/src/config.py` — `LLM_TRIAGE`

| Key | Default | Meaning |
|---|---|---|
| `model_id` | `fdtn-ai/Foundation-Sec-8B-Instruct` | security-tuned engine |
| `control_model_id` | `NousResearch/Meta-Llama-3.1-8B-Instruct` | general-purpose A/B control |
| `backend` | `vllm` | `vllm` (cluster GPU) or `ollama` (local HTTP fallback) |
| `max_new_tokens` | `800` | generation budget per row |
| `sampling_seed` | `42` | fixed for reproducibility |
| `temperature` | `0.0` | deterministic decoding |
| `max_cases_per_dataset` | `400` | stratified cap on contradiction rows sent to the LLM |
| `cascade_low_thresh` / `cascade_high_thresh` | `0.3` / `0.7` | routing band — mirrors `hybrid_cascade.cascade_predict`, keep in sync |
| `schema_version` | `v1` | prompt/schema version stamp |
| `vllm_cpu_offload_gb` | `0` | GB of weights vLLM streams from host RAM (fits a big model on a small GPU; slower) |
| `vllm_dtype` | `bfloat16` | `bfloat16` needs compute capability >= 8.0; use `float16` on pre-Volta GPUs |

The script overrides `dtype` and `cpu_offload_gb` per run via CLI flags — you
normally don't edit `config.py`.

### `llm_triage_arbitrate.py` CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | (required) | `camlds` or `casino` |
| `--data-dir` | `.` | `data/v3` for CAM-LDS, `data` for Casino |
| `--out` | (required) | `.../results/current/llm_triage/<ds>/` |
| `--model` | config `model_id` | HF repo id (vllm) or Ollama tag |
| `--backend` | config `backend` | `vllm` / `ollama` |
| `--max-cases` | `400` | honoured on both the fresh and CSV-reuse paths |
| `--seed` | `42` | |
| `--cpu-offload-gb` | config value | vLLM only |
| `--dtype` | config `vllm_dtype` | vLLM only — `bfloat16` / `float16` |

`llm_triage_dump.py`: `--dataset --data-dir --out [--seed --n-splits --max-cases]`.
`llm_triage_evaluate.py`: `--dataset --arbitration <csv> --oof <csv> --out <dir>`.

---

## 3. Running

### Tier 1 — verify wiring (no GPU, ~5 s)

```bash
python -m pytest milestone3_pipeline -m "not slow"      # 43 passed, 1 deselected
```

### Tier 2 — local smoke, one real call (needs a running Ollama with `llama3.1:8b`)

```bash
python -m pytest milestone3_pipeline -m slow -v
```

### Tier 3 — the real run (SLURM)

```bash
git pull origin feat/llm-triage-contextual-arbitration     # get the latest fixes first

# sanity-check ONE task before the full array
sbatch --array=0 milestone3_pipeline/run_scripts/run_llm_triage.sh
squeue -u $USER
tail -f logs/m3_llm_triage_<jobid>_0.out

# once task 0 produces a report_section.md with parse_status = ok/repaired (not call_error):
sbatch milestone3_pipeline/run_scripts/run_llm_triage.sh
```

Run a stage by hand (debugging):

```bash
D=milestone3_pipeline/src ; OUT=milestone3_pipeline/results/current/llm_triage/camlds
$PYTHON $D/llm_triage_dump.py      --dataset camlds --data-dir data/v3 --out $OUT
$PYTHON $D/llm_triage_arbitrate.py --dataset camlds --data-dir data/v3 --out $OUT \
        --model fdtn-ai/Foundation-Sec-8B-Instruct --backend vllm --dtype float16 --cpu-offload-gb 8
$PYTHON $D/llm_triage_evaluate.py  --dataset camlds --out $OUT \
        --arbitration $OUT/camlds_foundation-sec-8b-instruct_llm_arbitration.csv \
        --oof $OUT/camlds_cascade_oof.csv
```

The **dump** stage fits XGBoost + CNN over 5 folds (CPU/TensorFlow, several
minutes, no GPU). Only **arbitrate** needs the GPU. It checkpoints every 25 rows
and resumes on restart, so a preemption costs minutes, not the task.

---

## 4. GPU / partition reality on this cluster

| Partition (account `gpu-students`) | GPUs | bf16? | Verdict |
|---|---|---|---|
| **`studentkillable`** (what you have) | `titan` = **Titan Xp**, CC 6.1, 12 GB · `rtx_2080`, 8 GB | **no** (CC < 8.0) | `--dtype float16 --cpu-offload-gb 8` on the titan |
| `killable` (staff must grant) | RTX 3090 / A5000 (24 GB), A6000 / L40S (48 GB), V100 (32 GB) | yes (>= Ampere) | bf16, no offload; `--gres=gpu:geforce_rtx_3090:1 --partition=killable` |

`bfloat16` requires compute capability **>= 8.0** (Ampere). `float16` works on
Pascal. To ask about `killable`: `sacctmgr -n show assoc user=$USER format=account,partition`
shows what you can submit to; the request goes to the course staff.

Check a GPU's real VRAM (only works inside an allocation):

```bash
srun --partition=studentkillable --gres=gpu:titan:1 --time=0:05:00 \
     nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

---

## 5. Outputs — `milestone3_pipeline/results/current/llm_triage/<ds>/`

| File | Read it for |
|---|---|
| `<ds>_contradictions.csv` | how many disagreement rows (CAM-LDS ~400 capped; Casino may be ~0) |
| `<ds>_cascade_oof.csv` | full out-of-fold XGB/CNN calls (input to `pipeline_effect`) |
| `<ds>_<model>_llm_arbitration.csv` | per-row `llm_verdict`, `rationale_steps`, `parse_status`, `raw` (exception text on failure) |
| `<ds>_<model>_arbitration_metrics.csv` | **headline** — LLM accuracy vs always-XGB / always-CNN / trust-more-confident / cascade@0.5thr / majority |
| `<ds>_<model>_pipeline_effect.csv` | per-fold + pooled dF1 / dFPR / dRecall from substituting the LLM's calls |
| `<ds>_<model>_reasoning_quality.csv` | ECE, `key_feature_overlap`, `parse_error_rate`, `call_error_rate`, `failure_rate` |
| `<ds>_<model>_report_section.md` | paste-ready prose + tables for the Milestone-3 report |
| `<ds>_<model>_accuracy.png`, `_calibration.png` | the two charts |

The Deliverable-4 finding is the **difference between the two `report_section.md`
files** (Foundation-Sec vs Llama-3.1) on the same dataset.

`model` tag in filenames = the `--model` value's last path segment, lowercased,
non-alphanumeric runs -> `-` (e.g. `foundation-sec-8b-instruct`,
`meta-llama-3-1-8b-instruct`).

---

## 6. Troubleshooting

Every failure below has actually happened on this cluster.

| Symptom | Cause | Fix |
|---|---|---|
| `parse_status` all `call_error`, `raw` = `ValueError: Bfloat16 is only supported on GPUs with compute capability >= 8.0` | titan is Pascal (CC 6.1) | `--dtype float16` |
| `call_error`, `raw` = `ConnectionError` / `Engine core initialization failed` after ~35 s | vLLM engine couldn't start | read the SLURM `.err` — the real cause is printed above "Engine core initialization failed" (usually `CUDA out of memory` -> raise `--cpu-offload-gb`, or a driver/kernel mismatch) |
| `call_error`, `raw` = connection refused to `localhost:11434` | Ollama backend, but no server (the binary can't be installed here — `github.com`/`ollama.com` are network-blocked) | use `--backend vllm` |
| `EmptyDataError: No columns to parse` / `ValueError: Found empty input array` in evaluate | a dataset had 0 usable LLM rows (fixed in code — `git pull`) | pull latest; evaluate now writes NaN rows instead of crashing |
| `mkdir -p failed ... .config/matplotlib: Disk quota exceeded` | home quota full | `export MPLCONFIGDIR=$LAB/tmp/mpl` (the script does this) |
| `hf download ... Access denied. This repository requires approval` | gated `meta-llama/*` repo | use `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| `hf download` fetches a 9-byte "Not Found" | wrong domain / network block, not a 404 | HF works; `github.com` does not — don't route downloads through GitHub |
| `sbatch: Invalid account or account/partition combination` | asked for `killable` without access | use `studentkillable`, or get staff to add the account |
| Job runs for hours doing nothing, all rows `call_error` | *(old bug, fixed)* engine construction retried once per row | pull latest — a doomed engine now fails once and the batch fails fast |

The `raw` column in `<ds>_<model>_llm_arbitration.csv` holds the exact exception
for every `call_error` / `parse_error` row — always look there first:

```bash
python -c "import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); print(d['parse_status'].value_counts()); print(d['raw'].dropna().unique()[:5])" \
  milestone3_pipeline/results/current/llm_triage/camlds/camlds_foundation-sec-8b-instruct_llm_arbitration.csv
```
