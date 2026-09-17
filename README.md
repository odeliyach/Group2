# Host-Based Behavioral Anomaly Detection on Linux Systems
## Tel Aviv University | Workshop on Intrusion Detection Using ML Techniques | Group 2

**Attack category:** Linux host-based privilege escalation, targeting MITRE ATT&CK
T1548 (Abuse Elevation Control Mechanism: Setuid/Setgid), T1059 (Command and Scripting
Execution), T1222 (File/Directory Permissions Modification), and T1595 (Active Scanning),
evaluated on two independent telemetry sources: **CAM-LDS** and **CasinoLimit**.

**Group members:**
- Alin Loshevsky ([alinl@mail.tau.ac.il](mailto:alinl@mail.tau.ac.il))
- Lior Pernik ([liorpernik@mail.tau.ac.il](mailto:liorpernik@mail.tau.ac.il))
- Odeliya Charitonova ([odeliyac@mail.tau.ac.il](mailto:odeliyac@mail.tau.ac.il))

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── dataset1_features.csv       # CAM-LDS, process-level features (label col: `label`)
│   └── casino_process_level_ds.csv # CasinoLimit, process-level (label col: `is_privesc`)
├── docs/
│   ├── milestone1/         # Individual attack profiling, breach analysis, paper/dataset review
│   ├── milestone2/         # EDA, feature engineering, model selection report + dataset build notes
│   └── milestone3/         # Pipeline implementation & evaluation — changelog + experiment log
├── milestone2_pipeline/
│   ├── scripts/            # unified_eda.py, unified_validation.py, cross_dataset_harmonization.py
│   │   ├── caml/            # CAM-LDS raw-log extraction (reads data/raw/, which is not committed)
│   │   └── dataset/         # CasinoLimit feature build
│   ├── run_pipeline.sh     # SLURM: full Ch3/4/5 pipeline, both datasets
│   └── results/            # EDA, validation, and cross-dataset harmonization outputs
└── milestone3_pipeline/
    ├── src/                 # config.py, ingestion.py, models.py, train_eval.py, etc.
    ├── run_scripts/         # SLURM job scripts (headline CV, sensitivity, operating curve, error analysis)
    └── results/
        ├── current/         # Latest results (post-fix, regenerated CAM-LDS v3)
        └── archive/         # Pre-fix baseline results, kept for before/after comparison
```

## Data

Three processed feature CSVs are committed directly in `data/` (original and regenerated cam-lds, casino)(~15MB combined) — they're
the direct `--input` to every Milestone 2 and Milestone 3 script, so anyone cloning this
repo can run the pipelines immediately without regenerating anything.

| Dataset | Source | Rows | Label col |
|---|---|---|---|
| CAM-LDS (`dataset1_features.csv`) | Zenodo record 18861762, 28 scenario captures | 90,805 | `label` |
| CasinoLimit (`casino_process_level_ds.csv`) | internal build | 91,692 | `is_privesc` |

## Quick start

```bash
# Milestone 2 — EDA + validation + cross-dataset harmonization, both datasets
sbatch milestone2_pipeline/run_pipeline.sh

# Milestone 3 — headline model CV, both datasets, all 4 models
sbatch milestone3_pipeline/run_scripts/run_milestone3.sh
```

**Note:** several scripts have a hardcoded conda interpreter path from the original
account. Update the `PYTHON=` line in each `.sh` before submitting under a different
account.

## Current state

See `docs/milestone3/CHANGELOG.md` for the full decision log (every fix, experiment,
and finding) and `docs/milestone3/experiment_log.html` for a summary.

- **Milestone 1**: complete — individual attack/breach/paper/dataset analyses in `docs/milestone1/`.
- **Milestone 2**: complete — report, EDA/validation/harmonization scripts and outputs.
- **Milestone 3**: headline pipeline, sensitivity analysis, operating-curve derivation, and
  cross-dataset evaluation done. In progress: sample-level error analysis (CasinoLimit vs.
  regenerated CAM-LDS). 

  Step 8 capstone:
- hybrid cascade — implemented (`milestone3_pipeline/src/hybrid_cascade*.py`; XGBoost→CNN,
  plus RF-stage-2 and weighted variants)
- LLM triage layer — implemented in feature/milestone3-llm-arbitration
