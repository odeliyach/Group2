# AI-Driven Intrusion Detection - Workshop Project
## Tel Aviv University | Group 2

**Attack Category:** IAM Privilege Abuse & Cloud Lateral Movement

**Group Members:**
- Alin Ioshevsky (alinl@mail.tau.ac.il)
- Lior Pernik (liorpernik@mail.tau.ac.il)
- Odeliya Charitonova (odeliyac@mail.tau.ac.il)

---

## Repository Structure

```
Group2/
├── README.md                  ← this file
├── code/
│   ├── features/              ← Period 5: per-student feature extraction scripts
│   └── final/                 ← Period 8: final codebase + run_all.sh
├── data/
│   └── README.md              ← dataset download instructions (data not committed)
├── deliverables/
│   ├── 01_group_and_incidents.md
│   ├── 02_paper_analyses.zip  ← Period 2
│   ├── 03_midsemester_package.zip ← Period 3 (mid-semester gate)
│   ├── 04_EDA_and_feature_spec.zip ← Period 4
│   ├── 05_feature_extraction_spec.md ← Period 5
│   ├── 06_crossdataset_summary.pdf ← Period 6 (end-of-semester)
│   ├── 07_error_and_ensemble.zip ← Period 7
│   └── 08_final_report.pdf   ← Period 8 (final)
└── experiments/
    └── pipelines/             ← Period 6: per-student pipeline folders
```

---

## Attack Category

**IAM Privilege Abuse & Cloud Lateral Movement** covers initial privilege escalation, lateral movement across cloud resources, and persistence via IAM misconfigurations in AWS and Azure environments.

**Primary MITRE ATT&CK techniques (IaaS/SaaS matrix):**
- T1548 - Abuse Elevation Control Mechanism
- T1098 - Account Manipulation
- T1078.004 - Valid Accounts: Cloud Accounts
- T1550.001 - Use Alternate Authentication Material: Application Access Token

**Primary telemetry:** AWS CloudTrail audit logs, Azure AD sign-in/audit logs

---

## Datasets

| Student | Dataset | Source |
|---|---|---|
| Odeliya | flaws.cloud CloudTrail Logs (1.9M events, real attackers) | https://summitroute.com/downloads/flaws_cloudtrail_logs.tar |
| Alin | TBD | TBD |
| Lior | BOTS V3 | https://github.com/splunk/botsv3 |

> Data files are NOT committed to this repository. See `data/README.md` for download instructions.

---

## Reproducibility

- Python version: 3.12
- Dependency file: `requirements.txt` (to be added in Period 5)
- All random seeds must be fixed and documented in `model_configs.json` per student
- Final demo must run end-to-end in under 15 minutes on sampled data via `code/final/run_all.sh`

---

## Milestones

| Milestone | Due | Deliverable |
|---|---|---|
| Period 1 | Week 2 | `01_group_and_incidents.md` + this README, tag v0.1 |
| Period 2 | Week 4 | `02_paper_analyses.zip` (3 individual PDFs + feature map CSV) |
| **Mid-semester** | **Week 6** | `03_midsemester_package.zip` (breach analyses + dataset commitments) |
| Period 4 | Week 8 | `04_EDA_and_feature_spec.zip` (3 EDA notebooks + harmonized spec) |
| Period 5 | Week 10 | `code/features/` (3 extraction scripts + sample CSVs) |
| **End-of-semester** | **Week 12** | `experiments/pipelines.zip` + `06_crossdataset_summary.pdf` |
| Period 7 | Week 14 | `07_error_and_ensemble.zip` + ensemble design + LLM plan |
| **Final** | **Week 18** | `GroupX/final_release.zip` (report + code + slides + demo video) |
