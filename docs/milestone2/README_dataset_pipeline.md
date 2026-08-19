# Dataset 1 Construction Pipeline — Final Version
## AI-Driven Linux Host-Based Anomaly Detection — Milestone 2

---

## Overview

This pipeline constructs **Dataset 1** — a same-source CAM-LDS corpus for
general Linux host anomaly detection, with an emphasis on privilege escalation
(LPE) — used for EDA (Ch. 3), feature importance analysis (Ch. 4), and the
Random Forest baseline (Ch. 6.2).

Both benign and attack processes are drawn from the **same 28 attack-scenario
captures**, distinguished only by presence or absence of a MITRE-technique
`auditd` rule tag. This same-source design was adopted after an earlier
AIT-LDS(benign)+CAM-LDS(attack) combination was found to leak dataset identity
as a trivial classification shortcut (AUROC=1.000) — see `Design Decisions`
below.

**Final output:** `dataset1_features_v3_final.csv`
- 90,805 processes
- 41,783 benign / 49,022 attack (46% / 54%)
- 20 features (full Ch. 1 feature set, kept fixed — see Design Decisions)
- 19 MITRE techniques in the attack class

A second, narrower extraction (`raw_labeled_logs_lpe_byproc.csv`) restricts the
attack class to strict privilege-escalation techniques only (T1166, T1169,
T1068) and is used specifically for the Ch. 4.3 case study on aggregate-feature
limitations against scripted attack replay.

---

## Pipeline — Run in This Order

### Step 1 — Extract (broad, 19-technique corpus)
**Script:** `scripts/caml/extract_caml_full_v3.py`

Reads every `audit.log` under `data/caml/cam_lds/scenario_*/`. For each file:

1. Groups raw lines by `seq_id` (auditd's own event-grouping key) to correctly
   parse multi-line events (`SYSCALL` + companion `PATH`/`EXECVE` records).
2. **Regroups by PID**, reconstructing each process's complete event history
   *before* any sampling occurs — sampling at the event-group level was found
   to fragment real processes, since a process's `seq_id`s could be picked
   independently rather than together.
3. Stamps the real `pid` and `host` onto every record in a process, including
   companion record types that don't carry their own `pid=` field in raw
   auditd — without this, such records default to a null/zero PID and
   silently merge into an unrelated process during later grouping.
4. Classifies each process as **benign** (no technique tag on any event) or
   **attack** (technique tag present — labeled with that technique).
5. Applies **symmetric capping**: benign is capped per-scenario, attack is
   capped per-technique (both at 15,000), preventing any single
   high-volume scenario (e.g. `scenario_6_screensharing_cron`) or technique
   (e.g. `T1078_Valid_Accounts`, `T1043_Commonly_Used_Port`) from dominating
   either class.

```bash
sbatch scripts/caml/extract_caml_full_v3.sh
```

**Output:** `combined/raw_labeled_logs_v3.csv` — 20 columns:
`record_type, timestamp, seq_id, pid, ppid, uid, euid, auid, gid, comm, exe,
syscall_name, key_tag, success, failed, source, scenario, host, label,
technique`

---

### Step 2 — Extract (strict LPE-only corpus, for the Ch. 4.3 case study)
**Script:** `scripts/caml/lpe_proc.py`

Same PID-level, host-safe extraction logic as Step 1, but the attack class is
restricted to `{T1166_Seuid_and_Setgid, T1169_Sudo,
T1068_Exploitation_for_Privilege_Escalation}`; any process tagged with a
different technique is excluded entirely (neither benign nor attack).

```bash
sbatch scripts/caml/lpe_proc.sh
```

**Output:** `combined/raw_labeled_logs_lpe_byproc.csv`

---

### Step 3 — Build Features and Sequences
**Script:** `scripts/dataset/prepare_dataset.py`

Takes a raw labeled CSV (from Step 1 or Step 2) and produces three
representations:

1. **`dataset1_fixed.csv`** — cleaned raw events, NaN-filled with neutral
   defaults, ready for EDA.
2. **`dataset1_features_v3_final.csv`** — one row per process, grouped by
   `(source, scenario, host, pid)` (host included specifically to prevent PID
   collisions across the multiple hosts present within a single scenario
   capture), aggregated into the full 20-feature Ch. 1 set. `MIN_EVENTS=2`.
3. **`sequences_X.npy` / `sequences_y.npy`** — padded token sequences (shape
   `(N, 50, 7)`) for LSTM/1D-CNN, preserving syscall order within each process.

```bash
sbatch scripts/dataset/prep_ds.sh
```

Run once per raw CSV (Step 1's broad output → `dataset1_features_v3_final.csv`
+ `sequences_X.npy`; Step 2's LPE-only output → the `_lpe_only` suffixed
files) — edit the `CSV=` line in `prep_ds.sh` to point at the desired input
before each run.

---

### Step 4 — EDA
**Script:** `scripts/eda/run_eda.py`

Runs on `dataset1_features_v3_final.csv` (the full 20-feature set, matching
Step 5 exactly, so Ch. 4.1's RF importance and Ch. 6.2's baseline are directly
comparable). Produces class distribution plots, per-feature distributions,
correlation matrix, Mann-Whitney U + Cohen's *d* per feature, RF feature
importance, and the domain-vs-empirical discrepancy analysis.

```bash
sbatch scripts/eda/eda.sh
```

**Output:** `eda_output/` — feeds Ch. 3 and Ch. 4.1/4.2.

---

### Step 5 — Model Validation (leak-free)
**Script:** `scripts/validation/validate_features_v3.py`

Trains a Random Forest on the same full 20-feature set as Step 4. Evaluation
uses `StratifiedGroupKFold` (5 folds) grouped by **exact feature-vector
identity** — every row sharing an identical feature vector with another row is
guaranteed to land entirely on one side of the train/test split, eliminating
duplicate-row leakage without discarding any data. Metrics are averaged across
all 5 folds and reported as mean ± std, since a single split can be
unrepresentative when the corpus has a high duplication rate (see
`Design Decisions`).

```bash
/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python \
    scripts/validation/validate_features_v3.py \
    --features combined/dataset1_features_v3_final.csv \
    --out validation_output_final/
```

**Output:** `validation_output_final/` — feeds Ch. 4.2 and Ch. 6.2.

**Final result:** AUROC = 0.881 ± 0.130, F1 = 0.823 ± 0.156 (5-fold mean ± std).

---

## Design Decisions

**Same-source CAM-LDS, not AIT-LDS+CAM-LDS.** An initial design combined
AIT-LDS (benign) with CAM-LDS (attack). This was abandoned after diagnosing
AUROC=1.000 as dataset-identity leakage: AIT-LDS contributes almost entirely
`USER_AUTH`/`LOGIN` record types while CAM-LDS contributes `SYSCALL` records,
so any model trivially learns *which dataset a row came from* rather than
genuine attack behavior. AIT-LDS and BETH were both evaluated as candidate
benign baselines and are excluded from this work entirely — no unsupervised
one-class architecture is included in the current model scope, so there is no
role in which they could be used without reintroducing the same
cross-dataset schema leakage.

**Host-safe process grouping.** PID values are reused independently across
the multiple hosts present within a single scenario capture (e.g.
`videoserver`, `inetfw`, `corpdns`). Grouping by `(source, scenario, pid)`
alone can silently merge unrelated processes from different hosts that happen
to share a PID number into a single fabricated "mega-process" — diagnosed via
inspection of anomalously large process groups with internally inconsistent
`comm` values. Fixed by including `host` in the grouping key throughout.

**PID-level (not event-group-level) sampling.** Sampling individual `seq_id`
event-groups independently at random can fragment a single real process
across many sampled groups, so that only 1 of a process's true 10 events gets
selected — making it look like a single-event process downstream even though
it wasn't. Fixed by reconstructing each process's full event history via PID
regrouping *before* any capping or sampling occurs.

**Symmetric class capping.** Initial extraction capped only the benign class
per-scenario, leaving attack uncapped; since two techniques (T1078, T1043)
are far more common than others, this produced an extreme 9:1 attack/benign
imbalance. The final extraction caps both classes symmetrically —
per-scenario for benign, per-technique for attack.

**Full 20-feature set kept fixed.** The Ch. 1 feature table is not pruned
based on empirical performance mid-pipeline. Features that show weaker or
inverted behavior in isolation (e.g. `max_uid_euid_delta`, `uid_changed`) are
reported and discussed as Ch. 4.2 discrepancy findings, not silently excluded
— excluding them was tried and found to *reduce* corpus diversity (2,006 →
8,098 unique feature-vector patterns when restored) and *increase* evaluation
instability, in addition to breaking direct comparability between Ch. 4.1
(EDA) and Ch. 6.2 (validation) results.

**91.1% row duplication is a documented corpus property, not a defect.**
Verified via manual inspection: it reflects (a) twelve near-identical scripted
exploit variants replayed automatically hundreds of times each, and (b)
highly repetitive background service activity. This motivates sequence-based
(LSTM) modeling as future work — aggregate features cannot distinguish replay
run *N* from run *N+1* of an identical script, since exact syscall order and
timing are discarded during aggregation.

---

## Output Schema Reference

**`dataset1_features_v3_final.csv`** (20 features + metadata):

`lifetime_seconds, events_per_second, seq_length, unique_syscalls,
uid_is_root, uid_changed, euid_root, max_uid_euid_delta, is_suid_exec,
auid_euid_mismatch, ephemeral_privileged, failed_call_rate,
failed_call_count, priv_op_count, exec_count, file_access_count,
network_count, shell_from_service, sensitive_path_access,
parent_child_rarity` — plus `label`, `source`, `technique`, `proc_name`
(metadata, not used as model input).

---

## Data Sources

**CAM-LDS** — Zenodo record 18861762. 28 scenario captures across privilege
escalation (PwnKit, sudo abuse, race condition, valid-account abuse ×
localaccount/pam/sshkey persistence), lateral movement (SSH/VNC), and
persistence/rootkit/macro scenarios.

**AIT-LDS v2.0** (evaluated and excluded — see `Design Decisions`) —
Zenodo record 5789064. Not used in this pipeline's final output.

**Linux-APT-2024** (Dataset 2, cross-telemetry-layer evaluation) — Wazuh
agent export; schema mapping documented in Ch. 5.2 of the report.
