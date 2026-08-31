# LLM-Driven Triage and Contextual Arbitration — Design Spec

**Project:** Host-Based Behavioral Anomaly Detection on Linux Systems — TAU Workshop, Group 2
**Milestone 3, Step 8 capstone:** "LLM-Driven Triage and Contextual Arbitration"
**Date:** 2026-08-31
**Status:** Design approved, pending spec read → implementation plan

---

## 1. Purpose and scope

### 1.1 What this adds

The Milestone 3 pipeline already has a **Hybrid Behavioural Cascade** (`milestone3_pipeline/src/hybrid_cascade.py`):
XGBoost is stage 1 (all rows), and rows whose XGBoost probability lands in the uncertain band
(`0.3 ≤ p ≤ 0.7`) are routed to a 1D-CNN stage 2, then the two probabilities are blended.

This spec adds the **third capstone paradigm**: a locally deployed, open-source, cybersecurity-tuned
LLM that acts as a **contextual arbitrator** for the cascade. When the cascade's two models produce
**contradicting binary classifications** on a routed row (stage-1 XGBoost call ≠ stage-2 CNN call),
that row's behavioural context is formatted into a compact prompt and passed to the LLM, which
performs structured chain-of-thought reasoning and returns a final, explainable
MALICIOUS / BENIGN decision.

The two required focus areas, per the refined task statement, are:

1. **Finding a relevant LLM engine** — a Hugging Face model trained on cybersecurity data.
2. **Generating the right context window** — how the process telemetry is rendered for the model,
   within that model's context limit.

### 1.2 Locked decisions (from brainstorming)

| Decision | Value |
|---|---|
| LLM host | TAU SLURM cluster (GPU node; vLLM / transformers). Ollama GGUF is the fallback path. |
| Integration depth | **Offline arbitrator** — run the cascade once, dump contradiction rows, batch the LLM over them, evaluate. Not wired live into the CV loop in v1. |
| Trigger | **Model disagreement only** — specifically, cascade stage-1 (XGBoost) binary call ≠ stage-2 (CNN) binary call, on rows the cascade routes to stage 2. |
| Datasets | Both CAM-LDS v3 and CasinoLimit. |
| Approach | **A — single-model zero-shot arbitrator with structured CoT + enriched context.** One LLM call per contradiction row. Deterministic decoding. |

### 1.3 Explicitly out of scope (YAGNI)

- Live wiring into `hybrid_cascade.py`'s CV folds (v1 is offline dump → batch → evaluate). The
  arbitrator core is written as a pure function so a later version *can* import it as stage 3.
- Self-consistency / multi-sample voting, multi-agent triage (CORTEX-style), fine-tuning, RAG,
  raw-log retrieval.
- More than one LLM call per contradiction row.
- Any change to `error_analysis.py`, the four base models, or the cascade's routing logic.

---

## 2. LLM engine selection

### 2.1 Chosen engine

**`fdtn-ai/Foundation-Sec-8B-Instruct`** (Cisco Foundation AI) — Hugging Face:
`https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct`

| Property | Value | Consequence for this design |
|---|---|---|
| Backbone | Meta Llama-3.1-8B, continued-pretrained + instruct-tuned on a cybersecurity corpus | Same model family as the `llama3.1:8b` already pulled locally → clean security-vs-general comparison |
| Security benchmark lift | +3 to +11 points over stock Llama-3.1-8B; CTI-RCM 75.3 (≈ Llama-3.1-70B) | Real domain gain at 8B, no 70B infrastructure needed |
| Stated intended use | SOC acceleration — "triage, summarization, evidence collection" | Exactly this task |
| **Context window** | **4,096 tokens** | **Binding constraint.** Drives the compact context-window design in §4. No raw-log dumps; ≤ 1 few-shot example. |
| Serving | transformers / vLLM / SGLang; GGUF quant for llama.cpp / Ollama / LM Studio | vLLM batched offline on one cluster GPU (~16 GB bf16); Ollama GGUF fallback |
| Safety | 91.98% HarmBench, 99.25% with LlamaGuard | Will not refuse to reason over security telemetry |
| Knowledge cutoff / caveats | No post-April-2025 vulnerability knowledge; vendor doc: "not for autonomous decisions without human review" | Acceptable — decisions are graded against ground-truth labels offline, never deployed |
| License | "Other" (permissive research use; see repo NOTICE.md) | Fine for coursework |

### 2.2 Paired control model

**`meta-llama/Llama-3.1-8B-Instruct`** — identical architecture, no security tuning. Run side by side
so the evaluation isolates whether cybersecurity pre-training actually changes arbitration quality on
these contradiction cases. This is the headline Deliverable-4 finding.

### 2.3 Optional upgrade run (only if the 4k window proves too tight)

- **`fdtn-ai/Foundation-Sec-8B-Reasoning`** (Jan 2026) — same family, explicit reasoning traces.
- **`Qwen3-14B`** — 32k context, strongest instruction-following / JSON-adherence in the 8–14B band
  per current (2026) benchmarks.

One extra run, decided at implementation time based on observed parse-failure and truncation rates.

### 2.4 Rejected alternatives

| Model | Reason rejected |
|---|---|
| `ZySec-AI/SecurityLLM` (ZySec-7B), `segolilylabs/Lily-Cybersecurity-7B` | Less rigorously benchmarked; no published safety evaluation; smaller community validation than Foundation-Sec |
| `RavichandranJ/Dolphin3-Cyber-8B`, community merges | Unclear provenance / training data |
| Llama-3.3-70B, Foundation-Sec at 70B-equivalent | Unjustified GPU/infra cost for a few hundred low-dimensional numeric rows |
| DeepSeek-V4 and other API-scale models | Not a clean "locally deployed open-source" story |
| Anything < 7B | Insufficient multi-step reasoning reliability |

### 2.5 Serving configuration

- **Primary:** vLLM on a SLURM GPU node, model loaded once, all contradiction rows for a dataset
  batched in one process.
- **Decoding:** `temperature = 0`, fixed seed, `guided_json` (or `outlines` grammar) to force
  schema-valid JSON.
- **Fallback:** `backend = "ollama"` against a locally served GGUF quant (`llama3.1:8b` is already
  pulled for smoke tests; a Foundation-Sec GGUF is pulled for the real fallback run).
- Residual non-determinism from vLLM batch ordering is documented, not eliminated.

---

## 3. Architecture

### 3.1 Package layout

New package: `milestone3_pipeline/src/llm_triage/`

| File | Responsibility | Reads | Writes |
|---|---|---|---|
| `cascade_contradiction_dump.py` | One fixed-seed `StratifiedGroupKFold` pass (same split style as `error_analysis.py`). Per fold, fit XGBoost and the CNN via the existing `train_eval._FITTERS["xgb"]` / `["cnn"]` — each returns that model's own F1-optimal-thresholded binary call **and** probability. Separately derive the cascade's routing band from the XGBoost probability (`low_thresh ≤ p_xgb ≤ high_thresh`, defaults `0.3`/`0.7` from `hybrid_cascade.py`). Record every row that is **both routed and contradicting** (`pred_xgb != pred_cnn`): all 20 raw feature values, both probabilities, both binary calls, disagreement direction, fold id, true label, `technique` (post-hoc reporting only). Then stratified-subsample to `max_cases_per_dataset` (§3.5). | `dataset1_features.csv` / `casino_process_level_ds.csv` via `ingestion.ingest`; `train_eval._FITTERS` | `results/current/llm_triage/<ds>/<ds>_contradictions.csv` |
| `population_stats.py` | Compute per-feature percentile breakpoints from the full labelled population — separately for benign rows and attack rows — once per dataset. | ingested dataset | `results/current/llm_triage/<ds>/<ds>_percentiles.json` |
| `context_builder.py` | Pure function: one contradiction row + percentile tables → one compact prompt string (§4). Dataset-agnostic. | contradiction row, percentile JSON, `config.FEATURE_GLOSS`, known-failure block | — (in-memory) |
| `prompts.py` | Versioned constants: system prompt, CoT rubric, output JSON schema, the single hand-authored few-shot example. | — | — |
| `llm_client.py` | Thin wrapper over vLLM (primary) / Ollama (fallback). Batched, `temperature=0`, fixed seed, `guided_json`, one repair retry on malformed output, enum validation. | prompt strings | raw + parsed responses |
| `arbitrate.py` | Orchestrator CLI: dump (or load) contradictions → build percentiles → build context per row → query LLM → parse → write arbitration table. | all of the above | `results/current/llm_triage/<ds>/<ds>_<model>_llm_arbitration.csv` |
| `evaluate_arbitration.py` | Deliverable-4 metrics, charts, and a paste-ready report section from the arbitration tables. | arbitration CSVs, contradiction CSV | `arbitration_metrics.csv`, `pipeline_effect.csv`, `calibration.csv`, `reasoning_quality.csv`, `*.png`, `report_section.md` |

### 3.2 Config additions (`milestone3_pipeline/src/config.py`)

```python
LLM_TRIAGE = {
    "model_id": "fdtn-ai/Foundation-Sec-8B-Instruct",  # control run: "meta-llama/Llama-3.1-8B-Instruct"
    "backend": "vllm",                                  # or "ollama"
    "max_new_tokens": 800,
    "sampling_seed": 42,
    "temperature": 0.0,
    "max_cases_per_dataset": 400,   # stratified cap (§3.5); bounds cost. CAM-LDS likely under this naturally.
    "cascade_low_thresh": 0.3,      # routing band, mirrors hybrid_cascade.py defaults
    "cascade_high_thresh": 0.7,
    "schema_version": "v1",
}

# One-line analyst gloss per feature, used verbatim in the context window (§4.4).
FEATURE_GLOSS = {
    "lifetime_seconds": "process wall-clock lifetime",
    "events_per_second": "syscall throughput",
    "seq_length": "length of the syscall sequence",
    "unique_syscalls": "distinct syscall types used",
    "uid_is_root": "process real UID is root",
    "uid_changed": "UID changed during the process lifetime",
    "euid_root": "effective UID is root",
    "max_uid_euid_delta": "largest gap between real and effective UID; large = privilege transition",
    "is_suid_exec": "executed a setuid/setgid binary",
    "auid_euid_mismatch": "audit (login) UID differs from effective UID; identity inconsistency",
    "ephemeral_privileged": "short-lived process that held privilege",
    "failed_call_rate": "fraction of syscalls that returned an error",
    "failed_call_count": "absolute count of failed syscalls",
    "priv_op_count": "count of privileged operations (setuid, capset, ...)",
    "exec_count": "number of exec() calls",
    "file_access_count": "file-access syscalls",
    "network_count": "network-related syscalls",
    "shell_from_service": "a shell was spawned from a service/daemon context",
    "sensitive_path_access": "touched a sensitive path (/etc/shadow, /etc/sudoers, ...)",
    "parent_child_rarity": "rarity of this parent->child exec pair (1 = never seen before)",
}
```

Contradiction is defined **structurally** (`pred_xgb != pred_cnn` on routed rows, §3.5) — there is
no new threshold to tune.

### 3.3 Data flow

```
error_analysis.py is NOT invoked; its per-model fitters (train_eval._FITTERS) are reused directly.

ingestion.ingest(<ds>)
      |
      v
cascade_contradiction_dump.py  --(one fixed-seed CV pass; per fold fit xgb + cnn via _FITTERS;
      |                            derive routing band from p_xgb; keep routed AND pred_xgb != pred_cnn;
      |                            stratified-subsample to max_cases_per_dataset)-->
      v
<ds>_contradictions.csv   +   population_stats.py --> <ds>_percentiles.json
      |                                                     |
      +--------------------------+--------------------------+
                                 v
                        context_builder.py            (per row: compact prompt string)
                                 |
                                 v
                 prompts.py  +  llm_client.py         (vLLM, guided JSON, temp 0, 1 repair retry)
                                 v
              <ds>_<model>_llm_arbitration.csv
              (row id, fold, true label, xgb call+prob, cnn call+prob,
               llm verdict, llm confidence, rationale_steps, key_features,
               agrees_with, parse_status)
                                 |
                                 v
                     evaluate_arbitration.py
                                 v
   results/current/llm_triage/<ds>/  ->  metrics CSVs + PNGs + report_section.md
```

### 3.4 Prerequisite

No pre-existing artifact is required. `cascade_contradiction_dump.py` fits XGBoost and the CNN
itself per fold via `train_eval._FITTERS`, the same fitters `error_analysis.py` and the headline
pipeline use, so the contradiction rows are scored under the exact same F1-threshold / `MAX_FPR`
logic as everything else. `hybrid_cascade.py` is imported only for its routing-band constants
(`low_thresh`, `high_thresh`); its `cascade_predict` blending is **not** used, and it is not modified.

### 3.5 Contradiction set definition

A row enters the contradiction set for a dataset iff **both** hold, evaluated out-of-fold:

1. **Routed** — the row's XGBoost probability is in the cascade's uncertain band:
   `cascade_low_thresh ≤ p_xgb ≤ cascade_high_thresh` (defaults `0.3`/`0.7`).
2. **Contradicting** — XGBoost's own binary call ≠ the CNN's own binary call, where each call uses
   that model's F1-optimal train-fold threshold (the `_FITTERS` path, capped at `MAX_FPR`).

Each contradiction row is tagged with a **disagreement direction**:
`xgb_benign_cnn_malicious` or `xgb_malicious_cnn_benign`.

**Stratified subsampling.** If the set exceeds `max_cases_per_dataset`, sample down with a fixed
seed (`sampling_seed`), stratifying jointly on (disagreement direction × `technique` for attack
rows / `"benign"` for benign rows) so no direction or technique is dropped. If the set is at or
under the cap, use all rows. Expected: CAM-LDS v3 lands near or under 400 naturally; CasinoLimit is
capped (its XGBoost/CNN disagreement is inflated by CNN instability).

---

## 4. Context window design (focus area 2)

### 4.1 Token budget (Foundation-Sec-8B-Instruct, 4,096-token window)

| Segment | ~Tokens | Static? |
|---|---:|---|
| System prompt (role + task) | 250 | yes |
| CoT rubric (5 steps) | 350 | yes |
| Output schema spec | 200 | yes |
| 1 hand-authored few-shot example (context + ideal JSON) | 700 | yes |
| Live row context (feature table + votes + scope + known-failure block) | ~750 | per row |
| **Input subtotal** | **~2,250** | |
| Generation budget (`max_new_tokens`) | 800 | |
| Headroom | ~1,050 | |

All static segments are versioned constants in `prompts.py` (`schema_version = "v1"`) so the report
can quote them verbatim. Only the live row context is built per row.

### 4.2 System prompt (constant, v1)

> You are a Tier-2 SOC analyst on a Linux host-intrusion team. Two ML models in an automated
> detection pipeline have produced **contradicting** classifications for one process-execution
> record. Your job is to arbitrate: decide **MALICIOUS** (a privilege-escalation attempt) or
> **BENIGN**, using structured reasoning over the process's behavioural features. The monitored
> environment contains normal system and administrative activity plus privilege-escalation attempts
> in the MITRE ATT&CK families T1548 (Abuse Elevation Control Mechanism), T1059 (Command and
> Scripting Interpreter), T1222 (File/Directory Permissions Modification), and T1595 (Active
> Scanning). Attacks are a minority of records (roughly 17–29%). Follow the reasoning rubric exactly
> and return only the specified JSON object.

The row's actual `technique` value is **never** included in the prompt — it is populated only for
attack rows and would be a label leak.

### 4.3 CoT rubric (constant, v1) — tuned to privilege escalation

1. **Privilege state** — is there a genuine UID/EUID transition or root context?
   (`uid_is_root`, `euid_root`, `max_uid_euid_delta`, `uid_changed`, `auid_euid_mismatch`)
2. **Mechanism** — setuid/setgid execution, an unusual exec chain, or a shell spawned by a service?
   (`is_suid_exec`, `exec_count`, `shell_from_service`, `priv_op_count`)
3. **Behavioural anomaly** — failed-syscall rate, rare parent→child pair, ephemeral privileged
   process, sensitive-path access, network activity.
   (`failed_call_rate`, `parent_child_rarity`, `ephemeral_privileged`, `sensitive_path_access`,
   `network_count`)
4. **Benign-explanation test** — could this plausibly be admin scripting, a cron job, package
   management, or a health-check? Weigh `events_per_second`, `lifetime_seconds`, `seq_length`,
   `unique_syscalls`.
5. **Decide** — weigh steps 1–4, state which model's call the evidence supports, and emit a verdict
   with a calibrated confidence.

### 4.4 Live row context — format

Rendered per contradiction row:

```
=== RECORD UNDER REVIEW ===
Model votes:  XGBoost -> BENIGN (p_malicious=0.34)   |   CNN -> MALICIOUS (p_malicious=0.71)

Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)
  <feature_name>   <raw value>   p<NN>   p<NN>   <FEATURE_GLOSS[feature]>
  ... all 20 features, fixed order = config.FULL_FEATURE_LIST ...

Notable deviations (largest |value - benign median| first):
  <feat>, <feat>, <feat>, <feat>, <feat>          # top 5

Known failure patterns in this dataset (from labelled error analysis):
  - Benign rows misflagged as attack (FP) tend to have elevated failed_call_count and network_count.
  - Missed attacks (FN) tend to have LOW max_uid_euid_delta and priv_op_count.
```

Design choices:

- **Raw human-readable values**, not the pipeline's `log1p` / scaled values. The LLM reasons better
  over `lifetime_seconds = 42022` than over `10.6`.
- **Dual percentiles** (vs the benign population and vs the attack population) give a calibrated
  sense of "how unusual this value is" in each direction, without handing over the label.
  Percentile breakpoints come from `population_stats.py`, computed once per dataset.
- **Auto-flagged "notable deviations"** — the five features whose value is furthest (absolute
  difference) from the benign-population median — focuses attention inside the 20-row table.
- **"Known failure patterns" block** — the aggregate domain knowledge a real analyst carries,
  lifted from `error_analysis.py`'s FP-vs-TN and FN-vs-TP comparison tables (`camlds/*_fp_vs_tn_*`,
  `*_feature_comparison.csv`). Aggregate, not row-specific → not a leak. Two to four short bullets
  per dataset, authored from those CSVs at implementation time and stored as a constant.
- **Model probabilities** are shown so the LLM can distinguish a near-miss (0.34) from a confident
  call (0.71).

### 4.5 Few-shot example (constant, v1)

Exactly **one** hand-authored, fabricated-but-realistic worked example: a clear setuid-shell
privilege transition classified as `MALICIOUS`, showing the exact five `rationale_steps` and the
full JSON object. Hand-authored so it leaks nothing from either dataset. Kept to ~700 tokens.

### 4.6 Output schema (`guided_json`-enforced, v1)

```json
{
  "verdict": "MALICIOUS | BENIGN",
  "confidence": 0.0,
  "rationale_steps": ["step 1 text", "... 3 to 6 steps total"],
  "key_features": ["max_uid_euid_delta", "... <= 5 feature names"],
  "agrees_with": "xgb | cnn | neither"
}
```

`agrees_with` is cross-checked in evaluation against `verdict` vs each model's recorded call — a
free self-consistency signal. Values outside the enums, or a `key_features` entry not in
`FULL_FEATURE_LIST`, are treated as a parse failure (§5.2).

---

## 5. LLM client behaviour

### 5.1 Determinism

`temperature = 0`, `seed = LLM_TRIAGE["sampling_seed"]`, `guided_json` schema binding. The same
contradiction CSV + same model + same `schema_version` should reproduce the same arbitration CSV
up to documented vLLM batch-order effects.

### 5.2 Malformed output handling

1. Parse the model output as JSON against the v1 schema.
2. On failure: one **repair retry** — re-prompt with the offending output and a "return only valid
   JSON matching this schema" instruction.
3. On second failure: record the row with `parse_status = "parse_error"` and a null verdict. It is
   **excluded from accuracy metrics** but **counted in the robustness statistics** (§6E).
4. Enum / unknown-feature violations are handled identically to a JSON parse failure.

### 5.3 Batching

All contradiction rows for one (dataset, model) pair are submitted in a single vLLM run with the
model loaded once. Expected volume: ~400–800 rows → minutes on one GPU.

---

## 6. Evaluation (`evaluate_arbitration.py`) — Deliverable 4

### A. Arbitration accuracy on the contradiction set

LLM verdict vs true label, compared against baselines:

| Baseline | Definition |
|---|---|
| always-trust-XGB | take the stage-1 call on every contradiction row |
| always-trust-CNN | take the stage-2 call |
| trust-more-confident | pick whichever model has the larger \|p − 0.5\| |
| cascade blend | the current `0.5·p_xgb + 0.5·p_cnn ≥ threshold` decision |
| majority class | predict the majority label of the contradiction set |

Report accuracy, precision, recall, F1 on the contradiction subset for the LLM and each baseline,
per dataset, per model (Foundation-Sec vs Llama-3.1).

### B. Net effect on the full cascade pipeline

Substitute the LLM verdict for the cascade's decision **only on contradiction rows**, then recompute
F1 / FPR / Recall over the whole test population. Report ΔF1 / ΔFPR / ΔRecall vs cascade-alone, both
per-fold and pooled, in the **same table shape** as `hybrid_cascade`'s CV results so the report can
place them side by side. This subsumes any separate "cascade uncertain-band vs LLM" comparison,
since the trigger is now the cascade contradiction itself.

### C. Security-tuned vs general

Deltas on A and B between `Foundation-Sec-8B-Instruct` and `Llama-3.1-8B-Instruct`. This is the
headline Deliverable-4 result: does cybersecurity pre-training measurably change arbitration quality
on these edge cases?

### D. Reasoning quality (not just outcome)

- `key_features` overlap with the auto-flagged "notable deviations" and with `error_analysis.py`'s
  discriminative features — hit-rate and Jaccard.
- `confidence` calibration — reliability curve across confidence bins + Expected Calibration Error.
- `agrees_with` self-consistency — does the model correctly state which base model it sided with.
- Faithfulness spot-check on N = 20 sampled rows — the cited `key_features` actually hold the values
  the rationale claims.

### E. Robustness

JSON parse-failure rate (before and after the repair retry), enum-violation rate, unknown-feature
rate. Reported separately from accuracy.

### F. Per-technique breakdown

Attacks only, using the `technique` column **post-hoc only** — which specific missed techniques the
LLM recovers relative to trusting XGBoost or CNN alone.

### G. Per-dataset expectation (stated up front in the report)

The LLM is expected to help **CAM-LDS v3** (XGBoost vs CNN disagreement there is genuinely
complementary — ~2.8% of rows, ~50/50 on which model is right) and to help **little on
CasinoLimit** (disagreement there is dominated by CNN's known instability, F1 0.518 ± 0.369). That
asymmetry is a reportable finding consistent with the cascade module's own docstring, not a failure
of the LLM layer.

### Charts

- Grouped bar: accuracy on the contradiction set — LLM-sec / LLM-general / five baselines, per dataset.
- ΔF1 / ΔFPR waterfall: cascade-alone → + LLM-general → + LLM-sec, per dataset.
- Reliability curve (confidence calibration).
- `key_features` frequency vs `error_analysis.py` discriminators.

### Outputs

`results/current/llm_triage/<ds>/`:
`arbitration_metrics.csv`, `pipeline_effect.csv`, `calibration.csv`, `reasoning_quality.csv`,
`*.png`, and `report_section.md` — prose plus tables in the honest-reporting style of
`docs/milestone3/CHANGELOG.md`, ready to paste into the Milestone 3 report.

---

## 7. Testing (TDD — tests written before implementation)

| Module | Tests |
|---|---|
| `cascade_contradiction_dump` | synthetic fold predictions → correct `(routed) ∩ (pred_xgb != pred_cnn)` mask; routing band boundary values (exactly `0.3` / `0.7`) included; disagreement-direction tag correct; empty contradiction set handled; single-class (degenerate) fold skipped; stratified subsample keeps every direction × technique and is seed-stable |
| `population_stats` | percentile breakpoints correct on a known array; benign and attack populations computed separately; deterministic output |
| `context_builder` | golden-file test of the full rendered string for a fixed row + fixed percentile table; percentile lookup correctness; "notable deviations" ordering; all 20 features present in `FULL_FEATURE_LIST` order; `technique` never appears in output |
| `prompts` | the v1 schema constant validates as a JSON Schema; the few-shot example's JSON parses and validates against it |
| `llm_client` | mocked backend — valid JSON passes; malformed → repair-retry path exercised; still-malformed → `parse_status="parse_error"`; enum violation and unknown-feature name both caught |
| `evaluate_arbitration` | metric math (accuracy, ΔF1, ECE) on a hand-built confusion table with known expected values |
| smoke (`@pytest.mark.slow`, optional) | 3 real contradiction rows through local Ollama `llama3.1:8b`; asserts schema-valid output only, not correctness |

---

## 8. SLURM deployment

`milestone3_pipeline/run_scripts/run_llm_triage.sh`:

- SLURM array over `{camlds, casino} × {foundation-sec, llama-3.1}` (4 tasks).
- One GPU per task (request ~24 GB), vLLM loads the model once, batches all contradiction rows for
  the dataset.
- `PYTHON=` header note matching the other run scripts (hardcoded conda interpreter path caveat from
  the README).
- `--open-mode=truncate`, `python -u`, `flush=True` on progress prints — same conventions as the
  existing scripts (see CHANGELOG §2).
- If the cluster has no usable GPU partition: set `backend=ollama` and run against a locally served
  Foundation-Sec GGUF; document the substitution.

---

## 9. Disclosures for the Milestone 3 report

1. **Labelled-data-derived context.** The percentile tables and the "known failure patterns" block
   are computed from labelled data. This is **descriptive context, not model fitting** — the LLM
   never sees a row's label and is never trained. Same disclosure class as `IFOREST_NOVELTY_MODE`
   in `config.py`.
2. **Pooled OOF contradiction rows.** The four base models and the cascade are trained leak-free
   per fold, but the percentile tables span the entire dataset. Stated explicitly.
3. **Decoding determinism.** `temperature = 0` + fixed seed, with schema-guided decoding where
   the backend supports it (vLLM `GuidedDecodingParams`, Ollama `format`), backed by JSON-schema
   validation and a single repair retry; residual batch-order non-determinism from vLLM is
   possible and is documented rather than eliminated.
4. **Engine caveats.** Foundation-Sec-8B-Instruct has a 4,096-token context window and no
   post-April-2025 knowledge; the vendor states it is not intended for autonomous security
   decisions without human review. Here it is graded offline against ground truth, never deployed.

---

## 10. Deliverable-4 checklist (from the brief)

| Brief requirement | Where satisfied |
|---|---|
| Chosen open-source LLM + deployment constraints | §2 (engine), §5, §8 |
| Exact prompt-engineering templates used for triage | §4.2–4.6 (versioned constants in `prompts.py`) |
| LLM performs structured chain-of-thought over the security telemetry | §4.3 rubric, §4.6 `rationale_steps` |
| Pipeline formats and passes anomalous feature context / log strings to the LLM | §4.4 context window, `context_builder.py` |
| Final, explainable classification decision | §4.6 schema (`verdict` + `rationale_steps` + `key_features`) |
| Acts as arbitrator on direct disagreements between models | §1.1, §3.1 `cascade_contradiction_dump.py` |
| Evaluation of reasoning accuracy on edge cases | §6 A–G, `evaluate_arbitration.py` |
| Software block diagram + code for cascading + LLM-arbitration logic | §3.3 data flow (diagram to be drawn in the report), whole `llm_triage/` package |
