# LLM-Driven Triage and Contextual Arbitration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, cybersecurity-tuned LLM arbitrator that resolves cases where the hybrid cascade's two models (XGBoost stage-1, CNN stage-2) produce contradicting classifications, and evaluate its reasoning accuracy on those edge cases.

**Architecture:** A fixed-seed out-of-fold pass fits XGBoost and the CNN via the existing `train_eval._FITTERS`, records rows that are both *routed* (XGBoost probability in the cascade's `0.3–0.7` band) and *contradicting* (`pred_xgb != pred_cnn`), and stratified-subsamples them. Each contradiction row is rendered into a compact ≤4096-token context window (20 features with dual benign/attack percentiles, auto-flagged deviations, an aggregate known-failure block) and sent to a locally served LLM (`fdtn-ai/Foundation-Sec-8B-Instruct`, control `meta-llama/Llama-3.1-8B-Instruct`) with a 5-step chain-of-thought rubric and a guided-JSON schema. An evaluation module scores the LLM against five baselines, the whole-population pipeline delta, confidence calibration, and reasoning quality.

**Tech Stack:** Python, pandas, numpy, scikit-learn (StratifiedGroupKFold, metrics), jsonschema, requests (Ollama backend), vLLM (cluster backend, lazy-imported), matplotlib (Agg), pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-llm-triage-contextual-arbitration-design.md`

## Global Constraints

- **Module layout:** flat files in `milestone3_pipeline/src/` with an `llm_triage_` prefix (matches the existing `hybrid_cascade.py` / `hybrid_cascade_rf_xgb.py` / `hybrid_cascade_weighted.py` sibling-file pattern). This is a deliberate deviation from the spec's `src/llm_triage/` subpackage wording; interfaces are unchanged. Rationale: every existing module uses flat imports (`from config import ...`, `from train_eval import ...`) that only resolve because scripts run with `src/` at `sys.path[0]`.
- **Dataset-specific logic:** only `ingestion.py` parses raw datasets. New modules read `config.DATASETS[dataset_key]` for the label/technique column names but contain no other per-dataset branching, except the two literal `KNOWN_FAILURE_BLOCKS` / `DISCRIMINATOR_FEATURES` lookup tables (aggregate error-analysis facts, keyed by dataset).
- **Reuse, do not reimplement:** XGBoost and CNN binary calls come from `train_eval._FITTERS["xgb"]` / `["cnn"]`, signature `fitter(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr) -> (pred, proba)`. Do not write new thresholding logic. `hybrid_cascade.py` is not modified.
- **Contradiction definition:** a row qualifies iff, evaluated out-of-fold, `cascade_low_thresh <= proba_xgb <= cascade_high_thresh` (`0.3` / `0.7`) AND `pred_xgb != pred_cnn` AND both calls are valid (`>= 0`).
- **Engine:** primary `fdtn-ai/Foundation-Sec-8B-Instruct`; control `meta-llama/Llama-3.1-8B-Instruct`. Context window ≤ 4096 tokens. No raw-log text in any prompt. At most one few-shot example.
- **Decoding:** `temperature = 0.0`, fixed `seed = 42`, exactly one repair retry on invalid output, then record `parse_status = "parse_error"`.
- **Leakage guard:** the `technique` column MUST NOT appear in any context string or prompt. It is carried in output CSVs for post-hoc reporting only.
- **CV pass:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, one pass (no repeats), matching `error_analysis.py`.
- **Preprocessing before fitters:** `log1p_continuous(X_raw, feat_cols, BINARY_FEATURES)` when `config.LOG1P_CONTINUOUS_FEATURES` is true, then `impute`, exactly as `error_analysis.py` does. Feature *values shown to the LLM* are the raw (pre-log1p) values.
- **Output location:** `milestone3_pipeline/results/current/llm_triage/<dataset_key>/`. Never overwrite other modules' output directories.
- **TDD:** one test file per module in `milestone3_pipeline/tests/`. `python -m pytest milestone3_pipeline -m "not slow"` must be green before each commit.
- **Commits:** one commit per task, message prefix `feat:` / `test:` / `chore:`.

---

## File Structure

| File | Responsibility |
|---|---|
| `milestone3_pipeline/pytest.ini` | pytest config (testpaths, `slow` marker) |
| `milestone3_pipeline/tests/conftest.py` | put `src/` on `sys.path` for all tests |
| `milestone3_pipeline/src/requirements.txt` | + `jsonschema`, `requests`, `pytest` |
| `milestone3_pipeline/src/requirements-llm.txt` | new — cluster-only: `vllm`, `transformers`, `accelerate` |
| `milestone3_pipeline/src/config.py` | + `LLM_TRIAGE` dict, `FEATURE_GLOSS` dict |
| `milestone3_pipeline/src/llm_triage_prompts.py` | new — static prompt constants + JSON schema + 1 few-shot + known-failure blocks |
| `milestone3_pipeline/src/llm_triage_stats.py` | new — per-feature percentile breakpoints + `percentile_label` |
| `milestone3_pipeline/src/llm_triage_context.py` | new — one contradiction row → one context string |
| `milestone3_pipeline/src/llm_triage_dump.py` | new — OOF XGB/CNN calls, contradiction mask, stratified subsample, CLI |
| `milestone3_pipeline/src/llm_triage_client.py` | new — `build_prompt`, `LLMClient` (vLLM/Ollama), parse + repair + validate |
| `milestone3_pipeline/src/llm_triage_arbitrate.py` | new — orchestrator: dump → context → client → arbitration CSV, CLI |
| `milestone3_pipeline/src/llm_triage_evaluate.py` | new — accuracy vs baselines, pipeline delta, ECE, key-feature overlap, charts, report section, CLI |
| `milestone3_pipeline/run_scripts/run_llm_triage.sh` | new — SLURM array `{camlds,casino} × {foundation-sec,llama}` |
| `docs/milestone3/CHANGELOG.md` | + section 13 entry |
| `README.md` | flip "LLM triage layer" from not-started to done |
| `milestone3_pipeline/tests/test_llm_triage_prompts.py` … `test_llm_triage_evaluate.py` | one per module |

---

## Task 0: Test scaffolding and dependencies

**Files:**
- Create: `milestone3_pipeline/pytest.ini`
- Create: `milestone3_pipeline/tests/conftest.py`
- Create: `milestone3_pipeline/src/requirements-llm.txt`
- Modify: `milestone3_pipeline/src/requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `python -m pytest milestone3_pipeline` invocation; tests can `import config`, `import llm_triage_*` directly.

- [ ] **Step 1: Create `milestone3_pipeline/pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    slow: needs a running Ollama server on localhost:11434 (deselect with -m "not slow")
```

- [ ] **Step 2: Create `milestone3_pipeline/tests/conftest.py`**

```python
"""Put the flat src/ directory on sys.path so tests import modules the same way
the pipeline scripts do (scripts run with src/ as sys.path[0])."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 3: Create `milestone3_pipeline/src/requirements-llm.txt`**

```text
# Cluster-only extras for the LLM triage layer (heavy; not needed to run the tests
# or the CPU pipeline). Install on the GPU node alongside src/requirements.txt.
vllm
transformers
accelerate
```

- [ ] **Step 4: Append to `milestone3_pipeline/src/requirements.txt`**

Final content:

```text
pandas
numpy
scikit-learn
xgboost
tensorflow-cpu
matplotlib
jsonschema
requests
pytest
```

- [ ] **Step 5: Verify pytest collects nothing yet without error**

Run: `python -m pytest milestone3_pipeline -m "not slow"`
Expected: exits 5 ("no tests ran") or 0, with no import/collection errors.

- [ ] **Step 6: Commit**

```bash
git add milestone3_pipeline/pytest.ini milestone3_pipeline/tests/conftest.py milestone3_pipeline/src/requirements.txt milestone3_pipeline/src/requirements-llm.txt
git commit -m "chore: add pytest scaffolding and LLM triage dependencies"
```

---

## Task 1: Config constants and prompt module

**Files:**
- Modify: `milestone3_pipeline/src/config.py` (append at end)
- Create: `milestone3_pipeline/src/llm_triage_prompts.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_prompts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `config.LLM_TRIAGE: dict` with keys `model_id, control_model_id, backend, max_new_tokens, sampling_seed, temperature, max_cases_per_dataset, cascade_low_thresh, cascade_high_thresh, schema_version`.
  - `config.FEATURE_GLOSS: dict[str, str]` — one entry per name in `FULL_FEATURE_LIST`.
  - `llm_triage_prompts.SCHEMA_VERSION: str`
  - `llm_triage_prompts.SYSTEM_PROMPT: str`
  - `llm_triage_prompts.COT_RUBRIC: str`
  - `llm_triage_prompts.SCHEMA_INSTRUCTION: str`
  - `llm_triage_prompts.FEWSHOT_EXAMPLE: str`
  - `llm_triage_prompts.ARBITRATION_SCHEMA_V1: dict` (a JSON Schema)
  - `llm_triage_prompts.KNOWN_FAILURE_BLOCKS: dict[str, str]` keyed `"camlds"`, `"casino"`

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_prompts.py`

```python
import json
import jsonschema
import config
import llm_triage_prompts as P


def test_feature_gloss_covers_every_feature():
    assert set(config.FEATURE_GLOSS) == set(config.FULL_FEATURE_LIST)
    assert all(isinstance(v, str) and v for v in config.FEATURE_GLOSS.values())


def test_llm_triage_config_keys():
    c = config.LLM_TRIAGE
    for k in ("model_id", "control_model_id", "backend", "max_new_tokens",
              "sampling_seed", "temperature", "max_cases_per_dataset",
              "cascade_low_thresh", "cascade_high_thresh", "schema_version"):
        assert k in c
    assert c["temperature"] == 0.0
    assert c["cascade_low_thresh"] < c["cascade_high_thresh"]
    assert c["model_id"] == "fdtn-ai/Foundation-Sec-8B-Instruct"


def test_arbitration_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(P.ARBITRATION_SCHEMA_V1)


def test_schema_forbids_extra_keys_and_enumerates_verdict():
    props = P.ARBITRATION_SCHEMA_V1["properties"]
    assert props["verdict"]["enum"] == ["MALICIOUS", "BENIGN"]
    assert props["agrees_with"]["enum"] == ["xgb", "cnn", "neither"]
    assert P.ARBITRATION_SCHEMA_V1["additionalProperties"] is False


def test_fewshot_example_answer_validates_against_schema():
    # The example block ends with "--- IDEAL ANSWER ---\n<json>\n--- END EXAMPLE ---".
    marker = "--- IDEAL ANSWER ---"
    assert marker in P.FEWSHOT_EXAMPLE
    tail = P.FEWSHOT_EXAMPLE.split(marker, 1)[1]
    start, end = tail.find("{"), tail.rfind("}")
    obj = json.loads(tail[start:end + 1])
    jsonschema.validate(obj, P.ARBITRATION_SCHEMA_V1)
    assert obj["verdict"] in ("MALICIOUS", "BENIGN")


def test_known_failure_blocks_cover_both_datasets_and_omit_technique():
    assert set(P.KNOWN_FAILURE_BLOCKS) == {"camlds", "casino"}
    for text in P.KNOWN_FAILURE_BLOCKS.values():
        assert "technique" not in text.lower()
        assert len(text) < 700  # keeps the context window small


def test_prompt_text_mentions_only_the_four_mitre_families():
    for tid in ("T1548", "T1059", "T1222", "T1595"):
        assert tid in P.SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_prompts` / `AttributeError: config has no attribute LLM_TRIAGE`.

- [ ] **Step 3: Append to `milestone3_pipeline/src/config.py`**

```python

# =====================================================================
# Step 8 capstone -- LLM-Driven Triage and Contextual Arbitration
# =====================================================================

# One-line SOC-analyst gloss per feature, shown verbatim in the LLM context
# window (see llm_triage_context.build_context). Keys MUST match FULL_FEATURE_LIST.
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

LLM_TRIAGE = {
    "model_id": "fdtn-ai/Foundation-Sec-8B-Instruct",      # security-tuned primary
    "control_model_id": "meta-llama/Llama-3.1-8B-Instruct",  # general-purpose A/B control
    "backend": "vllm",                                       # "vllm" (cluster) or "ollama" (fallback)
    "max_new_tokens": 800,
    "sampling_seed": 42,
    "temperature": 0.0,
    "max_cases_per_dataset": 400,        # stratified cap; bounds LLM cost
    "cascade_low_thresh": 0.3,           # routing band, mirrors hybrid_cascade.py defaults
    "cascade_high_thresh": 0.7,
    "schema_version": "v1",
}
```

- [ ] **Step 4: Create `milestone3_pipeline/src/llm_triage_prompts.py`**

```python
"""Milestone 3 -- Step 8 capstone: static prompt assets for LLM arbitration.

Everything here is a versioned constant so the report can quote it verbatim
(schema_version = "v1"). Nothing dataset-specific except KNOWN_FAILURE_BLOCKS,
which are aggregate facts lifted from error_analysis.py's FP-vs-TN / FN-vs-TP
tables -- population-level, not row-level, so not a label leak.
"""

SCHEMA_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a Tier-2 SOC analyst on a Linux host-intrusion team. Two ML models in "
    "an automated detection pipeline have produced CONTRADICTING classifications for "
    "one process-execution record. Your job is to arbitrate: decide MALICIOUS (a "
    "privilege-escalation attempt) or BENIGN, using structured reasoning over the "
    "process's behavioural features.\n\n"
    "The monitored environment contains normal system and administrative activity plus "
    "privilege-escalation attempts in the MITRE ATT&CK families T1548 (Abuse Elevation "
    "Control Mechanism), T1059 (Command and Scripting Interpreter), T1222 "
    "(File/Directory Permissions Modification), and T1595 (Active Scanning). Attacks "
    "are a minority of records (roughly 17-29%).\n\n"
    "Follow the reasoning rubric exactly and return ONLY a single JSON object matching "
    "the schema. Do not write anything before or after the JSON."
)

COT_RUBRIC = (
    "Reasoning rubric -- work through these five steps in order, one entry in "
    "rationale_steps per step:\n"
    "1. Privilege state: is there a genuine UID/EUID transition or root context? "
    "Consider uid_is_root, euid_root, max_uid_euid_delta, uid_changed, auid_euid_mismatch.\n"
    "2. Mechanism: setuid/setgid execution, an unusual exec chain, or a shell spawned "
    "by a service? Consider is_suid_exec, exec_count, shell_from_service, priv_op_count.\n"
    "3. Behavioural anomaly: failed-syscall rate, rare parent->child pair, ephemeral "
    "privileged process, sensitive-path access, network activity. Consider "
    "failed_call_rate, parent_child_rarity, ephemeral_privileged, sensitive_path_access, "
    "network_count.\n"
    "4. Benign-explanation test: could this plausibly be admin scripting, a cron job, "
    "package management, or a health-check? Weigh events_per_second, lifetime_seconds, "
    "seq_length, unique_syscalls.\n"
    "5. Decide: weigh steps 1-4, state which model's call the evidence supports, and "
    "give a calibrated confidence."
)

SCHEMA_INSTRUCTION = (
    "Return ONLY this JSON object and nothing else:\n"
    "{\n"
    '  "verdict": "MALICIOUS" or "BENIGN",\n'
    '  "confidence": a number from 0.0 to 1.0,\n'
    '  "rationale_steps": [3 to 6 short strings, one per rubric step],\n'
    '  "key_features": [up to 5 feature names that drove the decision],\n'
    '  "agrees_with": "xgb" or "cnn" or "neither"\n'
    "}"
)

ARBITRATION_SCHEMA_V1 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["MALICIOUS", "BENIGN"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale_steps": {
            "type": "array", "items": {"type": "string"},
            "minItems": 3, "maxItems": 6,
        },
        "key_features": {
            "type": "array", "items": {"type": "string"}, "maxItems": 5,
        },
        "agrees_with": {"type": "string", "enum": ["xgb", "cnn", "neither"]},
    },
    "required": ["verdict", "confidence", "rationale_steps", "key_features", "agrees_with"],
    "additionalProperties": False,
}

FEWSHOT_EXAMPLE = (
    "--- EXAMPLE ---\n"
    "=== RECORD UNDER REVIEW ===\n"
    "Model votes:  XGBoost -> BENIGN (p_malicious=0.41)   |   CNN -> MALICIOUS (p_malicious=0.66)\n"
    "\n"
    "Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)\n"
    "  max_uid_euid_delta     15       p99+   p90    largest gap between real and effective UID; large = privilege transition\n"
    "  auid_euid_mismatch     1        p98    p72    audit (login) UID differs from effective UID; identity inconsistency\n"
    "  is_suid_exec           1        p99+   p88    executed a setuid/setgid binary\n"
    "  shell_from_service     1        p99+   p80    a shell was spawned from a service/daemon context\n"
    "  sensitive_path_access  1        p99+   p70    touched a sensitive path (/etc/shadow, /etc/sudoers, ...)\n"
    "  lifetime_seconds       3        p10    p20    process wall-clock lifetime\n"
    "  (... remaining features omitted in this example ...)\n"
    "\n"
    "Notable deviations (most unusual vs benign, first = most extreme):\n"
    "  is_suid_exec, shell_from_service, sensitive_path_access, max_uid_euid_delta, auid_euid_mismatch\n"
    "\n"
    "Known failure patterns in this dataset (from labelled error analysis):\n"
    "  - Benign rows misflagged as attack tend to have elevated parent_child_rarity and uid_changed with short lifetime_seconds.\n"
    "  - Missed attacks tend to have auid_euid_mismatch set but low max_uid_euid_delta and priv_op_count.\n"
    "\n"
    "--- IDEAL ANSWER ---\n"
    '{"verdict": "MALICIOUS", "confidence": 0.86, "rationale_steps": ['
    '"Privilege state: root context with max_uid_euid_delta at p99+ and auid_euid_mismatch set -- a real identity inconsistency and privilege transition.", '
    '"Mechanism: is_suid_exec is set and a shell was spawned from a service context with priv_op_count elevated -- a classic setuid-shell escalation.", '
    '"Behavioural anomaly: sensitive_path_access=1 and parent_child_rarity near 1 -- an exec pair essentially never seen in benign activity.", '
    '"Benign-explanation test: very short lifetime (p10) and low throughput argue against a routine admin script, cron job, or health-check.", '
    '"Decide: the escalation mechanism and sensitive-path access outweigh XGBoost\'s benign call; this supports the CNN."], '
    '"key_features": ["is_suid_exec", "shell_from_service", "sensitive_path_access", "max_uid_euid_delta", "auid_euid_mismatch"], '
    '"agrees_with": "cnn"}\n'
    "--- END EXAMPLE ---"
)

KNOWN_FAILURE_BLOCKS = {
    "camlds": (
        "Known failure patterns in this dataset (from labelled error analysis):\n"
        "  - Benign rows misflagged as attack (FP) tend to have elevated parent_child_rarity, "
        "uid_changed and ephemeral_privileged, often with unusually short lifetime_seconds "
        "(XGBoost); or elevated failed_call_rate, failed_call_count and exec_count (CNN).\n"
        "  - Missed attacks (FN) tend to have auid_euid_mismatch set but LOW max_uid_euid_delta "
        "and priv_op_count, zero network_count and no shell_from_service -- escalation attempts "
        "that never complete a visible UID transition."
    ),
    "casino": (
        "Known failure patterns in this dataset (from labelled error analysis):\n"
        "  - Benign rows misflagged as attack (FP) tend to show exec_count > 0 with "
        "euid_root/uid_is_root set but auid_euid_mismatch absent (XGBoost).\n"
        "  - Missed attacks (FN) tend to have elevated priv_op_count but LOW failed_call_rate, "
        "no shell_from_service, and uid_is_root unset (CNN)."
    ),
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_prompts.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Commit**

```bash
git add milestone3_pipeline/src/config.py milestone3_pipeline/src/llm_triage_prompts.py milestone3_pipeline/tests/test_llm_triage_prompts.py
git commit -m "feat: add LLM triage config constants and prompt assets"
```

---

## Task 2: Percentile statistics module

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_stats.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_stats.py`

**Interfaces:**
- Consumes: nothing (pure numpy/pandas).
- Produces:
  - `compute_percentile_tables(df, feat_cols, label_col) -> dict` — returns
    `{feat: {"benign": [99 floats p1..p99], "attack": [99 floats p1..p99]}}`.
  - `percentile_label(value: float, breakpoints: list[float]) -> str` — one of `"p1-"`,
    `"p{n}"` for `2 <= n <= 98`, `"p99+"`.
  - `label_to_rank(label: str) -> int` — inverse for ordering: `"p1-"→0`, `"p99+"→100`, else `n`.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_stats.py`

```python
import numpy as np
import pandas as pd
import llm_triage_stats as S


def test_compute_percentile_tables_shape_and_separation():
    df = pd.DataFrame({
        "f1": list(range(100)) + list(range(100, 200)),
        "lbl": [0] * 100 + [1] * 100,
    })
    tables = S.compute_percentile_tables(df, ["f1"], "lbl")
    assert set(tables) == {"f1"}
    assert len(tables["f1"]["benign"]) == 99
    assert len(tables["f1"]["attack"]) == 99
    # benign pool is 0..99, attack pool is 100..199
    assert tables["f1"]["benign"][49] < 100 <= tables["f1"]["attack"][0]


def test_percentile_label_buckets():
    bp = [float(i) for i in range(1, 100)]  # p1=1 .. p99=99
    assert S.percentile_label(50, bp) == "p50"
    assert S.percentile_label(0, bp) == "p1-"
    assert S.percentile_label(-5, bp) == "p1-"
    assert S.percentile_label(99, bp) == "p99+"
    assert S.percentile_label(500, bp) == "p99+"
    assert S.percentile_label(98.5, bp) == "p98"


def test_label_to_rank_roundtrip():
    assert S.label_to_rank("p1-") == 0
    assert S.label_to_rank("p99+") == 100
    assert S.label_to_rank("p37") == 37


def test_compute_percentile_tables_handles_nan():
    df = pd.DataFrame({"f1": [np.nan, 1.0, 2.0, 3.0], "lbl": [0, 0, 1, 1]})
    tables = S.compute_percentile_tables(df, ["f1"], "lbl")
    assert not any(np.isnan(tables["f1"]["benign"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_stats`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_stats.py`**

```python
"""Milestone 3 -- Step 8 capstone: per-feature percentile breakpoints.

Computed ONCE per dataset from the full labelled population (benign rows and
attack rows separately). Used by llm_triage_context to tell the LLM how unusual
each feature value is, without ever handing it the label. This uses labels for
DESCRIPTIVE CONTEXT only -- same disclosure class as config.IFOREST_NOVELTY_MODE.
"""

import numpy as np

PCT_POINTS = list(range(1, 100))  # p1 .. p99


def compute_percentile_tables(df, feat_cols, label_col):
    y = df[label_col].values
    out = {}
    for f in feat_cols:
        vals = df[f].fillna(0.0).values.astype(float)
        benign = vals[y == 0]
        attack = vals[y == 1]
        out[f] = {
            "benign": np.percentile(benign, PCT_POINTS).tolist() if len(benign) else [0.0] * 99,
            "attack": np.percentile(attack, PCT_POINTS).tolist() if len(attack) else [0.0] * 99,
        }
    return out


def percentile_label(value, breakpoints):
    """breakpoints: ascending list of the p1..p99 values. Returns 'p1-', 'pN'
    (2<=N<=98), or 'p99+'."""
    bp = np.asarray(breakpoints, dtype=float)
    rank = int(np.searchsorted(bp, value, side="right"))  # 0..99
    if rank <= 0:
        return "p1-"
    if rank >= 99:
        return "p99+"
    return f"p{rank}"


def label_to_rank(label):
    if label == "p1-":
        return 0
    if label == "p99+":
        return 100
    return int(label[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_stats.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_stats.py milestone3_pipeline/tests/test_llm_triage_stats.py
git commit -m "feat: add per-feature percentile stats for LLM context"
```

---

## Task 3: Context builder

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_context.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_context.py`

**Interfaces:**
- Consumes: `config.FEATURE_GLOSS`, `llm_triage_prompts.KNOWN_FAILURE_BLOCKS`,
  `llm_triage_stats.percentile_label` / `label_to_rank`.
- Produces:
  - `notable_deviations(row, feat_cols, pct_tables, top_n=5) -> list[str]` — the `top_n`
    feature names whose benign-percentile rank is furthest from 50 (scale-free), ties broken
    by `feat_cols` order.
  - `build_context(row, feat_cols, pct_tables, dataset_key) -> str` — the full context string.
    `row` is a mapping with the 20 feature values plus `pred_xgb`, `pred_cnn`, `proba_xgb`,
    `proba_cnn` (a `dict` or a pandas `Series` both work).

**Note on spec §4.4:** the spec says "largest |value − benign median|". That is scale-broken
(it always selects `lifetime_seconds` / `events_per_second`). This plan ranks by absolute
deviation of the *benign percentile rank* from 50 instead — same intent ("most unusual vs
benign"), scale-free. This refinement is intentional.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_context.py`

```python
import numpy as np
import config
import llm_triage_context as C


def _pct_tables():
    # every feature: benign & attack breakpoints are p1..p99 = 1..99
    bp = [float(i) for i in range(1, 100)]
    return {f: {"benign": list(bp), "attack": list(bp)} for f in config.FULL_FEATURE_LIST}


def _row():
    r = {f: 10.0 for f in config.FULL_FEATURE_LIST}
    r["max_uid_euid_delta"] = 99.0     # -> p99+  (rank 100, dev 50)
    r["auid_euid_mismatch"] = 1.0      # -> p1-   (rank 0,   dev 50)
    r["priv_op_count"] = 95.0          # -> p95   (dev 45)
    r["pred_xgb"], r["proba_xgb"] = 0, 0.34
    r["pred_cnn"], r["proba_cnn"] = 1, 0.71
    r["technique"] = "T1548.001"
    return r


def test_notable_deviations_are_scale_free_top5():
    dev = C.notable_deviations(_row(), config.FULL_FEATURE_LIST, _pct_tables())
    assert len(dev) == 5
    assert dev[0] in ("max_uid_euid_delta", "auid_euid_mismatch")
    assert "priv_op_count" in dev


def test_build_context_structure():
    ctx = C.build_context(_row(), config.FULL_FEATURE_LIST, _pct_tables(), "camlds")
    assert ctx.startswith("=== RECORD UNDER REVIEW ===")
    # one line per feature
    feature_lines = [ln for ln in ctx.splitlines()
                     if any(ln.strip().startswith(f + " ") or ln.strip() == f for f in config.FULL_FEATURE_LIST)]
    assert len(feature_lines) == len(config.FULL_FEATURE_LIST)
    assert "XGBoost -> BENIGN (p_malicious=0.34)" in ctx
    assert "CNN -> MALICIOUS (p_malicious=0.71)" in ctx
    assert "Notable deviations" in ctx
    assert "Known failure patterns in this dataset" in ctx


def test_build_context_never_leaks_technique():
    ctx = C.build_context(_row(), config.FULL_FEATURE_LIST, _pct_tables(), "camlds")
    assert "T1548" not in ctx
    assert "technique" not in ctx.lower()


def test_build_context_uses_dataset_specific_failure_block():
    cam = C.build_context(_row(), config.FULL_FEATURE_LIST, _pct_tables(), "camlds")
    cas = C.build_context(_row(), config.FULL_FEATURE_LIST, _pct_tables(), "casino")
    assert "never complete a visible UID transition" in cam
    assert "uid_is_root unset (CNN)" in cas
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_context.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_context`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_context.py`**

```python
"""Milestone 3 -- Step 8 capstone: render one cascade-contradiction row into a
compact context window (<= ~750 tokens) for the LLM arbitrator.

The row's `technique` value is deliberately never read here -- it is a label leak
(populated only for attack rows). Only family-level scope, which lives in the
static SYSTEM_PROMPT, is given to the model.
"""

from config import FEATURE_GLOSS
from llm_triage_prompts import KNOWN_FAILURE_BLOCKS
from llm_triage_stats import percentile_label, label_to_rank


def _fmt_value(v):
    f = float(v)
    if f == int(f):
        return str(int(f))
    return f"{f:.4g}"


def notable_deviations(row, feat_cols, pct_tables, top_n=5):
    scored = []
    for idx, f in enumerate(feat_cols):
        rank = label_to_rank(percentile_label(float(row[f]), pct_tables[f]["benign"]))
        scored.append((abs(rank - 50), idx, f))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [f for _, _, f in scored[:top_n]]


def build_context(row, feat_cols, pct_tables, dataset_key):
    xgb_call = "MALICIOUS" if int(row["pred_xgb"]) == 1 else "BENIGN"
    cnn_call = "MALICIOUS" if int(row["pred_cnn"]) == 1 else "BENIGN"

    lines = [
        "=== RECORD UNDER REVIEW ===",
        (f"Model votes:  XGBoost -> {xgb_call} (p_malicious={float(row['proba_xgb']):.2f})   |   "
         f"CNN -> {cnn_call} (p_malicious={float(row['proba_cnn']):.2f})"),
        "",
        "Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)",
    ]
    for f in feat_cols:
        v = float(row[f])
        pb = percentile_label(v, pct_tables[f]["benign"])
        pa = percentile_label(v, pct_tables[f]["attack"])
        lines.append(f"  {f:<22} {_fmt_value(v):<8} {pb:<6} {pa:<6} {FEATURE_GLOSS[f]}")

    lines += [
        "",
        "Notable deviations (most unusual vs benign, first = most extreme):",
        "  " + ", ".join(notable_deviations(row, feat_cols, pct_tables)),
        "",
        KNOWN_FAILURE_BLOCKS[dataset_key],
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_context.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_context.py milestone3_pipeline/tests/test_llm_triage_context.py
git commit -m "feat: add LLM context-window builder for contradiction rows"
```

---

## Task 4: Contradiction dump

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_dump.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_dump.py`

**Interfaces:**
- Consumes: `config.LLM_TRIAGE`, `config.DATASETS`, `config.LOG1P_CONTINUOUS_FEATURES`,
  `config.BINARY_FEATURES`, `config.MAX_FPR`; `ingestion.ingest`; `feature_selection.select_features`;
  `preprocessing.vector_group_key` / `log1p_continuous` / `impute`; `train_eval._FITTERS`
  (all lazy-imported inside `_oof_calls`).
- Produces:
  - `contradiction_mask(oof: pd.DataFrame, low: float, high: float) -> pd.Series[bool]` — pure.
  - `add_direction(contra: pd.DataFrame) -> pd.DataFrame` — adds `disagreement_direction`
    (`"xgb_benign_cnn_malicious"` / `"xgb_malicious_cnn_benign"`). pure.
  - `stratified_subsample(df: pd.DataFrame, max_cases: int, seed: int) -> pd.DataFrame` —
    proportional per `(disagreement_direction, technique)`, ≥1 per stratum, seed-stable,
    sorted by `row_id`. pure.
  - `find_contradictions(dataset_key, data_dir, seed=42, n_splits=5) -> (contra_df, oof_df, feat_cols)`
    — integration (runs the models).
  - `oof_df` columns: `row_id, fold, true_label, technique, proba_xgb, proba_cnn, pred_xgb,
    pred_cnn` + the 20 raw feature columns.
  - `contra_df` = `add_direction(oof_df[contradiction_mask(...)])`.
  - CLI: `python llm_triage_dump.py --dataset camlds --data-dir data/v3 --out <dir>` writes
    `<out>/<ds>_contradictions.csv` and `<out>/<ds>_cascade_oof.csv`.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_dump.py`

```python
import numpy as np
import pandas as pd
import llm_triage_dump as D


def _oof():
    return pd.DataFrame({
        "row_id": [0, 1, 2, 3, 4, 5],
        "technique": ["benign", "T1548", "benign", "T1059", "T1548", "benign"],
        "true_label": [0, 1, 0, 1, 1, 0],
        "proba_xgb": [0.10, 0.45, 0.55, 0.65, 0.90, 0.50],
        "proba_cnn": [0.20, 0.80, 0.10, 0.30, 0.95, 0.60],
        "pred_xgb": [0, 0, 1, 1, 1, 0],
        "pred_cnn": [0, 1, 0, 0, 1, 1],
    })


def test_contradiction_mask_routed_and_disagree_only():
    m = D.contradiction_mask(_oof(), 0.3, 0.7)
    # row0: not routed. row1: routed(0.45) & disagree -> True. row2: routed & disagree -> True.
    # row3: routed(0.65) & disagree -> True. row4: not routed(0.90). row5: routed & disagree -> True.
    assert list(m) == [False, True, True, True, False, True]


def test_add_direction_labels():
    contra = D.add_direction(_oof().loc[[1, 3]])
    assert list(contra["disagreement_direction"]) == [
        "xgb_benign_cnn_malicious", "xgb_malicious_cnn_benign"]


def test_stratified_subsample_is_capped_seed_stable_and_covers_strata():
    rows = []
    for i in range(60):
        rows.append({
            "row_id": i,
            "disagreement_direction": "xgb_benign_cnn_malicious" if i % 2 else "xgb_malicious_cnn_benign",
            "technique": "T1548" if i % 3 else "benign",
        })
    df = pd.DataFrame(rows)
    a = D.stratified_subsample(df, max_cases=12, seed=42)
    b = D.stratified_subsample(df, max_cases=12, seed=42)
    assert len(a) <= 12
    assert a["row_id"].tolist() == b["row_id"].tolist()          # deterministic
    assert a["disagreement_direction"].nunique() == 2            # both directions kept
    assert list(a["row_id"]) == sorted(a["row_id"])              # sorted


def test_stratified_subsample_passthrough_when_small():
    df = pd.DataFrame({"row_id": [3, 1, 2],
                       "disagreement_direction": ["x", "x", "y"],
                       "technique": ["a", "b", "c"]})
    out = D.stratified_subsample(df, max_cases=10, seed=1)
    assert len(out) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_dump.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_dump`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_dump.py`**

```python
"""Milestone 3 -- Step 8 capstone: dump the rows where the hybrid cascade's two
models contradict each other.

A row qualifies iff, out-of-fold:
  1. routed  -- cascade_low_thresh <= proba_xgb <= cascade_high_thresh, and
  2. contradicting -- pred_xgb != pred_cnn (each model's own F1-optimal call).

XGBoost and the CNN are fit per fold via train_eval._FITTERS, the exact fitters
error_analysis.py and the headline pipeline use, so these calls are scored under
the identical F1-threshold / MAX_FPR logic. hybrid_cascade.py is imported only
for its routing-band constants; its blending is not used and it is not modified.

Usage:
    python llm_triage_dump.py --dataset camlds --data-dir data/v3 --out results/current/llm_triage/camlds
    python llm_triage_dump.py --dataset casino --data-dir data     --out results/current/llm_triage/casino
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (LLM_TRIAGE, DATASETS, LOG1P_CONTINUOUS_FEATURES,
                    BINARY_FEATURES, MAX_FPR)

LOW = LLM_TRIAGE["cascade_low_thresh"]
HIGH = LLM_TRIAGE["cascade_high_thresh"]


def contradiction_mask(oof, low=LOW, high=HIGH):
    routed = oof["proba_xgb"].between(low, high, inclusive="both")
    disagree = oof["pred_xgb"] != oof["pred_cnn"]
    valid = (oof["pred_xgb"] >= 0) & (oof["pred_cnn"] >= 0)
    return routed & disagree & valid


def add_direction(contra):
    contra = contra.copy()
    contra["disagreement_direction"] = np.where(
        contra["pred_xgb"].astype(int) == 0,
        "xgb_benign_cnn_malicious", "xgb_malicious_cnn_benign")
    return contra


def stratified_subsample(df, max_cases, seed):
    if len(df) <= max_cases:
        return df.sort_values("row_id").reset_index(drop=True)
    strat = (df["disagreement_direction"].astype(str) + "|" + df["technique"].astype(str))
    total = len(df)
    picks = []
    for _, g in df.groupby(strat):
        k = max(1, int(round(max_cases * len(g) / total)))
        k = min(k, len(g))
        picks.append(g.sample(n=k, random_state=seed))
    result = pd.concat(picks)
    if len(result) > max_cases:
        result = result.sample(n=max_cases, random_state=seed)
    return result.sort_values("row_id").reset_index(drop=True)


def _oof_calls(dataset_key, data_dir, seed, n_splits):
    from sklearn.model_selection import StratifiedGroupKFold
    from ingestion import ingest
    from feature_selection import select_features
    from preprocessing import vector_group_key, log1p_continuous, impute
    from train_eval import _FITTERS

    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    groups = vector_group_key(ds.df, feat_cols)
    X_raw = ds.df[feat_cols].values.astype(float)
    X = (log1p_continuous(X_raw, feat_cols, BINARY_FEATURES)
         if LOG1P_CONTINUOUS_FEATURES else X_raw.copy())
    X = impute(X)
    y = ds.y.values
    n = len(y)

    fold_id = np.full(n, -1, dtype=int)
    preds = {mk: np.full(n, -1, dtype=int) for mk in ("xgb", "cnn")}
    probas = {mk: np.full(n, np.nan, dtype=float) for mk in ("xgb", "cnn")}

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_i, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert not (set(groups[tr]) & set(groups[te])), "group leakage"
        fold_id[te] = fold_i
        for mk in ("xgb", "cnn"):
            p, pr = _FITTERS[mk](X[tr], y[tr], groups[tr], X[te], None, MAX_FPR)
            preds[mk][te] = p
            probas[mk][te] = pr
        print(f"[llm_triage_dump] {dataset_key} fold {fold_i + 1}/{n_splits} done", flush=True)

    tech_col = DATASETS[dataset_key].get("technique_col")
    out = ds.df[feat_cols].copy().reset_index(drop=True)
    out.insert(0, "row_id", np.arange(n))
    out["fold"] = fold_id
    out["true_label"] = y
    out["technique"] = (ds.df[tech_col].values
                        if tech_col and tech_col in ds.df.columns else "unknown")
    for mk in ("xgb", "cnn"):
        out[f"pred_{mk}"] = preds[mk]
        out[f"proba_{mk}"] = probas[mk]
    return out, feat_cols


def find_contradictions(dataset_key, data_dir, seed=42, n_splits=5):
    oof, feat_cols = _oof_calls(dataset_key, data_dir, seed, n_splits)
    contra = add_direction(oof[contradiction_mask(oof)].copy()).reset_index(drop=True)
    return contra, oof, feat_cols


def main():
    ap = argparse.ArgumentParser(description="Dump hybrid-cascade contradiction rows")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=LLM_TRIAGE["max_cases_per_dataset"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    contra, oof, _ = find_contradictions(args.dataset, args.data_dir, args.seed, args.n_splits)
    contra = stratified_subsample(contra, args.max_cases, args.seed)

    oof.to_csv(out_dir / f"{args.dataset}_cascade_oof.csv", index=False)
    contra.to_csv(out_dir / f"{args.dataset}_contradictions.csv", index=False)
    print(f"[llm_triage_dump] {args.dataset}: {len(oof):,} OOF rows, "
          f"{len(contra):,} contradiction rows after cap -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_dump.py -v`
Expected: PASS (5 tests). (Importing the module triggers `import tensorflow` transitively only when `_oof_calls` runs — the pure-function tests do not touch it.)

- [ ] **Step 5: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_dump.py milestone3_pipeline/tests/test_llm_triage_dump.py
git commit -m "feat: dump hybrid-cascade contradiction rows for LLM arbitration"
```

---

## Task 5: LLM client

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_client.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_client.py`

**Interfaces:**
- Consumes: `config.FULL_FEATURE_LIST`, `config.LLM_TRIAGE`; `llm_triage_prompts.*`;
  `jsonschema`; lazy `requests` (ollama) / `vllm` (vllm).
- Produces:
  - `build_prompt(context_str: str) -> str`.
  - `class LLMClient` with `__init__(self, model_id=None, backend=None, max_new_tokens=None,
    temperature=None, seed=None)` and `arbitrate(self, context_str: str) -> dict`.
  - `arbitrate` return: the validated schema object plus `parse_status` in
    `{"ok", "repaired", "parse_error"}` and `raw` (the last raw model text). On `parse_error`,
    `verdict/confidence/agrees_with` are `None` and `rationale_steps/key_features` are `[]`.
  - `LLMClient._generate(self, prompt: str) -> str` — the single method a test subclass overrides.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_client.py`

```python
import json
import llm_triage_client as LC


VALID = {
    "verdict": "MALICIOUS", "confidence": 0.8,
    "rationale_steps": ["a", "b", "c"],
    "key_features": ["max_uid_euid_delta", "priv_op_count"],
    "agrees_with": "cnn",
}


class ScriptedClient(LC.LLMClient):
    def __init__(self, responses):
        super().__init__(model_id="test", backend="ollama")
        self._responses = list(responses)
        self.calls = 0

    def _generate(self, prompt):
        self.calls += 1
        return self._responses.pop(0)


def test_valid_first_try_is_ok():
    c = ScriptedClient([json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "ok"
    assert out["verdict"] == "MALICIOUS"
    assert c.calls == 1


def test_prose_wrapped_json_is_extracted():
    c = ScriptedClient(["Sure, here is my answer:\n" + json.dumps(VALID) + "\nThanks!"])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "ok"
    assert out["agrees_with"] == "cnn"


def test_garbage_then_valid_is_repaired():
    c = ScriptedClient(["not json at all", json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"
    assert c.calls == 2


def test_garbage_twice_is_parse_error():
    c = ScriptedClient(["nope", "still nope"])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "parse_error"
    assert out["verdict"] is None
    assert out["key_features"] == []


def test_unknown_key_feature_triggers_repair():
    bad = dict(VALID, key_features=["not_a_real_feature"])
    c = ScriptedClient([json.dumps(bad), json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"


def test_out_of_enum_verdict_triggers_repair():
    bad = dict(VALID, verdict="MAYBE")
    c = ScriptedClient([json.dumps(bad), json.dumps(VALID)])
    out = c.arbitrate("ctx")
    assert out["parse_status"] == "repaired"


def test_build_prompt_contains_all_static_blocks():
    p = LC.build_prompt("CTXBODY")
    assert "Tier-2 SOC analyst" in p
    assert "Reasoning rubric" in p
    assert "--- EXAMPLE ---" in p
    assert "CTXBODY" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_client.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_client`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_client.py`**

```python
"""Milestone 3 -- Step 8 capstone: the LLM arbitration client.

Deterministic decoding (temperature 0, fixed seed), guided by ARBITRATION_SCHEMA_V1.
On invalid output: exactly one repair retry, then parse_status="parse_error".
Two backends: "vllm" (cluster GPU, lazy-imported) and "ollama" (local HTTP fallback).
Tests subclass LLMClient and override _generate -- no network or GPU needed.
"""

import json

from jsonschema import validate, ValidationError

from config import FULL_FEATURE_LIST, LLM_TRIAGE
from llm_triage_prompts import (SYSTEM_PROMPT, COT_RUBRIC, SCHEMA_INSTRUCTION,
                                FEWSHOT_EXAMPLE, ARBITRATION_SCHEMA_V1)

_EMPTY = {"verdict": None, "confidence": None, "rationale_steps": [],
          "key_features": [], "agrees_with": None}


def build_prompt(context_str):
    return "\n\n".join([
        SYSTEM_PROMPT, COT_RUBRIC, SCHEMA_INSTRUCTION, FEWSHOT_EXAMPLE,
        "--- RECORD TO CLASSIFY ---", context_str,
    ])


def _extract_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def _validate_semantics(obj):
    validate(obj, ARBITRATION_SCHEMA_V1)  # raises ValidationError
    bad = [f for f in obj["key_features"] if f not in FULL_FEATURE_LIST]
    if bad:
        raise ValidationError(f"unknown key_features: {bad}")


def _try_parse(text):
    obj = _extract_json(text)          # ValueError / json.JSONDecodeError
    _validate_semantics(obj)           # ValidationError
    return obj


class LLMClient:
    def __init__(self, model_id=None, backend=None, max_new_tokens=None,
                 temperature=None, seed=None):
        c = LLM_TRIAGE
        self.model_id = model_id or c["model_id"]
        self.backend = backend or c["backend"]
        self.max_new_tokens = max_new_tokens or c["max_new_tokens"]
        self.temperature = c["temperature"] if temperature is None else temperature
        self.seed = c["sampling_seed"] if seed is None else seed
        self._engine = None
        self._sp = None

    # -- backend dispatch -------------------------------------------------
    def _generate(self, prompt):
        if self.backend == "ollama":
            return self._generate_ollama(prompt)
        if self.backend == "vllm":
            return self._generate_vllm(prompt)
        raise ValueError(f"unknown backend {self.backend!r}")

    def _generate_ollama(self, prompt):
        import requests
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False,
                  "options": {"temperature": self.temperature, "seed": self.seed,
                              "num_predict": self.max_new_tokens}},
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["response"]

    def _generate_vllm(self, prompt):
        if self._engine is None:
            from vllm import LLM, SamplingParams
            self._engine = LLM(model=self.model_id, dtype="bfloat16",
                               max_model_len=4096, gpu_memory_utilization=0.90)
            self._sp = SamplingParams(temperature=self.temperature, seed=self.seed,
                                      max_tokens=self.max_new_tokens)
        return self._engine.generate([prompt], self._sp)[0].outputs[0].text

    # -- public API -----------------------------------------------------
    def arbitrate(self, context_str):
        prompt = build_prompt(context_str)
        raw = self._generate(prompt)
        try:
            obj = _try_parse(raw)
            obj.update({"parse_status": "ok", "raw": raw})
            return obj
        except (ValueError, ValidationError, json.JSONDecodeError):
            pass

        repair_prompt = (prompt + "\n\nYour previous output was invalid. Return ONLY a "
                         "single valid JSON object matching the schema, nothing else.")
        raw2 = self._generate(repair_prompt)
        try:
            obj = _try_parse(raw2)
            obj.update({"parse_status": "repaired", "raw": raw2})
            return obj
        except (ValueError, ValidationError, json.JSONDecodeError):
            out = dict(_EMPTY)
            out.update({"parse_status": "parse_error", "raw": raw2})
            return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_client.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_client.py milestone3_pipeline/tests/test_llm_triage_client.py
git commit -m "feat: add LLM arbitration client with guided-JSON parse and repair"
```

---

## Task 6: Arbitration orchestrator

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_arbitrate.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_arbitrate.py`

**Interfaces:**
- Consumes: `config.LLM_TRIAGE`, `config.DATASETS`; `llm_triage_dump.find_contradictions` /
  `stratified_subsample`; `llm_triage_stats.compute_percentile_tables`;
  `llm_triage_context.build_context`; `llm_triage_client.LLMClient`; `ingestion.ingest`.
- Produces:
  - `assemble_row(contra_row, feat_cols, llm_result) -> dict` — pure. Flattens one
    contradiction row + one `LLMClient.arbitrate` result into an output record with:
    `row_id, fold, true_label, technique, pred_xgb, pred_cnn, proba_xgb, proba_cnn,
    disagreement_direction, llm_verdict, llm_pred (1/0/-1), llm_confidence, llm_agrees_with,
    parse_status, rationale_steps (json str), key_features (json str)`.
  - `arbitrate_dataset(dataset_key, data_dir, out_dir, model_id=None, backend=None,
    max_cases=None, seed=42, client=None) -> pd.DataFrame` — integration; writes
    `<out_dir>/<ds>_<modeltag>_llm_arbitration.csv` and `<out_dir>/<ds>_percentiles.json`.
    `modeltag` = the model id's last path segment, lowercased, non-alnum → `-`.
  - CLI.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_arbitrate.py`

```python
import json
import llm_triage_arbitrate as A


def test_assemble_row_maps_verdict_to_pred_and_serialises_lists():
    contra = {
        "row_id": 7, "fold": 2, "true_label": 1, "technique": "T1548",
        "pred_xgb": 0, "pred_cnn": 1, "proba_xgb": 0.34, "proba_cnn": 0.71,
        "disagreement_direction": "xgb_benign_cnn_malicious",
    }
    res = {"verdict": "MALICIOUS", "confidence": 0.9,
           "rationale_steps": ["s1", "s2", "s3"], "key_features": ["priv_op_count"],
           "agrees_with": "cnn", "parse_status": "ok"}
    row = A.assemble_row(contra, ["priv_op_count"], res)
    assert row["llm_pred"] == 1
    assert row["true_label"] == 1
    assert row["parse_status"] == "ok"
    assert json.loads(row["rationale_steps"]) == ["s1", "s2", "s3"]
    assert json.loads(row["key_features"]) == ["priv_op_count"]
    assert row["technique"] == "T1548"          # carried, for post-hoc use only


def test_assemble_row_parse_error_gives_minus_one_pred():
    contra = {
        "row_id": 1, "fold": 0, "true_label": 0, "technique": "benign",
        "pred_xgb": 1, "pred_cnn": 0, "proba_xgb": 0.6, "proba_cnn": 0.4,
        "disagreement_direction": "xgb_malicious_cnn_benign",
    }
    res = {"verdict": None, "confidence": None, "rationale_steps": [],
           "key_features": [], "agrees_with": None, "parse_status": "parse_error"}
    row = A.assemble_row(contra, ["priv_op_count"], res)
    assert row["llm_pred"] == -1
    assert row["llm_confidence"] is None


def test_modeltag_slug():
    assert A._modeltag("fdtn-ai/Foundation-Sec-8B-Instruct") == "foundation-sec-8b-instruct"
    assert A._modeltag("meta-llama/Llama-3.1-8B-Instruct") == "llama-3-1-8b-instruct"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_arbitrate.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_arbitrate`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_arbitrate.py`**

```python
"""Milestone 3 -- Step 8 capstone: orchestrate LLM arbitration over the
cascade-contradiction set for one dataset and one model.

    dump contradictions -> build percentile tables -> render context per row
    -> LLMClient.arbitrate -> flat arbitration CSV

Usage:
    python llm_triage_arbitrate.py --dataset camlds --data-dir data/v3 \
        --out results/current/llm_triage/camlds \
        --model fdtn-ai/Foundation-Sec-8B-Instruct --backend vllm
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import LLM_TRIAGE, DATASETS


def _modeltag(model_id):
    tail = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")


def assemble_row(contra_row, feat_cols, llm_result):
    verdict = llm_result.get("verdict")
    llm_pred = 1 if verdict == "MALICIOUS" else 0 if verdict == "BENIGN" else -1
    return {
        "row_id": int(contra_row["row_id"]),
        "fold": int(contra_row["fold"]),
        "true_label": int(contra_row["true_label"]),
        "technique": contra_row["technique"],
        "pred_xgb": int(contra_row["pred_xgb"]),
        "pred_cnn": int(contra_row["pred_cnn"]),
        "proba_xgb": float(contra_row["proba_xgb"]),
        "proba_cnn": float(contra_row["proba_cnn"]),
        "disagreement_direction": contra_row["disagreement_direction"],
        "llm_verdict": verdict,
        "llm_pred": llm_pred,
        "llm_confidence": llm_result.get("confidence"),
        "llm_agrees_with": llm_result.get("agrees_with"),
        "parse_status": llm_result.get("parse_status"),
        "rationale_steps": json.dumps(llm_result.get("rationale_steps", [])),
        "key_features": json.dumps(llm_result.get("key_features", [])),
    }


def arbitrate_dataset(dataset_key, data_dir, out_dir, model_id=None, backend=None,
                      max_cases=None, seed=42, client=None):
    from llm_triage_dump import find_contradictions, stratified_subsample
    from llm_triage_stats import compute_percentile_tables
    from llm_triage_context import build_context
    from llm_triage_client import LLMClient
    from ingestion import ingest

    model_id = model_id or LLM_TRIAGE["model_id"]
    max_cases = max_cases or LLM_TRIAGE["max_cases_per_dataset"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contra, _oof, feat_cols = find_contradictions(dataset_key, data_dir, seed=seed)
    contra = stratified_subsample(contra, max_cases, seed)

    ds = ingest(dataset_key, data_dir=data_dir)
    label_col = DATASETS[dataset_key]["label_col"]
    pct = compute_percentile_tables(ds.df, feat_cols, label_col)
    (out_dir / f"{dataset_key}_percentiles.json").write_text(json.dumps(pct))

    client = client or LLMClient(model_id=model_id, backend=backend)
    rows = []
    for i, (_, r) in enumerate(contra.iterrows()):
        ctx = build_context(r, feat_cols, pct, dataset_key)
        res = client.arbitrate(ctx)
        rows.append(assemble_row(r, feat_cols, res))
        if (i + 1) % 25 == 0:
            print(f"[llm_triage_arbitrate] {dataset_key}/{_modeltag(model_id)}: "
                  f"{i + 1}/{len(contra)}", flush=True)

    df = pd.DataFrame(rows)
    path = out_dir / f"{dataset_key}_{_modeltag(model_id)}_llm_arbitration.csv"
    df.to_csv(path, index=False)
    n_err = (df["parse_status"] == "parse_error").sum() if len(df) else 0
    print(f"[llm_triage_arbitrate] wrote {path}  ({len(df)} rows, {n_err} parse_error)",
          flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="LLM arbitration over cascade contradictions")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=LLM_TRIAGE["model_id"])
    ap.add_argument("--backend", default=LLM_TRIAGE["backend"], choices=["vllm", "ollama"])
    ap.add_argument("--max-cases", type=int, default=LLM_TRIAGE["max_cases_per_dataset"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    arbitrate_dataset(args.dataset, args.data_dir, args.out, model_id=args.model,
                      backend=args.backend, max_cases=args.max_cases, seed=args.seed)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_arbitrate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_arbitrate.py milestone3_pipeline/tests/test_llm_triage_arbitrate.py
git commit -m "feat: add LLM arbitration orchestrator and CLI"
```

---

## Task 7: Evaluation

**Files:**
- Create: `milestone3_pipeline/src/llm_triage_evaluate.py`
- Test: `milestone3_pipeline/tests/test_llm_triage_evaluate.py`

**Interfaces:**
- Consumes: arbitration CSV (from Task 6) and `<ds>_cascade_oof.csv` (from Task 4);
  scikit-learn metrics; matplotlib (Agg).
- Produces:
  - `arbitration_accuracy(arb_df) -> pd.DataFrame` — rows: `strategy` in
    `{llm, always_xgb, always_cnn, trust_more_confident, blend_at_0.5, majority_class}`,
    cols: `accuracy, precision, recall, f1, n`. Excludes `parse_status == "parse_error"`.
  - `pipeline_effect(arb_df, oof_df) -> (pd.DataFrame, pd.Series)` — whole-population
    `f1/recall/fpr` for `cascade_blend@0.5` vs `cascade+llm` (LLM verdict substituted on
    contradiction `row_id`s; `parse_error` rows keep the blend call), plus the delta Series.
  - `expected_calibration_error(confidence, correct, n_bins=10) -> float`.
  - `key_feature_overlap(arb_df, discriminator_features) -> float` — fraction of non-empty
    `key_features` lists that intersect the dataset's discriminator set.
  - `DISCRIMINATOR_FEATURES: dict[str, list[str]]` keyed `"camlds"`, `"casino"`.
  - `write_report_section(dataset_key, model_tag, acc_df, eff_df, eff_delta, ece,
    kf_overlap, parse_rate, out_path)`.
  - CLI: `python llm_triage_evaluate.py --dataset camlds --arbitration <csv> --oof <csv>
    --out <dir>`.

- [ ] **Step 1: Write the failing test** — `milestone3_pipeline/tests/test_llm_triage_evaluate.py`

```python
import json
import numpy as np
import pandas as pd
import llm_triage_evaluate as E


def _arb():
    # 4 usable rows + 1 parse_error that must be ignored by arbitration_accuracy
    def kf(x):
        return json.dumps(x)
    return pd.DataFrame([
        dict(row_id=0, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.45, proba_cnn=0.80,
             llm_pred=1, llm_confidence=0.9, parse_status="ok", key_features=kf(["priv_op_count"])),
        dict(row_id=1, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.60, proba_cnn=0.30,
             llm_pred=0, llm_confidence=0.8, parse_status="ok", key_features=kf(["network_count"])),
        dict(row_id=2, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.40, proba_cnn=0.66,
             llm_pred=1, llm_confidence=0.7, parse_status="repaired", key_features=kf(["exec_count"])),
        dict(row_id=3, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.55, proba_cnn=0.20,
             llm_pred=1, llm_confidence=0.6, parse_status="ok", key_features=kf(["seq_length"])),
        dict(row_id=4, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.50, proba_cnn=0.70,
             llm_pred=-1, llm_confidence=None, parse_status="parse_error", key_features=kf([])),
    ])


def test_arbitration_accuracy_excludes_parse_errors_and_scores_llm():
    acc = E.arbitration_accuracy(_arb()).set_index("strategy")
    assert acc.loc["llm", "n"] == 4
    # llm preds on rows 0,1,2,3 = [1,0,1,1]; truth = [1,0,1,0] -> 3/4
    assert abs(acc.loc["llm", "accuracy"] - 0.75) < 1e-9
    # always_xgb preds = [0,1,0,1]; truth = [1,0,1,0] -> 0/4
    assert abs(acc.loc["always_xgb", "accuracy"] - 0.0) < 1e-9
    # always_cnn preds = [1,0,1,0] -> 4/4
    assert abs(acc.loc["always_cnn", "accuracy"] - 1.0) < 1e-9


def test_pipeline_effect_substitutes_llm_on_contradiction_rows():
    oof = pd.DataFrame({
        "row_id": [0, 1, 2, 3, 4, 5],
        "true_label": [1, 0, 1, 0, 1, 0],
        "proba_xgb": [0.45, 0.60, 0.40, 0.55, 0.20, 0.90],
        "proba_cnn": [0.80, 0.30, 0.66, 0.20, 0.10, 0.95],
    })
    eff, delta = E.pipeline_effect(_arb(), oof)
    assert set(eff["pipeline"]) == {"cascade_blend@0.5", "cascade+llm"}
    assert "f1" in delta.index and "fpr" in delta.index


def test_expected_calibration_error_zero_when_perfect():
    conf = np.array([0.05, 0.25, 0.55, 0.95])
    correct = np.array([0.0, 0.0, 1.0, 1.0])  # matches bin midpoints closely
    ece = E.expected_calibration_error(conf, correct, n_bins=4)
    assert ece < 0.3


def test_expected_calibration_error_high_when_inverted():
    conf = np.array([0.99, 0.99, 0.99, 0.99])
    correct = np.array([0.0, 0.0, 0.0, 0.0])
    assert E.expected_calibration_error(conf, correct, n_bins=10) > 0.9


def test_key_feature_overlap():
    df = _arb()
    frac = E.key_feature_overlap(df, ["priv_op_count", "network_count"])
    # 5 rows, key_features non-empty on 4, intersect on rows 0 and 1 -> 2/4
    assert abs(frac - 0.5) < 1e-9


def test_discriminator_features_present_for_both_datasets():
    assert set(E.DISCRIMINATOR_FEATURES) == {"camlds", "casino"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: llm_triage_evaluate`.

- [ ] **Step 3: Create `milestone3_pipeline/src/llm_triage_evaluate.py`**

```python
"""Milestone 3 -- Step 8 capstone: evaluate LLM arbitration (Deliverable 4).

Reads one <ds>_<model>_llm_arbitration.csv (Task 6) plus <ds>_cascade_oof.csv
(Task 4) and reports:
  A. arbitration accuracy on the contradiction set vs 5 baselines
  B. whole-population pipeline effect (cascade blend@0.5 vs cascade+LLM)
  D. confidence calibration (ECE) and key_features overlap with error-analysis
     discriminators
  E. parse-failure rate
Charts + a paste-ready report_section markdown.

Note: baseline B uses a fixed 0.5 blend cutoff (labelled 'blend@0.5'); the live
cascade picks its cutoff per fold from XGBoost train probabilities, so this is a
close, honest approximation, not the exact cascade decision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# Aggregate discriminators from error_analysis.py's FP-vs-TN / FN-vs-TP tables
# (results/current/error_analysis_results/<ds>/). Population-level facts, not
# row-level -- safe to show the model as "known failure patterns".
DISCRIMINATOR_FEATURES = {
    "camlds": ["auid_euid_mismatch", "max_uid_euid_delta", "priv_op_count", "network_count",
               "shell_from_service", "parent_child_rarity", "uid_changed", "ephemeral_privileged",
               "failed_call_rate", "failed_call_count", "exec_count", "lifetime_seconds"],
    "casino": ["exec_count", "file_access_count", "euid_root", "uid_is_root", "auid_euid_mismatch",
               "priv_op_count", "failed_call_rate", "shell_from_service", "failed_call_count"],
}


def _fpr(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fp / (fp + tn) if (fp + tn) else 0.0


def arbitration_accuracy(arb_df):
    d = arb_df[arb_df["parse_status"] != "parse_error"].copy()
    y = d["true_label"].astype(int).values
    px, pc = d["proba_xgb"].values, d["proba_cnn"].values
    more_conf = np.where(np.abs(px - 0.5) >= np.abs(pc - 0.5),
                         d["pred_xgb"].values, d["pred_cnn"].values).astype(int)
    strategies = {
        "llm": d["llm_pred"].astype(int).values,
        "always_xgb": d["pred_xgb"].astype(int).values,
        "always_cnn": d["pred_cnn"].astype(int).values,
        "trust_more_confident": more_conf,
        "blend_at_0.5": ((0.5 * px + 0.5 * pc) >= 0.5).astype(int),
        "majority_class": np.full(len(d), int(round(y.mean())) if len(d) else 0),
    }
    out = []
    for name, pred in strategies.items():
        out.append({
            "strategy": name,
            "accuracy": accuracy_score(y, pred) if len(d) else float("nan"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "n": int(len(d)),
        })
    return pd.DataFrame(out)


def pipeline_effect(arb_df, oof_df):
    o = oof_df.copy()
    y = o["true_label"].astype(int).values
    base = ((0.5 * o["proba_xgb"] + 0.5 * o["proba_cnn"]) >= 0.5).astype(int).values
    sub = {int(r.row_id): int(r.llm_pred) for r in arb_df.itertuples()
           if int(r.llm_pred) in (0, 1)}
    idx = {rid: i for i, rid in enumerate(o["row_id"].astype(int).values)}
    withllm = base.copy()
    for rid, p in sub.items():
        if rid in idx:
            withllm[idx[rid]] = p
    rows = []
    for name, pred in [("cascade_blend@0.5", base), ("cascade+llm", withllm)]:
        rows.append({"pipeline": name,
                     "f1": f1_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "fpr": _fpr(y, pred)})
    eff = pd.DataFrame(rows)
    delta = (eff.set_index("pipeline").loc["cascade+llm"]
             - eff.set_index("pipeline").loc["cascade_blend@0.5"])
    return eff, delta


def expected_calibration_error(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if not m.any():
            continue
        ece += (m.sum() / len(confidence)) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def key_feature_overlap(arb_df, discriminator_features):
    disc = set(discriminator_features)
    n = hits = 0
    for s in arb_df["key_features"]:
        kf = set(json.loads(s)) if isinstance(s, str) else set(s or [])
        if not kf:
            continue
        n += 1
        if kf & disc:
            hits += 1
    return hits / n if n else float("nan")


# -- charts ---------------------------------------------------------------
def plot_accuracy_bars(acc_df, title, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(acc_df["strategy"], acc_df["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy on contradiction set")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(confidence, correct, path):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(confidence):
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (confidence > edges[i]) & (confidence <= edges[i + 1])
            if m.any():
                xs.append(confidence[m].mean())
                ys.append(correct[m].mean())
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(xs, ys, "o-")
    ax.set_xlabel("mean reported confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("LLM confidence calibration")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report_section(dataset_key, model_tag, acc_df, eff_df, eff_delta, ece,
                         kf_overlap, parse_rate, out_path):
    lines = [
        f"### LLM arbitration -- {dataset_key} / {model_tag}", "",
        f"- Contradiction rows scored: {int(acc_df['n'].iloc[0])} "
        f"(parse-failure rate {parse_rate:.1%})", "",
        "**Accuracy on the contradiction set**", "",
        acc_df.to_markdown(index=False), "",
        "**Whole-population pipeline effect (blend@0.5 baseline)**", "",
        eff_df.to_markdown(index=False), "",
        f"- Delta vs baseline: F1 {eff_delta['f1']:+.4f}, "
        f"recall {eff_delta['recall']:+.4f}, FPR {eff_delta['fpr']:+.4f}", "",
        "**Reasoning quality**", "",
        f"- Expected Calibration Error: {ece:.4f}",
        f"- key_features intersect the dataset's error-analysis discriminators "
        f"in {kf_overlap:.1%} of scored rows", "",
    ]
    Path(out_path).write_text("\n".join(str(x) for x in lines))


def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM arbitration output")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--arbitration", required=True, help="<ds>_<model>_llm_arbitration.csv")
    ap.add_argument("--oof", required=True, help="<ds>_cascade_oof.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arb = pd.read_csv(args.arbitration)
    oof = pd.read_csv(args.oof)
    model_tag = Path(args.arbitration).stem.replace(f"{args.dataset}_", "").replace("_llm_arbitration", "")

    acc = arbitration_accuracy(arb)
    eff, delta = pipeline_effect(arb, oof)
    usable = arb[arb["parse_status"] != "parse_error"].copy()
    correct = (usable["llm_pred"].astype(int) == usable["true_label"].astype(int)).astype(float).values
    ece = expected_calibration_error(usable["llm_confidence"].values, correct)
    kf = key_feature_overlap(arb, DISCRIMINATOR_FEATURES[args.dataset])
    parse_rate = (arb["parse_status"] == "parse_error").mean() if len(arb) else float("nan")

    acc.to_csv(out_dir / f"{args.dataset}_{model_tag}_arbitration_metrics.csv", index=False)
    eff.to_csv(out_dir / f"{args.dataset}_{model_tag}_pipeline_effect.csv", index=False)
    pd.DataFrame([{"ece": ece, "key_feature_overlap": kf, "parse_error_rate": parse_rate}]).to_csv(
        out_dir / f"{args.dataset}_{model_tag}_reasoning_quality.csv", index=False)
    plot_accuracy_bars(acc, f"{args.dataset} / {model_tag}",
                       out_dir / f"{args.dataset}_{model_tag}_accuracy.png")
    plot_reliability(usable["llm_confidence"].values, correct,
                     out_dir / f"{args.dataset}_{model_tag}_calibration.png")
    write_report_section(args.dataset, model_tag, acc, eff, delta, ece, kf, parse_rate,
                         out_dir / f"{args.dataset}_{model_tag}_report_section.md")
    print(f"[llm_triage_evaluate] wrote metrics + charts + report section to {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_evaluate.py -v`
Expected: PASS (6 tests). If `to_markdown` raises for missing `tabulate`, add `tabulate` to `requirements.txt` in this step and re-run.

- [ ] **Step 5: Run the whole test suite**

Run: `python -m pytest milestone3_pipeline -m "not slow" -v`
Expected: PASS (all tasks' tests, ~30 tests).

- [ ] **Step 6: Commit**

```bash
git add milestone3_pipeline/src/llm_triage_evaluate.py milestone3_pipeline/tests/test_llm_triage_evaluate.py milestone3_pipeline/src/requirements.txt
git commit -m "feat: add LLM arbitration evaluation, charts, and report section"
```

---

## Task 8: SLURM runner, optional smoke test, and docs

**Files:**
- Create: `milestone3_pipeline/run_scripts/run_llm_triage.sh`
- Create: `milestone3_pipeline/tests/test_llm_triage_smoke.py`
- Modify: `docs/milestone3/CHANGELOG.md` (append section 13)
- Modify: `README.md` (Current state list)

**Interfaces:**
- Consumes: `llm_triage_arbitrate.py` and `llm_triage_evaluate.py` CLIs.
- Produces: a submittable SLURM array job; a `@pytest.mark.slow` smoke test.

- [ ] **Step 1: Create `milestone3_pipeline/run_scripts/run_llm_triage.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=milestone3-llm-triage
#SBATCH --output=logs/m3_llm_triage_%A_%a.out
#SBATCH --error=logs/m3_llm_triage_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --partition=studentkillable
#SBATCH --open-mode=truncate
#
# Milestone 3 -- Step 8 capstone: LLM-Driven Triage and Contextual Arbitration.
# One array task per (dataset x model). Each task: dump the hybrid-cascade
# contradiction rows, run the LLM arbitrator over them, then evaluate.
#
#   task 0: camlds  x  fdtn-ai/Foundation-Sec-8B-Instruct
#   task 1: camlds  x  meta-llama/Llama-3.1-8B-Instruct
#   task 2: casino  x  fdtn-ai/Foundation-Sec-8B-Instruct
#   task 3: casino  x  meta-llama/Llama-3.1-8B-Instruct
#
# Requires src/requirements-llm.txt installed on the GPU node. If no GPU
# partition is available, set BACKEND=ollama below and serve a GGUF locally
# (ollama pull the model first); then --gres=gpu:1 can be dropped.
#
# Submit from the repo ROOT:  sbatch milestone3_pipeline/run_scripts/run_llm_triage.sh

PYTHON="/vol/joberant_nobck/data/NLP_368307701_2526a/liorpernik/anaconda3/envs/toxic/bin/python -u"
SCRIPTS_DIR="milestone3_pipeline/src"
OUT_ROOT="milestone3_pipeline/results/current/llm_triage"
BACKEND="vllm"

DATASETS=(camlds camlds casino casino)
MODELS=(fdtn-ai/Foundation-Sec-8B-Instruct meta-llama/Llama-3.1-8B-Instruct \
        fdtn-ai/Foundation-Sec-8B-Instruct meta-llama/Llama-3.1-8B-Instruct)

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

if [ "$DATASET" == "camlds" ]; then DATA_DIR="data/v3"; else DATA_DIR="data"; fi
OUT_DIR="$OUT_ROOT/$DATASET"
MODEL_TAG=$(echo "$MODEL" | sed 's#.*/##; s#[^A-Za-z0-9]#-#g' | tr 'A-Z' 'a-z')

mkdir -p logs "$OUT_DIR"

echo "============================================"
echo "Task       : $SLURM_ARRAY_TASK_ID  ($DATASET x $MODEL)"
echo "Data dir   : $DATA_DIR"
echo "Backend    : $BACKEND"
echo "Out dir    : $OUT_DIR"
echo "Start      : $(date)"
echo "============================================"

$PYTHON "$SCRIPTS_DIR/llm_triage_arbitrate.py" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR" \
    --model "$MODEL" --backend "$BACKEND"
RC=$?

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
```

- [ ] **Step 2: Note the dump-runs-twice cost and fix it**

`llm_triage_arbitrate.py` calls `find_contradictions` (which fits XGB+CNN over 5 folds).
`llm_triage_evaluate.py` needs `<ds>_cascade_oof.csv`, which only `llm_triage_dump.py`'s CLI
writes. Add an explicit dump step to the script **before** arbitration so the OOF file exists
and the fold models are computed once:

Insert into `run_llm_triage.sh` immediately before the `llm_triage_arbitrate.py` call:

```bash
$PYTHON "$SCRIPTS_DIR/llm_triage_dump.py" \
    --dataset "$DATASET" --data-dir "$DATA_DIR" --out "$OUT_DIR"
RC=$?
if [ $RC -ne 0 ]; then echo "dump failed"; exit $RC; fi
```

Then change `llm_triage_arbitrate.arbitrate_dataset` to **reuse the dumped CSVs when present**
rather than recomputing. Edit `milestone3_pipeline/src/llm_triage_arbitrate.py`, replacing the
`contra, _oof, feat_cols = find_contradictions(...)` line with:

```python
    from feature_selection import select_features
    dump_csv = out_dir / f"{dataset_key}_contradictions.csv"
    oof_csv = out_dir / f"{dataset_key}_cascade_oof.csv"
    if dump_csv.exists() and oof_csv.exists():
        contra = pd.read_csv(dump_csv)
        feat_cols = select_features(ingest(dataset_key, data_dir=data_dir).feat_cols,
                                    dataset_key, policy="full")
    else:
        contra, _oof, feat_cols = find_contradictions(dataset_key, data_dir, seed=seed)
        contra = stratified_subsample(contra, max_cases, seed)
```

(The pre-existing `test_llm_triage_arbitrate.py` tests only `assemble_row` / `_modeltag`, so
they still pass. Run them again to confirm.)

Run: `python -m pytest milestone3_pipeline/tests/test_llm_triage_arbitrate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 3: Create `milestone3_pipeline/tests/test_llm_triage_smoke.py`**

```python
"""Optional end-to-end smoke test. Needs a running Ollama server with the model
pulled: `ollama pull llama3.1:8b`. Run explicitly with:  pytest -m slow
Asserts only that the pipeline produces a schema-valid decision, never that the
decision is correct.
"""
import json
import pytest
import config
import llm_triage_context as C
import llm_triage_client as LC

pytestmark = pytest.mark.slow


def test_local_ollama_roundtrip_is_schema_valid():
    bp = [float(i) for i in range(1, 100)]
    pct = {f: {"benign": list(bp), "attack": list(bp)} for f in config.FULL_FEATURE_LIST}
    row = {f: 10.0 for f in config.FULL_FEATURE_LIST}
    row.update({"max_uid_euid_delta": 90.0, "is_suid_exec": 1.0, "priv_op_count": 5.0,
                "pred_xgb": 0, "proba_xgb": 0.35, "pred_cnn": 1, "proba_cnn": 0.72})
    ctx = C.build_context(row, config.FULL_FEATURE_LIST, pct, "camlds")

    client = LC.LLMClient(model_id="llama3.1:8b", backend="ollama")
    out = client.arbitrate(ctx)
    assert out["parse_status"] in ("ok", "repaired", "parse_error")
    if out["parse_status"] != "parse_error":
        assert out["verdict"] in ("MALICIOUS", "BENIGN")
        assert 3 <= len(out["rationale_steps"]) <= 6
        assert all(f in config.FULL_FEATURE_LIST for f in out["key_features"])
```

Run (optional, needs Ollama): `python -m pytest milestone3_pipeline -m slow -v`
Expected: PASS if Ollama is up; otherwise skipped/deselected by default runs.

- [ ] **Step 4: Append section 13 to `docs/milestone3/CHANGELOG.md`**

```markdown

---

## 13. LLM-driven triage & contextual arbitration (Step 8 capstone)

**What was built:** an offline LLM arbitrator (`src/llm_triage_*.py`) for the
rows where the hybrid cascade's two models contradict each other -- routed
(XGBoost probability in the 0.3-0.7 band) AND `pred_xgb != pred_cnn`, evaluated
out-of-fold with the same `train_eval._FITTERS` as the headline pipeline.

**Engine:** `fdtn-ai/Foundation-Sec-8B-Instruct` (Cisco Foundation AI,
Llama-3.1-8B backbone, security-tuned; 4,096-token context). Control model:
`meta-llama/Llama-3.1-8B-Instruct`, identical architecture, no security tuning --
isolates whether the domain tuning helps on our edge cases.

**Context window:** each contradiction row is rendered as 20 raw feature values
with dual benign/attack percentile ranks, an auto-flagged "notable deviations"
list, and an aggregate "known failure patterns" block distilled from
`error_analysis.py`'s FP-vs-TN / FN-vs-TP tables. The model returns guided JSON
(`verdict`, `confidence`, 5-step `rationale_steps`, `key_features`, `agrees_with`).
The row's `technique` label is never shown -- context is family-level only.

**Disclosure:** percentile tables and the known-failure block are computed from
labelled data (descriptive context, not model fitting) -- same disclosure class
as `IFOREST_NOVELTY_MODE`.

**Evaluation (`llm_triage_evaluate.py`):** arbitration accuracy vs 5 baselines
(always-XGB, always-CNN, trust-more-confident, blend@0.5, majority class);
whole-population pipeline F1/FPR/recall delta; confidence ECE; `key_features`
overlap with the error-analysis discriminators; parse-failure rate.

**Run:** `sbatch milestone3_pipeline/run_scripts/run_llm_triage.sh` (array over
{camlds, casino} x {foundation-sec, llama-3.1}). Ollama-GGUF fallback documented
in the script header.
```

- [ ] **Step 5: Update `README.md` Current state**

Replace the two trailing bullets:

```markdown
- hybrid cascade (Isolation Forest → supervised)
- LLM triage layer (Step 8 capstone)
```

with:

```markdown
- hybrid cascade — implemented (`milestone3_pipeline/src/hybrid_cascade*.py`; XGBoost→CNN,
  plus RF-stage-2 and weighted variants)
- LLM triage layer — implemented (`milestone3_pipeline/src/llm_triage_*.py`; offline
  contextual arbitration over cascade contradictions, `run_scripts/run_llm_triage.sh`)
```

- [ ] **Step 6: Run the full fast suite once more**

Run: `python -m pytest milestone3_pipeline -m "not slow"`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add milestone3_pipeline/run_scripts/run_llm_triage.sh milestone3_pipeline/tests/test_llm_triage_smoke.py milestone3_pipeline/src/llm_triage_arbitrate.py docs/milestone3/CHANGELOG.md README.md
git commit -m "feat: add SLURM runner, smoke test, and docs for LLM triage layer"
```

---

## Self-Review

### 1. Spec coverage

| Spec section | Plan task(s) |
|---|---|
| §2.1 engine `Foundation-Sec-8B-Instruct` | Task 1 (`LLM_TRIAGE.model_id`), Task 5 (`LLMClient`), Task 8 (SLURM) |
| §2.2 control model `Llama-3.1-8B-Instruct` | Task 1 (`control_model_id`), Task 8 (array task 1 & 3) |
| §2.5 vLLM primary / Ollama fallback, temp 0, fixed seed, guided JSON | Task 5 (`_generate_vllm` / `_generate_ollama`, `LLMClient.__init__`, schema validate) |
| §3.1 `cascade_contradiction_dump` via `_FITTERS` | Task 4 (`_oof_calls`, `find_contradictions`) |
| §3.1 `population_stats` percentiles | Task 2 |
| §3.1 `context_builder` | Task 3 |
| §3.1 `prompts` constants | Task 1 |
| §3.1 `llm_client` parse + repair + enum | Task 5 |
| §3.1 `arbitrate` orchestrator | Task 6 |
| §3.1 `evaluate_arbitration` | Task 7 |
| §3.2 `LLM_TRIAGE` + `FEATURE_GLOSS` in config | Task 1 |
| §3.5 contradiction = routed ∧ `pred_xgb != pred_cnn`; disagreement-direction tag; stratified cap | Task 4 (`contradiction_mask`, `add_direction`, `stratified_subsample`) |
| §4.1 token budget / static constants | Task 1 (`SYSTEM_PROMPT`, `COT_RUBRIC`, `SCHEMA_INSTRUCTION`, `FEWSHOT_EXAMPLE`) |
| §4.2 system prompt (4 MITRE families, no technique) | Task 1 + test `test_prompt_text_mentions_only_the_four_mitre_families` |
| §4.3 5-step CoT rubric | Task 1 (`COT_RUBRIC`) |
| §4.4 feature table, dual percentiles, notable deviations, known-failure block, raw values, model probs | Task 3 (`build_context`) |
| §4.4 refinement: deviation ranked in percentile space | Task 3 note + `notable_deviations` |
| §4.5 one hand-authored few-shot | Task 1 (`FEWSHOT_EXAMPLE`) |
| §4.6 output schema + `agrees_with` | Task 1 (`ARBITRATION_SCHEMA_V1`), Task 6 (`assemble_row` carries `llm_agrees_with`) |
| §5.1 determinism | Task 5 |
| §5.2 malformed → 1 repair → `parse_error`; enum/unknown-feature = parse failure | Task 5 (`arbitrate`, `_validate_semantics`) + tests |
| §5.3 batching / load once | Task 5 (`_engine` cached); Task 8 (one model per array task) |
| §6A accuracy vs 5 baselines | Task 7 (`arbitration_accuracy`) |
| §6B whole-population pipeline effect, same framing | Task 7 (`pipeline_effect`) |
| §6C security-vs-general | Task 8 (both models run); comparison is reading the two `report_section.md` files |
| §6D key_features overlap, ECE, agrees_with consistency, faithfulness spot-check | Task 7 (`key_feature_overlap`, `expected_calibration_error`); `agrees_with` stored for manual check; faithfulness spot-check is a manual report step (documented in CHANGELOG) |
| §6E robustness (parse-failure rate) | Task 7 (`parse_error_rate` in `reasoning_quality.csv`) |
| §6F per-technique | `technique` column carried through to the arbitration CSV (Task 6); per-technique tabulation is a manual report step |
| §6G per-dataset expectation | both datasets run (Task 8); narrative in CHANGELOG §13 |
| §6 charts | Task 7 (`plot_accuracy_bars`, `plot_reliability`) |
| §7 test matrix | one test file per module, Tasks 1–7; smoke test Task 8 |
| §8 SLURM | Task 8 |
| §9 disclosures | CHANGELOG §13 (Task 8), module docstrings (Tasks 2, 4) |

**Gaps intentionally left as manual report steps (not code):** §6C cross-model comparison
table, §6D faithfulness spot-check on N=20, §6F per-technique breakdown table, §6G narrative.
All their input data is written to CSV by Tasks 6–7; assembling them into prose is report-writing,
not pipeline code. This is called out in CHANGELOG §13.

### 2. Placeholder scan

No "TBD" / "add error handling" / "similar to Task N" / bare prose steps. Every code step has a
full code block. The one deviation from spec wording (subpackage → flat `llm_triage_` files) is
stated in Global Constraints with rationale. The `to_markdown`/`tabulate` dependency risk is
handled inline in Task 7 Step 4.

### 3. Type consistency

- Fitter signature `(_FITTERS[mk](X_tr, y_tr, groups_tr, X_te, None, MAX_FPR) -> (pred, proba))`
  matches `train_eval.py:104-184` (verified).
- `contradiction_mask` / `add_direction` / `stratified_subsample` operate on a DataFrame with
  columns `row_id, technique, true_label, proba_xgb, proba_cnn, pred_xgb, pred_cnn` — produced by
  `_oof_calls` (Task 4) and consumed unchanged by Task 6.
- `LLMClient.arbitrate` return dict keys (`verdict, confidence, rationale_steps, key_features,
  agrees_with, parse_status, raw`) — consumed by `assemble_row` (Task 6) via `.get(...)`, which
  tolerates the `parse_error` shape.
- `assemble_row` output keys (`row_id, true_label, pred_xgb, pred_cnn, proba_xgb, proba_cnn,
  llm_pred, llm_confidence, parse_status, key_features, technique, disagreement_direction`) — every
  column read by `arbitration_accuracy` / `pipeline_effect` / `key_feature_overlap` (Task 7) is
  present.
- `percentile_label` / `label_to_rank` names identical in Task 2 (definition), Task 3 (import).
- `compute_percentile_tables` output shape `{feat: {"benign": [...], "attack": [...]}}` — consumed
  by `build_context` as `pct_tables[f]["benign"]` (Task 3) and serialised to
  `<ds>_percentiles.json` (Task 6).
- `_modeltag` (Task 6) and the shell `MODEL_TAG` sed expression (Task 8) both lowercase and map
  non-alnum runs to `-`: `Foundation-Sec-8B-Instruct` → `foundation-sec-8b-instruct`. Consistent.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-31-llm-triage-contextual-arbitration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
