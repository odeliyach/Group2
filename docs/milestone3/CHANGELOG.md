# Milestone 3 — Changelog & Decision Log

Running record of every change made to the pipeline, why it was made, what
it found, and what's still open. Read top to bottom for the story; use the
headers to jump to a specific decision when writing the report.

---

## 1. Initial pipeline build (Step 7)

Built a modular pipeline matching the brief's required sequence:
```
Data Ingestion → Preprocessing & Normalization → Feature Selection & Engineering → Model Training → Evaluation
```
Dependency Rule: only `ingestion.py` is dataset-specific; every other module
(`preprocessing.py`, `feature_selection.py`, `models.py`, `train_eval.py`)
is dataset-agnostic.

Four models, one per required paradigm, all with **explicit** hyperparameters
(no library defaults) in `config.py`:
- Random Forest, XGBoost — traditional supervised
- 1D-CNN — deep learning
- Isolation Forest — unsupervised/anomaly detection

CV methodology ported from M2's `unified_validation.py`: leak-free
vector-group `StratifiedGroupKFold`, pooled-per-repeat scoring (not raw
per-fold averaging).

`feature_selection.py`'s `ABLATION_DROPPABLE` is intentionally empty for
both datasets — M2's own ablation found dropping features costs real F1 on
CAM-LDS and is a wash on Casino. Policy: use all available features.

---

## 2. SLURM deployment fixes

Several real bugs surfaced only once running on the actual cluster:

- **`DATA_DIR` mismatches** — scripts initially pointed at `data/` (a
  placeholder) instead of the real `combined/` path. Fixed across all
  `.sh` scripts.
- **Output buffering** — Python's stdout is fully buffered when redirected
  to a file (not a terminal), making logs appear to "hang" for a long time
  even when the job was running fine. Fixed with `python -u` in every
  script + `flush=True` on key print statements.
- **Resource right-sizing** — initial `--cpus-per-task=8 --mem=32G` was
  copied from M2's much heavier multi-stage job and queued for a long time
  under `(Resources)`. Reduced to `4 cpus / 12G`, which fits into
  partially-allocated nodes far more easily.
- **Array concurrency cap** — `--array=0-7%3` limits how many of our own
  tasks run simultaneously, so the scheduler only needs to find 3 small
  slots instead of 8 at once.
- **XGBoost thread oversubscription** — `XGB_PARAMS` never set `n_jobs`,
  so XGBoost defaulted to spawning threads for every core it could detect
  on the *node*, not the 4 cores SLURM actually allocated to the task. With
  3 concurrent array tasks sharing a node, this caused severe thread
  thrashing: XGBoost completed only 2–3 of 15 repeats in a 2-hour budget
  where RF finished all 15 with an identical (fit-twice-per-fold) workload.
  Fixed by pinning `XGB_PARAMS["n_jobs"] = 4` to match `--cpus-per-task`.
  Post-fix: 8–12/15 repeats in 3 hours — a 4–5x speedup. Time budgets were
  further raised (6hr) to guarantee completion.
- **Stale log content** — SLURM appends to output files by default rather
  than truncating; a cancelled/resubmitted job could show a *previous*
  job's `CANCELLED` message inside what looked like the current job's log.
  Fixed with `--open-mode=truncate`.

---

## 3. F1-optimal thresholding (replacing the naive 0.5 cutoff)

**Problem found:** every model originally used a flat 0.5 probability
cutoff (or Isolation Forest's `contamination='auto'`, which blindly assumes
~10% outliers). On CAM-LDS's near-balanced 54% attack rate, this let
Isolation Forest's F1 look deceptively non-trivial (~0.70) while it was
actually flagging almost everything as an attack (FPR≈0.996).

**The math:** for a classifier with *zero* discriminative skill that
predicts positive for everything, F1 = 2·prevalence/(1+prevalence). At
CAM-LDS's 54% prevalence, that's F1≈0.70 — indistinguishable from what a
coin that always says "attack" would score. F1 alone is not a reliable
headline metric when one class is a large majority.

**Fix:** `_best_f1_threshold()` in `train_eval.py` sweeps every possible
cutoff (via sorted-probability cumulative sums) and picks the one
maximizing F1, calibrated on a held-out slice of the *training* fold only
— never the test fold. Verified with a regression test: on pure noise
input, the unconstrained search converges to the same degenerate
always-positive pattern (proving the math above is real), while adding an
FPR cap (next section) correctly collapses F1 instead of hiding behind it.

---

## 4. FPR cap — from a guess to an evidence-based number

**Initial state:** `MAX_FPR = 0.10`, a starting guess with no
justification.

**Built `operating_curve.py`** to sweep several candidate caps and report
the actual F1/Precision/Recall/FPR trade-off, instead of guessing.

**Real sweep result on CAM-LDS/RF:**
| Cap | F1 | Actual FPR |
|---|---|---|
| 0.01 | 0.829 | 0.066 |
| 0.02 | 0.840 | 0.069 |
| **0.05** | **0.863 (peak)** | **0.094** |
| 0.10 | 0.861 | 0.141 |
| 0.20–0.50 | 0.838–0.839 | 0.27–0.30 |
| 1.0 (uncapped) | 0.861 | 0.141 |

F1 peaks at cap=0.05 and *declines* for looser caps — loosening trades
away precision faster than it buys recall past that point.

**Decision:** `MAX_FPR = 0.05`, justified directly by this sweep, not a
guess. One caveat documented for the report: the cap only constrains
threshold *selection* on an inner validation slice — the *achieved* FPR on
a genuinely unseen test fold can still exceed the cap (e.g. cap=0.01
achieved actual FPR=0.066), since it's a target, not a hard guarantee.

---

## 5. Isolation Forest novelty-detection mode

**Hypothesis tested:** CAM-LDS's F1=0.32/AUROC=0.53 might be caused by
training on a *mixed* 46/54 population diluting what the model learns as
"normal."

**Fix implemented:** `IFOREST_NOVELTY_MODE = True` — fit only on rows
where `y_train==0` (confirmed benign), instead of the full training fold.
Standard technique in anomaly-detection literature.

**Disclosure required:** this uses training-fold *labels* to pick the
`.fit()` subset (never to fit the trees themselves) — changes the method
from purely unsupervised to semi-supervised "novelty detection." Must be
stated explicitly in the report, not left implicit.

**Result: the hypothesis was WRONG.** AUROC before: 0.5324. AUROC after:
0.5547 — essentially unchanged (within noise). This is a genuinely useful
negative result: it proves the failure isn't about *what data* Isolation
Forest trains on, it's the **random-split isolation mechanism itself**
being unable to find structure that RF/XGBoost (label-guided splits) can
(AUROC 0.94 on the identical features). Confirmed by the same model
scoring 0.821 AUROC on Casino, where attack genuinely is a 17% minority —
the paradigm's assumption holds there.

---

## 6. CNN degenerate-split guard

**Problem found:** Casino is 99.6% duplicated at the feature-vector level
(only 356 unique patterns across ~91k rows, per M2's own report). The
CNN's inner validation split (used for both early stopping and threshold
selection) could unluckily land almost all of one dominant duplicate-heavy
group on one side, leaving the validation slice with near-zero positive
examples — producing a garbage threshold. Observed: F1 swinging 0.11–0.86,
FPR swinging 0.003–0.98 across repeats at identical hyperparameters.

**Fix:** `_degenerate_split_guard()` in `train_eval.py` retries the
`GroupShuffleSplit` across up to 8 random seeds, requiring at least 5
examples of each class in the validation slice before accepting a split.

**Result:** reduced but did NOT eliminate the collapse. Stress-tested on
synthetic heavily-duplicated data (27 unique groups / 2000 rows): F1
stayed in a 0.72–0.96 range post-fix vs. catastrophic single-repeat
failures pre-fix. On real data, one repeat out of 15 still collapsed
(FPR=0.981) — documented honestly as a **partial mitigation**, not a full
fix.

---

## 7. log1p preprocessing

**Rationale:** RF/XGBoost choose splits by information gain over *sorted*
values — any monotonic transform is a no-op for their decisions. Isolation
Forest picks split points **uniformly at random** between a feature's min
and max at each node — a heavily right-skewed feature wastes most random
splits in the dense low region, rarely probing the tail.

**Fix:** `log1p_continuous()` in `preprocessing.py`, applied to every
non-binary feature, gated by `LOG1P_CONTINUOUS_FEATURES = True`.

**Verified on real data:**
- RF: AUROC 0.9994 → 0.9994 (unaffected, confirms the no-op theory)
- Isolation Forest: changed, but in one synthetic test got *slightly
  worse* (0.9501 → 0.9322) — direction is data-dependent, not a guaranteed
  win. Reported honestly rather than oversold.

**Correction issued mid-project:** the "no-op for XGBoost" claim was
**overstated**. XGBoost here uses `tree_method="hist"` (approximate
histogram-based splitting), which builds bins from the *observed value
distribution* — log1p reshapes that distribution and can shift where bin
edges fall, even though it never changes rank order. This likely explains
why XGBoost's single-run cross-dataset AUROC swung dramatically between
runs (0.70 → 0.08 in one direction) after the log1p change — a real,
mechanistically-explainable shift caused by using an *approximate* (not
exact) split-finding method, not a bug or noise.

---

## 8. Cross-dataset generalization module (`cross_dataset_eval.py`)

**Why built:** re-reading the brief's Step 8 language closely — *"diagnose
whether this is driven by Environmental Distribution Shift... or systemic
Model Overfitting to the primary training set"* — only makes sense if
there's a literal single training set being tested on a different one.
`run_pipeline.py`'s within-dataset CV never does this; it runs the same CV
separately on each dataset and compares summary numbers, which is a
different (and initially wrong) reading of the requirement.

**What it does:** trains fully on one dataset, evaluates on the ENTIRE
other dataset (both directions). Restricted to the 17 features both
datasets share (Casino lacks `is_suid_exec`, `ephemeral_privileged`,
`parent_child_rarity`).

**First single-run result — dramatic and asymmetric:**
- CAM-LDS → Casino: AUROC crashes to near 0 (0.006–0.08 across models) —
  **ranking actively inverted**, not just miscalibrated.
- Casino → CAM-LDS: AUROC near chance (0.45–0.53) — uninformative, but not
  inverted.

**Repeats added** (`--repeats N`, varies random seed per repeat) after
recognizing a single fit can't distinguish "genuinely stable failure" from
"one unlucky run" — especially for CNN (no fixed seed on weight init) and
XGBoost (approximate-histogram sensitivity, see §7).

**Stability results (5 repeats):**
| Model | CAM-LDS→Casino AUROC | Casino→CAM-LDS AUROC |
|---|---|---|
| RF | 0.024 ± 0.002 | 0.530 ± 0.101 |
| XGBoost | 0.203 ± 0.281 | 0.506 ± 0.028 |
| CNN | 0.462 ± 0.296 | 0.447 ± 0.130 |
| Isolation Forest | 0.014 ± 0.007 | 0.503 ± 0.062 |

**Interpretation:** RF/IsoForest fail *consistently* (tiny std) — stable,
reproducible near-total inversion, strong evidence of genuine overfitting
to CAM-LDS-specific feature relationships. XGBoost/CNN fail *unstably*
(large std) — same core failure, compounded by model-specific instability
(§7's histogram sensitivity; CNN's known training variance).

**Recalibration experiment added** (`--calibration-frac`): picks a new
threshold from a small labeled sample of the *target* domain, evaluates on
the rest. Tests whether the failure is fixable by recalibration alone (a
pure miscalibration problem) vs. not (a ranking/overfitting problem, which
recalibration can't fix since it never touches the model's predictions,
only the cutoff).

---

## 9. Root cause discovered: CAM-LDS's attack rate is a curation artifact

Reviewed the upstream dataset-construction scripts
(`extract_caml_full_v3.py`, `prepare_dataset.py`) for the first time this
session. Key finding:

```python
parser.add_argument('--benign_ratio', type=float, default=2.5, ...)
MAX_PER_SCENARIO_BENIGN = 15000   # also caps attack per technique
```

CAM-LDS's ~54% attack rate is **not** a natural property of the monitored
environment — it's the direct, deliberate output of `benign_ratio=2.5`
sampling combined with per-technique capping (pushing all 19 techniques
toward equal representation). This was built as a curated, class-balanced
*training* benchmark, not a realistic-prevalence corpus. Casino's 17.1%
attack rate is much closer to a plausible real-world base rate.

**This one fact explains two separate findings with the same mechanism:**
1. **Isolation Forest's failure (§5):** its sparse-minority assumption
   isn't broken by "real" data behaving unusually — it's broken because
   the extraction script explicitly engineered a near-balanced set.
2. **The cross-dataset collapse (§8):** models trained on CAM-LDS learned
   to expect ~54% prevalence across 19 equally-represented techniques;
   Casino looks nothing like that at "deployment." This reframes the
   Step 8 finding from "mysterious distribution shift between two real
   environments" to the well-documented phenomenon of **models trained on
   a curated, balanced benchmark failing to generalize to realistic,
   imbalanced target conditions.**

**New experiment queued** (not yet run): regenerate CAM-LDS at a more
realistic attack rate (`--benign_ratio 15.0`, targeting ~17% to match
Casino) via `run_camlds_v2_regen.sh`, then rerun the M3 pipeline on it via
`run_milestone3_camlds_v2.sh` — output kept in a separate
`outputs_camlds_v2/` folder, never overwriting the original results, since
M2's entire EDA/ablation/RF-importance analysis was computed on the
*original* 54%-attack extraction and describes that specific corpus.

### 9b. First regeneration attempt failed — benign is not the limiting factor

Ran `run_camlds_v2_regen.sh` with `--benign_ratio 15.0` (6x the original
2.5). **Result: identical output** — 90,805 rows, 54.0% attack, unchanged.
Diagnosis from the log:

- **Attack pool: 49,022 — a hard ceiling.** "ALL viable" attack processes
  in the raw corpus, no technique even hits its per-technique cap. Cannot
  be increased without more raw data.
- **Benign pool: only 41,783 reached output, but 62,990 are viable.** Two
  scenarios (`screensharing_cron`: 30,016; `screensharing_binary`: 21,191)
  were individually clipped by the 15,000-per-scenario cap, losing
  ~21,000 processes combined.
- **Even fixing that cap isn't enough.** Best case with the FULL 62,990
  benign pool: attack rate = 49,022/(49,022+62,990) ≈ **43.8%** — nowhere
  near Casino's 17.1%. `benign_ratio` was the wrong lever the whole time;
  49,022 attack processes is simply too large relative to how much benign
  data exists in this corpus to dilute via upsampling alone.

**Corrected approach:** wrote `extract_caml_realistic_ratio.py`, which (1)
raises the benign-per-scenario cap to unlock the full pool, and (2)
**stratified-subsamples attack** down to hit the target rate — proportional
to each technique's original share, preserving relative technique
diversity instead of an uncontrolled cut. Solved directly:
`target_attack = benign_pool * rate / (1 - rate)` ≈ 12,993 attack processes
needed (down from 49,022) to hit 17.1% using the full benign pool.
Verified the stratified sampling logic on synthetic data (100/50/20 split
across 3 mock techniques, requesting 34/170 total) — got exactly the
proportional 20/10/4 split expected.

**This is a genuine methodological choice to disclose in the report, not
a data-source workaround**: *"we subsampled attack processes, stratified
by technique, to reach a target prevalence matching CasinoLimit."*
`run_camlds_v2_regen.sh` updated to call the new script.

### 9c. Size vs. rate tradeoff — can't have both ~90k rows AND 17.1%

Requirement clarified: keep the dataset near the original ~90,805 rows,
not just hit an exact rate. The two goals conflict given the real pool
ceilings:

```
Hit rate=17.1% exactly  -> total caps at 62,990/(1-0.171) ≈ 75,983 rows
Hit size=90,805 exactly -> best achievable rate = 27,815/90,805 ≈ 30.6%
```

Benign pool (62,990) simply isn't large enough to dilute 90k+ total rows
down to Casino-level prevalence. **Decision at the time: prioritize
matching the original row count.** `extract_caml_realistic_ratio.py`
extended with a `--target_total_size` mode (mutually exclusive with
`--target_attack_rate`) that uses the full benign pool + stratified
attack subsampling to hit an exact total size. `run_camlds_v2_regen.sh`
used `--target_total_size 90805`, expected ~30.6% attack. This tradeoff
was later superseded by §9d below, once it became clear the "hard"
ceilings weren't as hard as they looked.

### 9d. Full corpus discovery — the extraction only ever used 1.2% of available data

While investigating the size-vs-rate tradeoff, checked whether the "hard"
benign/attack pool ceilings (§9c) were real limits of the underlying data
or just limits of what the extraction script scans. They were the latter,
partially:

```
Extraction script found:  98 audit.log files  (28 scenario_* dirs)
Actually on disk:       8,103 audit.log files
```

`find_audit_logs()` only globs directories literally named `scenario_*` —
it has never looked inside `manifestations_filtered/` (2,252 files) or
`manifestations_raw/` (5,753 files), two sibling directories with
entirely different internal organization.

**Verified this is genuinely additive data, not a duplicate/reorganized
copy**, via three checks:
1. Audit.log format inside `manifestations_filtered` matches
   `parse_line()`'s expected conventions exactly (`type=SYSCALL`,
   `msg=audit(...)`, `key="T1078_Valid_Accounts"`).
2. Host-path structure (`<trial>/<host>/logs/log/audit/audit.log`) is
   already compatible with `get_host_name()`'s existing "find the `logs`
   path segment, take the preceding component" logic — no changes needed.
3. **Decisive check**: `manifestations_filtered/sequences/` contains 19
   separate numbered trial directories for `3_ssh_healthcheck` alone
   (trials 1, 3, 6, 7, 8, 25–36, some grouped e.g. `33_34`), against the
   single `scenario_3_ssh_healthcheck` directory currently in use.
   `md5sum` confirms distinct file content — independent repeated
   executions of the same scripted scenario, not duplicates. This roughly
   matches the corpus-wide multiplier (2,252/98 ≈ 23x).

**`manifestations_raw/`** (5,753 files) has a different internal
structure (`sequences/`, `steps/`, `techniques/` subdirectories) — not
yet explored for overlap with `manifestations_filtered`. Left out of the
new extraction to avoid a double-counting risk until checked.

**New script: `extract_caml_full_corpus.py`** — combines both sources
(`scenario_*` + `manifestations_filtered/sequences/`), normalizes
trial-grouped directory names to a scenario-family label (e.g.
`3_ssh_healthcheck-33_34` → `3_ssh_healthcheck`, verified against 6 real
directory names from this corpus) so per-scenario capping still groups
all trials of the same scenario together, and reuses the
`--target_total_size`/`--target_attack_rate` modes + stratified attack
subsampling from `extract_caml_realistic_ratio.py`. With ~23x more raw
data, hitting both the ~90.8k row target AND a meaningfully lower attack
rate simultaneously may now be achievable — untested at time of writing.
`run_camlds_full_corpus.sh` submits this as a SLURM job
(`--time=04:00:00`, unmeasured guess given ~23x more files to parse).

### 9e. Results — full corpus regeneration confirmed the hypothesis, dramatically

**Achieved**: 90,805 rows (exact target), 28.7% attack rate (down from
54.0%). Benign pool only grew ~3% (62,990→64,787) despite scanning 23x
more files — the additional trials are predominantly repeated *attack*
scenario executions, not extra benign background traffic. A property of
the testbed's scenario design, not an extraction limitation.

**Within-dataset IsoForest (CAM-LDS)**: AUROC 0.555→0.655 (+0.10, ~2.5
std above original mean) — real, moderate improvement. F1 barely moved
(+0.015) because precision fell (−0.24) as attack became a genuine
minority — expected math, not a null result; AUROC is the metric that
actually tests the hypothesis, and it moved clearly.

**Cross-dataset — this is the headline result of the entire project**:

| Model | CAM→Casino AUROC (orig) | CAM→Casino AUROC (v3) | Delta |
|---|---|---|---|
| RF | 0.024±0.002 | 0.897±0.151 | **+0.873** |
| XGBoost | 0.203±0.281 | 0.961±0.001 | **+0.758** |
| CNN | 0.462±0.296 | 0.765±0.001 | **+0.303** |
| Isolation Forest | 0.014±0.007 | 0.015±0.007 | +0.001 (unchanged) |

RF/XGBoost/CNN went from near-total ranking inversion to genuinely strong
transfer — XGBoost's 0.961 is nearly as good as within-dataset
performance, and far more STABLE too (std 0.281→0.001). Direct,
dramatic confirmation that the curated 54% training distribution was
actively teaching these models CAM-LDS-specific patterns that inverted
on Casino.

**Isolation Forest is the clean negative control**: the identical fix
that transformed the other three models did nothing for IsoForest's
cross-dataset AUROC (0.014→0.015). This independently corroborates §5's
novelty-mode finding from a completely different angle — two separate
experiments now agree IsoForest's failure is mechanistic (the isolation
splitting approach itself), not primarily a training-distribution
artifact.

**The remaining gap is a different problem**: XGBoost now
has AUROC=0.961 but F1≈0.0008 — not a ranking failure, apparently a
**threshold miscalibration** (still calibrated for CAM-LDS's 28.7%
prevalence, wrong for Casino's 17.1%). Added `--calibration-frac 0.05` to
`run_cross_dataset_v3.sh` to test this — see §9f, result was negative,
with an important caveat.

### 9f. Recalibration result — bug fixed, then a clean and decisive finding

The initial `--calibration-frac 0.05` test (documented in an earlier draft
of this section) had a real bug: `GroupShuffleSplit` allocates a fraction
of *unique groups*, not rows. Casino's 356 unique groups (99.6%
duplication) meant 5% of groups landed on just **35 rows** — far too few
to derive a reliable threshold from, even given excellent AUROC.

**Fix**: added `row_proportional_group_split()` to `cross_dataset_eval.py`
— shuffles groups randomly and accumulates whole groups (preserving
leak-freeness) until the cumulative ROW count reaches the target fraction,
rather than the group count. Verified on synthetic Casino-scale data (356
groups, 91,692 rows, Dirichlet-skewed sizes): went from ~35 rows to 7,586
rows for the same 5% request — no group leakage. Real run achieved 5,398
rows (CAM-LDS→Casino) and 8,259 rows (Casino→CAM-LDS).

**With a proper sample, the result is clean and theory-confirming**:

| Model | Direction | AUROC | Before F1 | After F1 | Delta |
|---|---|---|---|---|---|
| RF | CAM→Casino | 0.897 | 0.000 | 0.644 ± 0.270 | **+0.644** |
| XGBoost | CAM→Casino | 0.961 | 0.001 | 0.846 ± 0.116 | **+0.845** |
| CNN | CAM→Casino | 0.765 | 0.064 | 0.799 ± 0.049 | **+0.734** |
| Isolation Forest | CAM→Casino | 0.015 | 0.292 | 0.002 ± 0.002 | −0.290 |
| RF | Casino→CAM | 0.605 | 0.445 | 0.021 ± 0.015 | −0.424 |
| XGBoost | Casino→CAM | 0.480 | 0.446 | 0.005 ± 0.000 | −0.441 |
| CNN | Casino→CAM | 0.418 | 0.448 | 0.037 ± 0.023 | −0.410 |
| Isolation Forest | Casino→CAM | 0.482 | 0.446 | 0.107 ± 0.069 | −0.338 |

**Every model with strong CAM→Casino AUROC (RF, XGBoost, CNN — all
0.76–0.96) recovered massively (+0.64 to +0.85 F1)** once given a real
calibration sample — decisive confirmation that the entire prior gap in
this direction was pure threshold miscalibration, not a ranking problem.

**Isolation Forest is the one CAM→Casino model that did NOT recover**
(0.29→0.002) — its AUROC there was still near-zero (0.015), so there was
no real ranking underneath for any threshold to exploit. Third
independent confirmation (after novelty mode, §5, and the corpus
regeneration negative control, §9e) that IsoForest's failure is
mechanistic, not distributional or a calibration artifact.

**Every model got WORSE in Casino→CAM-LDS**, consistently. Not a
methodological artifact this time — that direction's AUROC was always
unstable/weak (0.41–0.61), and a trustworthy calibration sample now
reliably finds a degenerate near-zero-recall solution rather than
helping. Confirms recalibration specifically requires a genuinely good
ranking to exploit; it cannot manufacture one.

---

## 10. Files and what they do (current state)

| File | Role |
|---|---|
| `config.py` | All hyperparameters, `MAX_FPR=0.05`, `IFOREST_NOVELTY_MODE=True`, `LOG1P_CONTINUOUS_FEATURES=True` |
| `ingestion.py` | Only dataset-specific module |
| `preprocessing.py` | Vector-group keys, log1p, scaling, class weights |
| `feature_selection.py` | Full vs. reduced feature policy (full is used — see §1) |
| `models.py` | 4 model builders, explicit hyperparameters only |
| `train_eval.py` | Core CV loop, F1-threshold logic, all 4 fitters, degenerate-split guard |
| `run_pipeline.py` | CLI entry point — within-dataset headline CV |
| `sensitivity_analysis.py` | Hyperparameter sensitivity sweep |
| `operating_curve.py` | FPR-cap sweep (used to derive `MAX_FPR=0.05`) |
| `cross_dataset_eval.py` | Train-on-one/test-on-other, with repeats + recalibration experiment |
| `run_milestone3.sh` | SLURM array: headline CV, 8 combos |
| `run_sensitivity.sh` | SLURM array: sensitivity sweep, 8 combos |
| `run_operating_curve.sh` | SLURM array: FPR sweep, RF only, both datasets |
| `run_cross_dataset.sh` | SLURM array: cross-dataset, 4 models, now with `REPEATS=5` |
| `run_cross_dataset_repeats.sh` | Ad-hoc single-model repeated cross-dataset run |
| `run_camlds_v2_regen.sh` | **NEW** — regenerates CAM-LDS at realistic attack rate |
| `run_milestone3_camlds_v2.sh` | **NEW** — runs headline pipeline on the regenerated variant |
| `results_report.html` | Standalone results explainer for the report |

---

## 11. Current best results (for reference — see `results_report.html` for full context)

**Within-dataset headline** (MAX_FPR=0.05, log1p on, novelty mode on, all fixes applied):
| Dataset | Model | F1 | AUROC |
|---|---|---|---|
| CAM-LDS | RF | 0.855±0.020 | 0.941±0.007 |
| CAM-LDS | XGBoost | 0.850±0.034 | 0.935±0.005 |
| CAM-LDS | CNN | 0.816±0.043 | 0.878±0.059 |
| CAM-LDS | IsoForest | 0.221±0.008 | 0.555±0.041 |
| Casino | RF | 0.965±0.038 | 0.998±0.003 |
| Casino | XGBoost | 0.919±0.105 | 0.995±0.002 |
| Casino | CNN | 0.518±0.369 | 0.901±0.126 |
| Casino | IsoForest | 0.773±0.154 | 0.821±0.178 |

Pre-log1p / pre-fix results preserved in `outputs_v1/`,
`sensitivity_results_v1/`, `operating_curve_results_v1/` for before/after
comparison in the report.

---

## 12. Open items — status update

- **Hybrid cascade** (Isolation Forest first-pass filter → supervised
  classifier) — **done.** `hybrid_cascade.py` / `hybrid_cascade_weighted.py`
  implemented and validated; see `#15` for the confound-free A/B follow-ups
  (blend formula, Stage-2 model choice) and the added FNR metric.

- **Sample-level error analysis** — **done.** Per-technique FN rates,
  benign FP rates, FN-vs-TP and FP-vs-TN feature comparisons, and pairwise
  model agreement/disagreement counts computed for all four models on both
  datasets (`results/current/error_analysis_results/`). Notable finding
  worth citing: Isolation Forest misses 96.2% of `T1166_Seuid_and_Setgid`
  and 94.3% of `T1078_Valid_Accounts` on CAM-LDS — far higher FN rates than
  the other three models on those techniques.

- **CAM-LDS regeneration** — done, under a v3 pipeline
  (`data/v3`, `results/current/outputs_camlds_v3/`,
  `run_scripts/run_cross_dataset_v3_recal.sh`, 15-repeat protocol). No v2
  was ever built — the project went directly to v3.

- **XGBoost sensitivity sweep** — the original 3-repeat sweep never
  finished (5hr timeout, partial grid). Resolved differently:
  `verify_hyperparameter_candidate.py` tested XGBoost (and RF, MLP,
  Isolation Forest) candidates directly against production under the full
  15-repeat headline protocol instead of finishing the sweep. Verdict: no
  real difference for XGBoost on either dataset (CAM-LDS Δ F1=−0.030,
  signal/noise=0.25; CasinoLimit `learning_rate=0.1`, Δ F1=+0.010,
  signal/noise=0.07) — production config kept. Same check found one
  genuine win (Isolation Forest, CAM-LDS, `n_estimators=400/max_samples=512`,
  Δ F1=+0.045, signal/noise=1.41 — adopted, see `#14`); RF and MLP
  candidates also showed no real difference on both datasets.

---

## 13. Deep Learning model swap: MLP for CasinoLimit

**What changed:** added `MLP_PARAMS` (`src/config.py`) — a deep MLP with Batch
Normalization and Dropout — as the Deep Learning paradigm representative for
CasinoLimit, replacing the 1D-CNN used in the Milestone 2 report.

**Why:** per M2 reviewer feedback, CasinoLimit's process traces lack the
sequential depth a CNN/LSTM-style architecture needs (98.2% single-event
processes). Head-to-head comparison (`compare_cnn_vs_mlp.py`, 15-repeat
protocol, CasinoLimit):

| Model | F1 | AUROC |
| --- | --- | --- |
| CNN | 0.635 ± 0.264 | 0.793 ± 0.255 |
| MLP | 0.862 ± 0.089 | 0.993 ± 0.019 |

MLP wins on both mean performance and stability by a wide margin.

**Scope:** CasinoLimit only. CAM-LDS keeps its CNN — it has genuine sequence
depth (10.87 events/process average vs. CasinoLimit's 1.08), so the original
M2 Ch6.1 justification for a sequence-aware architecture still applies there
and was not re-tested.

**Related:** `src/models.py`, `src/config.py` (`SENSITIVITY_GRIDS["mlp"]`),
`results/current/cnn_vs_mlp/`, `results/current/error_analysis_results/{camlds,casino}_mlp/`.

---

## 14. Isolation Forest hyperparameter retune (CAM-LDS)

**What changed:** `IFOREST_PARAMS.n_estimators` 200→400 and `max_samples`
256→512.

**Why:** verified via `verify_hyperparameter_candidate.py` under the full
15-repeat protocol on CAM-LDS: F1 improved 0.261 → 0.306 (+0.045,
signal/noise ratio 1.41 — a genuine improvement, not sweep noise).

**Scope:** applies dataset-wide, since `IFOREST_PARAMS` isn't currently split
per-dataset — but the supporting evidence is CAM-LDS-specific. CasinoLimit's
own sweep found no defensible candidate (every configuration tested had
std > 0.2, too noisy to act on), so its Isolation Forest is unchanged in
substance, just inheriting the new shared values.

**Related:** `results/current/hyperparameter_verification/camlds_iforest_*`.

---

---

## 15. Cascade design validation: confound-free A/B tests + missing FNR metric

Three follow-up scripts closing gaps the report template requires but the
original `hybrid_cascade.py` / `hybrid_cascade_weighted.py` split didn't
cleanly answer, because those are separate scripts that each independently
retrain XGBoost + CNN per fold — with CNN's run-to-run instability
(FP rate observed 0.33%–98.00% across 6 runs at the same seed, per Report
Sec. 2.1), naively comparing their numbers conflates "which design is
better" with "which run got luckier." All three scripts below fix this by
training Stage 1 (and, where relevant, Stage 2) **once per fold** and
reusing those exact fitted models/predictions across every variant being
compared, so any metric difference is attributable only to the thing
actually being tested.

**`cascade_blend_formula_ab.py` — flat 50/50 vs. distance-weighted blend**
(single seed, 5-fold, XGBoost+CNN trained once per fold, reused for both
blend formulas):

| Dataset | flat F1 | weighted F1 | Δ |
| --- | --- | --- | --- |
| CAM-LDS | 0.8475 ± 0.1327 | 0.8627 ± 0.1237 | +0.0152 |
| CasinoLimit (4/5 usable folds — fold 0 degenerate, single-class test fold) | 0.9832 ± 0.0134 | 0.9900 ± 0.0102 | +0.0068 |

Distance-weighted blending (full Stage-2 weight at the band center, fading
to zero at the edges) beats flat 50/50 on both datasets under this
confound-free setup — confirms `hybrid_cascade_weighted.py`'s design choice
was a genuine improvement, not an artifact of a lucky CNN run.

**`cascade_stage2_model_ab.py` — CNN vs. MLP as the Stage-2 model**
(same confound control, weighted-blend formula held fixed, Stage 1 +
fold split shared):

| Dataset | CNN F1 | MLP F1 | Δ |
| --- | --- | --- | --- |
| CAM-LDS | 0.8478 ± 0.1266 | 0.8355 ± 0.1353 | −0.0123 |
| CasinoLimit (4/5 usable folds) | 0.9900 ± 0.0101 | 0.9900 ± 0.0101 | +0.0000 |

Confirms Ch. 6's dataset-specific model choice is directionally correct at
the cascade level too: CNN is (slightly) better on CAM-LDS's sequence-rich
data, the two are indistinguishable on CasinoLimit under this single-seed
check (contrast with the standalone 15-repeat MLP-vs-CNN comparison in
`#13`, which is the more statistically reliable number for that dataset).

**`compute_cascade_fnr.py` — adds the FNR metric the report template asks
for** (S3.2), which the original cascade CV loop never recorded. Reuses
`hybrid_cascade.py`'s actual `cascade_predict()` unmodified, full 15-repeat
pooled protocol (matching every other headline number in the project, not
a cheap exploratory sweep):

| Dataset | F1 | AUROC | FPR | FNR |
| --- | --- | --- | --- | --- |
| CAM-LDS | 0.8524 ± 0.0278 | 0.9296 ± 0.0151 | 0.2143 ± 0.0572 | 0.1221 ± 0.0370 |
| CasinoLimit | 0.9393 ± 0.0671 | 0.9357 ± 0.0453 | 0.1298 ± 0.0678 | 0.0747 ± 0.0829 |

**`cascade_threshold_sweep.py`** also ships in this branch — sweeps the
cascade's uncertain-band half-width (0.10–0.30, vs. the never-validated
default of 0.20) with per-config repeats to average out CNN instability.
⚠️ **Caveat:** its result files (`results/current/cascade_threshold_sweep/`)
are byte-identical on `main` already — worth confirming with whoever ran it
whether that's a prior partial merge or the script reproducing the same
output, before citing sweep numbers as new in this entry. Left out here
rather than guessed at.

**Related files:** `src/cascade_blend_formula_ab.py`,
`src/cascade_stage2_model_ab.py`, `src/cascade_threshold_sweep.py`,
`src/compute_cascade_fnr.py`, `run_scripts/run_cascade_fnr.sh`,
`results/current/cascade_blend_formula_ab/`,
`results/current/cascade_stage2_model_ab/`, `results/current/cascade_final/`.
