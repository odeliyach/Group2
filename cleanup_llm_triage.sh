#!/bin/bash
# Run this from your repo root AFTER merging
# origin/feat/llm-triage-contextual-arbitration into main, BEFORE committing
# the merge (i.e. right after `git merge --no-commit --no-ff origin/feat/...`).
#
#   git checkout main
#   git pull
#   git merge --no-commit --no-ff origin/feat/llm-triage-contextual-arbitration
#   ./cleanup_llm_triage.sh
#   git add -A
#   git commit -m "Merge feat/llm-triage-contextual-arbitration; drop LLM triage layer (redundant)"
#
# If your merge auto-committed already, just run this script and commit the
# result as a normal follow-up commit instead.

set -e

echo "== Removing LLM-only files/dirs =="
git rm -rf --ignore-unmatch \
  milestone3_pipeline/src/llm_triage_arbitrate.py \
  milestone3_pipeline/src/llm_triage_client.py \
  milestone3_pipeline/src/llm_triage_context.py \
  milestone3_pipeline/src/llm_triage_dump.py \
  milestone3_pipeline/src/llm_triage_evaluate.py \
  milestone3_pipeline/src/llm_triage_prompts.py \
  milestone3_pipeline/src/llm_triage_stats.py \
  milestone3_pipeline/src/requirements-llm.txt \
  milestone3_pipeline/run_scripts/run_llm_triage.sh \
  milestone3_pipeline/results/current/llm_triage \
  milestone3_pipeline/tests \
  milestone3_pipeline/pytest.ini \
  docs/milestone3/llm_triage_runbook.md \
  docs/superpowers

echo "== Stripping LLM_TRIAGE / FEATURE_GLOSS block out of config.py =="
# Removes everything from the "Step 8 capstone -- LLM-Driven Triage" banner
# to the end of the file. Verify the marker text still matches your copy
# of config.py before trusting this blindly -- if it doesn't match, open
# config.py and delete the block by hand instead.
if grep -q "Step 8 capstone -- LLM-Driven Triage" milestone3_pipeline/src/config.py; then
  sed -i '/^# =====.*$/,$d' milestone3_pipeline/src/config.py
  # the line above deletes from the first "# =====..." separator onward,
  # which also eats the banner+FEATURE_GLOSS+LLM_TRIAGE block together.
else
  echo "  WARNING: marker not found in config.py -- edit it manually."
fi

echo "== Removing LLM-only deps from requirements.txt =="
sed -i '/^jsonschema/d;/^requests/d;/^pytest/d;/^tabulate/d' milestone3_pipeline/src/requirements.txt

echo "== Done removing files. Manual step still required: =="
echo "  - README.md: delete the 'LLM triage layer — implemented' bullet"
echo "    (keep the 'hybrid cascade — implemented' bullet above it)"
echo "  - docs/milestone3/CHANGELOG.md: delete the"
echo "    '## 13. LLM-driven triage & contextual arbitration' section at the end"
