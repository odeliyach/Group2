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


# Cap the rendered feature table to the most-deviating features rather than
# all of FULL_FEATURE_LIST (20). Most rows have ~15 features sitting near the
# 50th percentile in both populations -- uninformative for the rubric, but
# still tokens the model has to attend over on every single call, and every
# call's latency matters on CPU-only inference. notable_deviations() already
# ranks by |percentile - 50|, so reuse that ranking for the table itself
# instead of computing a separate 5-feature summary over the full list.
TOP_N_FEATURES_IN_TABLE = 8


def build_context(row, feat_cols, pct_tables, dataset_key, top_n=TOP_N_FEATURES_IN_TABLE):
    xgb_call = "MALICIOUS" if int(row["pred_xgb"]) == 1 else "BENIGN"
    cnn_call = "MALICIOUS" if int(row["pred_cnn"]) == 1 else "BENIGN"

    shortlist = notable_deviations(row, feat_cols, pct_tables, top_n=top_n)

    lines = [
        "=== RECORD UNDER REVIEW ===",
        (f"Model votes:  XGBoost -> {xgb_call} (p_malicious={float(row['proba_xgb']):.2f})   |   "
         f"CNN -> {cnn_call} (p_malicious={float(row['proba_cnn']):.2f})"),
        "",
        f"Behavioural features, top {len(shortlist)} most unusual vs benign "
        "(value | pctile vs benign pop | pctile vs attack pop | meaning)",
    ]
    for f in shortlist:
        v = float(row[f])
        pb = percentile_label(v, pct_tables[f]["benign"])
        pa = percentile_label(v, pct_tables[f]["attack"])
        lines.append(f"  {f:<22} {_fmt_value(v):<8} {pb:<6} {pa:<6} {FEATURE_GLOSS[f]}")

    lines += [
        "",
        "Most-deviating features, in order (first = most extreme):",
        "  " + ", ".join(shortlist[:5]),
        "",
        KNOWN_FAILURE_BLOCKS[dataset_key],
    ]
    return "\n".join(lines)
