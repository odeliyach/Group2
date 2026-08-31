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
