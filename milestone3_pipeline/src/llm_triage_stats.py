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
