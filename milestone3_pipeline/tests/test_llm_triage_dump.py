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
