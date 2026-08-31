import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
