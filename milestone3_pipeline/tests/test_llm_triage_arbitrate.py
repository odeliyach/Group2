import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

import config
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


# --- integration: reuse path, real context builder, stubbed model -------------

class _RecordingClient:
    """Records every context string it is handed; returns a schema-valid dict."""
    def __init__(self):
        self.contexts = []

    def arbitrate(self, context_str):
        self.contexts.append(context_str)
        return {"verdict": "MALICIOUS", "confidence": 0.8,
                "rationale_steps": ["s1", "s2", "s3"],
                "key_features": ["priv_op_count"],
                "agrees_with": "cnn", "parse_status": "ok"}


class _FakeDS:
    feat_cols = list(config.FULL_FEATURE_LIST)
    df = pd.DataFrame({f: [0.0] for f in config.FULL_FEATURE_LIST})


def _fake_pct(df, feat_cols, label_col):
    bp = [float(i) for i in range(1, 100)]
    return {f: {"benign": list(bp), "attack": list(bp)} for f in feat_cols}


def test_llm_triage_arbitrate_integration(tmp_path, monkeypatch):
    feats = list(config.FULL_FEATURE_LIST)
    contra = pd.DataFrame({
        "row_id": [0, 1],
        "fold": [0, 1],
        "true_label": [1, 0],
        "technique": ["T1548.001", "-"],           # label leak -- must NOT reach the model
        "pred_xgb": [0, 1],
        "pred_cnn": [1, 0],
        "proba_xgb": [0.45, 0.55],
        "proba_cnn": [0.70, 0.30],
        "disagreement_direction": ["xgb_benign_cnn_malicious", "xgb_malicious_cnn_benign"],
    })
    for j, f in enumerate(feats):
        contra[f] = [1.0 + j, 2.0 + j]
    contra.to_csv(tmp_path / "camlds_contradictions.csv", index=False)
    # Only needs to EXIST for the reuse branch; arbitrate never reads its contents.
    pd.DataFrame({"row_id": [0, 1]}).to_csv(tmp_path / "camlds_cascade_oof.csv", index=False)

    monkeypatch.setattr("ingestion.ingest", lambda *a, **k: _FakeDS())
    monkeypatch.setattr("llm_triage_stats.compute_percentile_tables", _fake_pct)

    stub = _RecordingClient()
    df = A.arbitrate_dataset(dataset_key="camlds", data_dir=str(tmp_path),
                             out_dir=tmp_path, client=stub)

    out_csv = tmp_path / f"camlds_{A._modeltag(config.LLM_TRIAGE['model_id'])}_llm_arbitration.csv"
    assert out_csv.exists()
    written = pd.read_csv(out_csv)
    assert len(written) == 2

    # label-leak guard on the REAL context path
    assert len(stub.contexts) == 2
    for ctx in stub.contexts:
        assert "technique" not in ctx
        assert "T1548" not in ctx

    # assemble_row's output columns are all present
    for col in ("row_id", "fold", "true_label", "technique", "llm_verdict", "llm_pred",
                "llm_confidence", "llm_agrees_with", "parse_status", "rationale_steps",
                "key_features"):
        assert col in written.columns
    assert list(df["llm_pred"]) == [1, 1]
