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
