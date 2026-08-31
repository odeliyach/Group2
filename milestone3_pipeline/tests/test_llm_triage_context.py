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
