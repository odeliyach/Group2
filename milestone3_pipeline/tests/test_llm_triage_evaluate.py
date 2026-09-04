import json
import numpy as np
import pandas as pd
import llm_triage_evaluate as E


def _arb():
    # 4 usable rows + 1 parse_error + 1 call_error that must be ignored by arbitration_accuracy
    def kf(x):
        return json.dumps(x)
    return pd.DataFrame([
        dict(row_id=0, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.45, proba_cnn=0.80,
             llm_pred=1, llm_confidence=0.9, parse_status="ok", key_features=kf(["priv_op_count"])),
        dict(row_id=1, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.60, proba_cnn=0.30,
             llm_pred=0, llm_confidence=0.8, parse_status="ok", key_features=kf(["network_count"])),
        dict(row_id=2, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.40, proba_cnn=0.66,
             llm_pred=1, llm_confidence=0.7, parse_status="repaired", key_features=kf(["exec_count"])),
        dict(row_id=3, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.55, proba_cnn=0.20,
             llm_pred=1, llm_confidence=0.6, parse_status="ok", key_features=kf(["seq_length"])),
        dict(row_id=4, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.50, proba_cnn=0.70,
             llm_pred=-1, llm_confidence=None, parse_status="parse_error", key_features=kf([])),
        dict(row_id=5, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.55, proba_cnn=0.20,
             llm_pred=-1, llm_confidence=None, parse_status="call_error", key_features=kf([])),
    ])


def test_arbitration_accuracy_excludes_parse_errors_and_scores_llm():
    acc = E.arbitration_accuracy(_arb()).set_index("strategy")
    assert acc.loc["llm", "n"] == 4
    # llm preds on rows 0,1,2,3 = [1,0,1,1]; truth = [1,0,1,0] -> 3/4
    assert abs(acc.loc["llm", "accuracy"] - 0.75) < 1e-9
    # always_xgb preds = [0,1,0,1]; truth = [1,0,1,0] -> 0/4
    assert abs(acc.loc["always_xgb", "accuracy"] - 0.0) < 1e-9
    # always_cnn preds = [1,0,1,0] -> 4/4
    assert abs(acc.loc["always_cnn", "accuracy"] - 1.0) < 1e-9


def test_arbitration_accuracy_excludes_call_errors():
    acc = E.arbitration_accuracy(_arb()).set_index("strategy")
    # Should exclude both parse_error (row 4) and call_error (row 5), leaving 4 usable rows
    assert acc.loc["llm", "n"] == 4


def test_arbitration_accuracy_handles_all_rows_failed_without_crashing():
    # Reproduces the real failure: every LLM call errored, so nothing survives
    # the parse_status filter. precision/recall/f1 must not raise on an empty
    # y_true/y_pred (sklearn does by default) -- they should read as NaN.
    cols = ["row_id", "true_label", "pred_xgb", "pred_cnn", "proba_xgb", "proba_cnn",
            "llm_pred", "llm_confidence", "parse_status", "key_features"]
    all_failed = pd.DataFrame([
        dict(row_id=0, true_label=1, pred_xgb=0, pred_cnn=1, proba_xgb=0.45, proba_cnn=0.80,
             llm_pred=-1, llm_confidence=None, parse_status="call_error", key_features="[]"),
        dict(row_id=1, true_label=0, pred_xgb=1, pred_cnn=0, proba_xgb=0.60, proba_cnn=0.30,
             llm_pred=-1, llm_confidence=None, parse_status="parse_error", key_features="[]"),
    ], columns=cols)
    acc = E.arbitration_accuracy(all_failed).set_index("strategy")
    assert acc.loc["llm", "n"] == 0
    assert np.isnan(acc.loc["llm", "accuracy"])
    assert np.isnan(acc.loc["llm", "precision"])
    assert np.isnan(acc.loc["llm", "recall"])
    assert np.isnan(acc.loc["llm", "f1"])


def test_arbitration_accuracy_handles_zero_contradiction_rows():
    # Reproduces the other real case: a dataset with no contradiction rows at
    # all this run -- an empty but properly-columned arbitration frame.
    cols = ["row_id", "true_label", "pred_xgb", "pred_cnn", "proba_xgb", "proba_cnn",
            "llm_pred", "llm_confidence", "parse_status", "key_features"]
    empty = pd.DataFrame(columns=cols)
    acc = E.arbitration_accuracy(empty)
    assert (acc["n"] == 0).all()
    assert acc["accuracy"].isna().all()


def test_pipeline_effect_substitutes_llm_on_contradiction_rows():
    oof = pd.DataFrame({
        "row_id": [0, 1, 2, 3, 4, 5],
        "fold": [0, 0, 1, 1, 0, 1],
        "true_label": [1, 0, 1, 0, 1, 0],
        "proba_xgb": [0.45, 0.60, 0.40, 0.55, 0.20, 0.90],
        "proba_cnn": [0.80, 0.30, 0.66, 0.20, 0.10, 0.95],
    })
    eff, delta = E.pipeline_effect(_arb(), oof)
    assert set(eff["pipeline"]) == {"cascade@0.5thr", "cascade+llm"}
    assert "fold" in eff.columns
    assert "pooled" in set(eff["fold"])
    # one (cascade@0.5thr, cascade+llm) pair per fold + the pooled pair
    assert (eff["fold"] == "pooled").sum() == 2
    assert set(eff.loc[eff["fold"] == "pooled", "pipeline"]) == {"cascade@0.5thr", "cascade+llm"}
    assert "f1" in delta.index and "fpr" in delta.index


def test_expected_calibration_error_zero_when_perfect():
    conf = np.array([0.05, 0.25, 0.55, 0.95])
    correct = np.array([0.0, 0.0, 1.0, 1.0])  # matches bin midpoints closely
    ece = E.expected_calibration_error(conf, correct, n_bins=4)
    assert ece < 0.3


def test_expected_calibration_error_high_when_inverted():
    conf = np.array([0.99, 0.99, 0.99, 0.99])
    correct = np.array([0.0, 0.0, 0.0, 0.0])
    assert E.expected_calibration_error(conf, correct, n_bins=10) > 0.9


def test_key_feature_overlap():
    df = _arb()
    frac = E.key_feature_overlap(df, ["priv_op_count", "network_count"])
    # 5 rows, key_features non-empty on 4, intersect on rows 0 and 1 -> 2/4
    assert abs(frac - 0.5) < 1e-9


def test_discriminator_features_present_for_both_datasets():
    assert set(E.DISCRIMINATOR_FEATURES) == {"camlds", "casino"}
