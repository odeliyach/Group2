"""
Milestone 3 -- M2 reviewer point 1: "Do not discard informative features;
instead, find a way to incorporate them into the unified schema."

CasinoLimit lacks 3 features CAM-LDS has natively (is_suid_exec,
ephemeral_privileged, parent_child_rarity). The current approach drops them
entirely from cross-dataset work (17-feature intersection). This tests the
standard alternative -- missingness indicators -- rather than just
asserting it's out of scope:

  - The 3 missing columns are added to Casino's feature matrix, filled with
    0 (matching this project's existing .fillna(0) convention everywhere
    else, not NaN, since the downstream fitters already expect that).
  - A single shared binary indicator (`camlds_only_features_available`) is
    added to BOTH datasets: 1 for CAM-LDS rows (real values), 0 for Casino
    rows (filled values). One shared indicator, not three, since all three
    features have identical availability -- they're missing or present
    together, tied to the dataset, not per-row -- so three separate
    indicator columns would just be three copies of the same constant.
  - RF/XGBoost (tree-based, tolerant of this kind of encoding) are used to
    test whether the model can actually exploit "this dataset lacks these
    features" as signal, or whether it's a no-op / actively hurts.

This is a genuine experiment, not a proxy/estimate for the missing values
(that would require unverifiable assumptions about what those columns
"should" contain for Casino rows that were never extracted with them) --
it tests whether TELLING the model the features are absent, rather than
either faking values or omitting the columns, changes cross-dataset
transfer performance.

Usage:
    python verify_unified_schema_missingness.py --data-dir data \
        --out results/current/unified_schema_missingness/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from config import CV_CONFIG, MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, impute
from train_eval import _fit_predict_rf, _fit_predict_xgb, _best_f1_threshold, _fpr

CAMLDS_ONLY_FEATURES = ['is_suid_exec', 'ephemeral_privileged', 'parent_child_rarity']


def build_unified_matrix(df, feat_cols, is_camlds):
    """Adds the 3 CAM-LDS-only columns (0-filled if not native to this
    dataset) plus one shared availability indicator. Returns the full
    unified feature matrix and its column list, in a fixed, shared order
    so both datasets produce directly comparable matrices."""
    unified_cols = feat_cols + [c for c in CAMLDS_ONLY_FEATURES if c not in feat_cols]
    X = pd.DataFrame(index=df.index)
    for c in unified_cols:
        X[c] = df[c] if c in df.columns else 0.0
    X['camlds_only_features_available'] = 1 if is_camlds else 0
    return X, unified_cols + ['camlds_only_features_available']


def cross_dataset_transfer(fit_fn, X_train, y_train, groups_train, X_test, y_test, max_fpr, n_calib_frac=0.1, seed=42):
    """Train on the full source dataset, recalibrate the F1 threshold on a
    small labeled slice of the target domain (matching this project's
    existing cross-dataset methodology -- calibration problem, not a
    ranking failure, per the established S2.3 finding), evaluate on the rest."""
    rng = np.random.RandomState(seed)
    n_calib = int(len(X_test) * n_calib_frac)
    calib_idx = rng.choice(len(X_test), size=n_calib, replace=False)
    eval_idx = np.setdiff1d(np.arange(len(X_test)), calib_idx)

    pred, proba = fit_fn(X_train, y_train, groups_train, X_test, None, max_fpr)
    threshold = _best_f1_threshold(y_test[calib_idx], proba[calib_idx], max_fpr=max_fpr)
    pred_recal = (proba >= threshold).astype(int)

    return {
        'auroc': roc_auc_score(y_test[eval_idx], proba[eval_idx]),
        'f1_raw': f1_score(y_test[eval_idx], pred[eval_idx], zero_division=0),
        'f1_recalibrated': f1_score(y_test[eval_idx], pred_recal[eval_idx], zero_division=0),
        'fpr_recalibrated': _fpr(y_test[eval_idx], pred_recal[eval_idx]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_fn = _fit_predict_rf if args.model == "rf" else _fit_predict_xgb

    camlds = ingest("camlds", data_dir=args.data_dir)
    casino = ingest("casino", data_dir=args.data_dir)
    camlds_feat = select_features(camlds.feat_cols, "camlds", policy="full")
    casino_feat = select_features(casino.feat_cols, "casino", policy="full")

    print(f"CAM-LDS native features: {len(camlds_feat)}, Casino native: {len(casino_feat)}")

    # -- BASELINE: current approach, 17-feature intersection, no indicator --
    shared = [c for c in camlds_feat if c in casino_feat]
    print(f"Baseline (current approach): {len(shared)}-feature intersection")

    X_cam_base = impute(camlds.df[shared].values)
    X_cas_base = impute(casino.df[shared].values)
    groups_cam = vector_group_key(camlds.df, shared)

    baseline_cam_to_cas = cross_dataset_transfer(
        fit_fn, X_cam_base, camlds.y.values, groups_cam, X_cas_base, casino.y.values, MAX_FPR)
    print(f"[baseline] CAM-LDS->Casino: AUROC={baseline_cam_to_cas['auroc']:.4f} "
          f"F1_recal={baseline_cam_to_cas['f1_recalibrated']:.4f}")

    # -- CANDIDATE: unified schema with missingness indicator --
    X_cam_df, unified_cols = build_unified_matrix(camlds.df, camlds_feat, is_camlds=True)
    X_cas_df, _ = build_unified_matrix(casino.df, casino_feat, is_camlds=False)
    X_cam_uni = impute(X_cam_df[unified_cols].values)
    X_cas_uni = impute(X_cas_df[unified_cols].values)
    groups_cam_uni = vector_group_key(camlds.df, camlds_feat)  # grouping still on real features only

    candidate_cam_to_cas = cross_dataset_transfer(
        fit_fn, X_cam_uni, camlds.y.values, groups_cam_uni, X_cas_uni, casino.y.values, MAX_FPR)
    print(f"[candidate] CAM-LDS->Casino: AUROC={candidate_cam_to_cas['auroc']:.4f} "
          f"F1_recal={candidate_cam_to_cas['f1_recalibrated']:.4f}")

    summary = pd.DataFrame([
        {'direction': 'camlds_to_casino', 'variant': 'baseline_17feat', **baseline_cam_to_cas},
        {'direction': 'camlds_to_casino', 'variant': 'unified_schema_missingness', **candidate_cam_to_cas},
    ])
    summary.to_csv(out_dir / f"{args.model}_unified_schema_comparison.csv", index=False)

    delta_auroc = candidate_cam_to_cas['auroc'] - baseline_cam_to_cas['auroc']
    delta_f1 = candidate_cam_to_cas['f1_recalibrated'] - baseline_cam_to_cas['f1_recalibrated']
    verdict = (
        "GENUINE IMPROVEMENT -- unified schema with missingness indicator "
        "transfers better; worth adopting and citing as the M2 reviewer point 1 response"
        if delta_auroc > 0.01 and delta_f1 > 0.01 else
        "NO MEANINGFUL DIFFERENCE -- the model doesn't exploit the missingness "
        "signal; the 17-feature intersection is not costing real transfer performance"
        if abs(delta_auroc) < 0.01 else
        "WORSE -- adding filled/indicator columns hurts transfer here; the "
        "17-feature intersection remains the better choice"
    )
    print(f"\nDelta AUROC: {delta_auroc:+.4f}  Delta F1(recal): {delta_f1:+.4f}")
    print(f"VERDICT: {verdict}")
    (out_dir / f"{args.model}_verdict.txt").write_text(
        f"Delta AUROC: {delta_auroc:+.4f}\nDelta F1(recalibrated): {delta_f1:+.4f}\n"
        f"VERDICT: {verdict}\n")
    print(f"\nSaved: {out_dir}/")


if __name__ == "__main__":
    main()
