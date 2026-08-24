"""
Milestone 3 -- Step 8 capstone: Hybrid Behavioural Cascading (confidence-routed)
=================================================================================
DEVIATES from the brief's example architecture (unsupervised IsolationForest
first-line filter -> supervised second stage), based on empirical evidence
from error_analysis.py's pairwise disagreement output:

  - IsoForest's standalone AUROC on CAM-LDS is 0.555 -- barely better than
    random. Using it as a "trusted" first-pass filter would mean discarding
    its flags almost arbitrarily; it is not fit for that role here.
  - RF vs XGB disagree on only 1.9% of CAM-LDS rows (and an even smaller
    0.1% of Casino rows) -- they make almost identical errors, so cascading
    them together adds ~nothing.
  - XGB vs CNN disagree on 2.8% of CAM-LDS rows, split almost exactly 50/50
    on which one is right when they do -- genuine complementary signal.
  - XGB vs CNN disagree on 85.8% of Casino rows, but XGB is right in nearly
    all of those cases (78,572 vs 71) -- this is CNN's known instability on
    Casino (standalone F1 = 0.518 +/- 0.369), not complementary insight.

Design: XGBoost is Stage 1 (fast, strong on both datasets) and produces a
per-row probability. Rows where XGBoost is CONFIDENT (below LOW_THRESH or
above HIGH_THRESH) are accepted directly. Rows in the UNCERTAIN band are
routed to Stage 2 (CNN) for a second opinion, and the two probabilities are
blended. This only pays the CNN's cost on the ambiguous slice of traffic,
and lets the models' genuine disagreements -- not just their individual
accuracy -- do the work.

Expected, and reported honestly either way: this should meaningfully help
CAM-LDS (real complementary pair) and add little-to-nothing on Casino
(CNN's instability dominates). That asymmetry is itself a finding for the
report, not a bug to chase away.

Usage:
    python hybrid_cascade.py --dataset camlds --data-dir data --out results/current/hybrid_cascade/
    python hybrid_cascade.py --dataset casino --data-dir data --out results/current/hybrid_cascade/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import (vector_group_key, log1p_continuous,
                            fit_scaler_on_train, xgb_scale_pos_weight)
from models import build_xgb, build_cnn, cnn_fit_kwargs
# NOTE: these two are underscore-prefixed (module-private) in train_eval.py,
# imported directly here to avoid duplicating the F1-optimal thresholding /
# FPR logic. If train_eval.py's interface changes, this import breaks first --
# consider promoting them to public functions if this module becomes permanent.
from train_eval import _best_f1_threshold, _fpr

# Same binary-feature convention used elsewhere in the pipeline (log1p should
# skip these, they're already 0/1).
BINARY_FEATURES = {
    'uid_is_root', 'uid_changed', 'euid_root', 'is_suid_exec',
    'auid_euid_mismatch', 'ephemeral_privileged', 'shell_from_service',
    'sensitive_path_access',
}


def cascade_predict(X_tr, y_tr, X_te, feat_cols, max_fpr,
                     low_thresh=0.3, high_thresh=0.7):
    """Fit XGB (stage 1) + CNN (stage 2) on the train fold, predict on the
    test fold with confidence-based routing.

    Returns (y_pred, final_proba, uncertain_mask) -- the mask lets the
    caller report what fraction of rows actually needed the expensive
    stage-2 model (this is the efficiency argument for the cascade design).
    """
    # -- Stage 1: XGBoost, same recipe as train_eval._fit_predict_xgb --
    X_tr_log = log1p_continuous(X_tr, feat_cols, BINARY_FEATURES)
    X_te_log = log1p_continuous(X_te, feat_cols, BINARY_FEATURES)
    spw = xgb_scale_pos_weight(y_tr)
    xgb = build_xgb(scale_pos_weight=spw)
    xgb.fit(X_tr_log, y_tr)
    xgb_proba = xgb.predict_proba(X_te_log)[:, 1]

    # -- Stage 2: CNN, only actually trained/run because we need it for the
    #    uncertain band -- but note the CNN is still fit on the FULL train
    #    fold (not just uncertain rows), same as it would be standalone.
    uncertain = (xgb_proba >= low_thresh) & (xgb_proba <= high_thresh)
    final_proba = xgb_proba.copy()

    if uncertain.sum() > 0:
        scaler = fit_scaler_on_train(X_tr_log)
        X_tr_scaled = scaler.transform(X_tr_log)
        X_te_scaled = scaler.transform(X_te_log)

        cnn = build_cnn(n_features=X_tr_scaled.shape[1])
        cnn.fit(X_tr_scaled, y_tr, **cnn_fit_kwargs())
        cnn_proba = cnn.predict(X_te_scaled[uncertain], verbose=0).ravel()

        # Blend: simple average on the uncertain band only. (A learned
        # blend weight is a reasonable follow-up if this simple version
        # underperforms XGBoost alone on CAM-LDS.)
        final_proba[uncertain] = 0.5 * xgb_proba[uncertain] + 0.5 * cnn_proba

    # Threshold chosen on TRAIN predictions only (never test) -- same
    # leak-free convention as the rest of the pipeline.
    xgb_train_proba = xgb.predict_proba(X_tr_log)[:, 1]
    thresh = _best_f1_threshold(y_tr, xgb_train_proba, max_fpr=max_fpr)
    y_pred = (final_proba >= thresh).astype(int)
    return y_pred, final_proba, uncertain


def run_cascade_cv(dataset_key, data_dir, out_dir, max_fpr, n_splits=5, seed=42):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        n_classes_te = len(set(y[te]))
        if n_classes_te < 2:
            # Degenerate fold: test split contains only one class (known risk
            # with StratifiedGroupKFold on CasinoLimit's ~356 unique
            # vector-groups -- see Milestone2_Report.docx Sec 5.3, and the
            # same reason train_eval.py/unified_validation.py use pooled
            # repeated CV instead of naive per-fold averaging). Scoring this
            # fold at all -- F1=0, AUROC=nan -- silently drags the mean down
            # with a number that reflects the SPLIT, not the model. Skip it
            # from the aggregate and say so explicitly, rather than average
            # a broken fold in.
            print(f"[{dataset_key}] fold {fold}: SKIPPED -- degenerate test "
                  f"fold ({n_classes_te} class present, {len(te)} rows). "
                  f"Not counted in the summary below.")
            rows.append({'fold': fold, 'f1': np.nan, 'auroc': np.nan, 'fpr': np.nan,
                         'pct_routed_to_stage2': np.nan, 'degenerate': True})
            continue

        y_pred, y_proba, uncertain = cascade_predict(
            X[tr], y[tr], X[te], feat_cols, max_fpr)
        f1 = f1_score(y[te], y_pred, zero_division=0)
        auc = roc_auc_score(y[te], y_proba)
        fpr = _fpr(y[te], y_pred)
        routed_pct = 100 * uncertain.mean()
        rows.append({'fold': fold, 'f1': f1, 'auroc': auc, 'fpr': fpr,
                     'pct_routed_to_stage2': routed_pct, 'degenerate': False})
        print(f"[{dataset_key}] fold {fold}: F1={f1:.4f} AUROC={auc:.4f} "
              f"FPR={fpr:.4f}  routed_to_CNN={routed_pct:.1f}%")

    result_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"{dataset_key}_cascade_cv_results.csv", index=False)

    valid = result_df[~result_df['degenerate']]
    n_skipped = result_df['degenerate'].sum()
    print(f"\n[{dataset_key}] Cascade summary ({len(valid)}/{len(result_df)} folds used, "
          f"{n_skipped} skipped as degenerate): "
          f"F1={valid['f1'].mean():.4f}+/-{valid['f1'].std():.4f}  "
          f"AUROC={valid['auroc'].mean():.4f}+/-{valid['auroc'].std():.4f}  "
          f"avg routed to Stage 2={valid['pct_routed_to_stage2'].mean():.1f}%")
    return result_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Hybrid cascade: XGBoost (stage 1, all rows) -> CNN "
                     "(stage 2, uncertain-confidence rows only)")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/current/hybrid_cascade")
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--low-thresh", type=float, default=0.3,
                    help="Below this XGB probability, accept 'benign' directly")
    p.add_argument("--high-thresh", type=float, default=0.7,
                    help="Above this XGB probability, accept 'malicious' directly")
    args = p.parse_args()

    run_cascade_cv(args.dataset, args.data_dir, args.out, args.max_fpr)
