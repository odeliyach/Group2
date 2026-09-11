"""
Milestone 3 -- Step 8: Hybrid Cascade variant -- confidence-WEIGHTED blend
=================================================================================
Follow-up to hybrid_cascade.py / hybrid_cascade_rf_xgb.py, which both use a
flat 50/50 blend for every row inside the uncertain band [0.3, 0.7]. Testing
the RF-vs-CNN stage-2 swap (hybrid_cascade_rf_xgb.py) DISPROVED the
hypothesis that stage-2 model choice was driving the CAM-LDS instability:
routing percentages were IDENTICAL fold-for-fold between the CNN and RF
variants (Stage 1/XGBoost decides routing, unchanged either way), and the
same folds (e.g. original CAM-LDS fold 1, 66.3% routed) were bad with BOTH
stage-2 models. That relocates the likely root cause from "which model
handles the uncertain band" to "the flat 50/50 blend dilutes XGBoost's own
signal too aggressively whenever a large fraction of a fold falls in the
band, regardless of what's doing the blending."

This variant tests that relocated hypothesis directly: instead of a flat
50/50 blend across the whole [0.3, 0.7] band, the stage-2 weight now scales
with distance from the band CENTER (0.5, XGBoost's most genuinely uncertain
point) down to 0.0 at the band EDGES (0.3 / 0.7, where XGBoost already has
a fairly clear lean and the cascade's "uncertain" flag is closer to a
technicality than genuine ambiguity):

    stage2_weight = 1 - |xgb_proba - 0.5| / 0.2      (0.2 = half-width of the band)
    final_proba   = (1 - stage2_weight) * xgb_proba + stage2_weight * stage2_proba

At proba=0.5 this is still a full 50/50 blend (same as before). At
proba=0.3 or 0.7 it smoothly degrades to trusting XGBoost alone, matching
the "accept directly" behavior just outside the nominal band. If the flat-
blend instability really was a dilution problem, this should measurably
reduce fold-to-fold variance -- especially on high-routing folds like
original CAM-LDS's fold 1, where routing volume was large.

Supports EITHER stage-2 model (--stage2 cnn|rf) so both prior variants get
the same fix tested; each stage-2 choice writes to its own output
directory, and neither overwrites the flat-blend results from
hybrid_cascade.py / hybrid_cascade_rf_xgb.py.

Usage:
    python hybrid_cascade_weighted.py --dataset camlds --stage2 cnn --data-dir data --out results/current/hybrid_cascade_weighted_cnn/
    python hybrid_cascade_weighted.py --dataset camlds --stage2 rf  --data-dir data --out results/current/hybrid_cascade_weighted_rf/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score

from config import MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import (vector_group_key, log1p_continuous, fit_scaler_on_train,
                            xgb_scale_pos_weight, fold_class_weights)
from models import build_xgb, build_cnn, build_rf, cnn_fit_kwargs
from train_eval import _best_f1_threshold, _fpr

BINARY_FEATURES = {
    'uid_is_root', 'uid_changed', 'euid_root', 'is_suid_exec',
    'auid_euid_mismatch', 'ephemeral_privileged', 'shell_from_service',
    'sensitive_path_access',
}


def cascade_predict(X_tr, y_tr, X_te, feat_cols, max_fpr, stage2,
                     low_thresh=0.3, high_thresh=0.7):
    """Fit XGB (stage 1) + stage2 model on the train fold, predict on the
    test fold with confidence-WEIGHTED blending (not flat 50/50).

    Returns (y_pred, final_proba, uncertain_mask, avg_stage2_weight) --
    avg_stage2_weight lets the caller report how much influence stage2
    actually had within the uncertain band, per fold (this is the direct
    evidence for whether the weighting is doing what it's meant to).
    """
    X_tr_log = log1p_continuous(X_tr, feat_cols, BINARY_FEATURES)
    X_te_log = log1p_continuous(X_te, feat_cols, BINARY_FEATURES)
    spw = xgb_scale_pos_weight(y_tr)
    xgb = build_xgb(scale_pos_weight=spw)
    xgb.fit(X_tr_log, y_tr)
    xgb_proba = xgb.predict_proba(X_te_log)[:, 1]

    uncertain = (xgb_proba >= low_thresh) & (xgb_proba <= high_thresh)
    final_proba = xgb_proba.copy()
    avg_stage2_weight = 0.0

    if uncertain.sum() > 0:
        band_half_width = (high_thresh - low_thresh) / 2.0
        band_center = (high_thresh + low_thresh) / 2.0
        dist_from_center = np.abs(xgb_proba[uncertain] - band_center)
        stage2_weight = np.clip(1.0 - dist_from_center / band_half_width, 0.0, 1.0)
        avg_stage2_weight = float(stage2_weight.mean())

        if stage2 == 'cnn':
            scaler = fit_scaler_on_train(X_tr_log)
            X_tr_scaled = scaler.transform(X_tr_log)
            X_te_scaled = scaler.transform(X_te_log)
            model = build_cnn(n_features=X_tr_scaled.shape[1])
            cw = fold_class_weights(y_tr)
            model.fit(X_tr_scaled, y_tr, class_weight=cw, **cnn_fit_kwargs())
            stage2_proba = model.predict(X_te_scaled[uncertain], verbose=0).ravel()
        elif stage2 == 'rf':
            model = build_rf()
            model.fit(X_tr, y_tr)
            stage2_proba = model.predict_proba(X_te[uncertain])[:, 1]
        else:
            raise ValueError(f"Unknown --stage2 '{stage2}' (use 'cnn' or 'rf')")

        final_proba[uncertain] = (1 - stage2_weight) * xgb_proba[uncertain] + stage2_weight * stage2_proba

    xgb_train_proba = xgb.predict_proba(X_tr_log)[:, 1]
    thresh = _best_f1_threshold(y_tr, xgb_train_proba, max_fpr=max_fpr)
    y_pred = (final_proba >= thresh).astype(int)
    return y_pred, final_proba, uncertain, avg_stage2_weight


def run_cascade_cv(dataset_key, data_dir, out_dir, max_fpr, stage2, n_splits=5, n_repeats=1, base_seed=42):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    rows = []
    for repeat in range(n_repeats):
        seed = base_seed + repeat
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            n_classes_te = len(set(y[te]))
            if n_classes_te < 2:
                print(f"[{dataset_key}/{stage2}] repeat {repeat} fold {fold}: SKIPPED -- degenerate test fold.")
                rows.append({'repeat': repeat, 'fold': fold, 'f1': np.nan, 'auroc': np.nan,
                             'fpr': np.nan, 'fnr': np.nan, 'pct_routed_to_stage2': np.nan,
                             'avg_stage2_weight': np.nan, 'degenerate': True})
                continue

            y_pred, y_proba, uncertain, avg_w = cascade_predict(
                X[tr], y[tr], X[te], feat_cols, max_fpr, stage2)
            f1 = f1_score(y[te], y_pred, zero_division=0)
            auc = roc_auc_score(y[te], y_proba)
            fpr = _fpr(y[te], y_pred)
            recall = recall_score(y[te], y_pred, zero_division=0)
            fnr = 1 - recall
            routed_pct = 100 * uncertain.mean()
            rows.append({'repeat': repeat, 'fold': fold, 'f1': f1, 'auroc': auc, 'fpr': fpr,
                         'fnr': fnr, 'pct_routed_to_stage2': routed_pct,
                         'avg_stage2_weight': avg_w, 'degenerate': False})
            print(f"[{dataset_key}/{stage2}] repeat {repeat} fold {fold}: F1={f1:.4f} AUROC={auc:.4f} "
                  f"FPR={fpr:.4f} FNR={fnr:.4f}  routed={routed_pct:.1f}%  avg_stage2_weight={avg_w:.3f}")

    result_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"{dataset_key}_cascade_weighted_{stage2}_cv_results.csv", index=False)

    valid = result_df[~result_df['degenerate']]
    n_skipped = result_df['degenerate'].sum()
    print(f"\n[{dataset_key}/{stage2}] Weighted cascade summary ({len(valid)}/{len(result_df)} "
          f"repeat-folds, {n_skipped} skipped): "
          f"F1={valid['f1'].mean():.4f}+/-{valid['f1'].std():.4f}  "
          f"AUROC={valid['auroc'].mean():.4f}+/-{valid['auroc'].std():.4f}  "
          f"avg routed={valid['pct_routed_to_stage2'].mean():.1f}%  "
          f"avg stage2 weight when routed={valid['avg_stage2_weight'].mean():.3f}")
    return result_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Hybrid cascade, confidence-weighted blend: XGBoost (stage 1) -> "
                     "stage-2 model, blend weight scales with distance from band center")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--stage2", choices=["cnn", "rf"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default=None,
                    help="Defaults to results/current/hybrid_cascade_weighted_<stage2>/ if omitted")
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--low-thresh", type=float, default=0.3)
    p.add_argument("--high-thresh", type=float, default=0.7)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    out = args.out or f"results/current/hybrid_cascade_weighted_{args.stage2}"
    run_cascade_cv(args.dataset, args.data_dir, out, args.max_fpr, args.stage2, n_repeats=args.repeats)
