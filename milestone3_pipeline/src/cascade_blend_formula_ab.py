"""
Milestone 3 -- Step 8 follow-up: flat-vs-weighted blend, confound-free A/B.

hybrid_cascade.py (flat 50/50) and hybrid_cascade_weighted.py (distance-
weighted) are SEPARATE scripts that each independently train their own
XGBoost + CNN per fold. Because CNN is not reproducible even at a fixed
seed (see report Sec 2.1: identical seed=42 produced FP rates from 0.33%
to 98.00% across 6 runs), comparing one script's single run against the
other's is confounded -- any F1 difference could be the blend fix working,
or could just be CNN training variance unrelated to blending at all.

This script removes that confound: for each fold, XGBoost and CNN are each
trained EXACTLY ONCE. Their raw probability outputs are reused to compute
BOTH the flat-blend prediction and the weighted-blend prediction. Any F1
difference between the two columns is now attributable ONLY to the blend
formula, since every other input (fold split, XGBoost model, CNN model,
CNN's raw predictions) is held perfectly identical between the two.

Usage:
    python cascade_blend_formula_ab.py --dataset camlds --data-dir ../data --out results/current/cascade_blend_formula_ab/
    python cascade_blend_formula_ab.py --dataset casino --data-dir ../data --out results/current/cascade_blend_formula_ab/
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
from models import build_xgb, build_cnn, cnn_fit_kwargs
from train_eval import _best_f1_threshold, _fpr

BINARY_FEATURES = {
    'uid_is_root', 'uid_changed', 'euid_root', 'is_suid_exec',
    'auid_euid_mismatch', 'ephemeral_privileged', 'shell_from_service',
    'sensitive_path_access',
}


def score(y_te, xgb_train_proba, y_tr, final_proba, max_fpr):
    thresh = _best_f1_threshold(y_tr, xgb_train_proba, max_fpr=max_fpr)
    y_pred = (final_proba >= thresh).astype(int)
    return {
        'f1': f1_score(y_te, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_te, final_proba),
        'fpr': _fpr(y_te, y_pred),
        'fnr': 1 - recall_score(y_te, y_pred, zero_division=0),
    }


def run_ab(dataset_key, data_dir, out_dir, max_fpr, low_thresh=0.3, high_thresh=0.7,
           n_splits=5, seed=42):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        if len(set(y[te])) < 2:
            print(f"[{dataset_key}] fold {fold}: SKIPPED -- degenerate test fold.")
            rows.append({'fold': fold, 'flat_f1': np.nan, 'weighted_f1': np.nan,
                         'routed_pct': np.nan, 'avg_stage2_weight': np.nan, 'degenerate': True})
            continue

        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        X_tr_log = log1p_continuous(X_tr, feat_cols, BINARY_FEATURES)
        X_te_log = log1p_continuous(X_te, feat_cols, BINARY_FEATURES)

        # -- Stage 1: XGBoost, trained ONCE --
        spw = xgb_scale_pos_weight(y_tr)
        xgb = build_xgb(scale_pos_weight=spw)
        xgb.fit(X_tr_log, y_tr)
        xgb_proba = xgb.predict_proba(X_te_log)[:, 1]
        xgb_train_proba = xgb.predict_proba(X_tr_log)[:, 1]

        uncertain = (xgb_proba >= low_thresh) & (xgb_proba <= high_thresh)
        routed_pct = 100 * uncertain.mean()

        flat_proba = xgb_proba.copy()
        weighted_proba = xgb_proba.copy()
        avg_stage2_weight = 0.0

        if uncertain.sum() > 0:
            # -- Stage 2: CNN, trained ONCE, reused for both blend formulas --
            scaler = fit_scaler_on_train(X_tr_log)
            X_tr_scaled = scaler.transform(X_tr_log)
            X_te_scaled = scaler.transform(X_te_log)
            cnn = build_cnn(n_features=X_tr_scaled.shape[1])
            cw = fold_class_weights(y_tr)
            cnn.fit(X_tr_scaled, y_tr, class_weight=cw, **cnn_fit_kwargs())
            cnn_proba = cnn.predict(X_te_scaled[uncertain], verbose=0).ravel()

            # Variant A: flat 50/50 (hybrid_cascade.py's formula)
            flat_proba[uncertain] = 0.5 * xgb_proba[uncertain] + 0.5 * cnn_proba

            # Variant B: distance-weighted (hybrid_cascade_weighted.py's formula)
            band_half_width = (high_thresh - low_thresh) / 2.0
            band_center = (high_thresh + low_thresh) / 2.0
            dist_from_center = np.abs(xgb_proba[uncertain] - band_center)
            stage2_weight = np.clip(1.0 - dist_from_center / band_half_width, 0.0, 1.0)
            avg_stage2_weight = float(stage2_weight.mean())
            weighted_proba[uncertain] = ((1 - stage2_weight) * xgb_proba[uncertain]
                                          + stage2_weight * cnn_proba)

        flat_scores = score(y_te, xgb_train_proba, y_tr, flat_proba, max_fpr)
        weighted_scores = score(y_te, xgb_train_proba, y_tr, weighted_proba, max_fpr)

        row = {
            'fold': fold, 'routed_pct': routed_pct, 'avg_stage2_weight': avg_stage2_weight,
            'flat_f1': flat_scores['f1'], 'flat_fpr': flat_scores['fpr'], 'flat_fnr': flat_scores['fnr'],
            'weighted_f1': weighted_scores['f1'], 'weighted_fpr': weighted_scores['fpr'],
            'weighted_fnr': weighted_scores['fnr'], 'degenerate': False,
        }
        rows.append(row)
        print(f"[{dataset_key}] fold {fold}: routed={routed_pct:.1f}%  "
              f"flat_F1={flat_scores['f1']:.4f}  weighted_F1={weighted_scores['f1']:.4f}  "
              f"delta={weighted_scores['f1']-flat_scores['f1']:+.4f}")

    result_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"{dataset_key}_cascade_blend_formula_ab.csv", index=False)

    valid = result_df[~result_df['degenerate']]
    print(f"\n[{dataset_key}] A/B summary ({len(valid)}/{len(result_df)} folds usable):")
    print(f"  flat_F1:     {valid['flat_f1'].mean():.4f} +/- {valid['flat_f1'].std():.4f}")
    print(f"  weighted_F1: {valid['weighted_f1'].mean():.4f} +/- {valid['weighted_f1'].std():.4f}")
    return result_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Confound-free A/B: flat vs. distance-weighted Stage-2 blend, "
                     "same trained XGB+CNN reused for both formulas per fold.")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/current/cascade_blend_formula_ab")
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--low-thresh", type=float, default=0.3)
    p.add_argument("--high-thresh", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_ab(args.dataset, args.data_dir, args.out, args.max_fpr,
           args.low_thresh, args.high_thresh, seed=args.seed)
