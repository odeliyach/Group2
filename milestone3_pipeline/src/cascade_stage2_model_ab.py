"""
Milestone 3 -- Step 8 follow-up: CNN-vs-MLP-at-Stage-2, confound-free A/B.

Direct sibling of cascade_blend_formula_ab.py, which A/B'd the BLEND
FORMULA (flat vs weighted) while holding the Stage-2 model fixed. This
script A/B's the STAGE-2 MODEL IDENTITY (CNN vs MLP) while holding the
blend formula (weighted, the adopted production design) fixed.

Why this is needed: a naive comparison of "CNN-cascade numbers already in
the report" vs "a freshly-run MLP-cascade" is confounded the same way the
original flat-vs-weighted comparison was -- two separate pipeline
executions have separate Stage-1 XGBoost fits, separate fold splits,
separate random draws. Any difference could be genuine model-choice signal
or could just be which run got luckier fold splits (exactly the mechanism
already confirmed for the blend-formula A/B, where a second run reversed
the verdict). This script eliminates that confound: XGBoost (Stage 1) and
the fold split are fit/drawn ONCE per fold and reused for both the CNN-blend
and MLP-blend predictions, so any F1/FPR/FNR difference is attributable
only to which model sits at Stage 2.

Usage:
    python cascade_stage2_model_ab.py --dataset camlds --data-dir ../data --out results/current/cascade_stage2_model_ab/
    python cascade_stage2_model_ab.py --dataset casino --data-dir ../data --out results/current/cascade_stage2_model_ab/
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
from models import build_xgb, build_cnn, build_mlp, cnn_fit_kwargs, mlp_fit_kwargs
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


def weighted_blend(xgb_proba, stage2_proba_full, uncertain, low_thresh, high_thresh):
    """Same distance-weighted formula as the adopted production cascade --
    full weight to Stage 2 at the band center, fading to zero at the edges."""
    final_proba = xgb_proba.copy()
    if uncertain.sum() > 0:
        band_half_width = (high_thresh - low_thresh) / 2.0
        band_center = (high_thresh + low_thresh) / 2.0
        dist_from_center = np.abs(xgb_proba[uncertain] - band_center)
        stage2_weight = np.clip(1.0 - dist_from_center / band_half_width, 0.0, 1.0)
        final_proba[uncertain] = ((1 - stage2_weight) * xgb_proba[uncertain]
                                   + stage2_weight * stage2_proba_full)
    return final_proba


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
            rows.append({'fold': fold, 'cnn_f1': np.nan, 'mlp_f1': np.nan,
                         'routed_pct': np.nan, 'degenerate': True})
            continue

        X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
        X_tr_log = log1p_continuous(X_tr, feat_cols, BINARY_FEATURES)
        X_te_log = log1p_continuous(X_te, feat_cols, BINARY_FEATURES)

        # -- Stage 1: XGBoost, trained ONCE, reused for both variants --
        spw = xgb_scale_pos_weight(y_tr)
        xgb = build_xgb(scale_pos_weight=spw)
        xgb.fit(X_tr_log, y_tr)
        xgb_proba = xgb.predict_proba(X_te_log)[:, 1]
        xgb_train_proba = xgb.predict_proba(X_tr_log)[:, 1]

        uncertain = (xgb_proba >= low_thresh) & (xgb_proba <= high_thresh)
        routed_pct = 100 * uncertain.mean()

        cnn_proba_full = np.zeros(uncertain.sum())
        mlp_proba_full = np.zeros(uncertain.sum())

        if uncertain.sum() > 0:
            scaler = fit_scaler_on_train(X_tr_log)
            X_tr_scaled = scaler.transform(X_tr_log)
            X_te_scaled = scaler.transform(X_te_log)
            cw = fold_class_weights(y_tr)

            # -- Stage 2, variant A: CNN, same fold split as MLP below --
            cnn = build_cnn(n_features=X_tr_scaled.shape[1])
            cnn.fit(X_tr_scaled[..., None], y_tr, class_weight=cw, **cnn_fit_kwargs())
            cnn_proba_full = cnn.predict(X_te_scaled[uncertain][..., None], verbose=0).ravel()

            # -- Stage 2, variant B: MLP, IDENTICAL fold split/Stage-1 fit --
            mlp = build_mlp(n_features=X_tr_scaled.shape[1])
            mlp.fit(X_tr_scaled, y_tr, class_weight=cw, **mlp_fit_kwargs())
            mlp_proba_full = mlp.predict(X_te_scaled[uncertain], verbose=0).ravel()

        cnn_final = weighted_blend(xgb_proba, cnn_proba_full, uncertain, low_thresh, high_thresh)
        mlp_final = weighted_blend(xgb_proba, mlp_proba_full, uncertain, low_thresh, high_thresh)

        cnn_scores = score(y_te, xgb_train_proba, y_tr, cnn_final, max_fpr)
        mlp_scores = score(y_te, xgb_train_proba, y_tr, mlp_final, max_fpr)

        row = {
            'fold': fold, 'routed_pct': routed_pct,
            'cnn_f1': cnn_scores['f1'], 'cnn_fpr': cnn_scores['fpr'], 'cnn_fnr': cnn_scores['fnr'],
            'mlp_f1': mlp_scores['f1'], 'mlp_fpr': mlp_scores['fpr'], 'mlp_fnr': mlp_scores['fnr'],
            'degenerate': False,
        }
        rows.append(row)
        print(f"[{dataset_key}] fold {fold}: routed={routed_pct:.1f}%  "
              f"cnn_F1={cnn_scores['f1']:.4f}  mlp_F1={mlp_scores['f1']:.4f}  "
              f"delta={mlp_scores['f1']-cnn_scores['f1']:+.4f}")

    result_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"{dataset_key}_stage2_model_ab.csv", index=False)

    valid = result_df[~result_df['degenerate']]
    print(f"\n[{dataset_key}] A/B summary ({len(valid)}/{len(result_df)} folds usable):")
    print(f"  cnn_F1: {valid['cnn_f1'].mean():.4f} +/- {valid['cnn_f1'].std():.4f}")
    print(f"  mlp_F1: {valid['mlp_f1'].mean():.4f} +/- {valid['mlp_f1'].std():.4f}")
    return result_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Confound-free A/B: CNN vs. MLP as Stage-2 model, same Stage-1 "
                     "XGBoost and fold split reused for both.")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/current/cascade_stage2_model_ab")
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--low-thresh", type=float, default=0.3)
    p.add_argument("--high-thresh", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_ab(args.dataset, args.data_dir, args.out, args.max_fpr,
           args.low_thresh, args.high_thresh, seed=args.seed)
