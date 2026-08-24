"""
Milestone 3 -- Step 8: Hybrid Cascade variant -- XGB (stage 1) -> RF (stage 2)
=================================================================================
Sibling to hybrid_cascade.py, NOT a replacement for it. Same confidence-routed
architecture, same thresholding/leak-free conventions -- the only change is
Stage 2: Random Forest instead of CNN. Kept as a separate file (rather than a
--stage2 flag on hybrid_cascade.py) so both variants' results sit in separate,
independently-inspectable output directories and neither run can silently
overwrite the other -- same pattern already used elsewhere in this project
(outputs_v1 vs outputs, error_analysis_results vs _original).

WHY THIS VARIANT, evidence from error_analysis.py's pairwise disagreement
output (original CAM-LDS, post CNN-fix):
  - RF vs XGB disagree on only 2.4% of rows -- small, but when they DO
    disagree, RF is right substantially more often (1,426 vs 712, ~67/33).
    A real, if modest, complementary signal -- unlike RF-vs-XGB on Casino
    (0.1% disagreement, effectively nothing to exploit there).
  - Critically, RF does NOT carry CNN's instability problem: no GPU/Keras
    non-determinism, no bimodal collapse behavior (see docs/milestone3
    Ch8.1's CNN reproducibility finding -- FP rate ranged from 0.33% to
    97.98% across two identical-seed re-runs). A cascade built entirely
    from two deterministic tree ensembles should be far more stable
    fold-to-fold than the XGB->CNN cascade, even if the mean improvement
    is smaller. This variant tests that trade-off directly: modest-but-
    reliable (RF stage 2) vs larger-but-volatile (CNN stage 2).
  - IForest is deliberately excluded from every cascade variant tested in
    this project: it disagrees with every other model 40-44% of the time
    on CAM-LDS, but is right in only ~7-11% of those disagreements --
    that's not complementary signal, it's IForest being wrong (AUROC=0.555
    standalone). Not worth testing as a cascade stage.

Usage:
    python hybrid_cascade_rf_xgb.py --dataset camlds --data-dir data --out results/current/hybrid_cascade_rf_xgb/
    python hybrid_cascade_rf_xgb.py --dataset casino --data-dir data --out results/current/hybrid_cascade_rf_xgb/
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
from preprocessing import vector_group_key, log1p_continuous, xgb_scale_pos_weight
from models import build_xgb, build_rf
from train_eval import _best_f1_threshold, _fpr

BINARY_FEATURES = {
    'uid_is_root', 'uid_changed', 'euid_root', 'is_suid_exec',
    'auid_euid_mismatch', 'ephemeral_privileged', 'shell_from_service',
    'sensitive_path_access',
}


def cascade_predict(X_tr, y_tr, X_te, feat_cols, max_fpr,
                     low_thresh=0.3, high_thresh=0.7):
    """Fit XGB (stage 1) + RF (stage 2) on the train fold, predict on the
    test fold with confidence-based routing. Mirrors hybrid_cascade.py's
    cascade_predict exactly except for the stage-2 model -- kept
    structurally identical so the two variants' results are comparable
    apples-to-apples, not just similarly-named.

    Returns (y_pred, final_proba, uncertain_mask).
    """
    # -- Stage 1: XGBoost, identical recipe to hybrid_cascade.py --
    X_tr_log = log1p_continuous(X_tr, feat_cols, BINARY_FEATURES)
    X_te_log = log1p_continuous(X_te, feat_cols, BINARY_FEATURES)
    spw = xgb_scale_pos_weight(y_tr)
    xgb = build_xgb(scale_pos_weight=spw)
    xgb.fit(X_tr_log, y_tr)
    xgb_proba = xgb.predict_proba(X_te_log)[:, 1]

    # -- Stage 2: Random Forest, only on the uncertain band. RF is
    #    deterministic given a fixed random_state (already set in
    #    RF_PARAMS via config.py, wired automatically since build_rf
    #    passes its params dict straight into the sklearn constructor)
    #    -- no equivalent to CNN's uncontrolled-seed non-determinism here.
    uncertain = (xgb_proba >= low_thresh) & (xgb_proba <= high_thresh)
    final_proba = xgb_proba.copy()

    if uncertain.sum() > 0:
        rf = build_rf()
        rf.fit(X_tr, y_tr)  # RF uses raw (non-log1p) features, matching
                             # train_eval._fit_predict_rf's own convention
        rf_proba = rf.predict_proba(X_te[uncertain])[:, 1]

        final_proba[uncertain] = 0.5 * xgb_proba[uncertain] + 0.5 * rf_proba

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
            # Same degenerate-fold guard as hybrid_cascade.py -- see that
            # file's comment for the full explanation (CasinoLimit's thin
            # ~356 unique feature-vector groups).
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
              f"FPR={fpr:.4f}  routed_to_RF={routed_pct:.1f}%")

    result_df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_dir / f"{dataset_key}_cascade_rf_xgb_cv_results.csv", index=False)

    valid = result_df[~result_df['degenerate']]
    n_skipped = result_df['degenerate'].sum()
    print(f"\n[{dataset_key}] RF+XGB cascade summary ({len(valid)}/{len(result_df)} folds used, "
          f"{n_skipped} skipped as degenerate): "
          f"F1={valid['f1'].mean():.4f}+/-{valid['f1'].std():.4f}  "
          f"AUROC={valid['auroc'].mean():.4f}+/-{valid['auroc'].std():.4f}  "
          f"avg routed to Stage 2={valid['pct_routed_to_stage2'].mean():.1f}%")
    return result_df


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Hybrid cascade variant: XGBoost (stage 1, all rows) -> "
                     "Random Forest (stage 2, uncertain-confidence rows only)")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="results/current/hybrid_cascade_rf_xgb",
                    help="Separate default from hybrid_cascade.py's output dir "
                         "-- the two variants never overwrite each other.")
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--low-thresh", type=float, default=0.3)
    p.add_argument("--high-thresh", type=float, default=0.7)
    args = p.parse_args()

    run_cascade_cv(args.dataset, args.data_dir, args.out, args.max_fpr)
