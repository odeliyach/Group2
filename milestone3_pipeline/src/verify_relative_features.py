"""
Milestone 3 -- test process-relative anomaly scores against the diagnosed
FPR-instability mechanism (52.4% of CAM-LDS rows in mixed-label groups,
concentrated in common utility processes like `dash`/`gpg` whose absolute
OS-level footprint doesn't reveal attacker vs. admin intent).

Idea: instead of raw counts (which collide across different real uses of
the same utility), score each numeric feature relative to that SAME
proc_name's own typical behavior. A `dash` invocation with an unusually
high failed_call_rate FOR A DASH PROCESS SPECIFICALLY may be informative
even when the absolute value collides with other processes' normal range.

Leak-free by construction: the per-proc_name baseline (mean/std) is
computed from the TRAINING FOLD ONLY, inside each fold of the CV loop, and
applied to that fold's test set -- never fit on data the model will be
evaluated against. proc_names with too little training support (<
min_support occurrences) fall back to a global baseline rather than a
noisy per-name estimate.

This is a standalone RF-based comparison (production 20 features vs.
20+relative-z augmented), using the same StratifiedGroupKFold + F1-optimal
threshold methodology as the rest of the project, but implemented directly
here rather than reusing train_eval.py's internals -- keeps this safely
self-contained rather than risking a signature-mismatch guess.

Usage:
    python verify_relative_features.py --data-dir data --out results/current/relative_features_test/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import CV_CONFIG, MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, impute
from train_eval import _fit_predict_rf, _fpr  # the REAL production fitter --
                                                # confirmed signature: (X_tr, y_tr,
                                                # groups_tr, X_te, hp_overrides, max_fpr)
                                                # -> (y_pred, y_proba). Reusing this
                                                # directly instead of reimplementing
                                                # threshold selection a second time.

# The behavioral features most plausibly informative RELATIVE to a
# process's own typical footprint, rather than in absolute terms.
RELATIVE_CANDIDATE_FEATURES = [
    'exec_count', 'unique_syscalls', 'failed_call_rate', 'failed_call_count',
    'lifetime_seconds', 'events_per_second', 'priv_op_count', 'network_count',
    'file_access_count', 'seq_length',
]


def add_relative_features(df_train, df_apply, base_features, proc_col='proc_name', min_support=30):
    """Compute per-proc_name z-scores using ONLY df_train's statistics,
    then apply to df_apply (which may be df_train itself, or a held-out
    test fold). Never touches df_apply's own values when building the
    baseline -- this is what keeps it leak-free."""
    global_mean = df_train[base_features].mean()
    global_std = df_train[base_features].std().replace(0, 1)

    proc_counts = df_train[proc_col].value_counts()
    supported_procs = set(proc_counts[proc_counts >= min_support].index)

    proc_stats = df_train[df_train[proc_col].isin(supported_procs)].groupby(proc_col)[base_features].agg(['mean', 'std'])

    out = pd.DataFrame(index=df_apply.index)
    for feat in base_features:
        proc_mean = df_apply[proc_col].map(proc_stats[(feat, 'mean')]).fillna(global_mean[feat])
        proc_std = df_apply[proc_col].map(proc_stats[(feat, 'std')]).fillna(global_std[feat]).replace(0, global_std[feat]).replace(0, 1)
        out[f'{feat}_relz'] = (df_apply[feat] - proc_mean) / proc_std
    return out


def run_variant(df, y, groups, feat_cols, use_relative, repeats, n_splits, base_seed, min_support):
    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_true_all, y_pred_all, y_proba_all = [], [], []
        for tr, te in cv.split(df[feat_cols].values, y, groups):
            df_tr, df_te = df.iloc[tr], df.iloc[te]
            y_tr, y_te = y[tr], y[te]
            groups_tr = groups[tr]

            X_tr = df_tr[feat_cols].copy()
            X_te = df_te[feat_cols].copy()
            if use_relative:
                rel_tr = add_relative_features(df_tr, df_tr, RELATIVE_CANDIDATE_FEATURES, min_support=min_support)
                rel_te = add_relative_features(df_tr, df_te, RELATIVE_CANDIDATE_FEATURES, min_support=min_support)
                X_tr = pd.concat([X_tr.reset_index(drop=True), rel_tr.reset_index(drop=True)], axis=1)
                X_te = pd.concat([X_te.reset_index(drop=True), rel_te.reset_index(drop=True)], axis=1)

            X_tr = impute(X_tr.values)
            X_te = impute(X_te.values)

            # The REAL production fitter -- same inner GroupShuffleSplit
            # threshold calibration, same MAX_FPR cap, as every headline number.
            pred, proba = _fit_predict_rf(X_tr, y_tr, groups_tr, X_te, None, MAX_FPR)

            y_true_all.extend(y_te); y_pred_all.extend(pred); y_proba_all.extend(proba)

        y_true_all, y_pred_all, y_proba_all = map(np.array, (y_true_all, y_pred_all, y_proba_all))
        rows.append({
            'repeat': rep, 'seed': seed,
            'f1': f1_score(y_true_all, y_pred_all, zero_division=0),
            'auroc': roc_auc_score(y_true_all, y_proba_all),
            'fpr': _fpr(y_true_all, y_pred_all),
        })
        print(f"  [{'relative' if use_relative else 'baseline'}] repeat {rep}: "
              f"F1={rows[-1]['f1']:.4f} FPR={rows[-1]['fpr']:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--min-support", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest("camlds", data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, "camlds", policy="full")
    df = ds.df.copy()
    y = ds.y.values
    groups = vector_group_key(df, feat_cols)
    base_seed = CV_CONFIG["seed"]

    missing = [f for f in RELATIVE_CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: {missing} not found in columns -- adjust "
              f"RELATIVE_CANDIDATE_FEATURES to match actual column names.")
        return
    if 'proc_name' not in df.columns:
        print("WARNING: 'proc_name' column not found -- cannot build relative features.")
        return

    print("=== BASELINE (production 20 features) ===")
    base_df = run_variant(df, y, groups, feat_cols, False, args.repeats, args.n_splits, base_seed, args.min_support)

    print("\n=== AUGMENTED (20 + relative-z features) ===")
    aug_df = run_variant(df, y, groups, feat_cols, True, args.repeats, args.n_splits, base_seed, args.min_support)

    base_df.to_csv(out_dir / "baseline_per_repeat.csv", index=False)
    aug_df.to_csv(out_dir / "augmented_per_repeat.csv", index=False)

    print(f"\nBaseline:  F1={base_df['f1'].mean():.4f}\u00b1{base_df['f1'].std():.4f}  "
          f"FPR={base_df['fpr'].mean():.4f}\u00b1{base_df['fpr'].std():.4f}")
    print(f"Augmented: F1={aug_df['f1'].mean():.4f}\u00b1{aug_df['f1'].std():.4f}  "
          f"FPR={aug_df['fpr'].mean():.4f}\u00b1{aug_df['fpr'].std():.4f}")

    print("\n--- Specifically the diagnosed bad repeats (3, 5, 12) ---")
    for rep in [3, 5, 12]:
        b = base_df[base_df['repeat'] == rep].iloc[0]
        a = aug_df[aug_df['repeat'] == rep].iloc[0]
        print(f"  Repeat {rep}: baseline FPR={b['fpr']:.4f} -> augmented FPR={a['fpr']:.4f}  "
              f"(F1 {b['f1']:.4f} -> {a['f1']:.4f})")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
