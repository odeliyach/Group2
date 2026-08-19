"""
Unified Validation — leak-free repeated CV + feature ablation
=================================================================
Replaces validate_features_v3.py (Dataset 1 only, single 5-fold split),
process_level_cv_final2.py / stability_fix_test2.py (Dataset 2, single or
3-repeat), and robust_reverify2.py (Dataset 2, 15-repeat) with one script
so BOTH datasets' headline numbers (Table 8) come from the identical
methodology and the same repeat count -- right now Dataset 1's number is
a single 5-fold split and Dataset 2's is a 15x-repeated 5-fold; that
asymmetry should be fixed by re-running Dataset 1 through this script
with --repeats matching whatever you use for Dataset 2.

Produces:
  duplication_report.txt   -- % duplicate feature vectors
  cv_results.csv           -- per-fold AUROC/F1 across all repeats
  validation_summary.txt   -- headline mean +/- std (the Table 8 numbers)
  feature_ablation.csv     -- drop-one-feature impact (Table 6's Ablation
                               DeltaF1 column, and the Ch3.2 pruning
                               justification), repeated across folds
  feature_ablation.png

Usage -- Dataset 1:
    python unified_validation.py \
        --input combined/dataset1_features.csv \
        --label-col label \
        --dataset-name "CAM-LDS" \
        --repeats 15 \
        --out validation_output/cam_lds/

Usage -- Dataset 2 (process-level, 4-technique subset):
    python unified_validation.py \
        --input casino_process_level_FINAL.csv \
        --label-col is_privesc \
        --technique-col technique \
        --exclude-techniques T1003,T1078 \
        --dataset-name "CasinoLimit" \
        --repeats 15 \
        --out validation_output/casino/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings('ignore')

# Must match unified_eda.py's FULL_FEATURE_LIST exactly -- same schema,
# same rule (auto-intersect with whatever columns exist in the CSV).
FULL_FEATURE_LIST = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]


def load_and_filter(args):
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {args.dataset_name}: {len(df):,} rows")

    if args.exclude_techniques and args.technique_col in df.columns:
        excl = [t.strip() for t in args.exclude_techniques.split(',') if t.strip()]
        before = len(df)
        keep_mask = ~df[args.technique_col].isin(excl) | (df[args.label_col] == 0)
        df = df[keep_mask].reset_index(drop=True)
        print(f"  Excluded techniques {excl}: {before - len(df):,} rows removed, {len(df):,} remain")

    feat_cols = [f for f in FULL_FEATURE_LIST if f in df.columns]
    missing = [f for f in FULL_FEATURE_LIST if f not in df.columns]
    print(f"  Features available: {len(feat_cols)}/{len(FULL_FEATURE_LIST)}")
    if missing:
        print(f"  Missing: {missing}")

    y = df[args.label_col].values
    print(f"  Benign: {(y==0).sum():,}  Attack: {(y==1).sum():,}")
    return df, feat_cols


def vector_group_key(df, feat_cols):
    """Deterministic grouping: round floats to 4 decimals (+0.0 normalizes
    -0.0 to 0.0, since str(-0.0) != str(0.0) and would silently split one
    true duplicate group into two -- see investigation report Section 9.7)
    and fillna('NA') for non-float columns, so raw-float-to-string
    formatting differences across pandas/numpy versions can't silently
    change which rows get grouped together."""
    parts = []
    for c in feat_cols:
        col = df[c]
        if col.dtype.kind in 'fc':
            parts.append((col.round(4) + 0.0).astype(str))
        else:
            parts.append(col.fillna('NA').astype(str))
    return parts[0].str.cat(parts[1:], sep='|').values


def check_duplication(df, feat_cols, out_dir, args):
    dup_count = df.duplicated(subset=feat_cols).sum()
    pct = 100 * dup_count / len(df)
    groups = vector_group_key(df, feat_cols)
    n_groups = len(set(groups))
    text = (f"Duplication report — {args.dataset_name}\n"
            f"{'='*50}\n"
            f"Duplicate rows: {dup_count:,} / {len(df):,} ({pct:.1f}%)\n"
            f"Unique feature-vector groups: {n_groups:,} "
            f"({len(df)/max(n_groups,1):.1f}x avg duplication)\n"
            f"Using vector-group split: identical rows always land on the\n"
            f"same side of the split, so duplication cannot leak.\n")
    print(f"  {dup_count:,} / {len(df):,} duplicate rows ({pct:.1f}%), "
          f"{n_groups:,} unique groups")
    with open(out_dir / 'duplication_report.txt', 'w') as f:
        f.write(text)
    return pct, groups


def repeated_cv(X, y, groups, feat_cols, repeats, n_splits=5, seed=42,
                 n_estimators=200, class_weight='balanced'):
    """Repeated StratifiedGroupKFold with POOLED-per-repeat scoring.

    FIX (this version): the previous version computed one F1/AUROC per
    INDIVIDUAL FOLD (e.g. 75 separate values across 15 repeats x 5 folds)
    and averaged those 75 raw numbers directly. With CasinoLimit's 70%+
    dominant feature-vector group and only ~400-600 unique groups total,
    individual folds can be tiny or degenerate (e.g. a handful of attack
    rows in one test fold), and those folds contributed to the average
    with equal weight to a fold holding tens of thousands of rows -- this
    is exactly what produced the misleadingly unstable-looking headline
    number (F1=0.7745+/-0.3981) that this fix addresses. It was not new
    instability in the data; it was a measurement artifact from the
    averaging method, already diagnosed and fixed once before in
    milestone3_models_casino_process_f4.py's run_rf() -- ported here so
    BOTH datasets go through the identical, correct methodology.

    Now: cross_val_predict pools every fold's held-out predictions into
    ONE array covering the full dataset within a single repeat, then
    computes ONE F1 and ONE AUROC per repeat from that pooled array (so
    repeats=15 -> 15 numbers averaged, not 75). This also fixes a second,
    related problem: previously F1 was averaged over every fold, but
    AUROC silently dropped folds with only one class present -- so the
    two metrics came from different, non-matching sets of folds. Pooling
    within a repeat guarantees F1 and AUROC always come from the exact
    same predictions, and no single small fold can dominate or distort
    either number -- only genuine repeat-to-repeat (seed) variance
    remains in the resulting std.

    Returns a per-REPEAT dataframe (repeats rows), not per-fold."""
    rows = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)

        # sanity check: no group leakage across the split (kept from original)
        for tr, te in cv.split(X, y, groups):
            train_groups = set(groups[tr])
            test_groups = set(groups[te])
            assert len(train_groups & test_groups) == 0, "Leakage: overlapping groups across split!"

        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                      n_jobs=-1, random_state=42)
        pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)
        proba = cross_val_predict(clf, X, y, groups=groups, cv=cv, method='predict_proba')[:, 1]

        f1 = f1_score(y, pred, zero_division=0)
        auc = roc_auc_score(y, proba) if len(set(y)) > 1 else np.nan
        rows.append({'repeat': rep, 'n_total': len(y), 'f1': f1, 'auroc': auc})
    return pd.DataFrame(rows)


def check_split(df, feat_cols, groups, args, out_dir):
    print(f"\n[1] Repeated vector-group K-fold CV, POOLED per repeat "
          f"({args.repeats} repeats x {args.n_splits} folds pooled into "
          f"{args.repeats} scored repeats -- not {args.repeats*args.n_splits} "
          f"flat fold scores)")

    X = df[feat_cols].fillna(0).values
    y = df[args.label_col].values

    cv_df = repeated_cv(X, y, groups, feat_cols, args.repeats, args.n_splits)
    cv_df.to_csv(out_dir / 'cv_results.csv', index=False)

    n_undefined = cv_df['auroc'].isna().sum()
    auroc_mean, auroc_std = cv_df['auroc'].mean(), cv_df['auroc'].std()
    f1_mean, f1_std = cv_df['f1'].mean(), cv_df['f1'].std()

    print(f"  AUROC = {auroc_mean:.4f} +/- {auroc_std:.4f} "
          f"({n_undefined}/{len(cv_df)} repeats undefined)")
    print(f"  F1    = {f1_mean:.4f} +/- {f1_std:.4f}")
    return cv_df, auroc_mean, auroc_std, f1_mean, f1_std


def check_feature_ablation(df, feat_cols, groups, args, out_dir, repeats=3):
    print(f"\n[2] Feature ablation (repeated vector-group split, {repeats} repeats each)")
    y = df[args.label_col].values

    def eval_subset(cols):
        X = df[cols].fillna(0).values
        cv_df = repeated_cv(X, y, groups, cols, repeats, args.n_splits, n_estimators=150)
        return cv_df['f1'].mean(), cv_df['auroc'].mean()

    base_f1, base_auc = eval_subset(feat_cols)
    print(f"  Baseline (all {len(feat_cols)} features): F1={base_f1:.4f} AUROC={base_auc:.4f}")

    results = []
    for feat in feat_cols:
        remaining = [f for f in feat_cols if f != feat]
        f1_ab, auc_ab = eval_subset(remaining)
        delta_f1 = f1_ab - base_f1
        delta_auc = auc_ab - base_auc
        tag = 'important' if delta_f1 < -0.005 else ('hurts' if delta_f1 > 0.005 else 'neutral')
        results.append({'feature': feat, 'f1_without': f1_ab, 'auc_without': auc_ab,
                        'delta_f1': delta_f1, 'delta_auc': delta_auc, 'tag': tag})
        print(f"  Remove {feat:<25} F1={f1_ab:.4f} (d{delta_f1:+.4f}) [{tag}]")

    abl_df = pd.DataFrame(results).sort_values('delta_f1')
    abl_df.to_csv(out_dir / 'feature_ablation.csv', index=False)

    reduce_feature_list(df, feat_cols, abl_df, groups, args, out_dir)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#E53935' if d < -0.005 else ('#43A047' if d > 0.005 else '#90A4AE')
              for d in abl_df['delta_f1'].values]
    ax.barh(range(len(abl_df)), abl_df['delta_f1'].values, color=colors)
    ax.set_yticks(range(len(abl_df)))
    ax.set_yticklabels(abl_df['feature'].values, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Delta F1 when feature removed (negative = feature helps)')
    ax.set_title(f'Feature Ablation: {args.dataset_name}\n(Red=important, Green=hurts, Gray=neutral)')
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_ablation.png')
    plt.close()
    print(f"  Saved: feature_ablation.png / .csv")
    return abl_df


def reduce_feature_list(df, feat_cols, abl_df, groups, args, out_dir, repeats=5):
    """Ch 3.2 -- 'optimize the feature list to maximize detection accuracy
    while minimizing computational overhead'. Per-feature ablation (above)
    only shows each feature's individual marginal value; this drops the
    WHOLE set of non-'important' features at once and re-measures, which
    is the actual claim the guideline is asking for: does the smaller
    model cost less to run while keeping the same detection power?"""
    print(f"\n[3] Full-vs-reduced feature set comparison ({repeats} repeats)")

    y = df[args.label_col].values
    reduced_cols = abl_df[abl_df['tag'] == 'important']['feature'].tolist()
    dropped_cols = [f for f in feat_cols if f not in reduced_cols]

    if len(reduced_cols) == 0:
        print("  No feature tagged 'important' by ablation -- skipping reduction "
              "(would leave zero features).")
        return None

    def eval_cols(cols):
        X = df[cols].fillna(0).values
        cv_df = repeated_cv(X, y, groups, cols, repeats, args.n_splits, n_estimators=200)
        return cv_df['f1'].mean(), cv_df['f1'].std(), cv_df['auroc'].mean(), cv_df['auroc'].std()

    full_f1, full_f1sd, full_auc, full_aucsd = eval_cols(feat_cols)
    red_f1, red_f1sd, red_auc, red_aucsd = eval_cols(reduced_cols)

    delta_f1 = red_f1 - full_f1
    delta_auc = red_auc - full_auc
    pct_dropped = 100 * len(dropped_cols) / len(feat_cols)

    text = (
        f"Feature List Optimization — {args.dataset_name}\n" + "=" * 60 + "\n\n"
        f"Full feature set ({len(feat_cols)} features):\n"
        f"  F1    = {full_f1:.4f} +/- {full_f1sd:.4f}\n"
        f"  AUROC = {full_auc:.4f} +/- {full_aucsd:.4f}\n\n"
        f"Reduced feature set ({len(reduced_cols)} features, "
        f"{pct_dropped:.0f}% fewer -- dropped {dropped_cols}):\n"
        f"  F1    = {red_f1:.4f} +/- {red_f1sd:.4f}\n"
        f"  AUROC = {red_auc:.4f} +/- {red_aucsd:.4f}\n\n"
        f"Delta: F1 {delta_f1:+.4f}  AUROC {delta_auc:+.4f}\n\n"
    )
    if delta_f1 >= -0.01 and delta_auc >= -0.01:
        text += (f"Reduced set matches or beats the full set (F1 {delta_f1:+.4f}, "
                 f"AUROC {delta_auc:+.4f}): the {len(dropped_cols)} dropped features "
                 f"can be removed without a meaningful loss in detection power, at "
                 f"{pct_dropped:.0f}% lower feature-extraction/training cost.\n")
    else:
        text += (f"Reduced set costs real detection power (F1 {delta_f1:+.4f}, "
                 f"AUROC {delta_auc:+.4f}): do not drop these {len(dropped_cols)} "
                 f"features purely for compute savings.\n")

    print(text)
    path = out_dir / 'feature_list_optimization.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f"  Saved: {path}")
    return reduced_cols


def main():
    p = argparse.ArgumentParser(description="Unified leak-free repeated CV + ablation")
    p.add_argument('--input', required=True)
    p.add_argument('--label-col', required=True)
    p.add_argument('--technique-col', default='technique')
    p.add_argument('--exclude-techniques', default=None)
    p.add_argument('--dataset-name', required=True)
    p.add_argument('--repeats', type=int, default=15,
                   help='CV repeats for the headline result (use the SAME value on both datasets)')
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Unified Validation — {args.dataset_name}")
    print("=" * 65)

    df, feat_cols = load_and_filter(args)
    dup_pct, groups = check_duplication(df, feat_cols, out_dir, args)
    cv_df, auroc_mean, auroc_std, f1_mean, f1_std = check_split(df, feat_cols, groups, args, out_dir)
    abl_df = check_feature_ablation(df, feat_cols, groups, args, out_dir)

    lines = [f"Validation Summary — {args.dataset_name}", "=" * 60,
             f"\nDataset: {len(df):,} rows, {len(feat_cols)} features",
             f"Duplicate feature vectors: {dup_pct:.1f}%",
             f"\nHeadline result ({args.repeats}x repeated {args.n_splits}-fold "
             f"vector-group CV, {args.repeats*args.n_splits} total folds -- "
             f"this is the number to report in Table 8):",
             f"  AUROC = {auroc_mean:.4f} +/- {auroc_std:.4f}",
             f"  F1    = {f1_mean:.4f} +/- {f1_std:.4f}",
             f"\nFeature ablation (sorted by importance):"]
    for _, row in abl_df.sort_values('delta_f1').iterrows():
        lines.append(f"  {row['feature']:<25} DeltaF1={row['delta_f1']:+.4f}  ({row['tag']})")

    text = '\n'.join(lines)
    print("\n" + text)
    with open(out_dir / 'validation_summary.txt', 'w') as f:
        f.write(text)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == '__main__':
    main()

