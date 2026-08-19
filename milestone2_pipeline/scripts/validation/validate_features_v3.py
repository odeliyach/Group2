"""
Deep Feature Validation v3 — vector-hash group split (no leakage, no over-dedup)
==================================================================================
Fixes from v2:
  - v2 either deduplicated the corpus outright (collapsed benign from
    37,017 -> 961 rows, unusable) or grouped by 'technique' (which
    collapsed all benign into one giant group, causing AUROC=nan).
  - v3 groups by a hash of the full feature vector itself. Identical
    rows (duplicates) are guaranteed to land on the same side of the
    split -- eliminating leakage -- while KEEPING every row (131,344)
    for training, preserving real class frequency/ratio signal.

Usage:
    python validate_features_v3.py \
        --features combined/dataset1_features.csv \
        --out validation_output_v5/
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
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score, ConfusionMatrixDisplay
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
})

ALL_FEATURES = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]

# Excludes lifetime_seconds / events_per_second -- prior analysis showed
# these are dominated by a near-zero-duration floor-value artifact for
# single-event benign groups rather than genuine behavioral signal.
# Full 20-feature set from Ch1's original feature table -- kept fixed and
# identical to run_eda.py's feature set. Features that show inverted or
# unstable behavior in ablation (max_uid_euid_delta, uid_changed,
# ephemeral_privileged, failed_call_rate, lifetime_seconds, events_per_second)
# are NOT excluded here -- their behavior is reported and discussed as a
# Ch4.2 discrepancy-analysis finding, not used as a silent exclusion
# criterion. Excluding them would make Ch4.1 (EDA importance, full feature
# set) and Ch6.2 (this validation) incomparable, which defeats the purpose
# of the discrepancy analysis.
SAFE_FEATURES = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]


def train_rf(X_train, y_train, **kwargs):
    defaults = dict(n_estimators=200, class_weight='balanced',
                    n_jobs=-1, random_state=42)
    defaults.update(kwargs)
    rf = RandomForestClassifier(**defaults)
    rf.fit(X_train, y_train)
    return rf


def evaluate(model, X_test, y_test, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    f1   = report['1']['f1-score']
    prec = report['1']['precision']
    rec  = report['1']['recall']
    return auc, f1, prec, rec, y_proba, y_pred


def get_all_kfold_splits(df, feat_cols, y, n_splits=5, seed=42):
    """
    Returns ALL StratifiedGroupKFold splits (not just the first) so
    metrics can be averaged across folds. This matters most for sparse
    datasets (few unique vector groups relative to row count) where a
    single split can be wildly unrepresentative -- e.g. the LPE-only
    dataset has only 2,399 unique groups for 163,678 rows, and one
    unlucky split put just 250 benign rows in test out of 42,264.
    Averaging across folds gives a stable, trustworthy estimate instead.
    """
    y = np.asarray(y)
    vector_group = df[feat_cols].astype(str).agg('|'.join, axis=1).values

    vg_series = pd.Series(vector_group)
    label_counts_per_group = pd.DataFrame({'group': vector_group, 'label': y}) \
        .groupby('group')['label'].nunique()
    ambiguous = (label_counts_per_group > 1).sum()
    if ambiguous > 0:
        print(f"  ℹ️  {ambiguous:,} feature-vector patterns appear under BOTH "
              f"labels (irreducible ambiguity under this feature set)")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in sgkf.split(df, y, groups=vector_group):
        train_groups = set(vector_group[train_idx])
        test_groups  = set(vector_group[test_idx])
        overlap = len(train_groups & test_groups)
        assert overlap == 0, f"Leakage detected: {overlap} overlapping vector groups!"
        splits.append((train_idx, test_idx))

    return splits, len(set(vector_group))


def get_vector_group_split(df, feat_cols, y, test_size=0.2, seed=42):
    """Single-fold convenience wrapper, kept for backward compatibility
    with the ablation function below (which needs speed over stability
    for the per-feature sweep)."""
    n_splits = max(2, round(1 / test_size))
    splits, n_groups = get_all_kfold_splits(df, feat_cols, y, n_splits, seed)
    train_idx, test_idx = splits[0]
    return train_idx, test_idx, n_groups


# ── Check 0 — Duplication report ──────────────────────────────────────────────

def check_duplication(df, out_dir):
    print("\n[0] Duplicate feature vector check")
    feat_cols = [f for f in ALL_FEATURES if f in df.columns]
    dup_count = df.duplicated(subset=feat_cols).sum()
    pct = 100 * dup_count / len(df)
    print(f"  Duplicate rows: {dup_count:,} / {len(df):,} ({pct:.1f}%)")
    print(f"  Using vector-hash group split (guarantees no leakage, "
          f"keeps all rows for training)")
    return pct


# ── Check 1 — Full evaluation on vector-group split ───────────────────────────

def check_split(df, out_dir, n_splits=5):
    print(f"\n[1] Vector-group K-fold evaluation (leak-free, {n_splits} folds, averaged)")

    cols = [f for f in SAFE_FEATURES if f in df.columns]
    X    = df[cols].fillna(0).values
    y    = df['label'].values

    splits, n_groups = get_all_kfold_splits(df, cols, y, n_splits=n_splits)
    print(f"  Unique vector groups: {n_groups:,}  ({n_splits} folds)")
    print(f"  Verified: zero overlap between train/test vector groups (all folds)")

    fold_results = []
    best_fold = None
    best_fold_auc = -1

    for i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        rf = train_rf(X_tr, y_tr)
        auc, f1, prec, rec, y_proba, y_pred = evaluate(rf, X_te, y_te)
        fold_results.append(dict(auc=auc, f1=f1, prec=prec, rec=rec,
                                 n_test=len(test_idx),
                                 benign_test=int(np.sum(y_te==0)),
                                 attack_test=int(np.sum(y_te==1))))
        print(f"  Fold {i+1}: test={len(test_idx):,} "
              f"(benign={np.sum(y_te==0):,} attack={np.sum(y_te==1):,})  "
              f"AUROC={auc:.4f}  F1={f1:.4f}")
        if auc > best_fold_auc:
            best_fold_auc = auc
            best_fold = (rf, X_te, y_te, y_pred, y_proba, test_idx)

    aucs = [r['auc'] for r in fold_results]
    f1s  = [r['f1']  for r in fold_results]
    print(f"\n  Mean AUROC={np.mean(aucs):.4f} (±{np.std(aucs):.4f})  "
          f"Mean F1={np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
    print(f"  This is the trustworthy result -- single-fold numbers above "
          f"are shown for transparency but averaged metrics should be reported.")

    # Return the fold closest to the median AUROC for downstream plots
    # (representative, not cherry-picked best or worst)
    median_auc = np.median(aucs)
    closest_idx = int(np.argmin([abs(a - median_auc) for a in aucs]))
    train_idx, test_idx = splits[closest_idx]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    rf = train_rf(X_tr, y_tr)
    _, _, _, _, y_proba, y_pred = evaluate(rf, X_te, y_te)

    return rf, X_te, y_te, y_pred, y_proba, test_idx, cols, fold_results


# ── Check 2 — Confusion matrix + per-technique detection ─────────────────────

def check_confusion_and_techniques(df, y_te, y_pred, y_proba, test_idx, out_dir):
    print("\n[2] Confusion matrix and per-technique detection")

    auc = roc_auc_score(y_te, y_proba)
    f1  = f1_score(y_te, y_pred, pos_label=1, zero_division=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm   = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Attack'])
    disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title(f'Confusion Matrix\nAUROC={auc:.3f}  F1={f1:.3f}')

    df_test = df.iloc[test_idx].copy()
    df_test['predicted'] = y_pred
    df_test['proba']     = y_proba

    attack_test = df_test[df_test['label'] == 1]
    if 'technique' in df_test.columns and len(attack_test) > 0:
        tech_detection = attack_test.groupby('technique').apply(
            lambda g: pd.Series({
                'total':     len(g),
                'detected':  g['predicted'].sum(),
                'det_rate':  g['predicted'].mean(),
            })
        ).reset_index().sort_values('det_rate')

        tech_detection.to_csv(out_dir / 'per_technique_detection.csv', index=False)
        print(f"\n  Per-technique detection rate:")
        for _, row in tech_detection.iterrows():
            bar = '█' * int(row['det_rate'] * 20)
            print(f"    {str(row['technique'])[:35]:<35} "
                  f"{row['det_rate']*100:5.1f}%  "
                  f"({int(row['detected'])}/{int(row['total'])})  {bar}")

        axes[1].barh(range(len(tech_detection)), tech_detection['det_rate'].values,
                     color=['#E53935' if r < 0.5 else '#43A047'
                            for r in tech_detection['det_rate'].values])
        axes[1].set_yticks(range(len(tech_detection)))
        axes[1].set_yticklabels([str(t)[:30] for t in tech_detection['technique']],
                                fontsize=8)
        axes[1].set_xlabel('Detection rate')
        axes[1].set_xlim(0, 1)
        axes[1].axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Detection Rate by Technique')

    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()
    print(f"  Saved confusion_matrix.png")


# ── Check 3 — Feature ablation (vector-group split, index-safe) ─────────────

def check_feature_ablation(df, out_dir):
    print("\n[3] Feature ablation (vector-group split, index-safe)")

    base_cols = [f for f in SAFE_FEATURES if f in df.columns]
    y = df['label'].values

    def eval_subset(cols):
        train_idx, test_idx, _ = get_vector_group_split(df, cols, y)
        X = df[cols].fillna(0).values
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        rf = train_rf(X_tr, y_tr)
        auc, f1, _, _, _, _ = evaluate(rf, X_te, y_te)
        return auc, f1

    base_auc, base_f1 = eval_subset(base_cols)
    print(f"  Baseline (all safe features): AUROC={base_auc:.4f} F1={base_f1:.4f}")

    results = []
    for feat in base_cols:
        remaining = [f for f in base_cols if f != feat]
        auc_ab, f1_ab = eval_subset(remaining)
        delta_f1  = f1_ab - base_f1
        delta_auc = auc_ab - base_auc
        results.append({
            'feature': feat, 'f1_without': f1_ab, 'auc_without': auc_ab,
            'delta_f1': delta_f1, 'delta_auc': delta_auc,
        })
        direction = ('important' if delta_f1 < -0.005
                     else ('hurts' if delta_f1 > 0.005 else 'neutral'))
        print(f"  Remove {feat:<25} F1={f1_ab:.4f} (Δ{delta_f1:+.4f}) [{direction}]")

    abl_df = pd.DataFrame(results).sort_values('delta_f1')
    abl_df.to_csv(out_dir / 'feature_ablation.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#E53935' if d < -0.005 else ('#43A047' if d > 0.005 else '#90A4AE')
              for d in abl_df['delta_f1'].values]
    ax.barh(range(len(abl_df)), abl_df['delta_f1'].values, color=colors)
    ax.set_yticks(range(len(abl_df)))
    ax.set_yticklabels(abl_df['feature'].values, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('ΔF1 when feature removed (negative = feature helps)')
    ax.set_title('Feature Ablation — Impact on F1 (vector-group split)\n'
                 'Red=important, Green=hurts model, Gray=neutral')
    plt.tight_layout()
    plt.savefig(out_dir / 'feature_ablation.png')
    plt.close()
    print(f"  Saved feature_ablation.png")
    return abl_df


# ── Check 4 — Threshold optimization ─────────────────────────────────────────

def check_threshold(rf, X_te, y_te, out_dir):
    print("\n[4] Threshold optimization for F1")

    y_proba = rf.predict_proba(X_te)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_te, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx  = int(np.argmax(f1_scores))
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1   = f1_scores[best_idx]
    best_prec = precisions[best_idx]
    best_rec  = recalls[best_idx]

    default_f1 = f1_score(y_te, (y_proba >= 0.5).astype(int),
                          pos_label=1, zero_division=0)
    print(f"  Default threshold (0.5): F1={default_f1:.4f}")
    print(f"  Optimal threshold ({best_thresh:.3f}): "
          f"F1={best_f1:.4f}  P={best_prec:.4f}  R={best_rec:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].scatter([best_rec], [best_prec], color='red', s=100, zorder=5,
                    label=f'Best F1={best_f1:.3f}\n@ threshold={best_thresh:.3f}')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend(fontsize=9); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    valid_thresh = thresholds[thresholds < 1]
    valid_f1     = f1_scores[:len(valid_thresh)]
    axes[1].plot(valid_thresh, valid_f1, 'g-', linewidth=2)
    axes[1].axvline(best_thresh, color='red', linestyle='--',
                    label=f'Best={best_thresh:.3f}')
    axes[1].axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default=0.5')
    axes[1].set_xlabel('Classification threshold'); axes[1].set_ylabel('F1')
    axes[1].set_title('F1 vs Threshold'); axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / 'precision_recall_curve.png')
    plt.close()
    print(f"  Saved precision_recall_curve.png")
    return best_thresh, best_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Deep Feature Validation v3 — Vector-Group Split (Leak-Free)")
    print("=" * 65)

    df = pd.read_csv(args.features, low_memory=False)
    print(f"Loaded: {len(df):,} processes  "
          f"Benign: {(df['label']==0).sum():,}  Attack: {(df['label']==1).sum():,}")

    dup_pct = check_duplication(df, out)
    rf, X_te, y_te, y_pred, y_proba, test_idx, cols, fold_results = check_split(df, out)
    check_confusion_and_techniques(df, y_te, y_pred, y_proba, test_idx, out)
    abl_df = check_feature_ablation(df, out)
    best_t, best_f1 = check_threshold(rf, X_te, y_te, out)

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    aucs = [r['auc'] for r in fold_results]
    f1s  = [r['f1']  for r in fold_results]

    lines = ["Deep Feature Validation Summary (v3 - vector-group K-fold)",
             "=" * 60]
    lines.append(f"\nDataset: {len(df):,} processes")
    lines.append(f"Duplicate feature vectors: {dup_pct:.1f}% "
                 f"(kept via vector-group split, not discarded)")
    lines.append(f"\nMain result (safe features, {len(fold_results)}-fold "
                 f"vector-group CV, AVERAGED -- this is the number to report):")
    lines.append(f"  AUROC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    lines.append(f"  F1    = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    lines.append(f"\nPer-fold detail:")
    for i, r in enumerate(fold_results):
        lines.append(f"  Fold {i+1}: n_test={r['n_test']:,} "
                     f"(benign={r['benign_test']:,} attack={r['attack_test']:,})  "
                     f"AUROC={r['auc']:.4f}  F1={r['f1']:.4f}")
    lines.append(f"\nOptimal threshold: {best_t:.3f} -> F1={best_f1:.4f}")

    lines.append(f"\nFeature importance (ablation):")
    for _, row in abl_df.sort_values('delta_f1').iterrows():
        tag = 'important' if row['delta_f1'] < -0.005 else (
              'hurts' if row['delta_f1'] > 0.005 else 'neutral')
        lines.append(f"  {row['feature']:<25} ΔF1={row['delta_f1']:+.4f}  ({tag})")

    text = '\n'.join(lines)
    print(text)
    with open(out / 'validation_summary.txt', 'w') as f:
        f.write(text)

    print(f"\nAll outputs saved to {out}/")


if __name__ == '__main__':
    main()
