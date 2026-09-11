"""
Milestone 3 -- confirm (not just hypothesize) why process-relative features
helped repeats 5/12 but not repeat 3, and broke previously-stable repeat 7.

Part 1 (repeat 3): reproduce its fold split, find the mixed-label group(s)
routed to test, check whether their proc_name had >=min_support occurrences
in that fold's TRAINING data. If not, the fallback-to-global-baseline
hypothesis is confirmed directly, not just plausible.

Part 2 (repeat 7): reproduce its fold split, fit BOTH the baseline and
augmented models, find attack rows the baseline caught but the augmented
model missed (the actual recall-loss rows), then check those rows'
proc_name and that process's TRAINING-fold label balance. If the process's
training occurrences were mostly/entirely benign, the "skewed baseline
masked a real attack" hypothesis is confirmed directly.

Usage:
    python confirm_relative_feature_mechanism.py --data-dir data \
        --out results/current/relative_features_test/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import CV_CONFIG, MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, impute
from train_eval import _fit_predict_rf

RELATIVE_CANDIDATE_FEATURES = [
    'exec_count', 'unique_syscalls', 'failed_call_rate', 'failed_call_count',
    'lifetime_seconds', 'events_per_second', 'priv_op_count', 'network_count',
    'file_access_count', 'seq_length',
]


def add_relative_features(df_train, df_apply, base_features, proc_col='proc_name', min_support=30):
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
    return out, supported_procs


def analyze_repeat3(df, y, groups, feat_cols, base_seed, min_support, out_dir):
    print("\n" + "="*60)
    print("PART 1: Repeat 3 -- why didn't relative features help?")
    print("="*60)
    seed = base_seed + 3
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    group_label_sets = {}
    for g, lbl in zip(groups, y):
        group_label_sets.setdefault(g, set()).add(lbl)
    mixed_groups = {g for g, labels in group_label_sets.items() if len(labels) > 1}

    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(df[feat_cols].values, y, groups)):
        df_tr = df.iloc[tr]
        te_groups_in_fold = set(groups[te])
        mixed_in_test = te_groups_in_fold & mixed_groups
        if not mixed_in_test:
            continue
        proc_counts_tr = df_tr['proc_name'].value_counts()
        for g in mixed_in_test:
            g_size = (groups == g).sum()
            if g_size < 100:  # focus on the large ones driving the FPR spike
                continue
            g_procs = df[groups == g]['proc_name'].value_counts()
            for proc, cnt in g_procs.items():
                support = proc_counts_tr.get(proc, 0) if pd.notna(proc) else 0
                rows.append({'fold': fold_idx, 'group_size': g_size, 'proc_name': proc,
                             'rows_in_group': cnt, 'train_fold_support': support,
                             'above_min_support': support >= min_support})

    result = pd.DataFrame(rows).sort_values('group_size', ascending=False)
    print(result.to_string(index=False))
    n_supported = result['above_min_support'].sum() if len(result) else 0
    print(f"\n>>> {n_supported}/{len(result)} (proc_name, group) pairs had enough training "
          f"support for a relative baseline.")
    if len(result) and n_supported == 0:
        print(">>> CONFIRMED: every large mixed-label group in repeat 3's test set involved "
              "processes with insufficient training support -- relative features had no "
              "usable signal to compute, hence no improvement. Hypothesis confirmed directly.")
    result.to_csv(out_dir / "repeat3_mechanism.csv", index=False)


def analyze_repeat7(df, y, groups, feat_cols, base_seed, min_support, out_dir):
    print("\n" + "="*60)
    print("PART 2: Repeat 7 -- why did relative features break a good fold?")
    print("="*60)
    seed = base_seed + 7
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    all_flipped = []
    for fold_idx, (tr, te) in enumerate(cv.split(df[feat_cols].values, y, groups)):
        df_tr, df_te = df.iloc[tr], df.iloc[te]
        y_tr, y_te = y[tr], y[te]
        groups_tr = groups[tr]

        X_tr_base = impute(df_tr[feat_cols].values)
        X_te_base = impute(df_te[feat_cols].values)
        pred_base, _ = _fit_predict_rf(X_tr_base, y_tr, groups_tr, X_te_base, None, MAX_FPR)

        rel_tr, supported = add_relative_features(df_tr, df_tr, RELATIVE_CANDIDATE_FEATURES, min_support=min_support)
        rel_te, _ = add_relative_features(df_tr, df_te, RELATIVE_CANDIDATE_FEATURES, min_support=min_support)
        X_tr_aug = impute(pd.concat([df_tr[feat_cols].reset_index(drop=True), rel_tr.reset_index(drop=True)], axis=1).values)
        X_te_aug = impute(pd.concat([df_te[feat_cols].reset_index(drop=True), rel_te.reset_index(drop=True)], axis=1).values)
        pred_aug, _ = _fit_predict_rf(X_tr_aug, y_tr, groups_tr, X_te_aug, None, MAX_FPR)

        # Rows baseline caught (true attack, predicted attack) that augmented MISSED
        flipped_mask = (y_te == 1) & (pred_base == 1) & (pred_aug == 0)
        if flipped_mask.sum() == 0:
            continue
        flipped_rows = df_te[flipped_mask].copy()
        flipped_rows['fold'] = fold_idx

        proc_counts_tr = df_tr['proc_name'].value_counts()
        proc_label_mean_tr = df_tr.groupby('proc_name').apply(lambda g: y_tr[df_tr.index.get_indexer(g.index)].mean())
        for _, row in flipped_rows.iterrows():
            proc = row['proc_name']
            all_flipped.append({
                'fold': fold_idx, 'proc_name': proc,
                'train_fold_support': proc_counts_tr.get(proc, 0) if pd.notna(proc) else 0,
                'train_fold_attack_rate_for_this_proc': proc_label_mean_tr.get(proc, np.nan) if pd.notna(proc) else np.nan,
            })

    result = pd.DataFrame(all_flipped)
    if len(result):
        print(result.to_string(index=False))
        low_attack_rate = result['train_fold_attack_rate_for_this_proc'] < 0.2
        print(f"\n>>> {low_attack_rate.sum()}/{len(result)} newly-missed attack rows belong to "
              f"processes whose training-fold occurrences were <20% attack (i.e. a "
              f"benign-skewed baseline that could plausibly mask a real attack).")
    else:
        print("No newly-missed attack rows found in this reproduction -- the F1 drop may be "
              "driven by something other than a simple baseline/recall-masking mechanism; "
              "worth checking false positives too, not just false negatives.")
    result.to_csv(out_dir / "repeat7_mechanism.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
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

    analyze_repeat3(df, y, groups, feat_cols, base_seed, args.min_support, out_dir)
    analyze_repeat7(df, y, groups, feat_cols, base_seed, args.min_support, out_dir)

    print(f"\nSaved detail CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
