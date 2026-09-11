"""
Milestone 3 -- diagnose the shared bad-repeat finding (repeats 1 and 8 are
bad for BOTH RF and XGBoost on v3 CAM-LDS, at the same seed-determined fold
split). Hypothesis: certain repeat-seeds create a StratifiedGroupKFold split
where a large duplicate-feature-vector group -- which the leak-free splitter
correctly keeps together -- lands almost entirely in one fold's TEST set for
some repeats, but is more evenly distributed across folds for others. If that
group is benign and its behavioural pattern isn't well-represented in TRAIN
for the unlucky fold, the model's F1-optimal threshold (tuned on train) may
not transfer to test, producing exactly the low-precision/high-FPR failure
mode observed (bad repeats: precision ~0.53, FPR ~0.32; good repeats:
precision ~0.95, FPR ~0.02).

This only inspects fold COMPOSITION (which groups land where) -- it does not
retrain anything, so it's fast even though the dataset is large.

Usage:
    python diagnose_bad_repeats.py --dataset camlds --data-dir data/v3 \
        --good-repeat 0 --bad-repeats 1,8 --out results/current/bad_repeat_diagnosis/
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import CV_CONFIG
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key


def diagnose_repeat(X, y, groups, n_splits, seed, label):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    print(f"\n=== {label} (seed={seed}) ===")

    group_sizes = Counter(groups)
    # Verify groups are label-homogeneous (duplicate feature vectors should
    # share a label; if not, that itself would be worth flagging).
    group_labels = {}
    mixed_label_groups = 0
    for g, lbl in zip(groups, y):
        if g in group_labels and group_labels[g] != lbl:
            mixed_label_groups += 1
        group_labels[g] = lbl
    if mixed_label_groups:
        print(f"  WARNING: {mixed_label_groups} rows found in groups with "
              f"inconsistent labels -- duplicate feature vectors do NOT always "
              f"share a label. This alone could explain threshold instability.")

    top_groups = [g for g, _ in Counter(groups).most_common(10)]

    fold_reports = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
        te_groups = set(groups[te])
        for g in top_groups:
            if g in te_groups:
                frac_in_test = (groups[te] == g).sum() / group_sizes[g]
                fold_reports.append({
                    'fold': fold_idx, 'group_id': g, 'group_size': group_sizes[g],
                    'group_label': group_labels[g],
                    'frac_of_group_in_this_test_fold': round(frac_in_test, 3),
                })

    df = pd.DataFrame(fold_reports).sort_values('group_size', ascending=False)
    print(df.to_string(index=False))

    # Flag any large group that's >90% concentrated in a single fold's test set
    concentrated = df[(df['group_size'] > 500) & (df['frac_of_group_in_this_test_fold'] > 0.9)]
    if len(concentrated):
        print(f"\n  >>> {len(concentrated)} large group(s) almost entirely in one "
              f"fold's TEST set (label={concentrated['group_label'].tolist()}):")
        print(f"  >>> {concentrated[['group_id','group_size','group_label','fold']].to_string(index=False)}")
    else:
        print(f"\n  No single large group is >90% concentrated in one test fold.")

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camlds")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--good-repeat", type=int, default=0)
    ap.add_argument("--bad-repeats", default="1,8")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest(args.dataset, data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, args.dataset, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    base_seed = CV_CONFIG["seed"]
    print(f"Base seed (from config.CV_CONFIG): {base_seed}")
    print(f"Total rows: {len(y):,}  Unique groups: {len(set(groups)):,}")

    good_df = diagnose_repeat(X, y, groups, args.n_splits, base_seed + args.good_repeat,
                               f"GOOD repeat {args.good_repeat}")
    good_df.to_csv(out_dir / f"good_repeat_{args.good_repeat}_composition.csv", index=False)

    for bad_rep in [int(r) for r in args.bad_repeats.split(",")]:
        bad_df = diagnose_repeat(X, y, groups, args.n_splits, base_seed + bad_rep,
                                  f"BAD repeat {bad_rep}")
        bad_df.to_csv(out_dir / f"bad_repeat_{bad_rep}_composition.csv", index=False)

    print(f"\nSaved all fold-composition CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
