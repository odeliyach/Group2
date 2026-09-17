"""
Follow-up to diagnose_bad_repeats.py. That script's original hypothesis
(large groups split unevenly across folds) was wrong by construction --
StratifiedGroupKFold NEVER splits a group across folds, so "is a group
concentrated in one fold" is trivially always true and can't discriminate
good vs. bad repeats.

What it DID surface, buried in a warning, is more useful: ~13% of CAM-LDS
rows (11,806/90,805) belong to feature-vector groups containing BOTH benign
and attack labels -- i.e. identical features, contradictory ground truth.
Since group integrity is guaranteed, an ENTIRE mixed-label group lands in
one fold together. If a large one lands in TEST for a given repeat, the
model never saw that ambiguous pattern in training and may misclassify the
whole group in a correlated burst -- a better candidate mechanism for the
observed FPR spikes (repeats 3, 5, 12) than fold-composition alone.

This script tests that directly: for each repeat, how many mixed-label-group
rows end up in the test fold, and does that count track the observed FPR.

Usage:
    python check_mixed_label_test_exposure.py --dataset camlds --data-dir data \
        --repeats 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
        --out results/current/bad_repeat_diagnosis/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import CV_CONFIG
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camlds")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repeats", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
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

    # Identify mixed-label groups directly (not just count them).
    group_label_sets = defaultdict(set)
    for g, lbl in zip(groups, y):
        group_label_sets[g].add(lbl)
    mixed_groups = {g for g, labels in group_label_sets.items() if len(labels) > 1}
    is_mixed_row = np.array([g in mixed_groups for g in groups])
    print(f"Mixed-label groups: {len(mixed_groups):,} groups, "
          f"{is_mixed_row.sum():,} rows ({100*is_mixed_row.mean():.1f}% of dataset)")

    base_seed = CV_CONFIG["seed"]
    rows = []
    for rep in [int(r) for r in args.repeats.split(",")]:
        seed = base_seed + rep
        cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        # Pool "test" across all folds within this repeat -- every row is
        # test exactly once per repeat (5-fold), matching how the headline
        # metric itself pools predictions.
        mixed_rows_in_test = 0
        for _, te in cv.split(X, y, groups):
            mixed_rows_in_test += is_mixed_row[te].sum()
        rows.append({'repeat': rep, 'seed': seed, 'mixed_label_rows_in_test': int(mixed_rows_in_test)})
        print(f"  repeat {rep} (seed={seed}): {mixed_rows_in_test:,} mixed-label rows in test")

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.dataset}_mixed_label_test_exposure.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print("\nCross-reference this 'mixed_label_rows_in_test' column against each")
    print("repeat's actual FPR from the headline CV CSV -- if repeats 3/5/12 show")
    print("notably higher mixed-label exposure than repeat 0 (and the other 11")
    print("'good' repeats), that confirms the mechanism.")


if __name__ == "__main__":
    main()
