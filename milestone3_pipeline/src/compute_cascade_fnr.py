"""
Milestone 3 report -- gap-filling script 3 of 3: cascade FNR.

hybrid_cascade.py's existing CV loop records F1/AUROC/FPR but never recall/FNR,
and the template explicitly asks for FNR alongside F1 and FPR (S3.2). Rather
than patch hybrid_cascade.py itself (risk of touching code the rest of the
pipeline depends on), this script reuses its exact cascade_predict() function
--the real production cascade logic, unmodified-- and just records the one
missing metric.

Uses the SAME 15-repeat pooled protocol as every other headline number in
this project, not a cheap 3-repeat sweep -- this is meant to be a final,
citable number for S3.2's table, not an exploratory result.

Usage:
    python compute_cascade_fnr.py --dataset camlds --data-dir data     --out results/current/hybrid_cascade_original/
    python compute_cascade_fnr.py --dataset camlds --data-dir data/v3 --out results/current/hybrid_cascade_v3_reverify/
    python compute_cascade_fnr.py --dataset casino --data-dir data     --out results/current/hybrid_cascade/
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
from preprocessing import vector_group_key
from train_eval import _fpr
from hybrid_cascade import cascade_predict  # the real, unmodified production cascade


def run(dataset_key, data_dir, max_fpr, repeats=15, n_splits=5, base_seed=42):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_f1, fold_auc, fold_fpr, fold_recall = [], [], [], []
        for tr, te in cv.split(X, y, groups):
            if len(set(y[te])) < 2:
                continue
            y_pred, y_proba, *_ = cascade_predict(X[tr], y[tr], X[te], feat_cols, max_fpr)
            fold_f1.append(f1_score(y[te], y_pred, zero_division=0))
            fold_auc.append(roc_auc_score(y[te], y_proba))
            fold_fpr.append(_fpr(y[te], y_pred))
            fold_recall.append(recall_score(y[te], y_pred, zero_division=0))
        if fold_f1:
            rows.append({
                "repeat": rep, "seed": seed,
                "f1": np.mean(fold_f1), "auroc": np.mean(fold_auc),
                "fpr": np.mean(fold_fpr), "recall": np.mean(fold_recall),
                "fnr": 1 - np.mean(fold_recall),
            })
        print(f"[{dataset_key}] repeat {rep+1}/{repeats}: "
              f"F1={rows[-1]['f1']:.4f} FPR={rows[-1]['fpr']:.4f} FNR={rows[-1]['fnr']:.4f}")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compute cascade FNR (missing metric for report S3.2)")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fpr", type=float, default=MAX_FPR)
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run(args.dataset, args.data_dir, args.max_fpr, args.repeats)
    csv_path = out_dir / f"{args.dataset}_cascade_fnr.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n=== {args.dataset}: report this in S3.2's table ===")
    print(f"F1    = {df['f1'].mean():.4f} \u00b1 {df['f1'].std():.4f}")
    print(f"FPR   = {df['fpr'].mean():.4f} \u00b1 {df['fpr'].std():.4f}")
    print(f"FNR   = {df['fnr'].mean():.4f} \u00b1 {df['fnr'].std():.4f}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
