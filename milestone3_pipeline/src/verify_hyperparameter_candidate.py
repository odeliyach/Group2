"""
Milestone 3 -- hyperparameter tuning verifier.

A cheap 3-repeat sensitivity sweep can only rank configs relative to each
other; it's too noisy to say a candidate genuinely beats production (see the
XGBoost case already in the report: sweep showed depth=9/lr=0.01 "marginally
better" at 3 repeats, but that was never strong enough evidence to justify
a production change). This script settles it properly: it runs BOTH the
current production config AND a candidate config under the exact same
15-repeat pooled protocol that produced your actual reported headline
numbers, so the comparison is apples-to-apples with what's already in the
report -- not a sweep-vs-headline mismatch.

Usage:
    python verify_hyperparameter_candidate.py --model rf --dataset camlds \
        --data-dir data --candidate-params '{"n_estimators": 500, "max_depth": null}' \
        --out results/current/hyperparameter_verification/

Reuses train_eval.py's repeated_cv exactly as the headline pipeline does --
this is not a new methodology, just the existing one applied to one more config.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import RF_PARAMS, XGB_PARAMS, CNN_PARAMS, IFOREST_PARAMS  # noqa: F401 -- kept for reference
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key
from train_eval import repeated_cv  # confirmed signature: (X, y, groups, model_key,
                                     # repeats, n_splits, hp_overrides, seed, max_fpr)


def run_config(model_key, dataset_key, data_dir, hp_overrides, repeats=15, n_splits=5, seed=42):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    # hp_overrides=None -> repeated_cv uses whatever config.py's current
    # PARAMS dict already specifies internally, i.e. genuine production.
    return repeated_cv(X, y, groups, model_key, repeats=repeats, n_splits=n_splits,
                        hp_overrides=hp_overrides, seed=seed)


def main():
    ap = argparse.ArgumentParser(description="Verify a candidate hyperparameter config "
                                              "against production under the headline protocol")
    ap.add_argument("--model", choices=["rf", "xgb", "cnn", "iforest"], required=True)
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--candidate-params", required=True,
                     help='JSON dict of ONLY the params that differ from production, '
                          'e.g. \'{"n_estimators": 500, "max_depth": null}\'')
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15,
                     help="Match the headline protocol's repeat count (default 15) -- "
                          "don't lower this just to save time, it defeats the point")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_overrides = json.loads(args.candidate_params)

    print(f"=== {args.model} / {args.dataset}: production vs. candidate, "
          f"{args.repeats}-repeat protocol (matches headline methodology) ===")
    print(f"Candidate overrides (delta from production): {candidate_overrides}")
    print()

    print("Running PRODUCTION config (hp_overrides=None -> current config.py values)...")
    prod_df = run_config(args.model, args.dataset, args.data_dir, None, args.repeats)
    print(f"  F1={prod_df['f1'].mean():.4f}\u00b1{prod_df['f1'].std():.4f}  "
          f"AUROC={prod_df['auroc'].mean():.4f}\u00b1{prod_df['auroc'].std():.4f}")

    print("Running CANDIDATE config...")
    cand_df = run_config(args.model, args.dataset, args.data_dir, candidate_overrides, args.repeats)
    print(f"  F1={cand_df['f1'].mean():.4f}\u00b1{cand_df['f1'].std():.4f}  "
          f"AUROC={cand_df['auroc'].mean():.4f}\u00b1{cand_df['auroc'].std():.4f}")

    delta_f1 = cand_df['f1'].mean() - prod_df['f1'].mean()
    # Simple two-sample t-test-style check: is the delta bigger than the combined noise?
    pooled_std = np.sqrt(prod_df['f1'].std()**2 + cand_df['f1'].std()**2)
    signal_to_noise = abs(delta_f1) / pooled_std if pooled_std > 0 else float('inf')

    verdict = (
        "CANDIDATE GENUINELY BETTER -- consider adopting, then re-run downstream "
        "pipeline (cross-dataset, cascade, error analysis) to match"
        if delta_f1 > 0.01 and signal_to_noise > 1.0 else
        "NO REAL DIFFERENCE -- delta is within noise, keep production config as-is, "
        "cite this verification in the report instead of leaving it unresolved"
        if signal_to_noise < 1.0 else
        "CANDIDATE WORSE -- keep production config"
    )

    summary = {
        "model": args.model, "dataset": args.dataset,
        "production_f1_mean": prod_df['f1'].mean(), "production_f1_std": prod_df['f1'].std(),
        "candidate_f1_mean": cand_df['f1'].mean(), "candidate_f1_std": cand_df['f1'].std(),
        "delta_f1": delta_f1, "signal_to_noise": signal_to_noise, "verdict": verdict,
    }
    print(f"\nDelta F1: {delta_f1:+.4f}  (signal/noise ratio: {signal_to_noise:.2f})")
    print(f"VERDICT: {verdict}")

    prod_df.to_csv(out_dir / f"{args.dataset}_{args.model}_production.csv", index=False)
    cand_df.to_csv(out_dir / f"{args.dataset}_{args.model}_candidate.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / f"{args.dataset}_{args.model}_verdict.csv", index=False)
    print(f"\nSaved: {out_dir}/{args.dataset}_{args.model}_verdict.csv")


if __name__ == "__main__":
    main()
