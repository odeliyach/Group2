"""
Milestone 3 — Step 7: Main pipeline entry point.

Usage:
    python run_pipeline.py --dataset camlds --model all --data-dir .
    python run_pipeline.py --dataset casino --model xgb --feature-policy reduced
"""

import argparse
from pathlib import Path

from ingestion import ingest
from preprocessing import vector_group_key, log1p_continuous
from feature_selection import select_features
from train_eval import repeated_cv
from config import CV_CONFIG, LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES


def run_one(dataset_key: str, model_key: str, data_dir: str, feature_policy: str, out_dir: Path):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy=feature_policy)
    print(f"[pipeline] Using {len(feat_cols)} features (policy='{feature_policy}')")

    groups = vector_group_key(ds.df, feat_cols)
    X = ds.df[feat_cols].values.astype(float)
    if LOG1P_CONTINUOUS_FEATURES:
        X = log1p_continuous(X, feat_cols, BINARY_FEATURES)
    y = ds.y.values

    cv_df = repeated_cv(X, y, groups, model_key,
                         repeats=CV_CONFIG["repeats"], n_splits=CV_CONFIG["n_splits"])

    out_path = out_dir / f"{dataset_key}_{model_key}_cv_results.csv"
    cv_df.to_csv(out_path, index=False)

    summary = (f"{ds.display_name} / {model_key}: "
               f"F1={cv_df['f1'].mean():.4f}+/-{cv_df['f1'].std():.4f}  "
               f"AUROC={cv_df['auroc'].mean():.4f}+/-{cv_df['auroc'].std():.4f}  "
               f"Precision={cv_df['precision'].mean():.4f}  "
               f"Recall={cv_df['recall'].mean():.4f}  "
               f"FPR={cv_df['fpr'].mean():.4f}")
    print(f"\n[SUMMARY] {summary}\n")
    return cv_df, summary


def main():
    p = argparse.ArgumentParser(description="Milestone 3 - Step 7 pipeline")
    p.add_argument("--dataset", choices=["camlds", "casino", "both"], default="both")
    p.add_argument("--model", choices=["rf", "xgb", "cnn", "iforest", "all"], default="all")
    p.add_argument("--feature-policy", choices=["full", "reduced"], default="full")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["camlds", "casino"] if args.dataset == "both" else [args.dataset]
    models = ["rf", "xgb", "cnn", "iforest"] if args.model == "all" else [args.model]

    all_summaries = []
    for ds_key in datasets:
        for model_key in models:
            print("=" * 70, flush=True)
            print(f"Dataset={ds_key}  Model={model_key}", flush=True)
            print("=" * 70, flush=True)
            try:
                _, summary = run_one(ds_key, model_key, args.data_dir, args.feature_policy, out_dir)
                all_summaries.append(summary)
            except FileNotFoundError as e:
                print(f"[pipeline] SKIPPED ({ds_key}/{model_key}): {e}")

    with open(out_dir / "run_summary.txt", "w") as f:
        f.write("\n".join(all_summaries))
    print("\n".join(all_summaries))


if __name__ == "__main__":
    main()
